"""Backtest API endpoints."""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.backtest.engine import BacktestEngine
from src.strategies.equity.momentum import MomentumStrategy
from src.strategies.credit.curve_strategy import CurveStrategy, CreditRotationStrategy
from src.strategies.cross_asset.divergence import CrossAssetSignals

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest", tags=["backtest"])

RESULTS_CACHE_PATH = Path("logs/backtest_results.json")


class BacktestRequest(BaseModel):
    strategy: str = "momentum"
    lookback: int = 63
    top_n: int = 5
    tickers: Optional[list[str]] = None
    transaction_cost_bps: float = 5.0


class CompareRequest(BaseModel):
    strategies: list[str]
    lookback: int = 63
    tickers: Optional[list[str]] = None
    transaction_cost_bps: float = 5.0


# ---------------------------------------------------------------------------
# Synthetic credit / cross-asset data helpers
# ---------------------------------------------------------------------------

def _synthetic_credit_data(n: int, dates: pd.DatetimeIndex) -> dict:
    """Generate synthetic spread and curve data for credit strategies."""
    rng = np.random.default_rng(42)
    return {
        "hy_spread": pd.Series(rng.normal(4.5, 0.5, n), index=dates).clip(lower=1),
        "ig_spread": pd.Series(rng.normal(1.2, 0.2, n), index=dates).clip(lower=0.3),
        "curve_slope": pd.Series(rng.normal(0.5, 0.3, n), index=dates),
        "vix": pd.Series(rng.normal(18, 3, n), index=dates).clip(lower=10),
    }


def _generate_strategy_signals(
    strategy_name: str,
    prices: pd.DataFrame,
    lookback: int,
    top_n: int,
) -> pd.DataFrame:
    """Generate signal DataFrame for a given strategy name."""
    n = len(prices)
    dates = prices.index
    credit = _synthetic_credit_data(n, dates)

    if strategy_name == "momentum":
        strat = MomentumStrategy(lookback=lookback)
        return strat.time_series_momentum(prices)
    elif strategy_name == "cross_sectional":
        strat = MomentumStrategy(lookback=lookback)
        return strat.cross_sectional_momentum(prices, top_n=top_n)
    elif strategy_name == "mean_reversion":
        strat = MomentumStrategy(lookback=lookback)
        return strat.mean_reversion(prices)
    elif strategy_name == "credit_rotation":
        crs = CreditRotationStrategy()
        rotation = crs.ig_hy_rotation(credit["ig_spread"], credit["hy_spread"], lookback=lookback)
        signals = pd.DataFrame(0.0, index=dates, columns=prices.columns)
        if "LQD" in signals.columns:
            signals["LQD"] = rotation["LQD"].reindex(dates).fillna(0)
        if "HYG" in signals.columns:
            signals["HYG"] = rotation["HYG"].reindex(dates).fillna(0)
        return signals
    elif strategy_name == "curve":
        cs = CurveStrategy()
        flat_steep = cs.flattener_steepener(credit["curve_slope"])
        signals = pd.DataFrame(0.0, index=dates, columns=prices.columns)
        if "TLT" in signals.columns:
            signals["TLT"] = flat_steep.reindex(dates).fillna(0)
        if "SHY" in signals.columns:
            signals["SHY"] = (-flat_steep).reindex(dates).fillna(0)
        return signals
    elif strategy_name == "cross_asset":
        cas = CrossAssetSignals()
        equity_returns = prices.mean(axis=1).pct_change().fillna(0)
        composite = cas.risk_on_off_composite(
            equity_returns,
            credit["hy_spread"],
            credit["vix"],
            credit["curve_slope"],
        )
        signals = pd.DataFrame(0.0, index=dates, columns=prices.columns)
        # Positive composite -> more SPY, negative -> more TLT
        if "SPY" in signals.columns:
            signals["SPY"] = composite.clip(lower=0).reindex(dates).fillna(0)
        if "TLT" in signals.columns:
            signals["TLT"] = (-composite).clip(lower=0).reindex(dates).fillna(0)
        return signals
    else:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available: momentum, cross_sectional, mean_reversion, "
            f"credit_rotation, curve, cross_asset"
        )


def _ensure_cache() -> Path:
    RESULTS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_CACHE_PATH.exists():
        RESULTS_CACHE_PATH.write_text("[]")
    return RESULTS_CACHE_PATH


_price_cache: dict = {}  # key -> data
_price_cache_ts: dict = {}  # key -> timestamp


def _fetch_backtest_prices(tickers: list[str]) -> pd.DataFrame:
    """Fetch prices for backtesting. Uses cache, skips yfinance if broken."""
    global _price_cache, _price_cache_ts
    import time as _time

    cache_key = ",".join(sorted(tickers))
    now = _time.monotonic()

    # Return cached if fresh (5 min)
    if cache_key in _price_cache and (now - _price_cache_ts.get(cache_key, 0)) < 300:
        return _price_cache[cache_key]

    # Check if yfinance is working (shared flag from _data_helper)
    from src.api.routes import _data_helper
    from src.api.routes._data_helper import suppress_yfinance

    if not _data_helper._yfinance_broken:
        try:
            import yfinance as yf

            with suppress_yfinance():
                data = yf.download(tickers, period="2y")

            if not data.empty and len(data) >= 100:
                if isinstance(data.columns, pd.MultiIndex):
                    if "Close" in data.columns.get_level_values(1):
                        prices = data.xs("Close", axis=1, level=1)
                    else:
                        prices = data
                else:
                    prices = data[["Close"]]
                    prices.columns = tickers
                prices = prices.dropna()
                if len(prices) >= 100:
                    _price_cache[cache_key] = prices
                    _price_cache_ts[cache_key] = now
                    return prices
        except Exception:
            pass

    # Synthetic fallback (instant, deterministic)
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=datetime.now(), periods=504)
    prices = pd.DataFrame(index=dates)
    base_params = {
        "SPY": (0.0004, 0.012), "QQQ": (0.0005, 0.015),
        "IWM": (0.0003, 0.014), "TLT": (0.0001, 0.010),
        "GLD": (0.0002, 0.008), "HYG": (0.0002, 0.006),
        "LQD": (0.0001, 0.005),
    }
    for t in tickers:
        mu, sigma = base_params.get(t, (0.0003, 0.01))
        prices[t] = 100 * np.exp(np.cumsum(rng.normal(mu, sigma, 504)))

    _price_cache[cache_key] = prices
    _price_cache_ts[cache_key] = now
    return prices


@router.post("/run")
async def run_backtest(request: BacktestRequest):
    """Run a strategy backtest, return metrics and cumulative returns as JSON."""
    try:
        tickers = request.tickers or ["SPY", "QQQ", "IWM", "TLT", "GLD", "HYG", "LQD"]
        prices = _fetch_backtest_prices(tickers)

        try:
            signals = _generate_strategy_signals(
                request.strategy, prices, request.lookback, request.top_n,
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        engine = BacktestEngine(transaction_cost_bps=request.transaction_cost_bps)
        result = engine.run(signals, prices)

        cumulative_list = [
            {"date": str(d), "value": round(float(v), 4)}
            for d, v in result["cumulative"].items()
        ]

        response = {
            "strategy": request.strategy,
            "lookback": request.lookback,
            "tickers": tickers,
            "metrics": result["metrics"],
            "cumulative_returns": cumulative_list[-252:],
            "timestamp": datetime.now().isoformat(),
        }

        path = _ensure_cache()
        try:
            cache = json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            cache = []
        cache.append(response)
        cache = cache[-100:]
        path.write_text(json.dumps(cache, indent=2, default=str))

        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Backtest failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/results")
async def backtest_results():
    """Return cached backtest results."""
    try:
        path = _ensure_cache()
        results = json.loads(path.read_text())
        return {"results": results, "count": len(results)}
    except Exception as exc:
        logger.exception("Failed to load cached results")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/compare")
async def compare_strategies(request: CompareRequest):
    """Run 2+ strategies side by side and return a comparison table."""
    try:
        if len(request.strategies) < 2:
            raise HTTPException(
                status_code=400,
                detail="Provide at least 2 strategies to compare.",
            )

        tickers = request.tickers or ["SPY", "QQQ", "IWM", "TLT", "GLD", "HYG", "LQD"]
        prices = _fetch_backtest_prices(tickers)

        strategy_signals: dict[str, pd.DataFrame] = {}
        for name in request.strategies:
            try:
                strategy_signals[name] = _generate_strategy_signals(
                    name, prices, request.lookback, top_n=5,
                )
            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve))

        engine = BacktestEngine(transaction_cost_bps=request.transaction_cost_bps)
        comparison_df = engine.compare_strategies(strategy_signals, prices)

        comparison = []
        for strat_name, row in comparison_df.iterrows():
            entry = {"strategy": str(strat_name)}
            for col in comparison_df.columns:
                entry[col] = round(float(row[col]), 4)
            comparison.append(entry)

        return {
            "comparison": comparison,
            "tickers": tickers,
            "lookback": request.lookback,
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Strategy comparison failed")
        raise HTTPException(status_code=500, detail=str(exc))
