"""Cross-asset signal API endpoints."""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.routes import _data_helper
from src.api.routes._data_helper import fetch_market_data
from src.risk.correlation import CorrelationRegimeTracker
from src.strategies.cross_asset.divergence import CrossAssetSignals

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cross-asset", tags=["cross-asset"])


def _classify_regime(score: float) -> str:
    if score > 0.3:
        return "risk_on"
    elif score < -0.3:
        return "risk_off"
    return "neutral"


@router.get("/signals")
async def cross_asset_signals():
    """Current cross-asset signal summary."""
    try:
        data = fetch_market_data(period="6mo")
        engine = CrossAssetSignals()

        # Credit-equity divergence: use 21-day rolling average for a smoother signal
        div_raw = engine.credit_equity_divergence(
            data["equity_returns"], data["hy_spread"],
        )
        div_clean = div_raw.dropna()
        if len(div_clean) > 21:
            div_score = float(div_clean.rolling(21).mean().iloc[-1])
        elif len(div_clean) > 0:
            div_score = float(div_clean.mean())
        else:
            div_score = 0.0

        # Classify divergence
        if div_score < -0.3:
            div_label = "BEARISH (credit leading down)"
        elif div_score > 0.3:
            div_label = "BULLISH (credit leading up)"
        elif div_score < -0.1:
            div_label = "MILDLY BEARISH"
        elif div_score > 0.1:
            div_label = "MILDLY BULLISH"
        else:
            div_label = "NO DIVERGENCE"

        # Composite risk-on/risk-off score
        composite = engine.risk_on_off_composite(
            data["equity_returns"],
            data["hy_spread"],
            data["vix"],
            data["curve_slope"],
        )
        comp_clean = composite.dropna()
        score_latest = float(comp_clean.iloc[-1]) if len(comp_clean) > 0 else 0.0

        # VIX
        vix_latest = float(data["vix"].dropna().iloc[-1]) if len(data["vix"].dropna()) > 0 else 0.0

        # Equity momentum (21d)
        eq_ret = data["equity_returns"]
        eq_mom_21 = float(eq_ret.iloc[-21:].sum()) if len(eq_ret) >= 21 else 0.0

        # Spread momentum (21d)
        hy = data["hy_spread"]
        spread_chg = float(hy.pct_change(21).dropna().iloc[-1]) if len(hy) > 21 else 0.0

        return {
            "credit_equity_divergence": round(div_score, 3),
            "divergence_label": div_label,
            "risk_on_off_score": round(score_latest, 3),
            "regime": _classify_regime(score_latest),
            "vix": round(vix_latest, 2),
            "equity_momentum_21d": round(eq_mom_21, 4),
            "hy_spread_change_21d": round(spread_chg, 4),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.exception("Cross-asset signals failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/history")
async def cross_asset_history(days: int = 252):
    """Cross-asset risk-on/off composite score history."""
    try:
        data = fetch_market_data(period="2y")
        engine = CrossAssetSignals()

        composite = engine.risk_on_off_composite(
            data["equity_returns"],
            data["hy_spread"],
            data["vix"],
            data["curve_slope"],
        )

        composite = composite.dropna().tail(days)

        history = [
            {"date": str(d.date()) if hasattr(d, "date") else str(d), "score": round(float(v), 4)}
            for d, v in composite.items()
        ]

        return {
            "history": history,
            "days": days,
            "count": len(history),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.exception("Cross-asset history failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Correlation regime endpoints
# ---------------------------------------------------------------------------

_corr_tracker = CorrelationRegimeTracker()

_etf_cache: dict = {}  # key -> data
_etf_cache_ts: dict = {}  # key -> timestamp
_ETF_CACHE_TTL = 300  # 5 minutes


def _fetch_etf_returns(period: str = "2y") -> pd.DataFrame:
    """Fetch ETF returns for correlation analysis, with synthetic fallback."""
    import time as _t
    now = _t.monotonic()
    cache_key = period
    if cache_key in _etf_cache and (now - _etf_cache_ts.get(cache_key, 0)) < _ETF_CACHE_TTL:
        return _etf_cache[cache_key]

    etfs = _corr_tracker.ASSET_ETFS

    if not _data_helper._yfinance_broken:
        try:
            import yfinance as yf
            from src.api.routes._data_helper import suppress_yfinance

            tickers = list(etfs.values())
            with suppress_yfinance():
                data = yf.download(tickers, period=period, progress=False)
                if isinstance(data.columns, pd.MultiIndex):
                    prices = data["Close"]
                else:
                    prices = data[["Close"]]
                    prices.columns = tickers

            if not prices.empty and len(prices) >= 60:
                returns = prices.pct_change().dropna()
                if len(returns) >= 30:
                    _etf_cache[cache_key] = returns
                    _etf_cache_ts[cache_key] = now
                    return returns
        except Exception:
            logger.warning("yfinance ETF download failed, using synthetic data")

    # Synthetic fallback
    rng = np.random.default_rng(99)
    n = 504
    dates = pd.bdate_range(end=datetime.now(), periods=n)
    synthetic = {}
    for name, ticker in etfs.items():
        if ticker in ("SPY", "QQQ"):
            synthetic[ticker] = rng.normal(0.0004, 0.012, n)
        elif ticker in ("TLT", "IEF"):
            synthetic[ticker] = rng.normal(0.0001, 0.008, n)
        elif ticker == "HYG":
            synthetic[ticker] = rng.normal(0.0002, 0.005, n)
        elif ticker == "GLD":
            synthetic[ticker] = rng.normal(0.0003, 0.009, n)
        else:
            synthetic[ticker] = rng.normal(0.0001, 0.004, n)
    result_df = pd.DataFrame(synthetic, index=dates)
    _etf_cache[cache_key] = result_df
    _etf_cache_ts[cache_key] = now
    return result_df


@router.get("/correlations")
async def cross_asset_correlations():
    """Current cross-asset correlation matrix and regime analysis."""
    try:
        returns = _fetch_etf_returns(period="2y")

        matrix = _corr_tracker.current_correlation_matrix(returns, window=63)
        matrix_dict = {
            row: {col: round(float(matrix.loc[row, col]), 4) for col in matrix.columns}
            for row in matrix.index
        }

        # Stock-bond correlation and regime
        try:
            sb_corr = _corr_tracker.stock_bond_correlation(returns, window=63)
            sb_current = round(float(sb_corr.iloc[-1]), 4) if len(sb_corr) > 0 else 0.0
            corr_regime = _corr_tracker.detect_correlation_regime(sb_corr)
        except ValueError:
            sb_current = 0.0
            corr_regime = {"regime": "unknown", "current_corr": 0.0,
                           "lookback_avg": 0.0, "percentile": 50.0}

        # PCA risk concentration
        pca = _corr_tracker.pca_risk_concentration(returns, window=63)

        return {
            "matrix": matrix_dict,
            "stock_bond_corr": sb_current,
            "correlation_regime": corr_regime,
            "pca": {
                "n_factors_90pct": pca["n_components_90pct"],
                "pc1_share": pca["first_component_share"],
                "explained_variance": pca["explained_variance"],
            },
            "source": "synthetic" if _data_helper._yfinance_broken else "live",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.exception("Cross-asset correlations failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/correlation-history")
async def cross_asset_correlation_history(days: int = 252):
    """Stock-bond correlation time series and regime change history."""
    try:
        returns = _fetch_etf_returns(period="2y")

        try:
            sb_corr = _corr_tracker.stock_bond_correlation(returns, window=63)
        except ValueError:
            return {
                "history": [],
                "regime_changes": [],
                "days": days,
                "timestamp": datetime.now().isoformat(),
            }

        sb_tail = sb_corr.tail(days)
        history = [
            {
                "date": str(d.date()) if hasattr(d, "date") else str(d),
                "correlation": round(float(v), 4),
            }
            for d, v in sb_tail.items()
        ]

        regime_changes = _corr_tracker.detect_regime_changes(sb_tail)

        return {
            "history": history,
            "regime_changes": regime_changes,
            "days": days,
            "count": len(history),
            "source": "synthetic" if _data_helper._yfinance_broken else "live",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.exception("Correlation history failed")
        raise HTTPException(status_code=500, detail=str(exc))
