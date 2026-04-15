"""Fixed-income API endpoints — bond pricing, portfolio duration, curve signals,
carry/roll-down analysis, and term-premium decomposition."""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.curve.duration import (
    bond_price,
    convexity,
    dv01,
    modified_duration,
    portfolio_duration,
    price_change_estimate,
)
from src.curve.carry import CarryRollDownAnalyzer
from src.curve.term_premium import TermPremiumEngine
from src.curve.yield_curve import YieldCurveEngine, MATURITY_LABELS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/fixed-income", tags=["fixed-income"])

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class BondPriceRequest(BaseModel):
    face: float = 1000.0
    coupon_rate: float = 0.05
    ytm: float = 0.04
    periods: int = 10


class BondInPortfolio(BaseModel):
    face: float = 1000.0
    coupon_rate: float = 0.05
    ytm: float = 0.04
    periods: int = 10
    quantity: int = 1


class PortfolioDurationRequest(BaseModel):
    bonds: list[BondInPortfolio]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

_BPS_SCENARIOS = [-100, -50, -25, 25, 50, 100]


@router.post("/bond-price")
async def calc_bond_price(request: BondPriceRequest):
    """Calculate bond price, duration, convexity, DV01 and price-change estimates."""
    try:
        px = bond_price(request.face, request.coupon_rate, request.ytm, request.periods)
        dur = modified_duration(request.face, request.coupon_rate, request.ytm, request.periods)
        cvx = convexity(request.face, request.coupon_rate, request.ytm, request.periods)
        d01 = dv01(request.face, request.coupon_rate, request.ytm, request.periods)

        estimates = []
        for bps in _BPS_SCENARIOS:
            pct = price_change_estimate(dur, cvx, bps)
            estimates.append({"bps": bps, "pct_change": round(pct, 6)})

        return {
            "price": round(px, 4),
            "modified_duration": round(dur, 4),
            "convexity": round(cvx, 4),
            "dv01": round(d01, 4),
            "price_change_estimates": estimates,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.exception("Bond price calculation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/portfolio-duration")
async def calc_portfolio_duration(request: PortfolioDurationRequest):
    """Calculate weighted portfolio duration and per-bond details."""
    try:
        bonds_dicts = [b.model_dump() for b in request.bonds]

        port_dur = portfolio_duration(bonds_dicts)

        individual = []
        for b in bonds_dicts:
            px = bond_price(b["face"], b["coupon_rate"], b["ytm"], b["periods"])
            dur = modified_duration(b["face"], b["coupon_rate"], b["ytm"], b["periods"])
            mv = px * b["quantity"]
            individual.append({
                "face": b["face"],
                "coupon_rate": b["coupon_rate"],
                "ytm": b["ytm"],
                "periods": b["periods"],
                "quantity": b["quantity"],
                "duration": round(dur, 4),
                "price": round(px, 4),
                "market_value": round(mv, 4),
            })

        return {
            "portfolio_duration": round(port_dur, 4),
            "individual_bonds": individual,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.exception("Portfolio duration calculation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/curve-signals")
async def curve_signals():
    """Generate flattener/steepener and IG/HY rotation signals.

    Uses FRED data when available; falls back to synthetic data.
    """
    try:
        from src.strategies.credit.curve_strategy import (
            CreditRotationStrategy,
            CurveStrategy,
        )

        # --- Curve slope data ---
        try:
            from src.api.routes._data_helper import get_fred_client
            client = get_fred_client()
            yield_df = client.get_treasury_curve()
            # 10y - 2y slope
            if "10Y" in yield_df.columns and "2Y" in yield_df.columns:
                curve_slope = (yield_df["10Y"] - yield_df["2Y"]).dropna()
            else:
                curve_slope = (yield_df.iloc[:, -2] - yield_df.iloc[:, 3]).dropna()

            spreads_df = client.get_credit_spreads()
            ig_spread = spreads_df["IG_OAS"].dropna()
            hy_spread = spreads_df["HY_OAS"].dropna()
        except Exception as exc:
            logger.warning("FRED unavailable, using synthetic curve data: %s", exc)
            n = 504
            dates = pd.bdate_range(end=datetime.now(), periods=n)
            np.random.seed(42)
            curve_slope = pd.Series(np.random.normal(0.5, 0.3, n), index=dates)
            hy_spread = pd.Series(np.random.normal(4.5, 0.5, n), index=dates).clip(lower=1)
            ig_spread = pd.Series(np.random.normal(1.2, 0.2, n), index=dates).clip(lower=0.3)

        # Flattener/steepener signal
        cs = CurveStrategy()
        flat_steep = cs.flattener_steepener(curve_slope)
        last_signal = float(flat_steep.dropna().iloc[-1]) if len(flat_steep.dropna()) > 0 else 0.0

        z_raw = (curve_slope - curve_slope.rolling(252).mean()) / curve_slope.rolling(252).std()
        last_z = float(z_raw.dropna().iloc[-1]) if len(z_raw.dropna()) > 0 else 0.0

        signal_label = "steepener" if last_signal > 0 else ("flattener" if last_signal < 0 else "neutral")

        # IG/HY rotation signal
        crs = CreditRotationStrategy()
        rotation = crs.ig_hy_rotation(ig_spread, hy_spread)
        hyg_weight = float(rotation["HYG"].dropna().iloc[-1]) if len(rotation["HYG"].dropna()) > 0 else 0.5
        lqd_weight = float(rotation["LQD"].dropna().iloc[-1]) if len(rotation["LQD"].dropna()) > 0 else 0.5

        return {
            "flattener_steepener": {
                "signal": signal_label,
                "z_score": round(last_z, 4),
            },
            "ig_hy_rotation": {
                "hyg_weight": round(hyg_weight, 4),
                "lqd_weight": round(lqd_weight, 4),
            },
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.exception("Curve signals failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Carry & Roll-Down
# ---------------------------------------------------------------------------

_carry_cache: dict = {"data": None, "ts": 0.0}
_CARRY_TTL = 600


@router.get("/carry-rolldown")
async def carry_rolldown():
    """Compute carry, roll-down, total return, and breakeven for each tenor."""
    import time as _t

    now = _t.monotonic()
    if _carry_cache["data"] is not None and (now - _carry_cache["ts"]) < _CARRY_TTL:
        return _carry_cache["data"]

    try:
        # Fetch treasury yields
        try:
            from src.api.routes._data_helper import get_fred_client

            client = get_fred_client()
            curve_df = client.get_treasury_curve()
            latest = curve_df.iloc[-1]
            yields_dict = {label: float(latest[label]) for label in MATURITY_LABELS}
        except Exception as exc:
            logger.warning("FRED yields unavailable, using synthetic: %s", exc)
            synthetic = [5.3, 5.25, 5.1, 4.9, 4.5, 4.3, 4.2, 4.25, 4.3, 4.5, 4.6]
            yields_dict = dict(zip(MATURITY_LABELS, synthetic))

        # Fetch SOFR rate
        sofr_rate = 5.3  # fallback default
        try:
            from src.api.routes._data_helper import get_fred_client

            client = get_fred_client()
            sofr_series = client.get_series("SOFR")
            if sofr_series is not None and not sofr_series.empty:
                sofr_rate = float(sofr_series.dropna().iloc[-1])
        except Exception as exc:
            logger.warning("SOFR fetch failed, using fallback %.2f: %s", sofr_rate, exc)

        # Fit Nelson-Siegel curve
        engine = YieldCurveEngine()
        yields_array = np.array([yields_dict[label] for label in MATURITY_LABELS])
        ns_curve = engine.fit_nelson_siegel(yields_array)

        # Analyse carry and roll-down
        analyzer = CarryRollDownAnalyzer()
        analysis = analyzer.compute(yields_dict, sofr_rate, ns_curve)

        result = {
            "sofr_rate": round(sofr_rate, 4),
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
        }
        _carry_cache["data"] = result
        _carry_cache["ts"] = now
        return result
    except Exception as exc:
        logger.exception("Carry/roll-down analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Term Premium Decomposition
# ---------------------------------------------------------------------------

_tp_cache: dict = {"data": None, "ts": 0.0}
_TP_TTL = 600


@router.get("/term-premium")
async def term_premium():
    """Decompose 10Y yield into expected rate and term premium."""
    import time as _t

    now = _t.monotonic()
    if _tp_cache["data"] is not None and (now - _tp_cache["ts"]) < _TP_TTL:
        return _tp_cache["data"]

    try:
        tp_engine = TermPremiumEngine()
        acm_df = tp_engine.fetch_acm_data()

        # 10Y term premium column
        tp_col = "ACMTP10"
        if tp_col not in acm_df.columns:
            # Pick the last column available
            tp_col = acm_df.columns[-1]
        term_premium_10y = acm_df[tp_col]

        # Fetch 10Y yield from FRED
        try:
            from src.api.routes._data_helper import get_fred_client

            client = get_fred_client()
            ten_year = client.get_series("DGS10")
        except Exception as exc:
            logger.warning("FRED 10Y unavailable, using synthetic: %s", exc)
            np.random.seed(77)
            dates = pd.bdate_range(start="2010-01-04", end=datetime.now())
            ten_year = pd.Series(
                np.random.normal(3.5, 0.8, len(dates)), index=dates
            ).clip(lower=0.5)

        decomp = tp_engine.decompose(ten_year, term_premium_10y)

        # Build history (last 252 business days)
        exp_rate = decomp["expected_rate"]
        tp_series = decomp["term_premium"]
        yld_series = decomp["ten_year_yield"]
        history_records = []
        for dt in exp_rate.index[-252:]:
            history_records.append({
                "date": str(dt.date()) if hasattr(dt, "date") else str(dt),
                "yield": round(float(yld_series.loc[dt]), 4),
                "term_premium": round(float(tp_series.loc[dt]), 4),
                "expected_rate": round(float(exp_rate.loc[dt]), 4),
            })

        result = {
            "latest": decomp["latest"],
            "history": history_records,
            "timestamp": datetime.now().isoformat(),
        }
        _tp_cache["data"] = result
        _tp_cache["ts"] = now
        return result
    except Exception as exc:
        logger.exception("Term premium decomposition failed")
        raise HTTPException(status_code=500, detail=str(exc))
