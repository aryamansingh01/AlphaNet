"""Yield curve API endpoints."""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.curve.yield_curve import YieldCurveEngine, MATURITIES, MATURITY_LABELS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/curve", tags=["curve"])


_yield_cache = {"data": None, "ts": 0.0}
_spread_cache = {"data": None, "ts": 0.0}
_FRED_TTL = 600  # cache FRED for 10 min


def _fetch_treasury_yields() -> pd.Series:
    """Fetch current Treasury yields. Cached for 10 min."""
    import time as _t
    now = _t.monotonic()
    if _yield_cache["data"] is not None and (now - _yield_cache["ts"]) < _FRED_TTL:
        return _yield_cache["data"]

    try:
        from src.data.fixed_income.fred_client import FredClient
        client = FredClient()
        curve_df = client.get_treasury_curve()
        latest = curve_df.iloc[-1]
        _yield_cache["data"] = latest
        _yield_cache["ts"] = now
        return latest
    except Exception as exc:
        logger.warning("FRED fetch failed, using synthetic yields: %s", exc)
        synthetic_yields = np.array([5.3, 5.25, 5.1, 4.9, 4.5, 4.3, 4.2, 4.25, 4.3, 4.5, 4.6])
        result = pd.Series(synthetic_yields, index=MATURITY_LABELS)
        _yield_cache["data"] = result
        _yield_cache["ts"] = now
        return result


def _fetch_credit_spreads() -> dict:
    """Fetch current IG/HY spreads with z-scores. Cached for 10 min."""
    import time as _t
    now = _t.monotonic()
    if _spread_cache["data"] is not None and (now - _spread_cache["ts"]) < _FRED_TTL:
        return _spread_cache["data"]

    try:
        from src.data.fixed_income.fred_client import FredClient
        client = FredClient()
        spreads_df = client.get_credit_spreads()
        ig_series = spreads_df["IG_OAS"].dropna()
        hy_series = spreads_df["HY_OAS"].dropna()

        ig_current = float(ig_series.iloc[-1])
        hy_current = float(hy_series.iloc[-1])
        ig_z = float((ig_current - ig_series.mean()) / ig_series.std()) if ig_series.std() > 0 else 0
        hy_z = float((hy_current - hy_series.mean()) / hy_series.std()) if hy_series.std() > 0 else 0

        result = {
            "ig_oas": ig_current,
            "hy_oas": hy_current,
            "ig_z_score": round(ig_z, 2),
            "hy_z_score": round(hy_z, 2),
        }
        _spread_cache["data"] = result
        _spread_cache["ts"] = now
        return result
    except Exception as exc:
        logger.warning("FRED spread fetch failed, using synthetic: %s", exc)
        result = {
            "ig_oas": 1.2,
            "hy_oas": 4.5,
            "ig_z_score": 0.1,
            "hy_z_score": -0.3,
        }
        _spread_cache["data"] = result
        _spread_cache["ts"] = now
        return result


@router.get("/current")
async def current_curve():
    """Fetch Treasury data, fit Nelson-Siegel, return curve and metrics."""
    try:
        yields = _fetch_treasury_yields()
        engine = YieldCurveEngine()

        yields_array = np.array([float(yields[label]) for label in MATURITY_LABELS])
        ns_curve = engine.fit_nelson_siegel(yields_array)
        interpolated = engine.interpolate_curve(ns_curve)
        metrics = engine.get_curve_metrics(yields)

        return {
            "observed_yields": {label: float(yields[label]) for label in MATURITY_LABELS},
            "nelson_siegel": {
                "beta0": float(ns_curve.beta0),
                "beta1": float(ns_curve.beta1),
                "beta2": float(ns_curve.beta2),
                "tau": float(ns_curve.tau),
            },
            "fitted_curve": {
                "maturities": interpolated.index.tolist(),
                "yields": interpolated.values.tolist(),
            },
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.exception("Failed to build yield curve")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/spreads")
async def credit_spreads():
    """Return current IG/HY spreads with z-scores."""
    try:
        spreads = _fetch_credit_spreads()
        return {
            "spreads": spreads,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.exception("Failed to fetch credit spreads")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/history")
async def curve_history(days: int = 252):
    """Return curve metrics history."""
    try:
        from src.data.fixed_income.fred_client import FredClient
        client = FredClient()
        yield_df = client.get_treasury_curve()
        yield_df = yield_df.tail(days)

        engine = YieldCurveEngine()
        history = engine.get_curve_history(yield_df)

        records = []
        for date, row in history.iterrows():
            records.append({
                "date": str(date),
                "level": float(row["level"]),
                "slope": float(row["slope"]),
                "curvature": float(row["curvature"]),
                "spread_2s10s": float(row["spread_2s10s"]),
                "inverted": bool(row["inverted"]),
            })

        return {"history": records, "days": days, "count": len(records)}
    except Exception as exc:
        logger.warning("FRED history unavailable, returning empty: %s", exc)
        return {"history": [], "days": days, "count": 0, "error": str(exc)}
