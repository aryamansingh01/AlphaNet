"""Funding-stress dashboard API endpoints."""

import logging
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/funding", tags=["funding"])

_stress_cache: dict = {"data": None, "ts": 0.0}
_history_cache: dict = {"data": None, "ts": 0.0}
_CACHE_TTL = 600  # 10 minutes


def _get_monitor():
    """Lazily construct a FundingStressMonitor with a FredClient."""
    from src.data.fixed_income.fred_client import FredClient
    from src.risk.funding_stress import FundingStressMonitor

    return FundingStressMonitor(FredClient())


@router.get("/stress")
async def funding_stress():
    """Return current funding-stress indicators, z-scores, and alerts."""
    import time as _t

    now = _t.monotonic()
    if _stress_cache["data"] is not None and (now - _stress_cache["ts"]) < _CACHE_TTL:
        return _stress_cache["data"]

    try:
        monitor = _get_monitor()
        indicators = monitor.fetch_indicators()
        z_scores = monitor.compute_z_scores(indicators)
        composite = monitor.composite_score(z_scores)
        alerts = monitor.get_alerts(z_scores)

        # Latest values
        latest_vals = indicators.iloc[-1] if not indicators.empty else pd.Series()
        latest_z = z_scores.iloc[-1] if not z_scores.empty else pd.Series()
        composite_val = float(composite.iloc[-1]) if not composite.empty else 0.0

        indicator_list = []
        for col in indicators.columns:
            val = float(latest_vals.get(col, 0)) if pd.notna(latest_vals.get(col)) else 0.0
            z = float(latest_z.get(col, 0)) if pd.notna(latest_z.get(col)) else 0.0
            indicator_list.append({
                "name": col,
                "value": round(val, 4),
                "z_score": round(z, 4),
            })

        result = {
            "composite_score": round(composite_val, 4),
            "indicators": indicator_list,
            "alerts": alerts,
            "timestamp": datetime.now().isoformat(),
        }
        _stress_cache["data"] = result
        _stress_cache["ts"] = now
        return result
    except Exception as exc:
        logger.exception("Funding stress dashboard failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/history")
async def funding_history(days: int = 252):
    """Return composite funding-stress score history."""
    import time as _t

    now = _t.monotonic()
    cache_key = f"history_{days}"
    if (
        _history_cache["data"] is not None
        and _history_cache.get("key") == cache_key
        and (now - _history_cache["ts"]) < _CACHE_TTL
    ):
        return _history_cache["data"]

    try:
        monitor = _get_monitor()
        indicators = monitor.fetch_indicators()
        z_scores = monitor.compute_z_scores(indicators)
        composite = monitor.composite_score(z_scores).dropna().tail(days)

        history = []
        for dt, score in composite.items():
            history.append({
                "date": str(dt.date()) if hasattr(dt, "date") else str(dt),
                "score": round(float(score), 4),
            })

        result = {
            "history": history,
            "days": days,
            "count": len(history),
        }
        _history_cache["data"] = result
        _history_cache["key"] = cache_key
        _history_cache["ts"] = now
        return result
    except Exception as exc:
        logger.exception("Funding stress history failed")
        raise HTTPException(status_code=500, detail=str(exc))
