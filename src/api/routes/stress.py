"""Stress testing API endpoints."""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.risk.stress_test import StressTestEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/stress", tags=["stress"])

_engine = StressTestEngine()


class HistoricalRequest(BaseModel):
    weights: dict[str, float]


class CustomRequest(BaseModel):
    weights: dict[str, float]
    rate_shock_bps: float = 0
    spread_shock_bps: float = 0
    equity_shock_pct: float = 0


@router.post("/historical")
async def stress_historical(req: HistoricalRequest):
    """Run all historical stress scenarios on a portfolio."""
    try:
        if not req.weights:
            raise HTTPException(status_code=400, detail="weights must not be empty")

        results = _engine.run_all_scenarios(req.weights)
        worst = results[0] if results else None
        best = results[-1] if results else None

        return {
            "scenarios": results,
            "worst_scenario": worst["scenario"] if worst else None,
            "best_scenario": best["scenario"] if best else None,
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Historical stress test failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/custom")
async def stress_custom(req: CustomRequest):
    """Run a custom stress scenario with factor shocks."""
    try:
        if not req.weights:
            raise HTTPException(status_code=400, detail="weights must not be empty")

        result = _engine.run_custom(
            portfolio_weights=req.weights,
            rate_shock_bps=req.rate_shock_bps,
            spread_shock_bps=req.spread_shock_bps,
            equity_shock_pct=req.equity_shock_pct,
        )
        result["timestamp"] = datetime.now().isoformat()
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Custom stress test failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/scenarios")
async def stress_scenarios():
    """List available historical stress scenarios."""
    try:
        return {
            "scenarios": _engine.list_scenarios(),
            "count": len(_engine.list_scenarios()),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.exception("Scenario listing failed")
        raise HTTPException(status_code=500, detail=str(exc))
