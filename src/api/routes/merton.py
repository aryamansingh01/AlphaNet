"""Merton credit-equity linkage API endpoints."""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.risk.merton import MertonModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/merton", tags=["merton"])

_model = MertonModel()


class AnalyzeRequest(BaseModel):
    ticker: str


@router.post("/analyze")
async def merton_analyze(req: AnalyzeRequest):
    """Run Merton structural credit model for a single ticker."""
    try:
        result = _model.analyze_ticker(req.ticker)

        # Market comparison
        leverage = result.get("leverage", 0)
        sector = "HY" if leverage > 0.6 else "IG"
        comparison = _model.compare_to_market_spread(
            result.get("implied_spread_bps", 0), sector=sector,
        )

        return {
            **result,
            "market_comparison": comparison,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.exception("Merton analyze failed for %s", req.ticker)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sample")
async def merton_sample():
    """Return pre-computed Merton results for 5 large companies."""
    try:
        companies = _model.sample_companies()
        enriched = []
        for c in companies:
            leverage = c.get("leverage", 0)
            sector = "HY" if leverage > 0.6 else "IG"
            comparison = _model.compare_to_market_spread(
                c.get("implied_spread_bps", 0), sector=sector,
            )
            enriched.append({**c, "market_comparison": comparison})

        return {
            "companies": enriched,
            "count": len(enriched),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.exception("Merton sample failed")
        raise HTTPException(status_code=500, detail=str(exc))
