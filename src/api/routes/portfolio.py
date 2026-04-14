"""Portfolio and paper trading API endpoints (paper_trade mode only)."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


class ExecuteRequest(BaseModel):
    """Request body for executing signals."""
    symbol: Optional[str] = None
    qty: Optional[float] = None
    side: str = "buy"
    close_all: bool = False


def _get_trader():
    """Lazy-load PaperTrader to avoid import errors when Alpaca keys are missing."""
    try:
        from src.execution.paper_trader import PaperTrader
        return PaperTrader()
    except Exception as exc:
        logger.error("Failed to initialize PaperTrader: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Paper trading unavailable. Check Alpaca API keys.",
        )


@router.get("/positions")
async def get_positions():
    """Get current paper trading positions from Alpaca."""
    try:
        trader = _get_trader()
        positions = trader.get_positions()
        return {"positions": positions, "count": len(positions)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to fetch positions")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/pnl")
async def get_pnl():
    """Get P&L from Alpaca account."""
    try:
        trader = _get_trader()
        account = trader.get_account()
        return {
            "equity": account["equity"],
            "cash": account["cash"],
            "buying_power": account["buying_power"],
            "pnl": account["pnl"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to fetch P&L")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/execute")
async def execute_signals(request: ExecuteRequest):
    """Execute signals via paper trading."""
    try:
        trader = _get_trader()

        if request.close_all:
            result = trader.close_all()
            return {"action": "close_all", "result": result}

        if not request.symbol or not request.qty:
            raise HTTPException(
                status_code=400,
                detail="Must provide symbol and qty, or set close_all=true",
            )

        result = trader.submit_order(
            symbol=request.symbol,
            qty=request.qty,
            side=request.side,
        )
        return {"action": "order_submitted", "result": result}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to execute order")
        raise HTTPException(status_code=500, detail=str(exc))
