"""Signal Council API endpoints — rule-based, no external dependencies needed."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.agents.council import SignalCouncil
from src.api.routes._data_helper import fetch_market_data

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/council", tags=["council"])

COUNCIL_LOG_PATH = Path("logs/council_log.json")


def _ensure_log_file() -> Path:
    COUNCIL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not COUNCIL_LOG_PATH.exists():
        COUNCIL_LOG_PATH.write_text("[]")
    return COUNCIL_LOG_PATH


def _append_decision(entry: dict) -> None:
    path = _ensure_log_file()
    try:
        history = json.loads(path.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        history = []
    history.append(entry)
    history = history[-100:]
    path.write_text(json.dumps(history, indent=2, default=str))


@router.post("/run")
async def run_council():
    """Run the full rule-based SignalCouncil, return detailed breakdown."""
    try:
        data = fetch_market_data()
        council = SignalCouncil()
        result = council.run(
            equity_returns=data["equity_returns"],
            hy_spread=data["hy_spread"],
            ig_spread=data["ig_spread"],
            curve_slope=data["curve_slope"],
            vix=data["vix"],
            regime=data["regime"],
        )

        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **result}
        _append_decision(entry)

        return {"council_result": result, "status": "ok"}
    except Exception as exc:
        logger.exception("Council run failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/history")
async def council_history(limit: int = 10):
    """Return past council decisions."""
    try:
        path = _ensure_log_file()
        history = json.loads(path.read_text())
        recent = history[-limit:] if len(history) > limit else history
        return {"decisions": recent, "limit": limit, "total": len(history)}
    except Exception as exc:
        logger.exception("Failed to load council history")
        raise HTTPException(status_code=500, detail=str(exc))
