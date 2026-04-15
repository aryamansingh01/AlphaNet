"""Signal API endpoints."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.agents.council import SignalCouncil
from src.api.routes._data_helper import fetch_market_data

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/signals", tags=["signals"])

SIGNAL_LOG_PATH = Path("logs/signal_log.json")


def _ensure_log_file() -> Path:
    SIGNAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not SIGNAL_LOG_PATH.exists():
        SIGNAL_LOG_PATH.write_text("[]")
    return SIGNAL_LOG_PATH


def _append_signal(entry: dict) -> None:
    path = _ensure_log_file()
    try:
        history = json.loads(path.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        history = []
    history.append(entry)
    history = history[-100:]
    path.write_text(json.dumps(history, indent=2, default=str))


@router.get("/")
async def list_signals():
    """Run the signal council with latest market data, return signal and all opinions."""
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
        _append_signal(entry)

        return {"signal": result, "status": "ok"}
    except Exception as exc:
        logger.exception("Failed to generate signals")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/history")
async def signal_history(days: int = 30):
    """Return signal history from the JSON log."""
    try:
        path = _ensure_log_file()
        history = json.loads(path.read_text())
        if days and history:
            cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
            filtered = []
            for entry in history:
                try:
                    ts = datetime.fromisoformat(entry.get("timestamp", "")).timestamp()
                    if ts >= cutoff:
                        filtered.append(entry)
                except (ValueError, TypeError):
                    filtered.append(entry)
            history = filtered
        return {"history": history, "days": days, "count": len(history)}
    except Exception as exc:
        logger.exception("Failed to load signal history")
        raise HTTPException(status_code=500, detail=str(exc))
