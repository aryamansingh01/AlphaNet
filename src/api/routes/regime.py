"""Regime API endpoints."""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException

from src.api.routes._data_helper import fetch_market_data

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/regime", tags=["regime"])


@router.get("/current")
async def current_regime():
    """Fetch latest data, run regime detector, return current regime."""
    try:
        from src.regime.detector import RegimeDetector

        data = fetch_market_data(period="2y", min_rows=100)
        detector = RegimeDetector(n_regimes=3)
        features = detector.build_features(
            equity_returns=data["equity_returns"],
            credit_spreads=data["hy_spread"],
            vix=data["vix"],
            curve_slope=data["curve_slope"],
        )

        if len(features) < 3:
            return {
                "regime": "unknown",
                "timestamp": datetime.now().isoformat(),
                "n_observations": len(features),
                "note": "insufficient data for regime detection",
            }

        detector.fit_gmm(features)
        current = detector.get_current_regime(features)

        return {
            "regime": current.value,
            "timestamp": datetime.now().isoformat(),
            "n_observations": len(features),
        }
    except Exception as exc:
        logger.exception("Failed to detect regime")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/history")
async def regime_history(days: int = 252):
    """Return regime labels over time."""
    try:
        from src.regime.detector import RegimeDetector

        data = fetch_market_data(period="2y", min_rows=100)
        detector = RegimeDetector(n_regimes=3)
        features = detector.build_features(
            equity_returns=data["equity_returns"],
            credit_spreads=data["hy_spread"],
            vix=data["vix"],
            curve_slope=data["curve_slope"],
        )

        if len(features) < 3:
            return {"history": [], "days": days, "count": 0}

        labels = detector.fit_gmm(features)

        named_labels = []
        for lbl in labels:
            regime = detector.regime_map.get(int(lbl))
            named_labels.append(regime.value if regime else "unknown")

        history = [
            {"date": str(features.index[i]), "regime": named_labels[i]}
            for i in range(max(0, len(named_labels) - days), len(named_labels))
        ]

        return {"history": history, "days": days, "count": len(history)}
    except Exception as exc:
        logger.exception("Failed to compute regime history")
        raise HTTPException(status_code=500, detail=str(exc))
