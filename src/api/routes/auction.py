"""Treasury auction API endpoints."""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException

from src.api.routes._data_helper import fetch_market_data
from src.data.fixed_income.auction_client import AuctionClient
from src.risk.auction_analytics import AuctionAnalytics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auction", tags=["auction"])

_client = AuctionClient()
_analytics = AuctionAnalytics()


@router.get("/recent")
async def recent_auctions():
    """Recent Treasury auction results with demand metrics."""
    try:
        df = _client.fetch_recent_auctions(days_back=90)
        metrics = _analytics.compute_metrics(df)
        weak = _analytics.flag_weak_auctions(df)

        records = df.to_dict(orient="records") if not df.empty else []

        return {
            "auctions": records,
            "metrics": metrics,
            "weak_auctions": weak,
            "count": len(records),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.exception("Recent auctions failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/upcoming")
async def upcoming_auctions():
    """Upcoming auction calendar with demand forecast."""
    try:
        upcoming = _client.fetch_upcoming_auctions()

        # Get market data for demand forecast
        data = fetch_market_data(period="6mo")
        vix = float(data["vix"].dropna().iloc[-1]) if len(data["vix"].dropna()) > 0 else 18.0
        slope = float(data["curve_slope"].dropna().iloc[-1]) if len(data["curve_slope"].dropna()) > 0 else 0.5
        spread = float(data["hy_spread"].dropna().iloc[-1]) if len(data["hy_spread"].dropna()) > 0 else 4.5

        forecast = _analytics.predict_demand(vix, slope, spread)

        return {
            "upcoming": upcoming,
            "demand_forecast": forecast,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.exception("Upcoming auctions failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/analytics")
async def auction_analytics():
    """Computed demand analytics: bid-to-cover trends, indirect bidder trends."""
    try:
        df = _client.fetch_recent_auctions(days_back=180)
        metrics = _analytics.compute_metrics(df)

        # Build trend data for charting
        btc_trend = []
        indirect_trend = []
        if not df.empty:
            for _, row in df.iterrows():
                entry_date = str(row.get("auction_date", ""))
                btc_trend.append({
                    "date": entry_date,
                    "security_type": row.get("security_type", ""),
                    "bid_to_cover": round(float(row.get("bid_to_cover_ratio", 0)), 3),
                })
                indirect_trend.append({
                    "date": entry_date,
                    "indirect_pct": round(float(row.get("indirect_bidder_pct", 0)), 2),
                })

        # Demand forecast
        data = fetch_market_data(period="6mo")
        vix = float(data["vix"].dropna().iloc[-1]) if len(data["vix"].dropna()) > 0 else 18.0
        slope = float(data["curve_slope"].dropna().iloc[-1]) if len(data["curve_slope"].dropna()) > 0 else 0.5
        spread = float(data["hy_spread"].dropna().iloc[-1]) if len(data["hy_spread"].dropna()) > 0 else 4.5
        forecast = _analytics.predict_demand(vix, slope, spread)

        return {
            "metrics": metrics,
            "btc_trend": btc_trend,
            "indirect_trend": indirect_trend,
            "demand_forecast": forecast,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.exception("Auction analytics failed")
        raise HTTPException(status_code=500, detail=str(exc))
