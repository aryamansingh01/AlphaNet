"""Treasury auction demand analytics -- metrics, flags, and demand forecasting."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AuctionAnalytics:
    """Compute auction demand metrics and flag weak results."""

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def compute_metrics(self, auctions_df: pd.DataFrame) -> dict:
        """Compute auction demand metrics.

        Returns avg bid-to-cover by type, indirect bidder trend,
        and a composite demand score.
        """
        if auctions_df.empty:
            return self._empty_metrics()

        # Average bid-to-cover by security type
        avg_btc = {}
        for sec_type in ("Bill", "Note", "Bond"):
            subset = auctions_df[auctions_df["security_type"] == sec_type]
            if not subset.empty:
                avg_btc[sec_type.lower()] = round(
                    float(subset["bid_to_cover_ratio"].mean()), 3
                )

        # Indirect bidder trend (rolling 3-auction average)
        indirect = auctions_df["indirect_bidder_pct"].dropna()
        if len(indirect) >= 3:
            indirect_trend = round(float(indirect.head(3).mean()), 2)
        else:
            indirect_trend = round(float(indirect.mean()), 2) if len(indirect) > 0 else 0.0

        # Tail estimate: deviation of high_yield from recent average
        yields = auctions_df["high_yield"].dropna()
        if len(yields) >= 3:
            recent_tail = round(float(yields.iloc[0] - yields.iloc[1:4].mean()), 3)
        else:
            recent_tail = 0.0

        # Demand score: composite of bid-to-cover z-score + indirect bidder z-score
        demand_score = self._compute_demand_score(auctions_df)

        return {
            "avg_bid_to_cover": avg_btc,
            "indirect_bidder_trend": indirect_trend,
            "recent_tail_bps": round(recent_tail * 100, 1),
            "demand_score": demand_score,
        }

    # ------------------------------------------------------------------
    # Weak auction flags
    # ------------------------------------------------------------------

    def flag_weak_auctions(
        self,
        auctions_df: pd.DataFrame,
        btc_threshold: float = 2.2,
    ) -> list[dict]:
        """Flag auctions with bid-to-cover below threshold or large tails."""
        flags: list[dict] = []
        if auctions_df.empty:
            return flags

        for _, row in auctions_df.iterrows():
            reasons: list[str] = []
            btc = row.get("bid_to_cover_ratio", 0)
            if btc and btc < btc_threshold:
                reasons.append(f"Low bid-to-cover ({btc:.2f})")

            indirect = row.get("indirect_bidder_pct", 0)
            if indirect and indirect < 50:
                reasons.append(f"Low indirect bidders ({indirect:.1f}%)")

            if reasons:
                flags.append({
                    "auction_date": str(row.get("auction_date", "")),
                    "security_type": row.get("security_type", ""),
                    "security_term": row.get("security_term", ""),
                    "bid_to_cover": round(float(btc), 2),
                    "reasons": reasons,
                })

        return flags

    # ------------------------------------------------------------------
    # Demand forecast (heuristic)
    # ------------------------------------------------------------------

    def predict_demand(
        self,
        vix: float,
        curve_slope: float,
        spread_level: float,
    ) -> dict:
        """Simple heuristic demand predictor.

        Low VIX + steep curve + tight spreads -> strong demand.
        High VIX + flat curve + wide spreads -> weak demand.
        """
        score = 0.0
        factors: list[str] = []

        # VIX component: below 15 is calm, above 25 is stressed
        if vix < 15:
            score += 1.0
            factors.append("Low VIX supports demand")
        elif vix > 25:
            score -= 1.0
            factors.append("Elevated VIX dampens demand")
        else:
            factors.append("Moderate VIX neutral for demand")

        # Curve slope: steep is positive for demand (term premium attractive)
        if curve_slope > 0.5:
            score += 1.0
            factors.append("Steep curve attracts buyers")
        elif curve_slope < -0.2:
            score -= 1.0
            factors.append("Flat/inverted curve reduces appeal")
        else:
            factors.append("Moderate slope neutral for demand")

        # Spreads: tight spreads mean risk appetite is strong
        if spread_level < 3.5:
            score += 1.0
            factors.append("Tight spreads reflect strong risk appetite")
        elif spread_level > 5.5:
            score -= 1.0
            factors.append("Wide spreads signal risk aversion")
        else:
            factors.append("Average spreads neutral for demand")

        if score >= 1.5:
            forecast = "strong"
            confidence = 0.75
        elif score <= -1.5:
            forecast = "weak"
            confidence = 0.70
        else:
            forecast = "moderate"
            confidence = 0.55

        return {
            "forecast": forecast,
            "confidence": round(confidence, 2),
            "score": round(score, 2),
            "factors": factors,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_demand_score(df: pd.DataFrame) -> dict:
        btc = df["bid_to_cover_ratio"].dropna()
        indirect = df["indirect_bidder_pct"].dropna()

        btc_z = 0.0
        if len(btc) >= 3:
            btc_z = float((btc.iloc[0] - btc.mean()) / btc.std()) if btc.std() > 0 else 0.0

        ind_z = 0.0
        if len(indirect) >= 3:
            ind_z = float((indirect.iloc[0] - indirect.mean()) / indirect.std()) if indirect.std() > 0 else 0.0

        composite = round((btc_z + ind_z) / 2, 3)
        label = "strong" if composite > 0.5 else ("weak" if composite < -0.5 else "moderate")

        return {
            "composite": composite,
            "label": label,
            "btc_z_score": round(btc_z, 3),
            "indirect_z_score": round(ind_z, 3),
        }

    @staticmethod
    def _empty_metrics() -> dict:
        return {
            "avg_bid_to_cover": {},
            "indirect_bidder_trend": 0.0,
            "recent_tail_bps": 0.0,
            "demand_score": {"composite": 0.0, "label": "unavailable", "btc_z_score": 0.0, "indirect_z_score": 0.0},
        }
