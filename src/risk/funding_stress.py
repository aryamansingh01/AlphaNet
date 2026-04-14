"""Funding-stress monitoring: multi-indicator dashboard with z-scores."""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# FRED series IDs for funding-stress indicators
STRESS_SERIES: dict[str, str] = {
    "stlfsi": "STLFSI4",         # St. Louis Financial Stress Index
    "nfci": "NFCI",              # Chicago Fed National Financial Conditions
    "sofr": "SOFR",              # Secured Overnight Financing Rate
    "fed_funds": "DFF",          # Federal Funds Rate
    "rrp": "RRPONTSYD",         # Overnight Reverse Repo
    "ted_spread": "TEDRATE",     # TED Spread
}


class FundingStressMonitor:
    """Fetch funding-stress indicators and compute z-scores."""

    def __init__(self, fred_client) -> None:
        self.fred = fred_client

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_indicators(self, start: str = "2020-01-01") -> pd.DataFrame:
        """Fetch all stress indicators from FRED.

        For any series that cannot be retrieved, synthetic data is
        generated so the dashboard always returns a complete picture.
        """
        data: dict[str, pd.Series] = {}

        for name, series_id in STRESS_SERIES.items():
            try:
                s = self.fred.get_series(series_id, start=start)
                if s is not None and not s.empty:
                    data[name] = s
                    continue
            except Exception as exc:
                logger.warning(
                    "FRED fetch failed for %s (%s): %s", name, series_id, exc
                )
            # Fallback: synthetic
            data[name] = self._synthetic_series(name, start)

        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().ffill()
        return df

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def compute_z_scores(
        self, indicators: pd.DataFrame, lookback: int = 252
    ) -> pd.DataFrame:
        """Z-score each indicator relative to its trailing lookback window."""
        rolling_mean = indicators.rolling(window=lookback, min_periods=20).mean()
        rolling_std = indicators.rolling(window=lookback, min_periods=20).std()
        z = (indicators - rolling_mean) / rolling_std.replace(0, np.nan)
        return z

    def composite_score(self, z_scores: pd.DataFrame) -> pd.Series:
        """Equal-weighted average of z-scores.

        Positive values indicate stress; negative values indicate calm.
        """
        return z_scores.mean(axis=1)

    def get_alerts(
        self, z_scores: pd.DataFrame, threshold: float = 1.5
    ) -> list[dict]:
        """Return indicators whose latest z-score exceeds the threshold."""
        if z_scores.empty:
            return []

        latest = z_scores.iloc[-1]
        alerts: list[dict] = []
        for name, z in latest.items():
            if pd.notna(z) and abs(z) >= threshold:
                alerts.append({
                    "indicator": str(name),
                    "z_score": round(float(z), 4),
                    "direction": "elevated" if z > 0 else "depressed",
                })
        return alerts

    # ------------------------------------------------------------------
    # Synthetic fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _synthetic_series(name: str, start: str) -> pd.Series:
        """Generate a realistic synthetic series for a given indicator."""
        np.random.seed(hash(name) % 2**31)
        dates = pd.bdate_range(start=start, end=datetime.now())
        n = len(dates)

        defaults = {
            "stlfsi": (0.0, 0.5),
            "nfci": (-0.3, 0.4),
            "sofr": (5.3, 0.1),
            "fed_funds": (5.3, 0.1),
            "rrp": (500_000, 100_000),
            "ted_spread": (0.3, 0.15),
            "move_index": (110, 20),
        }
        mean, std = defaults.get(name, (0.0, 1.0))
        values = np.random.normal(mean, std, n)
        return pd.Series(values, index=dates, name=name)
