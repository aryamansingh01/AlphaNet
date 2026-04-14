"""Equity momentum strategies."""

from typing import Optional

import pandas as pd
import numpy as np


class MomentumStrategy:
    """Cross-sectional and time-series momentum signals."""

    def __init__(self, lookback: int = 63, hold: int = 21):
        self.lookback = lookback
        self.hold = hold

    def time_series_momentum(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Generate time-series momentum signals (long if positive, flat if negative)."""
        returns = prices.pct_change(self.lookback)
        signals = (returns > 0).astype(int)
        return signals

    def cross_sectional_momentum(
        self, prices: pd.DataFrame, top_n: int = 5
    ) -> pd.DataFrame:
        """Rank assets by momentum, go long top N."""
        returns = prices.pct_change(self.lookback)
        ranks = returns.rank(axis=1, ascending=False)
        signals = (ranks <= top_n).astype(int)
        # Equal weight among selected
        signals = signals.div(signals.sum(axis=1), axis=0).fillna(0)
        return signals

    def mean_reversion(
        self,
        prices: pd.DataFrame,
        z_window: int = 21,
        z_threshold: float = 2.0,
    ) -> pd.DataFrame:
        """Z-score based mean reversion signals.

        When an asset's z-score drops below -threshold, go long (expect reversion up).
        When z-score rises above +threshold, go short / underweight (expect reversion down).
        Positions are sized inversely to the z-score magnitude for gradual entry.

        Args:
            prices: DataFrame of asset prices (index=dates, cols=tickers).
            z_window: Rolling window for computing z-scores.
            z_threshold: Absolute z-score level that triggers a signal.

        Returns:
            DataFrame of signal weights in [-1, 1].
        """
        # Shift by 1 to avoid look-ahead: stats are from [t-window, t-1]
        rolling_mean = prices.rolling(z_window).mean().shift(1)
        rolling_std = prices.rolling(z_window).std().shift(1)
        z_scores = (prices - rolling_mean) / rolling_std.replace(0, np.nan)

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        signals[z_scores <= -z_threshold] = 1.0
        signals[z_scores >= z_threshold] = -1.0

        # Fractional signals in the band between 1 and 2 standard deviations
        partial_long = (z_scores < -1.0) & (z_scores > -z_threshold)
        partial_short = (z_scores > 1.0) & (z_scores < z_threshold)
        signals[partial_long] = (-z_scores[partial_long] - 1.0) / (z_threshold - 1.0)
        signals[partial_short] = (-z_scores[partial_short] + 1.0) / (z_threshold - 1.0)

        return signals.fillna(0.0)

    def sector_rotation(
        self,
        sector_prices: pd.DataFrame,
        top_n: int = 3,
        bottom_n: int = 3,
        overweight: float = 0.6,
        underweight: float = 0.1,
    ) -> pd.DataFrame:
        """Rank sectors by momentum, overweight top performers, underweight bottom.

        Allocates ``overweight`` equally among the top-N sectors, ``underweight``
        equally among the bottom-N, and the residual equally among the middle.

        Args:
            sector_prices: DataFrame of sector ETF/index prices (cols=sector tickers).
            top_n: Number of top-momentum sectors to overweight.
            bottom_n: Number of bottom-momentum sectors to underweight.
            overweight: Total weight allocated to top sectors (split equally).
            underweight: Total weight allocated to bottom sectors (split equally).

        Returns:
            DataFrame of sector weights that sum to ~1.0 on each row.
        """
        n_sectors = sector_prices.shape[1]
        if top_n + bottom_n > n_sectors:
            raise ValueError("top_n + bottom_n exceeds number of available sectors")

        momentum_returns = sector_prices.pct_change(self.lookback)
        ranks = momentum_returns.rank(axis=1, ascending=False)

        weights = pd.DataFrame(0.0, index=sector_prices.index, columns=sector_prices.columns)

        is_top = ranks <= top_n
        is_bottom = ranks > (n_sectors - bottom_n)
        is_middle = ~is_top & ~is_bottom
        n_middle = n_sectors - top_n - bottom_n

        middle_weight = max(1.0 - overweight - underweight, 0.0)

        weights[is_top] = overweight / top_n
        weights[is_bottom] = underweight / bottom_n
        if n_middle > 0:
            weights[is_middle] = middle_weight / n_middle

        return weights.fillna(0.0)
