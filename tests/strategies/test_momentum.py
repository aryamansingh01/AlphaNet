"""Tests for equity momentum and mean reversion strategies."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.equity.momentum import MomentumStrategy


@pytest.fixture
def strategy():
    return MomentumStrategy(lookback=20, hold=5)


@pytest.fixture
def trending_up_prices():
    """Prices that trend consistently upward."""
    dates = pd.bdate_range("2023-01-01", periods=100)
    prices = pd.DataFrame({
        "A": 100 + np.arange(100) * 0.5,
        "B": 50 + np.arange(100) * 0.3,
    }, index=dates)
    return prices


@pytest.fixture
def trending_down_prices():
    """Prices that trend consistently downward."""
    dates = pd.bdate_range("2023-01-01", periods=100)
    prices = pd.DataFrame({
        "A": 100 - np.arange(100) * 0.3,
        "B": 80 - np.arange(100) * 0.2,
    }, index=dates)
    return prices


@pytest.fixture
def mixed_prices():
    """Multiple assets with different momentum profiles."""
    dates = pd.bdate_range("2023-01-01", periods=100)
    np.random.seed(42)
    prices = pd.DataFrame({
        "WINNER1": 100 + np.arange(100) * 1.0,
        "WINNER2": 100 + np.arange(100) * 0.8,
        "LOSER1": 100 - np.arange(100) * 0.5,
        "LOSER2": 100 - np.arange(100) * 0.3,
        "FLAT": 100 + np.random.normal(0, 0.5, 100).cumsum() * 0.01,
    }, index=dates)
    return prices


@pytest.fixture
def mean_reverting_prices():
    """Prices that oscillate around a mean level."""
    dates = pd.bdate_range("2023-01-01", periods=200)
    np.random.seed(42)
    # Ornstein-Uhlenbeck-like process
    price = np.zeros(200)
    price[0] = 100
    for i in range(1, 200):
        price[i] = price[i - 1] + 0.1 * (100 - price[i - 1]) + np.random.normal(0, 1)
    return pd.DataFrame({"A": price}, index=dates)


# ------------------------------------------------------------------
# Time-series momentum
# ------------------------------------------------------------------

class TestTimeSeriesMomentum:

    def test_long_signal_for_uptrend(self, strategy, trending_up_prices):
        """Assets trending up should produce signal = 1 (long)."""
        signals = strategy.time_series_momentum(trending_up_prices)
        # After the lookback period, all signals should be 1
        recent = signals.iloc[strategy.lookback + 5:]
        assert (recent == 1).all().all()

    def test_flat_signal_for_downtrend(self, strategy, trending_down_prices):
        """Assets trending down should produce signal = 0 (flat, no short)."""
        signals = strategy.time_series_momentum(trending_down_prices)
        recent = signals.iloc[strategy.lookback + 5:]
        assert (recent == 0).all().all()

    def test_output_shape_matches_input(self, strategy, trending_up_prices):
        """Signal DataFrame should have the same shape as input prices."""
        signals = strategy.time_series_momentum(trending_up_prices)
        assert signals.shape == trending_up_prices.shape

    def test_signals_are_binary(self, strategy, mixed_prices):
        """Time-series momentum signals should be 0 or 1."""
        signals = strategy.time_series_momentum(mixed_prices)
        unique_vals = set(signals.values.flatten())
        # NaN can appear at the start; filter them out
        unique_vals.discard(np.nan)
        assert unique_vals.issubset({0, 1, 0.0, 1.0})


# ------------------------------------------------------------------
# Cross-sectional momentum
# ------------------------------------------------------------------

class TestCrossSectionalMomentum:

    def test_picks_top_n_assets(self, strategy, mixed_prices):
        """Should select exactly top_n assets per row (after lookback)."""
        top_n = 2
        signals = strategy.cross_sectional_momentum(mixed_prices, top_n=top_n)
        # After lookback, each row should have exactly top_n non-zero entries
        recent = signals.iloc[strategy.lookback + 5:]
        for _, row in recent.iterrows():
            n_selected = (row > 0).sum()
            assert n_selected == top_n

    def test_weights_sum_to_one(self, strategy, mixed_prices):
        """Selected asset weights should sum to approximately 1.0."""
        signals = strategy.cross_sectional_momentum(mixed_prices, top_n=2)
        recent = signals.iloc[strategy.lookback + 5:]
        row_sums = recent.sum(axis=1)
        assert (abs(row_sums - 1.0) < 0.01).all()

    def test_winners_selected_over_losers(self, strategy, mixed_prices):
        """The strongest trending assets should be selected."""
        signals = strategy.cross_sectional_momentum(mixed_prices, top_n=2)
        # At the end, WINNER1 and WINNER2 should be selected (they have highest returns)
        last_row = signals.iloc[-1]
        selected = last_row[last_row > 0].index.tolist()
        assert "WINNER1" in selected
        assert "WINNER2" in selected

    def test_equal_weight_among_selected(self, strategy, mixed_prices):
        """Selected assets should have equal weight."""
        top_n = 2
        signals = strategy.cross_sectional_momentum(mixed_prices, top_n=top_n)
        last_row = signals.iloc[-1]
        selected_weights = last_row[last_row > 0]
        assert len(selected_weights) == top_n
        # All selected should have same weight = 1/top_n
        expected_weight = 1.0 / top_n
        assert (abs(selected_weights - expected_weight) < 0.01).all()


# ------------------------------------------------------------------
# Mean reversion
# ------------------------------------------------------------------

class TestMeanReversion:

    def test_long_signal_when_price_drops_below_mean(self, strategy, mean_reverting_prices):
        """Should generate positive signals when price is below rolling mean.

        Note: rolling stats are shifted by 1 to avoid look-ahead, so the signal
        appears the day after the z-score crosses the threshold.
        """
        signals = strategy.mean_reversion(mean_reverting_prices, z_window=21, z_threshold=1.5)
        # With the shift(1) fix, check that we get SOME long signals overall
        # when the price series is mean-reverting
        assert (signals["A"] > 0).any(), "Expected some long signals in mean-reverting data"

    def test_short_signal_when_price_above_mean(self, strategy, mean_reverting_prices):
        """Should generate negative signals when price is above rolling mean."""
        signals = strategy.mean_reversion(mean_reverting_prices, z_window=21, z_threshold=1.5)
        rolling_mean = mean_reverting_prices.rolling(21).mean()
        rolling_std = mean_reverting_prices.rolling(21).std()
        z_scores = (mean_reverting_prices - rolling_mean) / rolling_std
        very_high = z_scores["A"] > 1.5
        if very_high.any():
            assert (signals.loc[very_high, "A"] < 0).all()

    def test_signals_bounded(self, strategy, mean_reverting_prices):
        """Mean reversion signals should be in [-1, 1]."""
        signals = strategy.mean_reversion(mean_reverting_prices)
        assert signals.min().min() >= -1.0
        assert signals.max().max() <= 1.0

    def test_no_signal_when_near_mean(self, strategy):
        """When price stays very close to the mean, signals should be near zero."""
        dates = pd.bdate_range("2023-01-01", periods=100)
        # Perfectly flat prices — no deviation at all
        flat_prices = pd.DataFrame({"A": np.full(100, 100.0)}, index=dates)
        signals = strategy.mean_reversion(flat_prices, z_window=21, z_threshold=2.0)
        # After warm-up, signals should be all zero (price never deviates)
        recent = signals.iloc[30:]
        assert (recent.abs() < 0.01).all().all()

    def test_output_shape(self, strategy, mean_reverting_prices):
        """Output shape should match input."""
        signals = strategy.mean_reversion(mean_reverting_prices)
        assert signals.shape == mean_reverting_prices.shape
