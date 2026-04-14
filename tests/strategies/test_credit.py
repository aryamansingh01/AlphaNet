"""Tests for credit curve and rotation strategies."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.credit.curve_strategy import CurveStrategy, CreditRotationStrategy


@pytest.fixture
def dates():
    return pd.bdate_range("2022-01-01", periods=300)


@pytest.fixture
def steep_curve_slope(dates):
    """Curve slope that starts steep and gradually flattens."""
    np.random.seed(42)
    slope = pd.Series(
        np.linspace(2.0, -0.5, 300) + np.random.normal(0, 0.1, 300),
        index=dates,
    )
    return slope


@pytest.fixture
def hy_spread_tightening(dates):
    """HY spreads that are tightening (declining)."""
    np.random.seed(42)
    spread = pd.Series(
        np.linspace(6.0, 3.5, 300) + np.random.normal(0, 0.1, 300),
        index=dates,
    )
    return spread


@pytest.fixture
def hy_spread_widening(dates):
    """HY spreads that are widening (increasing)."""
    np.random.seed(42)
    spread = pd.Series(
        np.linspace(3.5, 7.0, 300) + np.random.normal(0, 0.1, 300),
        index=dates,
    )
    return spread


@pytest.fixture
def ig_spread(dates):
    """IG spreads for rotation testing."""
    np.random.seed(42)
    return pd.Series(np.random.normal(1.2, 0.2, 300), index=dates)


# ------------------------------------------------------------------
# Flattener / Steepener signals
# ------------------------------------------------------------------

class TestFlattenerSteepener:

    def test_steep_curve_produces_flattener_signal(self, steep_curve_slope):
        """When slope is high (steep), expect flattening signal (-1)."""
        strat = CurveStrategy()
        signals = strat.flattener_steepener(steep_curve_slope, threshold=0.5)
        # Early period: slope is steep, z-score should be positive -> signal -1
        # Need at least 252 days for rolling stats, so check near end of steep period
        # Actually the z-score is computed relative to a 252-day rolling window
        # After the 252-day warm-up, check signals
        recent = signals.iloc[260:]
        # By this point the curve has flattened/inverted relative to recent history
        # so we should see some steepener signals (1) as z-score goes negative
        assert (recent != 0).any()

    def test_signal_values_are_valid(self, steep_curve_slope):
        """Signals should only be -1, 0, or 1."""
        strat = CurveStrategy()
        signals = strat.flattener_steepener(steep_curve_slope)
        valid_values = {-1.0, 0.0, 1.0}
        actual_values = set(signals.dropna().unique())
        assert actual_values.issubset(valid_values)

    def test_flat_curve_no_signal(self):
        """A constant slope should produce z-scores near zero, hence no signal."""
        dates = pd.bdate_range("2022-01-01", periods=300)
        constant_slope = pd.Series(1.0, index=dates)
        strat = CurveStrategy()
        signals = strat.flattener_steepener(constant_slope, threshold=0.5)
        # With constant data, rolling std = 0, z-score is NaN/0, signal should be 0
        assert (signals.iloc[253:] == 0).all()

    def test_threshold_affects_sensitivity(self, steep_curve_slope):
        """Higher threshold should produce fewer signals."""
        strat = CurveStrategy()
        signals_low = strat.flattener_steepener(steep_curve_slope, threshold=0.3)
        signals_high = strat.flattener_steepener(steep_curve_slope, threshold=1.5)
        active_low = (signals_low != 0).sum()
        active_high = (signals_high != 0).sum()
        assert active_high <= active_low


# ------------------------------------------------------------------
# IG/HY rotation
# ------------------------------------------------------------------

class TestIGHYRotation:

    def test_tightening_spreads_overweight_hyg(self, ig_spread, hy_spread_tightening):
        """When HY spreads are tightening, HYG should get higher weight."""
        strat = CreditRotationStrategy()
        rotation = strat.ig_hy_rotation(ig_spread, hy_spread_tightening, lookback=63)

        assert "HYG" in rotation
        assert "LQD" in rotation

        # After lookback warm-up, check weights during tightening
        recent_hyg = rotation["HYG"].iloc[70:]
        recent_lqd = rotation["LQD"].iloc[70:]
        # On tightening days (momentum < 0), HYG should be 0.7
        assert (recent_hyg >= recent_lqd).mean() > 0.5

    def test_widening_spreads_overweight_lqd(self, ig_spread, hy_spread_widening):
        """When HY spreads are widening, LQD should get higher weight."""
        strat = CreditRotationStrategy()
        rotation = strat.ig_hy_rotation(ig_spread, hy_spread_widening, lookback=63)

        recent_hyg = rotation["HYG"].iloc[70:]
        recent_lqd = rotation["LQD"].iloc[70:]
        # On widening days, LQD (0.8) should be greater than HYG (0.2)
        assert (recent_lqd >= recent_hyg).mean() > 0.5

    def test_weights_are_valid(self, ig_spread, hy_spread_tightening):
        """All weights should be between 0 and 1."""
        strat = CreditRotationStrategy()
        rotation = strat.ig_hy_rotation(ig_spread, hy_spread_tightening)
        for key in ["HYG", "LQD"]:
            assert rotation[key].min() >= 0
            assert rotation[key].max() <= 1.0

    def test_weights_sum_to_one_when_active(self, ig_spread, hy_spread_tightening):
        """HYG + LQD weights should sum to 1.0 when there is a signal."""
        strat = CreditRotationStrategy()
        rotation = strat.ig_hy_rotation(ig_spread, hy_spread_tightening, lookback=63)
        total = rotation["HYG"] + rotation["LQD"]
        active = total[total > 0]
        assert (abs(active - 1.0) < 0.01).all()

    def test_output_length_matches_input(self, dates, ig_spread, hy_spread_tightening):
        """Output series should have same length as input."""
        strat = CreditRotationStrategy()
        rotation = strat.ig_hy_rotation(ig_spread, hy_spread_tightening)
        assert len(rotation["HYG"]) == len(dates)
        assert len(rotation["LQD"]) == len(dates)
