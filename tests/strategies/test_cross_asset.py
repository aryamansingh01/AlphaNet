"""Tests for cross-asset divergence signals."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.cross_asset.divergence import CrossAssetSignals


@pytest.fixture
def dates():
    return pd.bdate_range("2022-01-01", periods=300)


@pytest.fixture
def signals():
    return CrossAssetSignals()


@pytest.fixture
def bullish_equity_returns(dates):
    """Consistently positive equity returns."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.005, 300), index=dates)
    return returns


@pytest.fixture
def bearish_equity_returns(dates):
    """Consistently negative equity returns."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(-0.001, 0.005, 300), index=dates)
    return returns


@pytest.fixture
def widening_hy_spread(dates):
    """HY spreads widening (increasing) over time."""
    np.random.seed(42)
    return pd.Series(np.linspace(3.5, 7.0, 300) + np.random.normal(0, 0.05, 300), index=dates)


@pytest.fixture
def tightening_hy_spread(dates):
    """HY spreads tightening (decreasing) over time."""
    np.random.seed(42)
    return pd.Series(np.linspace(6.0, 3.0, 300) + np.random.normal(0, 0.05, 300), index=dates)


@pytest.fixture
def low_vix(dates):
    """Low, stable VIX."""
    np.random.seed(42)
    return pd.Series(np.random.normal(14, 1, 300).clip(10, 20), index=dates)


@pytest.fixture
def high_vix(dates):
    """Elevated VIX."""
    np.random.seed(42)
    return pd.Series(np.random.normal(30, 3, 300).clip(20, 45), index=dates)


@pytest.fixture
def positive_curve_slope(dates):
    """Positive (normal) curve slope."""
    return pd.Series(np.random.normal(1.5, 0.2, 300), index=dates)


@pytest.fixture
def negative_curve_slope(dates):
    """Negative (inverted) curve slope."""
    return pd.Series(np.random.normal(-0.5, 0.2, 300), index=dates)


# ------------------------------------------------------------------
# Credit-equity divergence
# ------------------------------------------------------------------

class TestCreditEquityDivergence:

    def test_bearish_divergence(self, signals, bullish_equity_returns, widening_hy_spread):
        """When equities are up but credit is widening, signal should be bearish (-1)."""
        div = signals.credit_equity_divergence(
            bullish_equity_returns, widening_hy_spread, lookback=21
        )
        # After warm-up, there should be bearish signals
        recent = div.iloc[30:]
        assert (recent == -1).any()

    def test_bullish_divergence(self, signals, bearish_equity_returns, tightening_hy_spread):
        """When equities are down but credit is tightening, signal should be bullish (+1)."""
        div = signals.credit_equity_divergence(
            bearish_equity_returns, tightening_hy_spread, lookback=21
        )
        recent = div.iloc[30:]
        assert (recent == 1).any()

    def test_signal_values_valid(self, signals, bullish_equity_returns, widening_hy_spread):
        """Divergence signals should be -1, 0, or 1."""
        div = signals.credit_equity_divergence(
            bullish_equity_returns, widening_hy_spread
        )
        unique_vals = set(div.dropna().unique())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_output_length(self, signals, bullish_equity_returns, widening_hy_spread):
        """Output should have same length as input."""
        div = signals.credit_equity_divergence(
            bullish_equity_returns, widening_hy_spread
        )
        assert len(div) == len(bullish_equity_returns)


# ------------------------------------------------------------------
# Risk-on/risk-off composite
# ------------------------------------------------------------------

class TestRiskOnOffComposite:

    def test_risk_on_environment(
        self, signals, bullish_equity_returns, tightening_hy_spread, low_vix, positive_curve_slope
    ):
        """Bullish equities + tight spreads + low VIX + positive slope should score positive."""
        score = signals.risk_on_off_composite(
            bullish_equity_returns, tightening_hy_spread, low_vix, positive_curve_slope
        )
        # After rolling window warm-up (252 for spread z-score)
        recent = score.iloc[260:]
        # Average score should be positive (risk-on)
        assert recent.mean() > 0

    def test_risk_off_environment(
        self, signals, bearish_equity_returns, widening_hy_spread, high_vix, negative_curve_slope
    ):
        """Bearish equities + wide spreads + high VIX + inverted slope should score negative."""
        score = signals.risk_on_off_composite(
            bearish_equity_returns, widening_hy_spread, high_vix, negative_curve_slope
        )
        recent = score.iloc[260:]
        assert recent.mean() < 0

    def test_score_is_bounded(
        self, signals, bullish_equity_returns, tightening_hy_spread, low_vix, positive_curve_slope
    ):
        """Composite score should be bounded (sum of components each in [-weight, +weight])."""
        score = signals.risk_on_off_composite(
            bullish_equity_returns, tightening_hy_spread, low_vix, positive_curve_slope
        )
        # Max possible: 0.25 + 0.30 + 0.25 + 0.20 = 1.0
        assert score.max() <= 1.01
        assert score.min() >= -1.01

    def test_output_length(
        self, signals, bullish_equity_returns, tightening_hy_spread, low_vix, positive_curve_slope
    ):
        """Output should have same length as input."""
        score = signals.risk_on_off_composite(
            bullish_equity_returns, tightening_hy_spread, low_vix, positive_curve_slope
        )
        assert len(score) == len(bullish_equity_returns)

    def test_components_contribute(self, signals, dates):
        """Each component should actually affect the score."""
        np.random.seed(42)
        eq = pd.Series(np.random.normal(0.001, 0.005, 300), index=dates)
        hy = pd.Series(np.random.normal(4.5, 0.5, 300), index=dates)
        vix = pd.Series(np.random.normal(18, 3, 300), index=dates)
        slope = pd.Series(np.random.normal(1.0, 0.5, 300), index=dates)

        score_base = signals.risk_on_off_composite(eq, hy, vix, slope)
        # Flip the equity returns to strongly negative
        eq_bear = pd.Series(np.random.normal(-0.003, 0.005, 300), index=dates)
        score_bear = signals.risk_on_off_composite(eq_bear, hy, vix, slope)

        # Score should decrease when equities are bearish
        assert score_bear.iloc[260:].mean() < score_base.iloc[260:].mean()
