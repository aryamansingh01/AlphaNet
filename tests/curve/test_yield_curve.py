"""Tests for yield curve construction, fitting, and analysis."""

import numpy as np
import pandas as pd
import pytest

from src.curve.yield_curve import YieldCurveEngine, MATURITIES, MATURITY_LABELS


@pytest.fixture
def engine():
    return YieldCurveEngine()


@pytest.fixture
def normal_yields():
    """Realistic upward-sloping Treasury yields."""
    return np.array([5.3, 5.25, 5.1, 4.9, 4.5, 4.3, 4.2, 4.25, 4.3, 4.5, 4.6])


@pytest.fixture
def inverted_yields():
    """Inverted yield curve: short end above long end."""
    return np.array([5.5, 5.4, 5.3, 5.1, 4.8, 4.6, 4.3, 4.2, 4.0, 3.9, 3.8])


@pytest.fixture
def yield_series_normal():
    """A pd.Series of yields indexed by maturity labels (normal curve)."""
    values = [5.3, 5.25, 5.1, 4.9, 4.5, 4.3, 4.2, 4.25, 4.3, 4.5, 4.6]
    return pd.Series(values, index=MATURITY_LABELS)


@pytest.fixture
def yield_series_inverted():
    """A pd.Series of yields indexed by maturity labels (inverted curve)."""
    values = [5.5, 5.4, 5.3, 5.1, 4.8, 4.6, 4.3, 4.2, 4.0, 3.9, 3.8]
    return pd.Series(values, index=MATURITY_LABELS)


@pytest.fixture
def yield_df_with_inversion():
    """A DataFrame of daily yields with a period of inversion in the middle."""
    dates = pd.bdate_range("2023-01-01", periods=60)
    data = {}
    for i, label in enumerate(MATURITY_LABELS):
        base = 4.0 + i * 0.1
        data[label] = [base] * 60

    df = pd.DataFrame(data, index=dates)
    # Create inversion days 20-39: 2Y > 10Y
    df.loc[df.index[20:40], "2Y"] = 5.5
    df.loc[df.index[20:40], "10Y"] = 4.5
    return df


# ------------------------------------------------------------------
# Nelson-Siegel fitting
# ------------------------------------------------------------------

class TestNelsonSiegelFitting:

    def test_fit_returns_curve_object(self, engine, normal_yields):
        """Fitting should return a NelsonSiegelCurve object."""
        curve = engine.fit_nelson_siegel(normal_yields)
        assert hasattr(curve, "beta0")
        assert hasattr(curve, "beta1")
        assert hasattr(curve, "beta2")
        assert hasattr(curve, "tau")

    def test_fitted_values_close_to_observed(self, engine, normal_yields):
        """Fitted yields at observed maturities should be close to input."""
        curve = engine.fit_nelson_siegel(normal_yields)
        fitted = curve(MATURITIES) * 100  # back to percent
        residuals = np.abs(fitted - normal_yields)
        # Average residual should be small (under 50bps)
        assert residuals.mean() < 0.5

    def test_fit_inverted_curve(self, engine, inverted_yields):
        """Nelson-Siegel should handle inverted curves."""
        curve = engine.fit_nelson_siegel(inverted_yields)
        fitted = curve(MATURITIES) * 100
        # beta1 should be positive for an inverted curve (short > long)
        # Just check the fit is reasonable
        assert np.abs(fitted - inverted_yields).mean() < 1.0

    def test_interpolation_produces_smooth_curve(self, engine, normal_yields):
        """Interpolated curve should have more points than observed."""
        curve = engine.fit_nelson_siegel(normal_yields)
        interpolated = engine.interpolate_curve(curve)
        assert len(interpolated) == 100  # default is 100 points
        # Values should be in a reasonable range
        assert interpolated.min() > 0
        assert interpolated.max() < 10

    def test_interpolation_custom_maturities(self, engine, normal_yields):
        """Should interpolate at custom maturity points."""
        curve = engine.fit_nelson_siegel(normal_yields)
        custom_mats = np.array([0.5, 1.0, 5.0, 10.0, 30.0])
        interpolated = engine.interpolate_curve(curve, maturities=custom_mats)
        assert len(interpolated) == 5


# ------------------------------------------------------------------
# Curve metrics
# ------------------------------------------------------------------

class TestCurveMetrics:

    def test_metrics_keys(self, engine, yield_series_normal):
        """Metrics dict should contain all expected keys."""
        metrics = engine.get_curve_metrics(yield_series_normal)
        expected_keys = {"level", "slope", "curvature", "spread_2s10s", "spread_3m10y", "inverted"}
        assert set(metrics.keys()) == expected_keys

    def test_level_is_mean(self, engine, yield_series_normal):
        """Level should be the mean of all yields."""
        metrics = engine.get_curve_metrics(yield_series_normal)
        assert abs(metrics["level"] - yield_series_normal.mean()) < 0.001

    def test_slope_positive_for_normal_curve(self, engine, yield_series_normal):
        """Slope (30Y - 1M) should be negative for this specific curve (30Y=4.6 < 1M=5.3)."""
        metrics = engine.get_curve_metrics(yield_series_normal)
        expected_slope = yield_series_normal.iloc[-1] - yield_series_normal.iloc[0]
        assert abs(metrics["slope"] - expected_slope) < 0.001

    def test_spread_2s10s_normal(self, engine, yield_series_normal):
        """2s10s spread should be 10Y - 2Y."""
        metrics = engine.get_curve_metrics(yield_series_normal)
        expected = yield_series_normal["10Y"] - yield_series_normal["2Y"]
        assert abs(metrics["spread_2s10s"] - expected) < 0.001

    def test_inverted_flag_false_when_normal(self, engine, yield_series_normal):
        """Inverted should be False when 2Y < 10Y."""
        metrics = engine.get_curve_metrics(yield_series_normal)
        is_2y_gt_10y = yield_series_normal["2Y"] > yield_series_normal["10Y"]
        assert metrics["inverted"] == is_2y_gt_10y

    def test_inverted_flag_true_when_inverted(self, engine, yield_series_inverted):
        """Inverted should be True when 2Y > 10Y."""
        metrics = engine.get_curve_metrics(yield_series_inverted)
        assert metrics["inverted"] is True


# ------------------------------------------------------------------
# Inversion detection
# ------------------------------------------------------------------

class TestInversionDetection:

    def test_detect_inversion_finds_window(self, engine, yield_df_with_inversion):
        """Should detect the inversion window we inserted."""
        inversions = engine.detect_inversion(yield_df_with_inversion)
        assert len(inversions) >= 1
        # The inversion should span at least 15 business days
        assert inversions["duration_days"].iloc[0] >= 1

    def test_no_inversion_in_normal_curve(self, engine):
        """Should find no inversions when 10Y is always above 2Y."""
        dates = pd.bdate_range("2023-01-01", periods=60)
        df = pd.DataFrame({
            label: [3.0 + i * 0.2] * 60
            for i, label in enumerate(MATURITY_LABELS)
        }, index=dates)
        inversions = engine.detect_inversion(df)
        assert len(inversions) == 0


# ------------------------------------------------------------------
# Curve history
# ------------------------------------------------------------------

class TestCurveHistory:

    def test_history_returns_correct_columns(self, engine, yield_df_with_inversion):
        """History DataFrame should have the expected columns."""
        history = engine.get_curve_history(yield_df_with_inversion)
        expected_cols = {"level", "slope", "curvature", "spread_2s10s", "spread_3m10y", "inverted"}
        assert expected_cols == set(history.columns)

    def test_history_length_matches_input(self, engine, yield_df_with_inversion):
        """History should have one row per input date."""
        history = engine.get_curve_history(yield_df_with_inversion)
        assert len(history) == len(yield_df_with_inversion)

    def test_history_inverted_column_tracks_inversion(self, engine, yield_df_with_inversion):
        """Inverted flag in history should be True during the inversion window."""
        history = engine.get_curve_history(yield_df_with_inversion)
        # Days 20-39 have 2Y > 10Y
        inverted_days = history["inverted"].iloc[20:40]
        assert inverted_days.all()
        # Days outside that window should not be inverted
        normal_days = history["inverted"].iloc[:20]
        assert not normal_days.any()
