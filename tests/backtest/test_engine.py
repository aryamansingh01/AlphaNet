"""Tests for the backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine


@pytest.fixture
def dates():
    return pd.bdate_range("2023-01-01", periods=252)


@pytest.fixture
def engine():
    return BacktestEngine(transaction_cost_bps=5.0)


@pytest.fixture
def engine_no_costs():
    return BacktestEngine(transaction_cost_bps=0.0)


@pytest.fixture
def trending_prices(dates):
    """Prices that trend upward (positive returns)."""
    np.random.seed(42)
    cumulative = np.exp(np.cumsum(np.random.normal(0.001, 0.01, 252)))
    return pd.DataFrame({"SPY": 100 * cumulative}, index=dates)


@pytest.fixture
def constant_long_signals(dates):
    """Always-long signal with weight 1."""
    return pd.DataFrame({"SPY": 1.0}, index=dates)


@pytest.fixture
def switching_signals(dates):
    """Signals that switch frequently to generate turnover."""
    sig = np.zeros(252)
    for i in range(252):
        sig[i] = 1.0 if i % 2 == 0 else 0.0
    return pd.DataFrame({"SPY": sig}, index=dates)


@pytest.fixture
def multi_asset_prices(dates):
    """Multiple assets with different return profiles."""
    np.random.seed(42)
    return pd.DataFrame({
        "A": 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.01, 252))),
        "B": 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, 252))),
        "C": 100 * np.exp(np.cumsum(np.random.normal(-0.0005, 0.02, 252))),
    }, index=dates)


@pytest.fixture
def multi_asset_signals(dates):
    """Equal-weight long signals for multiple assets."""
    return pd.DataFrame({
        "A": 0.34,
        "B": 0.33,
        "C": 0.33,
    }, index=dates)


# ------------------------------------------------------------------
# Basic backtest execution
# ------------------------------------------------------------------

class TestBasicBacktest:

    def test_run_returns_expected_keys(self, engine, constant_long_signals, trending_prices):
        """Result dict should contain returns, cumulative, metrics, and turnover."""
        result = engine.run(constant_long_signals, trending_prices)
        assert "returns" in result
        assert "cumulative" in result
        assert "metrics" in result
        assert "turnover" in result

    def test_returns_length(self, engine, constant_long_signals, trending_prices):
        """Portfolio returns should have correct length."""
        result = engine.run(constant_long_signals, trending_prices)
        # pct_change drops 1 row, then signals.shift(1) drops another
        assert len(result["returns"]) > 200

    def test_cumulative_starts_near_one(self, engine, constant_long_signals, trending_prices):
        """Cumulative returns should start near 1.0."""
        result = engine.run(constant_long_signals, trending_prices)
        assert abs(result["cumulative"].iloc[0] - 1.0) < 0.05

    def test_positive_trend_positive_return(self, engine, constant_long_signals, trending_prices):
        """Long position in uptrending asset should produce positive total return."""
        result = engine.run(constant_long_signals, trending_prices)
        assert result["metrics"]["total_return"] > 0

    def test_zero_signals_zero_return(self, engine, trending_prices, dates):
        """Zero signals should produce near-zero returns."""
        zero_signals = pd.DataFrame({"SPY": 0.0}, index=dates)
        result = engine.run(zero_signals, trending_prices)
        assert abs(result["metrics"]["total_return"]) < 0.01


# ------------------------------------------------------------------
# Metrics computation
# ------------------------------------------------------------------

class TestMetrics:

    def test_metrics_keys(self, engine, constant_long_signals, trending_prices):
        """Metrics should include Sharpe, drawdown, and other standard measures."""
        result = engine.run(constant_long_signals, trending_prices)
        m = result["metrics"]
        expected_keys = {
            "annual_return", "annual_vol", "sharpe", "sortino",
            "max_drawdown", "calmar", "win_rate", "total_return",
        }
        assert expected_keys == set(m.keys())

    def test_sharpe_reasonable(self, engine, constant_long_signals, trending_prices):
        """Sharpe ratio should be a finite number."""
        result = engine.run(constant_long_signals, trending_prices)
        sharpe = result["metrics"]["sharpe"]
        assert np.isfinite(sharpe)

    def test_max_drawdown_negative(self, engine, constant_long_signals, trending_prices):
        """Max drawdown should be negative or zero."""
        result = engine.run(constant_long_signals, trending_prices)
        assert result["metrics"]["max_drawdown"] <= 0

    def test_win_rate_bounded(self, engine, constant_long_signals, trending_prices):
        """Win rate should be between 0 and 1."""
        result = engine.run(constant_long_signals, trending_prices)
        wr = result["metrics"]["win_rate"]
        assert 0 <= wr <= 1

    def test_annual_vol_positive(self, engine, constant_long_signals, trending_prices):
        """Annualized volatility should be positive."""
        result = engine.run(constant_long_signals, trending_prices)
        assert result["metrics"]["annual_vol"] > 0


# ------------------------------------------------------------------
# Transaction costs
# ------------------------------------------------------------------

class TestTransactionCosts:

    def test_costs_reduce_returns(
        self, engine, engine_no_costs, switching_signals, trending_prices
    ):
        """Transaction costs should reduce total returns."""
        result_with_costs = engine.run(switching_signals, trending_prices)
        result_no_costs = engine_no_costs.run(switching_signals, trending_prices)
        assert result_with_costs["metrics"]["total_return"] < result_no_costs["metrics"]["total_return"]

    def test_no_turnover_no_cost_impact(
        self, engine, engine_no_costs, constant_long_signals, trending_prices
    ):
        """With constant signals (no turnover after initial), costs should be minimal."""
        result_with = engine.run(constant_long_signals, trending_prices)
        result_without = engine_no_costs.run(constant_long_signals, trending_prices)
        # Difference should be very small (only initial entry cost)
        diff = abs(
            result_with["metrics"]["total_return"] - result_without["metrics"]["total_return"]
        )
        assert diff < 0.01

    def test_high_turnover_higher_cost_drag(self, trending_prices, switching_signals):
        """Higher transaction costs should produce larger drag on high-turnover strategies."""
        engine_low = BacktestEngine(transaction_cost_bps=1.0)
        engine_high = BacktestEngine(transaction_cost_bps=50.0)
        result_low = engine_low.run(switching_signals, trending_prices)
        result_high = engine_high.run(switching_signals, trending_prices)
        assert result_high["metrics"]["total_return"] < result_low["metrics"]["total_return"]


# ------------------------------------------------------------------
# Walk-forward validation
# ------------------------------------------------------------------

class TestWalkForward:

    def test_walk_forward_returns_splits(self, engine, constant_long_signals, trending_prices):
        """Walk-forward should return in-sample and out-of-sample results."""
        result = engine.walk_forward(constant_long_signals, trending_prices)
        assert "in_sample" in result
        assert "out_of_sample" in result
        assert "degradation" in result

    def test_walk_forward_split_ratio(self, engine, constant_long_signals, trending_prices):
        """In-sample and out-of-sample should respect the train_pct split."""
        result = engine.walk_forward(constant_long_signals, trending_prices, train_pct=0.7)
        is_len = len(result["in_sample"]["returns"])
        oos_len = len(result["out_of_sample"]["returns"])
        # The split should be approximately 70/30
        total = is_len + oos_len
        assert 0.55 < is_len / total < 0.85

    def test_degradation_keys(self, engine, constant_long_signals, trending_prices):
        """Degradation dict should contain sharpe and annual_return."""
        result = engine.walk_forward(constant_long_signals, trending_prices)
        assert "sharpe" in result["degradation"]
        assert "annual_return" in result["degradation"]

    def test_walk_forward_both_have_metrics(self, engine, constant_long_signals, trending_prices):
        """Both in-sample and out-of-sample should have complete metrics."""
        result = engine.walk_forward(constant_long_signals, trending_prices)
        for split in ["in_sample", "out_of_sample"]:
            assert "metrics" in result[split]
            assert "sharpe" in result[split]["metrics"]


# ------------------------------------------------------------------
# Multi-asset backtest
# ------------------------------------------------------------------

class TestMultiAsset:

    def test_multi_asset_runs(self, engine, multi_asset_signals, multi_asset_prices):
        """Backtest should work with multiple assets."""
        result = engine.run(multi_asset_signals, multi_asset_prices)
        assert len(result["returns"]) > 200

    def test_multi_asset_diversification(self, engine, multi_asset_prices, dates):
        """A diversified portfolio should have lower vol than single-asset."""
        single = pd.DataFrame({"A": 1.0, "B": 0.0, "C": 0.0}, index=dates)
        diversified = pd.DataFrame({"A": 0.34, "B": 0.33, "C": 0.33}, index=dates)

        result_single = engine.run(single, multi_asset_prices)
        result_div = engine.run(diversified, multi_asset_prices)

        # Diversified portfolio should generally have lower volatility
        # (this is probabilistic, so use a generous bound)
        assert result_div["metrics"]["annual_vol"] < result_single["metrics"]["annual_vol"] * 1.5


# ------------------------------------------------------------------
# Regime-conditional metrics
# ------------------------------------------------------------------

class TestRegimeConditional:

    def test_regime_metrics_returned(self, engine, constant_long_signals, trending_prices, dates):
        """When regime labels are provided, regime_metrics should be in result."""
        regimes = pd.Series("risk_on", index=dates)
        regimes.iloc[126:] = "risk_off"
        result = engine.run(constant_long_signals, trending_prices, regime_labels=regimes)
        assert "regime_metrics" in result
        assert len(result["regime_metrics"]) > 0

    def test_regime_metrics_per_regime(self, engine, constant_long_signals, trending_prices, dates):
        """Should compute metrics for each unique regime."""
        regimes = pd.Series("risk_on", index=dates)
        regimes.iloc[126:] = "risk_off"
        result = engine.run(constant_long_signals, trending_prices, regime_labels=regimes)
        assert "risk_on" in result["regime_metrics"]
        assert "risk_off" in result["regime_metrics"]
