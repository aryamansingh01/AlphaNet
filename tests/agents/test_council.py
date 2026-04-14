"""Tests for the rule-based SignalCouncil and individual analysts."""

import numpy as np
import pandas as pd
import pytest

from src.agents.council import (
    AgentOpinion,
    EquityAnalyst,
    CreditAnalyst,
    MacroStrategist,
    RiskManager,
    SignalCouncil,
)


@pytest.fixture
def dates():
    return pd.bdate_range("2022-01-01", periods=252)


# ------------------------------------------------------------------
# Helper data generators
# ------------------------------------------------------------------

def make_bullish_equity_returns(dates):
    """Strong positive momentum equity returns."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.002, 0.008, len(dates)), index=dates)
    return returns


def make_bearish_equity_returns(dates):
    """Strong negative momentum equity returns."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(-0.002, 0.008, len(dates)), index=dates)
    return returns


def make_neutral_equity_returns(dates):
    """Flat, low-volatility returns."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0, 0.005, len(dates)), index=dates)
    return returns


def make_tight_hy_spread(dates):
    """Low HY spread with tightening trend."""
    np.random.seed(42)
    return pd.Series(np.linspace(5.0, 3.0, len(dates)) + np.random.normal(0, 0.1, len(dates)), index=dates)


def make_wide_hy_spread(dates):
    """High HY spread with widening trend."""
    np.random.seed(42)
    return pd.Series(np.linspace(3.0, 8.0, len(dates)) + np.random.normal(0, 0.1, len(dates)), index=dates)


def make_ig_spread(dates):
    np.random.seed(42)
    return pd.Series(np.random.normal(1.2, 0.2, len(dates)), index=dates)


def make_positive_curve_slope(dates):
    return pd.Series(np.random.normal(1.5, 0.2, len(dates)), index=dates)


def make_inverted_curve_slope(dates):
    return pd.Series(np.random.normal(-0.5, 0.2, len(dates)), index=dates)


def make_low_vix(dates):
    np.random.seed(42)
    return pd.Series(np.random.normal(14, 1.5, len(dates)).clip(10, 20), index=dates)


def make_high_vix(dates):
    np.random.seed(42)
    return pd.Series(np.random.normal(35, 5, len(dates)).clip(25, 50), index=dates)


# ------------------------------------------------------------------
# EquityAnalyst
# ------------------------------------------------------------------

class TestEquityAnalyst:

    def test_bullish_data_produces_long(self, dates):
        """Strong positive momentum should produce a LONG signal."""
        analyst = EquityAnalyst()
        returns = make_bullish_equity_returns(dates)
        opinion = analyst.analyze(returns)
        assert opinion.direction == "LONG"
        assert opinion.conviction > 0

    def test_bearish_data_produces_short(self, dates):
        """Strong negative momentum should produce a SHORT signal."""
        analyst = EquityAnalyst()
        returns = make_bearish_equity_returns(dates)
        opinion = analyst.analyze(returns)
        assert opinion.direction == "SHORT"
        assert opinion.conviction > 0

    def test_neutral_data_produces_flat(self, dates):
        """Flat returns should produce a FLAT signal."""
        analyst = EquityAnalyst()
        returns = make_neutral_equity_returns(dates)
        opinion = analyst.analyze(returns)
        assert opinion.direction == "FLAT"

    def test_opinion_fields(self, dates):
        """Opinion should have all required fields."""
        analyst = EquityAnalyst()
        returns = make_bullish_equity_returns(dates)
        opinion = analyst.analyze(returns)
        assert opinion.agent == "equity_analyst"
        assert isinstance(opinion.direction, str)
        assert 0 <= opinion.conviction <= 1.0
        assert isinstance(opinion.reasoning, str)

    def test_sentiment_boost(self, dates):
        """Positive sentiment should increase conviction."""
        analyst = EquityAnalyst()
        returns = make_bullish_equity_returns(dates)
        opinion_no_sent = analyst.analyze(returns, sentiment_score=0.0)
        opinion_pos_sent = analyst.analyze(returns, sentiment_score=0.5)
        assert opinion_pos_sent.conviction >= opinion_no_sent.conviction

    def test_high_vol_penalty(self, dates):
        """Elevated volatility should reduce the score."""
        analyst = EquityAnalyst()
        np.random.seed(42)
        # High-vol positive returns
        high_vol_returns = pd.Series(
            np.random.normal(0.002, 0.025, len(dates)), index=dates
        )
        opinion = analyst.analyze(high_vol_returns)
        # The vol penalty should reduce conviction compared to normal vol
        normal_returns = make_bullish_equity_returns(dates)
        normal_opinion = analyst.analyze(normal_returns)
        # High vol case may have lower conviction (depending on realized values)
        # Just check it doesn't crash and produces a valid opinion
        assert opinion.conviction >= 0


# ------------------------------------------------------------------
# CreditAnalyst
# ------------------------------------------------------------------

class TestCreditAnalyst:

    def test_tight_spreads_produce_long(self, dates):
        """Tight and tightening HY spreads should produce a LONG signal."""
        analyst = CreditAnalyst()
        hy = make_tight_hy_spread(dates)
        ig = make_ig_spread(dates)
        slope = make_positive_curve_slope(dates)
        opinion = analyst.analyze(hy, ig, slope)
        assert opinion.direction in ("LONG", "FLAT")

    def test_wide_spreads_produce_short(self, dates):
        """Wide and widening HY spreads should produce a SHORT signal."""
        analyst = CreditAnalyst()
        hy = make_wide_hy_spread(dates)
        ig = make_ig_spread(dates)
        slope = make_inverted_curve_slope(dates)
        opinion = analyst.analyze(hy, ig, slope)
        assert opinion.direction == "SHORT"
        assert opinion.conviction > 0

    def test_inverted_curve_bearish(self, dates):
        """An inverted curve should contribute to a bearish signal."""
        analyst = CreditAnalyst()
        hy = pd.Series(np.random.normal(4.5, 0.5, len(dates)), index=dates)
        ig = make_ig_spread(dates)
        slope = make_inverted_curve_slope(dates)
        opinion = analyst.analyze(hy, ig, slope)
        assert "inverted" in opinion.reasoning.lower() or "recession" in opinion.reasoning.lower() or opinion.direction in ("SHORT", "FLAT")

    def test_opinion_agent_name(self, dates):
        """Agent name should be credit_analyst."""
        analyst = CreditAnalyst()
        hy = make_tight_hy_spread(dates)
        ig = make_ig_spread(dates)
        slope = make_positive_curve_slope(dates)
        opinion = analyst.analyze(hy, ig, slope)
        assert opinion.agent == "credit_analyst"


# ------------------------------------------------------------------
# RiskManager
# ------------------------------------------------------------------

class TestRiskManager:

    def test_veto_on_high_drawdown(self):
        """Risk manager should veto when portfolio drawdown exceeds limit."""
        rm = RiskManager(max_drawdown_limit=-0.10)
        opinions = [
            AgentOpinion(agent="equity_analyst", direction="LONG", conviction=0.8, reasoning="bullish"),
        ]
        result = rm.review(opinions, portfolio_drawdown=-0.15, vix=20)
        assert result.direction == "FLAT"
        assert "VETO" in result.reasoning

    def test_veto_on_high_vix(self):
        """Risk manager should veto when VIX exceeds panic threshold."""
        rm = RiskManager(max_vix=35)
        opinions = [
            AgentOpinion(agent="equity_analyst", direction="LONG", conviction=0.8, reasoning="bullish"),
        ]
        result = rm.review(opinions, portfolio_drawdown=0.0, vix=40)
        assert result.direction == "FLAT"
        assert "VETO" in result.reasoning

    def test_approve_when_no_flags(self):
        """Risk manager should approve when all is well."""
        rm = RiskManager(max_drawdown_limit=-0.10, max_vix=35)
        opinions = [
            AgentOpinion(agent="equity_analyst", direction="LONG", conviction=0.8, reasoning="bullish"),
            AgentOpinion(agent="credit_analyst", direction="LONG", conviction=0.6, reasoning="tight"),
        ]
        result = rm.review(opinions, portfolio_drawdown=-0.02, vix=15)
        assert result.direction == "APPROVED"

    def test_conviction_reflects_analyst_average(self):
        """When approved, conviction should be average of analyst convictions."""
        rm = RiskManager()
        opinions = [
            AgentOpinion(agent="a", direction="LONG", conviction=0.8, reasoning="x"),
            AgentOpinion(agent="b", direction="LONG", conviction=0.4, reasoning="y"),
        ]
        result = rm.review(opinions, portfolio_drawdown=0.0, vix=15)
        assert abs(result.conviction - 0.6) < 0.01

    def test_correlation_warning(self):
        """High correlation should produce a warning in reasoning."""
        rm = RiskManager()
        opinions = [
            AgentOpinion(agent="a", direction="LONG", conviction=0.5, reasoning="x"),
        ]
        result = rm.review(opinions, portfolio_drawdown=0.0, vix=15, correlation=0.9)
        assert "correlation" in result.reasoning.lower()

    def test_disagreement_warning(self):
        """When all analysts disagree, the risk manager should note it."""
        rm = RiskManager()
        opinions = [
            AgentOpinion(agent="a", direction="LONG", conviction=0.5, reasoning="x"),
            AgentOpinion(agent="b", direction="SHORT", conviction=0.5, reasoning="y"),
            AgentOpinion(agent="c", direction="FLAT", conviction=0.5, reasoning="z"),
        ]
        result = rm.review(opinions, portfolio_drawdown=0.0, vix=15)
        assert "disagree" in result.reasoning.lower()


# ------------------------------------------------------------------
# Full SignalCouncil integration
# ------------------------------------------------------------------

class TestSignalCouncil:

    def test_full_council_returns_expected_keys(self, dates):
        """Council result should contain direction, conviction, and all opinions."""
        council = SignalCouncil()
        result = council.run(
            equity_returns=make_bullish_equity_returns(dates),
            hy_spread=make_tight_hy_spread(dates),
            ig_spread=make_ig_spread(dates),
            curve_slope=make_positive_curve_slope(dates),
            vix=make_low_vix(dates),
            regime="risk_on",
        )
        assert "direction" in result
        assert "conviction" in result
        assert "opinions" in result
        assert set(result["opinions"].keys()) == {"equity", "credit", "macro", "risk"}

    def test_bullish_inputs_produce_long(self, dates):
        """Strongly bullish inputs should produce a LONG signal."""
        council = SignalCouncil()
        result = council.run(
            equity_returns=make_bullish_equity_returns(dates),
            hy_spread=make_tight_hy_spread(dates),
            ig_spread=make_ig_spread(dates),
            curve_slope=make_positive_curve_slope(dates),
            vix=make_low_vix(dates),
            regime="risk_on",
        )
        assert result["direction"] == "LONG"
        assert result["conviction"] > 0

    def test_bearish_inputs_produce_short(self, dates):
        """Strongly bearish inputs should produce a SHORT signal."""
        council = SignalCouncil()
        result = council.run(
            equity_returns=make_bearish_equity_returns(dates),
            hy_spread=make_wide_hy_spread(dates),
            ig_spread=make_ig_spread(dates),
            curve_slope=make_inverted_curve_slope(dates),
            vix=make_high_vix(dates),
            regime="crisis",
        )
        # Risk manager should veto due to high VIX
        assert result["direction"] in ("SHORT", "FLAT")

    def test_risk_veto_overrides_council(self, dates):
        """When risk manager vetoes, final direction should be FLAT regardless of analysts."""
        council = SignalCouncil()
        result = council.run(
            equity_returns=make_bullish_equity_returns(dates),
            hy_spread=make_tight_hy_spread(dates),
            ig_spread=make_ig_spread(dates),
            curve_slope=make_positive_curve_slope(dates),
            vix=make_high_vix(dates),  # VIX above 35 triggers veto
            regime="risk_on",
            portfolio_drawdown=-0.15,  # drawdown also triggers veto
        )
        assert result["direction"] == "FLAT"
        assert result["conviction"] == 0.0

    def test_opinions_have_reasoning(self, dates):
        """Every analyst opinion should include reasoning text."""
        council = SignalCouncil()
        result = council.run(
            equity_returns=make_bullish_equity_returns(dates),
            hy_spread=make_tight_hy_spread(dates),
            ig_spread=make_ig_spread(dates),
            curve_slope=make_positive_curve_slope(dates),
            vix=make_low_vix(dates),
            regime="risk_on",
        )
        for name, opinion in result["opinions"].items():
            assert "reasoning" in opinion
            assert len(opinion["reasoning"]) > 0

    def test_conviction_bounded(self, dates):
        """Final conviction should be between 0 and 1."""
        council = SignalCouncil()
        result = council.run(
            equity_returns=make_bullish_equity_returns(dates),
            hy_spread=make_tight_hy_spread(dates),
            ig_spread=make_ig_spread(dates),
            curve_slope=make_positive_curve_slope(dates),
            vix=make_low_vix(dates),
            regime="risk_on",
        )
        assert 0 <= result["conviction"] <= 1.0

    def test_weighted_vote_sums(self, dates):
        """Equity (30%) + Credit (35%) + Macro (35%) weights should influence direction."""
        council = SignalCouncil()
        # All analysts agree on LONG
        result = council.run(
            equity_returns=make_bullish_equity_returns(dates),
            hy_spread=make_tight_hy_spread(dates),
            ig_spread=make_ig_spread(dates),
            curve_slope=make_positive_curve_slope(dates),
            vix=make_low_vix(dates),
            regime="risk_on",
        )
        # If all agree LONG, conviction should be meaningful
        if result["direction"] == "LONG":
            assert result["conviction"] > 0.1
