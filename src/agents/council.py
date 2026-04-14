"""Rule-Based Signal Council — 4 quantitative analysts score markets without any LLM."""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class AgentOpinion:
    agent: str
    direction: str  # "LONG", "SHORT", "FLAT"
    conviction: float  # 0.0 to 1.0
    reasoning: str


class EquityAnalyst:
    """Score equities based on momentum, mean reversion, and trend."""

    def analyze(self, equity_returns: pd.Series, sentiment_score: float = 0.0) -> AgentOpinion:
        # Exclude current observation to avoid look-ahead, with safe bounds
        n = len(equity_returns)
        end = max(n - 1, 1)
        mom_21 = equity_returns.iloc[max(end - 21, 0):end].sum()
        mom_63 = equity_returns.iloc[max(end - 63, 0):end].sum()
        vol = equity_returns.iloc[max(end - 21, 0):end].std() * np.sqrt(252)

        score = 0.0
        reasons = []

        # Momentum
        if mom_21 > 0.02:
            score += 0.3
            reasons.append(f"21d momentum positive ({mom_21:.2%})")
        elif mom_21 < -0.02:
            score -= 0.3
            reasons.append(f"21d momentum negative ({mom_21:.2%})")

        # Longer-term trend
        if mom_63 > 0.05:
            score += 0.3
            reasons.append(f"63d trend bullish ({mom_63:.2%})")
        elif mom_63 < -0.05:
            score -= 0.3
            reasons.append(f"63d trend bearish ({mom_63:.2%})")

        # Volatility penalty
        if vol > 0.25:
            score -= 0.2
            reasons.append(f"elevated volatility ({vol:.1%})")

        # Sentiment boost
        if abs(sentiment_score) > 0.3:
            score += sentiment_score * 0.2
            reasons.append(f"sentiment factor ({sentiment_score:.2f})")

        direction = "LONG" if score > 0.1 else "SHORT" if score < -0.1 else "FLAT"
        conviction = min(abs(score), 1.0)

        return AgentOpinion(
            agent="equity_analyst",
            direction=direction,
            conviction=round(conviction, 2),
            reasoning="; ".join(reasons) or "no strong signals",
        )


class CreditAnalyst:
    """Score credit markets based on spreads, curve shape, and spread momentum."""

    def analyze(
        self,
        hy_spread: pd.Series,
        ig_spread: pd.Series,
        curve_slope: pd.Series,
    ) -> AgentOpinion:
        score = 0.0
        reasons = []

        # HY spread level (z-score using t-1 stats to avoid look-ahead)
        n = len(hy_spread)
        idx = max(n - 2, 0)
        hy_roll_mean = hy_spread.rolling(min(252, n)).mean()
        hy_roll_std = hy_spread.rolling(min(252, n)).std()
        hy_mean = hy_roll_mean.iloc[idx] if idx < n else 0
        hy_std = hy_roll_std.iloc[idx] if idx < n else 1
        hy_z = (hy_spread.iloc[idx] - hy_mean) / hy_std if hy_std > 0 else 0

        if hy_z > 1.0:
            score -= 0.4
            reasons.append(f"HY spreads wide (z={hy_z:.1f}), stress signal")
        elif hy_z < -0.5:
            score += 0.3
            reasons.append(f"HY spreads tight (z={hy_z:.1f}), risk-on")

        # Spread momentum (use t-1, safe indexing)
        spread_pct = hy_spread.pct_change(min(21, n - 1))
        spread_change = spread_pct.iloc[idx] if idx < len(spread_pct) else 0
        if spread_change > 0.1:
            score -= 0.3
            reasons.append(f"spreads widening rapidly ({spread_change:.1%} in 21d)")
        elif spread_change < -0.05:
            score += 0.2
            reasons.append(f"spreads tightening ({spread_change:.1%} in 21d)")

        # Curve shape (use t-1, safe indexing)
        slope = curve_slope.iloc[max(len(curve_slope) - 2, 0)]
        if slope < 0:
            score -= 0.3
            reasons.append(f"curve inverted ({slope:.2f}%), recession risk")
        elif slope > 1.5:
            score += 0.2
            reasons.append(f"steep curve ({slope:.2f}%), expansion signal")

        direction = "LONG" if score > 0.1 else "SHORT" if score < -0.1 else "FLAT"
        conviction = min(abs(score), 1.0)

        return AgentOpinion(
            agent="credit_analyst",
            direction=direction,
            conviction=round(conviction, 2),
            reasoning="; ".join(reasons) or "credit markets neutral",
        )


class MacroStrategist:
    """Score macro environment based on regime, VIX, and cross-asset signals."""

    def analyze(
        self,
        regime: str,
        vix: pd.Series,
        equity_returns: pd.Series,
        hy_spread: pd.Series,
    ) -> AgentOpinion:
        score = 0.0
        reasons = []

        # Regime
        if regime == "risk_on":
            score += 0.4
            reasons.append(f"regime: {regime}")
        elif regime == "crisis":
            score -= 0.5
            reasons.append(f"regime: {regime}")
        else:
            reasons.append(f"regime: {regime}")

        # VIX level (use t-1 to avoid look-ahead, safe indexing)
        vix_idx = max(len(vix) - 2, 0)
        vix_current = vix.iloc[vix_idx]
        if vix_current > 30:
            score -= 0.3
            reasons.append(f"VIX elevated ({vix_current:.1f})")
        elif vix_current < 15:
            score += 0.2
            reasons.append(f"VIX low ({vix_current:.1f})")

        # Credit-equity divergence (exclude current day, safe bounds)
        n_eq = len(equity_returns)
        end_eq = max(n_eq - 1, 1)
        eq_trend = equity_returns.iloc[max(end_eq - 21, 0):end_eq].sum()
        n_hy = len(hy_spread)
        spread_pct = hy_spread.pct_change(min(21, n_hy - 1))
        spread_trend = spread_pct.iloc[max(n_hy - 2, 0)]
        if eq_trend > 0 and spread_trend > 0.05:
            score -= 0.3
            reasons.append("DIVERGENCE: equities up but credit widening — bearish")
        elif eq_trend < 0 and spread_trend < -0.05:
            score += 0.3
            reasons.append("DIVERGENCE: equities down but credit tightening — bullish")

        direction = "LONG" if score > 0.1 else "SHORT" if score < -0.1 else "FLAT"
        conviction = min(abs(score), 1.0)

        return AgentOpinion(
            agent="macro_strategist",
            direction=direction,
            conviction=round(conviction, 2),
            reasoning="; ".join(reasons) or "macro neutral",
        )


class RiskManager:
    """Veto or approve signals based on risk metrics."""

    def __init__(self, max_drawdown_limit: float = -0.10, max_vix: float = 35):
        self.max_dd = max_drawdown_limit
        self.max_vix = max_vix

    def review(
        self,
        opinions: list[AgentOpinion],
        portfolio_drawdown: float,
        vix: float,
        correlation: float = 0.0,
    ) -> AgentOpinion:
        reasons = []
        veto = False

        # Check drawdown limit
        if portfolio_drawdown < self.max_dd:
            veto = True
            reasons.append(
                f"VETO: portfolio drawdown ({portfolio_drawdown:.1%}) exceeds limit ({self.max_dd:.1%})"
            )

        # Check VIX panic
        if vix > self.max_vix:
            veto = True
            reasons.append(f"VETO: VIX at {vix:.1f} exceeds panic threshold ({self.max_vix})")

        # Check if analysts disagree
        directions = [o.direction for o in opinions]
        if len(set(directions)) == len(directions):
            reasons.append("WARNING: all analysts disagree — low conviction environment")

        # Check correlation risk
        if correlation > 0.8:
            reasons.append(f"WARNING: high cross-asset correlation ({correlation:.2f})")

        if veto:
            return AgentOpinion(
                agent="risk_manager",
                direction="FLAT",
                conviction=1.0,
                reasoning="; ".join(reasons),
            )

        avg_conviction = np.mean([o.conviction for o in opinions])
        return AgentOpinion(
            agent="risk_manager",
            direction="APPROVED",
            conviction=round(avg_conviction, 2),
            reasoning="; ".join(reasons) or "no risk flags — approved",
        )


class SignalCouncil:
    """Orchestrate all analysts and produce a final signal. No LLM required."""

    def __init__(self):
        self.equity = EquityAnalyst()
        self.credit = CreditAnalyst()
        self.macro = MacroStrategist()
        self.risk = RiskManager()

    def run(
        self,
        equity_returns: pd.Series,
        hy_spread: pd.Series,
        ig_spread: pd.Series,
        curve_slope: pd.Series,
        vix: pd.Series,
        regime: str,
        sentiment_score: float = 0.0,
        portfolio_drawdown: float = 0.0,
    ) -> dict:
        """Run the full council and return final signal."""
        # Each analyst scores independently
        eq_opinion = self.equity.analyze(equity_returns, sentiment_score)
        cr_opinion = self.credit.analyze(hy_spread, ig_spread, curve_slope)
        ma_opinion = self.macro.analyze(regime, vix, equity_returns, hy_spread)

        opinions = [eq_opinion, cr_opinion, ma_opinion]

        # Risk manager reviews
        risk_opinion = self.risk.review(
            opinions,
            portfolio_drawdown=portfolio_drawdown,
            vix=float(vix.iloc[-1]),
        )

        # Final signal
        if risk_opinion.direction == "FLAT":
            final_direction = "FLAT"
            final_conviction = 0.0
        else:
            # Weighted vote: equity 30%, credit 35%, macro 35%
            direction_scores = {
                "LONG": 0.0,
                "SHORT": 0.0,
                "FLAT": 0.0,
            }
            weights = {"equity_analyst": 0.30, "credit_analyst": 0.35, "macro_strategist": 0.35}
            for opinion in opinions:
                w = weights[opinion.agent]
                direction_scores[opinion.direction] += w * opinion.conviction

            final_direction = max(direction_scores, key=direction_scores.get)
            final_conviction = direction_scores[final_direction]

        return {
            "direction": final_direction,
            "conviction": round(final_conviction, 3),
            "opinions": {
                "equity": {
                    "direction": eq_opinion.direction,
                    "conviction": eq_opinion.conviction,
                    "reasoning": eq_opinion.reasoning,
                },
                "credit": {
                    "direction": cr_opinion.direction,
                    "conviction": cr_opinion.conviction,
                    "reasoning": cr_opinion.reasoning,
                },
                "macro": {
                    "direction": ma_opinion.direction,
                    "conviction": ma_opinion.conviction,
                    "reasoning": ma_opinion.reasoning,
                },
                "risk": {
                    "direction": risk_opinion.direction,
                    "conviction": risk_opinion.conviction,
                    "reasoning": risk_opinion.reasoning,
                },
            },
        }
