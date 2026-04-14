"""Historical and custom stress testing for multi-asset portfolios."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Historical scenario shock vectors (actual asset-class returns)
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, dict[str, Any]] = {
    "GFC 2008": {
        "description": "Global Financial Crisis (Sep-Nov 2008)",
        "shocks": {
            "SPY": -0.38, "TLT": 0.20, "IEF": 0.12,
            "HYG": -0.25, "LQD": -0.08, "GLD": 0.05,
        },
        "rate_shock_bps": -200,
        "spread_shock_bps": 300,
    },
    "COVID Mar 2020": {
        "description": "COVID-19 crash (Feb-Mar 2020)",
        "shocks": {
            "SPY": -0.34, "TLT": 0.15, "IEF": 0.05,
            "HYG": -0.20, "LQD": -0.12, "GLD": -0.03,
        },
        "rate_shock_bps": -150,
        "spread_shock_bps": 600,
    },
    "Rate Shock 2022": {
        "description": "2022 rate hiking cycle",
        "shocks": {
            "SPY": -0.19, "TLT": -0.31, "IEF": -0.15,
            "HYG": -0.11, "LQD": -0.17, "GLD": -0.01,
        },
        "rate_shock_bps": 236,
        "spread_shock_bps": 50,
    },
    "Tariff Turmoil 2025": {
        "description": "April 2025 tariff escalation",
        "shocks": {
            "SPY": -0.12, "TLT": 0.08, "IEF": 0.04,
            "HYG": -0.08, "LQD": -0.03, "GLD": 0.06,
        },
        "rate_shock_bps": -50,
        "spread_shock_bps": 120,
    },
}

# Approximate spread durations for credit ETFs (years)
_SPREAD_DURATION: dict[str, float] = {
    "HYG": 3.8,
    "LQD": 8.5,
    "JNK": 3.7,
    "EMB": 7.2,
}


class StressTestEngine:
    """Run historical and custom stress scenarios on a portfolio."""

    def run_historical(
        self, portfolio_weights: dict[str, float], scenario_name: str
    ) -> dict:
        """Apply a historical shock vector to a portfolio.

        Parameters
        ----------
        portfolio_weights : dict
            Mapping of ETF ticker to portfolio weight (e.g. {"SPY": 0.6, "TLT": 0.3}).
        scenario_name : str
            Key into SCENARIOS dict.

        Returns
        -------
        dict with scenario details and P&L breakdown.
        """
        if scenario_name not in SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. "
                f"Available: {list(SCENARIOS.keys())}"
            )

        scenario = SCENARIOS[scenario_name]
        shocks = scenario["shocks"]
        breakdown = []
        total_pnl = 0.0

        for asset, weight in portfolio_weights.items():
            shock = shocks.get(asset, 0.0)
            contribution = weight * shock
            total_pnl += contribution
            breakdown.append({
                "asset": asset,
                "weight": round(weight, 4),
                "shock": round(shock, 4),
                "contribution": round(contribution, 4),
            })

        # Identify worst-hit asset
        worst_asset = (
            min(breakdown, key=lambda x: x["contribution"])["asset"]
            if breakdown
            else None
        )

        return {
            "scenario": scenario_name,
            "description": scenario["description"],
            "portfolio_pnl_pct": round(total_pnl, 4),
            "breakdown": breakdown,
            "worst_asset": worst_asset,
            "rate_impact_bps": scenario["rate_shock_bps"],
            "spread_impact_bps": scenario["spread_shock_bps"],
        }

    def run_all_scenarios(
        self, portfolio_weights: dict[str, float]
    ) -> list[dict]:
        """Run all historical scenarios, sorted by worst impact first."""
        results = []
        for name in SCENARIOS:
            results.append(self.run_historical(portfolio_weights, name))
        results.sort(key=lambda r: r["portfolio_pnl_pct"])
        return results

    def run_custom(
        self,
        portfolio_weights: dict[str, float],
        rate_shock_bps: float = 0,
        spread_shock_bps: float = 0,
        equity_shock_pct: float = 0,
    ) -> dict:
        """Run a custom scenario using factor sensitivities.

        Applies:
        - rate shock via duration proxies (from etf_fetcher)
        - spread shock via spread duration for credit ETFs
        - equity shock directly to equity holdings
        """
        from src.data.fixed_income.etf_fetcher import ETFFetcher

        duration_proxy = ETFFetcher.get_duration_proxy()

        # Equity-like tickers (not in duration map)
        equity_tickers = {"SPY", "QQQ", "IWM", "EFA", "EEM", "VTI", "VOO", "DIA"}

        breakdown = []
        total_pnl = 0.0
        rate_contribution = 0.0
        spread_contribution = 0.0
        equity_contribution = 0.0

        for asset, weight in portfolio_weights.items():
            asset_pnl = 0.0

            # Rate shock via modified duration
            dur = duration_proxy.get(asset, 0.0)
            if dur > 0 and rate_shock_bps != 0:
                rate_impact = -dur * (rate_shock_bps / 10_000)
                asset_pnl += rate_impact
                rate_contribution += weight * rate_impact

            # Spread shock for credit ETFs
            spread_dur = _SPREAD_DURATION.get(asset, 0.0)
            if spread_dur > 0 and spread_shock_bps != 0:
                spread_impact = -spread_dur * (spread_shock_bps / 10_000)
                asset_pnl += spread_impact
                spread_contribution += weight * spread_impact

            # Equity shock
            if asset.upper() in equity_tickers and equity_shock_pct != 0:
                eq_impact = equity_shock_pct / 100.0
                asset_pnl += eq_impact
                equity_contribution += weight * eq_impact

            contribution = weight * asset_pnl
            total_pnl += contribution
            breakdown.append({
                "asset": asset,
                "weight": round(weight, 4),
                "asset_pnl_pct": round(asset_pnl, 4),
                "contribution": round(contribution, 4),
            })

        return {
            "portfolio_pnl_pct": round(total_pnl, 4),
            "breakdown": breakdown,
            "rate_contribution": round(rate_contribution, 4),
            "spread_contribution": round(spread_contribution, 4),
            "equity_contribution": round(equity_contribution, 4),
            "inputs": {
                "rate_shock_bps": rate_shock_bps,
                "spread_shock_bps": spread_shock_bps,
                "equity_shock_pct": equity_shock_pct,
            },
        }

    @staticmethod
    def list_scenarios() -> list[dict]:
        """Return available scenario names and descriptions."""
        return [
            {
                "name": name,
                "description": s["description"],
                "rate_shock_bps": s["rate_shock_bps"],
                "spread_shock_bps": s["spread_shock_bps"],
                "assets_shocked": list(s["shocks"].keys()),
            }
            for name, s in SCENARIOS.items()
        ]
