"""Carry and roll-down analysis for fixed-income curves."""

from __future__ import annotations

import logging

import numpy as np
from nelson_siegel_svensson import NelsonSiegelCurve

from src.curve.duration import modified_duration

logger = logging.getLogger(__name__)

# Maturities to analyse (skip sub-2Y since roll-down is less meaningful)
CARRY_MATURITIES = {
    "2Y": 2,
    "3Y": 3,
    "5Y": 5,
    "7Y": 7,
    "10Y": 10,
    "20Y": 20,
    "30Y": 30,
}


class CarryRollDownAnalyzer:
    """Compute carry, roll-down, total return, and breakeven for each tenor."""

    def compute(
        self,
        yields_dict: dict[str, float],
        sofr_rate: float,
        ns_curve: NelsonSiegelCurve,
        horizon_years: float = 1.0,
    ) -> list[dict]:
        """Analyse carry and roll-down for standard maturities.

        Parameters
        ----------
        yields_dict : dict
            Observed yields keyed by maturity label (e.g. ``{"2Y": 4.5, ...}``).
            Values are in *percent* (not decimal).
        sofr_rate : float
            Current SOFR rate in percent.
        ns_curve : NelsonSiegelCurve
            Fitted Nelson-Siegel curve (returns yields in *decimal*).
        horizon_years : float
            Investment horizon in years (default 1).

        Returns
        -------
        list[dict]
            One entry per maturity with keys: ``maturity``, ``carry``,
            ``roll_down``, ``total_return``, ``breakeven_bps``.
        """
        results: list[dict] = []

        for label, mat in CARRY_MATURITIES.items():
            # Current yield at this maturity (percent)
            current_yield = yields_dict.get(label)
            if current_yield is None:
                # Interpolate from NS curve if not in observed data
                current_yield = float(ns_curve(np.array([mat]))[0]) * 100

            # Carry: annualised income above financing cost
            carry = current_yield - sofr_rate

            # Roll-down: yield change from rolling down the curve
            rolled_mat = mat - horizon_years
            if rolled_mat <= 0:
                roll_down = 0.0
            else:
                rolled_yield = float(ns_curve(np.array([rolled_mat]))[0]) * 100
                roll_down = current_yield - rolled_yield

            total_return = carry + roll_down

            # Modified duration for a par bond at this maturity
            coupon_rate = current_yield / 100.0
            ytm = current_yield / 100.0
            mod_dur = modified_duration(1000.0, coupon_rate, ytm, mat)

            # Breakeven: how much yields must rise before total return goes negative
            if mod_dur > 0:
                breakeven_bps = total_return / mod_dur * 10_000
            else:
                breakeven_bps = 0.0

            results.append({
                "maturity": label,
                "carry": round(carry, 4),
                "roll_down": round(roll_down, 4),
                "total_return": round(total_return, 4),
                "breakeven_bps": round(breakeven_bps, 2),
            })

        return results
