"""Composite strategy that combines all sub-strategy signals into final portfolio weights."""

from typing import Optional

import pandas as pd
import numpy as np


# Default asset universe for the composite portfolio.
ASSET_UNIVERSE: list[str] = ["SPY", "TLT", "IEF", "LQD", "HYG", "SHY"]

# Regime-dependent strategy weightings.
# Keys are regime labels; values map strategy names to their blending weight.
REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "risk_on": {
        "equity_momentum": 0.35,
        "sector_rotation": 0.15,
        "credit_rotation": 0.20,
        "curve_signal": 0.10,
        "cross_asset": 0.10,
        "mean_reversion": 0.10,
    },
    "risk_off": {
        "equity_momentum": 0.10,
        "sector_rotation": 0.05,
        "credit_rotation": 0.15,
        "curve_signal": 0.30,
        "cross_asset": 0.25,
        "mean_reversion": 0.15,
    },
    "crisis": {
        "equity_momentum": 0.05,
        "sector_rotation": 0.00,
        "credit_rotation": 0.10,
        "curve_signal": 0.35,
        "cross_asset": 0.30,
        "mean_reversion": 0.20,
    },
}


class CompositeStrategy:
    """Combine all sub-strategy signals into a single set of portfolio weights.

    The blending is regime-adaptive: different regimes assign different
    importance to each strategy's output.
    """

    def __init__(
        self,
        assets: Optional[list[str]] = None,
        regime_weights: Optional[dict[str, dict[str, float]]] = None,
        max_single_asset: float = 0.40,
        min_cash_like: float = 0.05,
    ) -> None:
        """
        Args:
            assets: Ticker list for the output weight DataFrame.
            regime_weights: Override for per-regime strategy blending weights.
            max_single_asset: Hard cap on any single asset weight.
            min_cash_like: Minimum allocation to cash-equivalent (SHY).
        """
        self.assets = assets or ASSET_UNIVERSE
        self.regime_weights = regime_weights or REGIME_WEIGHTS
        self.max_single_asset = max_single_asset
        self.min_cash_like = min_cash_like

    def generate_weights(
        self,
        regime: pd.Series,
        signals: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Produce final portfolio weights from regime labels and strategy signals.

        IMPORTANT: The ``regime`` labels MUST be generated using only data
        available at each timestamp (e.g. via RegimeDetector.fit_gmm_expanding).
        Using full-sample fitted labels will cause severe look-ahead bias.

        Args:
            regime: Series of regime labels (values in ``regime_weights`` keys)
                    indexed by date. Must be point-in-time (no future data).
            signals: Dict mapping strategy name -> DataFrame of per-asset
                     signals. Each DataFrame must be indexed by date with
                     columns that are a subset of ``self.assets``.

        Returns:
            DataFrame (index=dates, columns=self.assets) of portfolio weights
            that sum to 1.0 on each row.
        """
        dates = regime.index
        weights = pd.DataFrame(0.0, index=dates, columns=self.assets)

        # Align all signal DataFrames to the master date index and asset list.
        aligned_signals: dict[str, pd.DataFrame] = {}
        for name, sig in signals.items():
            aligned = sig.reindex(index=dates, columns=self.assets).fillna(0.0)
            aligned_signals[name] = aligned

        # Blend signals per regime.
        for regime_label, strat_weights in self.regime_weights.items():
            mask = regime == regime_label
            if not mask.any():
                continue
            for strat_name, blend_w in strat_weights.items():
                if strat_name in aligned_signals:
                    weights.loc[mask] += blend_w * aligned_signals[strat_name].loc[mask]

        # Post-processing: clamp, floor, and normalise.
        weights = self._apply_constraints(weights)
        return weights

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def regime_summary(self, regime: pd.Series) -> pd.DataFrame:
        """Return a summary of how many days fall into each regime.

        Args:
            regime: Series of regime labels.

        Returns:
            DataFrame with columns [regime, days, pct].
        """
        counts = regime.value_counts()
        summary = pd.DataFrame({
            "regime": counts.index,
            "days": counts.values,
            "pct": (counts.values / len(regime) * 100).round(2),
        })
        return summary.reset_index(drop=True)

    def override_regime_weights(
        self, regime_label: str, new_weights: dict[str, float]
    ) -> None:
        """Hot-swap the blending weights for a specific regime.

        Args:
            regime_label: The regime to update (e.g. "risk_on").
            new_weights: New strategy->weight mapping. Should sum to ~1.0.
        """
        self.regime_weights[regime_label] = new_weights

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_constraints(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Clamp individual positions, enforce cash floor, and normalise to 1.0."""
        # Cap any single asset.
        weights = weights.clip(lower=0.0, upper=self.max_single_asset)

        # Ensure minimum cash-like (SHY) allocation.
        if "SHY" in weights.columns:
            weights["SHY"] = weights["SHY"].clip(lower=self.min_cash_like)

        # Normalise rows to sum to 1.0.
        row_sums = weights.sum(axis=1).replace(0, np.nan)
        weights = weights.div(row_sums, axis=0).fillna(0.0)

        return weights
