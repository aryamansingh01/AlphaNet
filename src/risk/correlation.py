"""Cross-asset correlation regime tracking and PCA risk concentration."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CorrelationRegimeTracker:
    """Track cross-asset correlations and detect regime shifts."""

    ASSET_ETFS: dict[str, str] = {
        "Equities": "SPY",
        "Long Treasury": "TLT",
        "Mid Treasury": "IEF",
        "HY Credit": "HYG",
        "Gold": "GLD",
        "USD": "UUP",
    }

    def compute_rolling_correlations(
        self,
        returns: pd.DataFrame,
        windows: list[int] | None = None,
    ) -> dict[int, pd.DataFrame]:
        """Compute rolling pairwise correlations at multiple lookback windows.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns with columns = asset names.
        windows : list[int]
            Lookback windows in trading days.

        Returns
        -------
        dict mapping window -> DataFrame of rolling correlations (flattened pairs).
        """
        if windows is None:
            windows = [21, 63, 252]

        result: dict[int, pd.DataFrame] = {}
        for w in windows:
            corr_series = {}
            cols = returns.columns.tolist()
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    pair = f"{cols[i]}__{cols[j]}"
                    corr_series[pair] = returns[cols[i]].rolling(w).corr(
                        returns[cols[j]]
                    )
            result[w] = pd.DataFrame(corr_series)
        return result

    def current_correlation_matrix(
        self, returns: pd.DataFrame, window: int = 63
    ) -> pd.DataFrame:
        """Return the latest correlation matrix over the trailing window."""
        recent = returns.tail(window)
        return recent.corr()

    def stock_bond_correlation(
        self,
        returns: pd.DataFrame,
        window: int = 63,
        stock_col: str = "SPY",
        bond_col: str = "TLT",
    ) -> pd.Series:
        """Extract the stock-bond correlation over time.

        This is the single most important cross-asset allocation metric.
        """
        if stock_col not in returns.columns or bond_col not in returns.columns:
            raise ValueError(
                f"Need columns '{stock_col}' and '{bond_col}' in returns DataFrame"
            )
        return returns[stock_col].rolling(window).corr(returns[bond_col]).dropna()

    def detect_correlation_regime(self, stock_bond_corr: pd.Series) -> dict:
        """Classify the current stock-bond correlation regime.

        Regimes:
        - 'negative_normal': corr < -0.2  (bonds diversify equities, pre-2022 norm)
        - 'transition':      -0.2 <= corr <= 0.2
        - 'positive_abnormal': corr > 0.2  (bonds and equities fall together)
        """
        if stock_bond_corr.empty:
            return {
                "regime": "unknown",
                "current_corr": None,
                "lookback_avg": None,
                "percentile": None,
            }

        current = float(stock_bond_corr.iloc[-1])
        lookback_avg = float(stock_bond_corr.mean())
        percentile = float(
            (stock_bond_corr < current).sum() / len(stock_bond_corr) * 100
        )

        if current < -0.2:
            regime = "negative_normal"
        elif current > 0.2:
            regime = "positive_abnormal"
        else:
            regime = "transition"

        return {
            "regime": regime,
            "current_corr": round(current, 4),
            "lookback_avg": round(lookback_avg, 4),
            "percentile": round(percentile, 1),
        }

    def pca_risk_concentration(
        self, returns: pd.DataFrame, window: int = 63
    ) -> dict:
        """Run PCA on recent returns to measure risk concentration.

        Returns
        -------
        dict with:
            n_components_90pct: how many factors explain 90% of variance
            first_component_share: % explained by PC1 (higher = more correlated)
            explained_variance: list of explained variance ratios
        """
        recent = returns.tail(window).dropna(axis=1, how="all").dropna()
        if recent.shape[0] < 10 or recent.shape[1] < 2:
            return {
                "n_components_90pct": recent.shape[1],
                "first_component_share": 1.0,
                "explained_variance": [1.0],
            }

        # Standardise
        std = recent.std()
        std[std == 0] = 1.0
        standardised = (recent - recent.mean()) / std

        # SVD-based PCA (no sklearn dependency needed)
        cov = standardised.cov().values
        eigenvalues, _ = np.linalg.eigh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        total = eigenvalues.sum()
        if total <= 0:
            return {
                "n_components_90pct": len(eigenvalues),
                "first_component_share": 0.0,
                "explained_variance": [],
            }

        explained = eigenvalues / total
        cumulative = np.cumsum(explained)
        n_90 = int(np.searchsorted(cumulative, 0.90) + 1)

        return {
            "n_components_90pct": n_90,
            "first_component_share": round(float(explained[0]), 4),
            "explained_variance": [round(float(e), 4) for e in explained],
        }

    def detect_regime_changes(
        self, stock_bond_corr: pd.Series
    ) -> list[dict]:
        """Identify dates where the correlation regime changed."""
        if stock_bond_corr.empty:
            return []

        def _classify(val: float) -> str:
            if val < -0.2:
                return "negative_normal"
            elif val > 0.2:
                return "positive_abnormal"
            return "transition"

        changes: list[dict] = []
        prev_regime = _classify(float(stock_bond_corr.iloc[0]))

        for dt, val in stock_bond_corr.items():
            cur_regime = _classify(float(val))
            if cur_regime != prev_regime:
                date_str = str(dt.date()) if hasattr(dt, "date") else str(dt)
                changes.append({
                    "date": date_str,
                    "from": prev_regime,
                    "to": cur_regime,
                })
                prev_regime = cur_regime

        return changes
