"""Market regime detection using GMM and HMM across asset classes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM
from enum import Enum


class Regime(str, Enum):
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    CRISIS = "crisis"
    TRANSITION = "transition"
    CREDIT_BENIGN = "credit_benign"
    CREDIT_STRESS = "credit_stress"
    CREDIT_CRISIS = "credit_crisis"


class RegimeDetector:
    """Detect market regimes across equity and credit markets."""

    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.gmm: GaussianMixture | None = None
        self.hmm: GaussianHMM | None = None
        self.regime_map: dict[int, Regime] = {}
        self.credit_gmm: GaussianMixture | None = None
        self.credit_regime_map: dict[int, Regime] = {}
        self._last_features: pd.DataFrame | None = None
        self._last_labels: np.ndarray | None = None
        self._last_credit_features: pd.DataFrame | None = None
        self._last_credit_labels: np.ndarray | None = None

    def build_features(
        self,
        equity_returns: pd.Series,
        credit_spreads: pd.Series,
        vix: pd.Series,
        curve_slope: pd.Series,
    ) -> pd.DataFrame:
        """Build feature matrix for regime detection."""
        # Align all series to a common index first
        common_idx = equity_returns.index
        credit_aligned = credit_spreads.reindex(common_idx, method="ffill").fillna(credit_spreads.mean())
        vix_aligned = vix.reindex(common_idx, method="ffill").fillna(20)
        slope_aligned = curve_slope.reindex(common_idx, method="ffill").fillna(0.5)

        features = pd.DataFrame({
            "equity_vol": equity_returns.rolling(21).std() * np.sqrt(252),
            "equity_momentum": equity_returns.rolling(63).mean(),
            "credit_spread": credit_aligned,
            "credit_spread_change": credit_aligned.pct_change(21),
            "vix": vix_aligned,
            "curve_slope": slope_aligned,
        }).dropna()
        return features

    def fit_gmm(self, features: pd.DataFrame) -> np.ndarray:
        """Fit GMM on full dataset. WARNING: causes look-ahead bias in backtests.

        Use fit_gmm_expanding() for backtest-safe regime labels.
        """
        self.gmm = GaussianMixture(
            n_components=self.n_regimes, covariance_type="full", random_state=42
        )
        labels = self.gmm.fit_predict(features)
        self._map_regimes(features, labels)
        self._last_features = features
        self._last_labels = labels
        return labels

    def fit_gmm_expanding(
        self, features: pd.DataFrame, min_train: int = 252
    ) -> np.ndarray:
        """Fit GMM with expanding window — no look-ahead bias.

        At each time step t, the model is trained only on data [0, t-1]
        and predicts the regime at t. This is safe for backtesting.

        Args:
            features: Feature matrix (rows=dates).
            min_train: Minimum training window before first prediction.

        Returns:
            Array of regime labels. Labels before min_train are -1 (unknown).
        """
        n = len(features)
        labels = np.full(n, -1, dtype=int)

        for t in range(min_train, n):
            train = features.iloc[:t]
            gmm = GaussianMixture(
                n_components=self.n_regimes, covariance_type="full", random_state=42
            )
            gmm.fit(train)
            labels[t] = gmm.predict(features.iloc[[t]])[0]

        # Map regimes using only the labeled portion
        labeled_mask = labels >= 0
        if labeled_mask.any():
            self._map_regimes(features.loc[labeled_mask], labels[labeled_mask])

        self.gmm = gmm  # keep the last fitted model
        self._last_features = features
        self._last_labels = labels
        return labels

    def fit_hmm(self, features: pd.DataFrame) -> np.ndarray:
        """Fit HMM on full dataset. WARNING: causes look-ahead bias in backtests.

        Use fit_gmm_expanding() for backtest-safe regime labels.
        """
        self.hmm = GaussianHMM(
            n_components=self.n_regimes, covariance_type="full", n_iter=100, random_state=42
        )
        self.hmm.fit(features)
        labels = self.hmm.predict(features)
        self._map_regimes(features, labels)
        self._last_features = features
        self._last_labels = labels
        return labels

    def _map_regimes(self, features: pd.DataFrame, labels: np.ndarray):
        """Map numeric labels to named regimes based on characteristics."""
        for label in range(self.n_regimes):
            mask = labels == label
            avg_vol = features.loc[mask, "equity_vol"].mean()
            avg_spread = features.loc[mask, "credit_spread"].mean()

            if avg_vol > features["equity_vol"].quantile(0.75):
                self.regime_map[label] = Regime.CRISIS
            elif avg_spread < features["credit_spread"].quantile(0.25):
                self.regime_map[label] = Regime.RISK_ON
            else:
                self.regime_map[label] = Regime.RISK_OFF

    def get_current_regime(self, features: pd.DataFrame) -> Regime:
        """Get the current regime from latest data point."""
        if self.gmm is None:
            raise ValueError("Model not fitted. Call fit_gmm() or fit_hmm() first.")
        latest = features.iloc[[-1]]
        label = self.gmm.predict(latest)[0]
        return self.regime_map.get(label, Regime.TRANSITION)

    # ------------------------------------------------------------------
    # Credit-specific regime detection
    # ------------------------------------------------------------------

    def fit_credit_regime(
        self,
        ig_oas: pd.Series,
        hy_oas: pd.Series,
        ted_spread: pd.Series,
        n_credit_regimes: int = 3,
    ) -> np.ndarray:
        """Detect credit-specific regimes using spread data.

        Parameters
        ----------
        ig_oas : pd.Series
            Investment-grade option-adjusted spread.
        hy_oas : pd.Series
            High-yield option-adjusted spread.
        ted_spread : pd.Series
            TED spread (3-month LIBOR minus T-bill).
        n_credit_regimes : int
            Number of regimes to detect (default 3).

        Returns
        -------
        np.ndarray
            Integer regime labels aligned to the feature index.
        """
        features = pd.DataFrame({
            "ig_oas": ig_oas,
            "hy_oas": hy_oas,
            "spread_momentum": hy_oas.pct_change(21),
            "ted_spread": ted_spread,
        }).dropna()

        self.credit_gmm = GaussianMixture(
            n_components=n_credit_regimes,
            covariance_type="full",
            random_state=42,
        )
        labels = self.credit_gmm.fit_predict(features)

        # Map labels to credit regime names by average HY spread
        for label in range(n_credit_regimes):
            mask = labels == label
            avg_hy = features.loc[mask, "hy_oas"].mean()

            if avg_hy > features["hy_oas"].quantile(0.75):
                self.credit_regime_map[label] = Regime.CREDIT_CRISIS
            elif avg_hy < features["hy_oas"].quantile(0.25):
                self.credit_regime_map[label] = Regime.CREDIT_BENIGN
            else:
                self.credit_regime_map[label] = Regime.CREDIT_STRESS

        self._last_credit_features = features
        self._last_credit_labels = labels
        return labels

    # ------------------------------------------------------------------
    # Combined regime
    # ------------------------------------------------------------------

    def combined_regime(
        self,
        equity_labels: pd.Series,
        credit_labels: pd.Series,
    ) -> pd.Series:
        """Merge equity and credit regime labels into a unified regime.

        The mapping rules:
        - If either is CRISIS / CREDIT_CRISIS -> CRISIS
        - If both are benign (RISK_ON + CREDIT_BENIGN) -> RISK_ON
        - Otherwise -> RISK_OFF

        Parameters
        ----------
        equity_labels : pd.Series
            Regime enum values indexed by date.
        credit_labels : pd.Series
            Regime enum values indexed by date.

        Returns
        -------
        pd.Series
            Unified Regime labels indexed by date.
        """
        common_idx = equity_labels.index.intersection(credit_labels.index)
        equity_aligned = equity_labels.loc[common_idx]
        credit_aligned = credit_labels.loc[common_idx]

        combined: list[Regime] = []
        for eq, cr in zip(equity_aligned, credit_aligned):
            if eq == Regime.CRISIS or cr == Regime.CREDIT_CRISIS:
                combined.append(Regime.CRISIS)
            elif eq == Regime.RISK_ON and cr == Regime.CREDIT_BENIGN:
                combined.append(Regime.RISK_ON)
            elif eq == Regime.TRANSITION or cr == Regime.CREDIT_STRESS:
                combined.append(Regime.TRANSITION)
            else:
                combined.append(Regime.RISK_OFF)

        return pd.Series(combined, index=common_idx, name="regime")

    # ------------------------------------------------------------------
    # Transition matrix
    # ------------------------------------------------------------------

    def get_transition_matrix(self, labels: np.ndarray) -> pd.DataFrame:
        """Compute the regime transition probability matrix.

        Parameters
        ----------
        labels : np.ndarray
            Sequence of integer regime labels (e.g. from fit_gmm).

        Returns
        -------
        pd.DataFrame
            Square matrix where entry (i, j) is P(next=j | current=i).
        """
        unique = sorted(set(labels))
        n = len(unique)
        idx_map = {v: i for i, v in enumerate(unique)}
        counts = np.zeros((n, n), dtype=float)

        for current, nxt in zip(labels[:-1], labels[1:]):
            counts[idx_map[current]][idx_map[nxt]] += 1

        # Normalise rows to probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        probs = counts / row_sums

        regime_names = [
            self.regime_map.get(u, Regime.TRANSITION).value for u in unique
        ]
        return pd.DataFrame(probs, index=regime_names, columns=regime_names)

    # ------------------------------------------------------------------
    # Regime history
    # ------------------------------------------------------------------

    def regime_history(self) -> pd.DataFrame:
        """Return a DataFrame of dates and their regime labels.

        Requires that :meth:`fit_gmm` or :meth:`fit_hmm` has been called
        so that ``_last_features`` and ``_last_labels`` are populated.

        Returns
        -------
        pd.DataFrame
            Columns: ``regime_id`` (int), ``regime`` (Regime enum value).
            Index matches the feature DataFrame's date index.
        """
        if self._last_features is None or self._last_labels is None:
            raise ValueError(
                "No regime history available. Call fit_gmm() or fit_hmm() first."
            )

        mapped = [
            self.regime_map.get(int(lbl), Regime.TRANSITION)
            for lbl in self._last_labels
        ]

        return pd.DataFrame(
            {"regime_id": self._last_labels, "regime": mapped},
            index=self._last_features.index,
        )
