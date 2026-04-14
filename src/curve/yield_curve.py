"""Yield curve construction, fitting, and analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from nelson_siegel_svensson import NelsonSiegelCurve
from nelson_siegel_svensson.calibrate import calibrate_ns_ols
from sklearn.decomposition import PCA


# Maturities in years for each Treasury series
MATURITIES = np.array([
    1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30
])

MATURITY_LABELS = [
    "1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"
]


class YieldCurveEngine:
    """Build, fit, and analyze yield curves."""

    def fit_nelson_siegel(self, yields: np.ndarray) -> NelsonSiegelCurve:
        """Fit Nelson-Siegel model to observed yields."""
        curve, _ = calibrate_ns_ols(MATURITIES, yields / 100)
        return curve

    def get_curve_metrics(self, yields: pd.Series) -> dict:
        """Calculate key curve metrics from a single day's yields."""
        return {
            "level": float(yields.mean()),
            "slope": float(yields.iloc[-1] - yields.iloc[0]),  # 30Y - 1M
            "curvature": float(
                2 * yields.get("5Y", 0) - yields.get("2Y", 0) - yields.get("10Y", 0)
            ),
            "spread_2s10s": float(yields.get("10Y", 0) - yields.get("2Y", 0)),
            "spread_3m10y": float(yields.get("10Y", 0) - yields.get("3M", 0)),
            "inverted": bool(yields.get("2Y", 0) > yields.get("10Y", 0)),
        }

    def interpolate_curve(
        self, curve: NelsonSiegelCurve, maturities: np.ndarray | None = None
    ) -> pd.Series:
        """Interpolate yields at arbitrary maturities."""
        if maturities is None:
            maturities = np.linspace(0.08, 30, 100)
        yields = curve(maturities) * 100
        return pd.Series(yields, index=maturities, name="yield")

    # ------------------------------------------------------------------
    # Curve history & inversion detection
    # ------------------------------------------------------------------

    def get_curve_history(self, yield_df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of daily curve metrics (slope, level, curvature).

        Parameters
        ----------
        yield_df : pd.DataFrame
            Rows are dates, columns are maturity labels matching
            ``MATURITY_LABELS`` (e.g. "2Y", "10Y").

        Returns
        -------
        pd.DataFrame
            Columns: level, slope, curvature, spread_2s10s, spread_3m10y,
            inverted.  Index matches the input date index.
        """
        records: list[dict] = []
        for date, row in yield_df.iterrows():
            metrics = self.get_curve_metrics(row)
            metrics["date"] = date
            records.append(metrics)

        history = pd.DataFrame(records).set_index("date")
        return history

    def detect_inversion(self, yield_df: pd.DataFrame) -> pd.DataFrame:
        """Identify periods where the 2s10s spread was negative.

        Parameters
        ----------
        yield_df : pd.DataFrame
            Same format as :meth:`get_curve_history` expects.

        Returns
        -------
        pd.DataFrame
            Each row describes one inversion window with columns
            ``start``, ``end``, and ``duration_days``.
        """
        spread = yield_df["10Y"] - yield_df["2Y"]
        inverted = spread < 0

        # Detect contiguous blocks of inversion
        blocks = inverted.ne(inverted.shift()).cumsum()
        inversions: list[dict] = []
        for _, group in inverted.groupby(blocks):
            if group.iloc[0]:  # inverted block
                start = group.index[0]
                end = group.index[-1]
                duration = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
                inversions.append({
                    "start": start,
                    "end": end,
                    "duration_days": duration,
                })

        return pd.DataFrame(inversions)

    # ------------------------------------------------------------------
    # PCA on yield changes
    # ------------------------------------------------------------------

    def curve_pca(
        self,
        yield_df: pd.DataFrame,
        n_components: int = 3,
    ) -> dict:
        """Run PCA on daily yield changes to extract level/slope/curvature factors.

        Parameters
        ----------
        yield_df : pd.DataFrame
            Rows are dates, columns are maturity labels.
        n_components : int
            Number of principal components to extract (default 3).

        Returns
        -------
        dict
            ``components`` – pd.DataFrame of principal component loadings,
            ``explained_variance`` – variance explained per component,
            ``factors`` – pd.DataFrame of daily factor scores.
        """
        changes = yield_df.diff().dropna()

        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(changes.values)

        component_names = [f"PC{i+1}" for i in range(n_components)]

        components = pd.DataFrame(
            pca.components_,
            index=component_names,
            columns=changes.columns,
        )

        factors = pd.DataFrame(
            scores,
            index=changes.index,
            columns=component_names,
        )

        return {
            "components": components,
            "explained_variance": pca.explained_variance_ratio_,
            "factors": factors,
        }
