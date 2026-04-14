"""Term premium decomposition using NY Fed ACM estimates."""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("logs")
_ACM_CACHE_FILE = _CACHE_DIR / "acm_cache.csv"
_ACM_URL = (
    "https://www.newyorkfed.org/medialibrary/media/research/"
    "data_indicators/ACMTermPremium.csv"
)
_CACHE_TTL_SECONDS = 86_400  # 24 hours


class TermPremiumEngine:
    """Fetch ACM term-premium data and decompose the 10-Year yield."""

    # ------------------------------------------------------------------
    # Data acquisition
    # ------------------------------------------------------------------

    def fetch_acm_data(self) -> pd.DataFrame:
        """Download ACM term-premium estimates from the NY Fed.

        Returns a DataFrame indexed by date with columns for the term
        premium at maturities 1Y through 10Y (``ACMTP01`` ... ``ACMTP10``).

        Results are cached on disk for 24 hours.  If the download fails,
        synthetic data is generated instead.
        """
        # Try disk cache first
        cached = self._load_cache()
        if cached is not None:
            return cached

        try:
            df = self._download_acm()
            self._save_cache(df)
            return df
        except Exception as exc:
            logger.warning("ACM download failed, using synthetic data: %s", exc)
            return self._synthetic_acm()

    # ------------------------------------------------------------------
    # Decomposition
    # ------------------------------------------------------------------

    def decompose(
        self,
        ten_year_yield: pd.Series,
        term_premium_10y: pd.Series,
    ) -> dict:
        """Decompose the 10-Year yield into expected rate and term premium.

        Parameters
        ----------
        ten_year_yield : pd.Series
            10-Year Treasury yield (percent).
        term_premium_10y : pd.Series
            ACM 10-Year term premium (percent).

        Returns
        -------
        dict
            ``expected_rate`` and ``term_premium`` as Series, plus the
            ``latest`` snapshot.
        """
        # Align on common dates
        common = ten_year_yield.dropna().index.intersection(
            term_premium_10y.dropna().index
        )
        tp = term_premium_10y.reindex(common)
        yld = ten_year_yield.reindex(common)
        expected = yld - tp

        latest_idx = common[-1] if len(common) else None
        latest = {}
        if latest_idx is not None:
            latest = {
                "ten_year_yield": round(float(yld.loc[latest_idx]), 4),
                "term_premium": round(float(tp.loc[latest_idx]), 4),
                "expected_rate": round(float(expected.loc[latest_idx]), 4),
            }

        return {
            "expected_rate": expected,
            "term_premium": tp,
            "ten_year_yield": yld,
            "latest": latest,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_acm(self) -> pd.DataFrame:
        """Download and parse the ACM CSV from the NY Fed."""
        import urllib.request

        req = urllib.request.Request(_ACM_URL, headers={"User-Agent": "AlphaNet/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw_bytes = resp.read()
            # Try multiple encodings — the NY Fed file sometimes uses non-UTF-8
            for encoding in ("utf-8", "latin-1", "cp1252", "iso-8859-1"):
                try:
                    raw = raw_bytes.decode(encoding)
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            else:
                raw = raw_bytes.decode("utf-8", errors="replace")

        # The CSV has a few header/comment rows before the actual data.
        # Find the row that starts with "DATE" (case-insensitive).
        lines = raw.splitlines()
        header_idx = 0
        for i, line in enumerate(lines):
            if line.strip().upper().startswith("DATE"):
                header_idx = i
                break

        csv_text = "\n".join(lines[header_idx:])
        df = pd.read_csv(io.StringIO(csv_text))
        # Find the date column (could be DATE, Date, date, etc.)
        date_col = None
        for col in df.columns:
            if col.strip().upper() == "DATE":
                date_col = col
                break
        if date_col is None:
            date_col = df.columns[0]  # assume first column is date
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.rename(columns={date_col: "date"}).set_index("date").sort_index()

        # Keep only term-premium columns (ACMTP01 .. ACMTP10)
        tp_cols = [c for c in df.columns if c.startswith("ACMTP")]
        if not tp_cols:
            raise ValueError("No ACMTP columns found in ACM CSV")
        return df[tp_cols].dropna(how="all")

    def _load_cache(self) -> pd.DataFrame | None:
        if not _ACM_CACHE_FILE.exists():
            return None
        age = time.time() - os.path.getmtime(_ACM_CACHE_FILE)
        if age > _CACHE_TTL_SECONDS:
            return None
        try:
            df = pd.read_csv(_ACM_CACHE_FILE, parse_dates=["date"], index_col="date")
            if df.empty:
                return None
            return df
        except Exception:
            return None

    def _save_cache(self, df: pd.DataFrame) -> None:
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(_ACM_CACHE_FILE)
        except Exception as exc:
            logger.warning("Failed to write ACM cache: %s", exc)

    @staticmethod
    def _synthetic_acm() -> pd.DataFrame:
        """Generate realistic synthetic term-premium data.

        Characteristics:
        - Roughly -1% to +2% range
        - Negative 2015-2021, turning positive 2022+
        """
        np.random.seed(99)
        dates = pd.bdate_range(start="2010-01-04", end="2025-12-31")
        n = len(dates)

        # Base trend: negative mid-decade, rising post-2021
        years = np.array([(d.year + d.month / 12) for d in dates])
        trend = -1.5 + 0.15 * (years - 2015)
        trend = np.clip(trend, -1.5, 2.5)

        tp10 = trend + np.random.normal(0, 0.15, n)

        # Build columns for 1Y through 10Y (shorter maturities have smaller TP)
        data = {}
        for mat in range(1, 11):
            scale = mat / 10.0
            col = f"ACMTP{mat:02d}"
            data[col] = tp10 * scale + np.random.normal(0, 0.05, n)

        return pd.DataFrame(data, index=dates)
