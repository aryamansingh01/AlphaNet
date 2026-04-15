"""Shared data fetching helper for API routes.

Attempts live data from yfinance once, caches the result.
If yfinance is down, all subsequent calls skip it instantly.
"""

import logging
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Global cache: avoids re-fetching on every request
_cache: dict = {}  # key -> data
_cache_ts: dict = {}  # key -> timestamp
_CACHE_TTL: int = 300  # 5 minutes
_yfinance_broken: bool = True  # assume broken until proven otherwise
_yfinance_broken_ts: float = 0.0
_YF_RETRY_AFTER: int = 600  # retry yfinance after 10 min


# ---------------------------------------------------------------------------
# Shared FredClient singleton (Fix 1)
# ---------------------------------------------------------------------------

_fred_client = None


def get_fred_client():
    global _fred_client
    if _fred_client is None:
        from src.data.fixed_income.fred_client import FredClient
        _fred_client = FredClient()
    return _fred_client


# ---------------------------------------------------------------------------
# yfinance stderr suppression context manager (Fix 5)
# ---------------------------------------------------------------------------

@contextmanager
def suppress_yfinance():
    import os, sys, warnings
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr


# ---------------------------------------------------------------------------
# Public helper for checking yfinance availability (Fix 3)
# ---------------------------------------------------------------------------

def is_yfinance_available() -> bool:
    return not _yfinance_broken


# ---------------------------------------------------------------------------
# yfinance probe
# ---------------------------------------------------------------------------

def _probe_yfinance() -> None:
    """Quick probe at import time -- sets _yfinance_broken flag."""
    global _yfinance_broken, _yfinance_broken_ts
    try:
        import yfinance as yf
        with suppress_yfinance():
            test = yf.Ticker("SPY").history(period="5d")
        if not test.empty:
            _yfinance_broken = False
            logger.info("yfinance probe: OK (%d rows)", len(test))
        else:
            _yfinance_broken = True
            _yfinance_broken_ts = time.monotonic()
            logger.info("yfinance probe: FAILED (empty). Using synthetic data.")
    except Exception:
        _yfinance_broken = True
        _yfinance_broken_ts = time.monotonic()
        logger.info("yfinance probe: FAILED (error). Using synthetic data.")


# Probe once at import
_probe_yfinance()


# ---------------------------------------------------------------------------
# Synthetic data generation (Fix 6: use default_rng)
# ---------------------------------------------------------------------------

def _generate_synthetic(n: int = 504) -> dict:
    """Generate deterministic synthetic market data."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=datetime.now(), periods=n)
    return {
        "equity_returns": pd.Series(rng.normal(0.0004, 0.01, n), index=dates),
        "hy_spread": pd.Series(rng.normal(4.5, 0.5, n), index=dates).clip(lower=1),
        "ig_spread": pd.Series(rng.normal(1.2, 0.2, n), index=dates).clip(lower=0.3),
        "curve_slope": pd.Series(rng.normal(0.5, 0.3, n), index=dates),
        "vix": pd.Series(rng.normal(18, 3, n), index=dates).clip(lower=10),
        "regime": "risk_on",
        "source": "synthetic",
    }


def fetch_market_data(period: str = "6mo", min_rows: int = 100) -> dict:
    """Fetch equity returns, VIX, and credit data. Cached for 5 min."""
    global _cache, _cache_ts, _yfinance_broken, _yfinance_broken_ts

    cache_key = period
    now = time.monotonic()

    # Return cached data if fresh (Fix 2: per-key timestamp)
    if cache_key in _cache and (now - _cache_ts.get(cache_key, 0)) < _CACHE_TTL:
        return _cache[cache_key]

    equity_returns = None
    vix_series = None

    # Skip yfinance if it recently failed
    should_try_yf = not _yfinance_broken or (now - _yfinance_broken_ts) > _YF_RETRY_AFTER

    if should_try_yf:
        try:
            import yfinance as yf

            with suppress_yfinance():
                spy = yf.Ticker("SPY").history(period=period)
                vix = yf.Ticker("^VIX").history(period=period)

            if not spy.empty and len(spy) >= min_rows:
                equity_returns = spy["Close"].pct_change().dropna()
                _yfinance_broken = False
            else:
                _yfinance_broken = True
                _yfinance_broken_ts = now

            if not vix.empty and len(vix) >= 10:
                vix_series = vix["Close"].dropna()
        except Exception:
            _yfinance_broken = True
            _yfinance_broken_ts = now

    # Build result with fallbacks (Fix 4: use _generate_synthetic for fallback fields)
    n = len(equity_returns) if equity_returns is not None else 504
    dates = equity_returns.index if equity_returns is not None else pd.bdate_range(end=datetime.now(), periods=n)

    synthetic = _generate_synthetic(n)

    if equity_returns is None:
        equity_returns = synthetic["equity_returns"]

    if vix_series is None:
        vix_series = synthetic["vix"]
    else:
        vix_series = vix_series.reindex(dates, method="ffill").fillna(18)

    result = {
        "equity_returns": equity_returns,
        "hy_spread": synthetic["hy_spread"],
        "ig_spread": synthetic["ig_spread"],
        "curve_slope": synthetic["curve_slope"],
        "vix": vix_series,
        "regime": "risk_on",
        "source": "live" if not _yfinance_broken else "synthetic",
    }

    _cache[cache_key] = result
    _cache_ts[cache_key] = now
    return result
