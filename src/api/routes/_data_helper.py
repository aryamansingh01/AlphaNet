"""Shared data fetching helper for API routes.

Attempts live data from yfinance once, caches the result.
If yfinance is down, all subsequent calls skip it instantly.
"""

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Global cache: avoids re-fetching on every request
_cache: dict = {}
_cache_ts: float = 0.0
_CACHE_TTL: int = 300  # 5 minutes
_yfinance_broken: bool = True  # assume broken until proven otherwise
_yfinance_broken_ts: float = 0.0
_YF_RETRY_AFTER: int = 600  # retry yfinance after 10 min


def _probe_yfinance() -> None:
    """Quick probe at import time — sets _yfinance_broken flag."""
    global _yfinance_broken, _yfinance_broken_ts
    try:
        import os, sys, warnings
        import yfinance as yf
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test = yf.Ticker("SPY").history(period="5d")
        finally:
            sys.stderr.close()
            sys.stderr = old_stderr
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


def _generate_synthetic(n: int = 504) -> dict:
    """Generate deterministic synthetic market data."""
    np.random.seed(42)
    dates = pd.bdate_range(end=datetime.now(), periods=n)
    return {
        "equity_returns": pd.Series(np.random.normal(0.0004, 0.01, n), index=dates),
        "hy_spread": pd.Series(np.random.normal(4.5, 0.5, n), index=dates).clip(lower=1),
        "ig_spread": pd.Series(np.random.normal(1.2, 0.2, n), index=dates).clip(lower=0.3),
        "curve_slope": pd.Series(np.random.normal(0.5, 0.3, n), index=dates),
        "vix": pd.Series(np.random.normal(18, 3, n), index=dates).clip(lower=10),
        "regime": "risk_on",
        "source": "synthetic",
    }


def fetch_market_data(period: str = "6mo", min_rows: int = 100) -> dict:
    """Fetch equity returns, VIX, and credit data. Cached for 5 min."""
    global _cache, _cache_ts, _yfinance_broken, _yfinance_broken_ts

    cache_key = period
    now = time.monotonic()

    # Return cached data if fresh
    if cache_key in _cache and (now - _cache_ts) < _CACHE_TTL:
        return _cache[cache_key]

    equity_returns = None
    vix_series = None

    # Skip yfinance if it recently failed
    should_try_yf = not _yfinance_broken or (now - _yfinance_broken_ts) > _YF_RETRY_AFTER

    if should_try_yf:
        try:
            import os
            import sys
            import warnings
            import yfinance as yf

            old_stderr = sys.stderr
            sys.stderr = open(os.devnull, "w")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    spy = yf.Ticker("SPY").history(period=period)
                    vix = yf.Ticker("^VIX").history(period=period)
            finally:
                sys.stderr.close()
                sys.stderr = old_stderr

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

    # Build result with fallbacks
    n = len(equity_returns) if equity_returns is not None else 504
    dates = equity_returns.index if equity_returns is not None else pd.bdate_range(end=datetime.now(), periods=n)

    if equity_returns is None:
        np.random.seed(42)
        equity_returns = pd.Series(np.random.normal(0.0004, 0.01, n), index=dates)

    if vix_series is None:
        np.random.seed(43)
        vix_series = pd.Series(np.random.normal(18, 3, n), index=dates).clip(lower=10)
    else:
        vix_series = vix_series.reindex(dates, method="ffill").fillna(18)

    np.random.seed(44)
    result = {
        "equity_returns": equity_returns,
        "hy_spread": pd.Series(np.random.normal(4.5, 0.5, n), index=dates).clip(lower=1),
        "ig_spread": pd.Series(np.random.normal(1.2, 0.2, n), index=dates).clip(lower=0.3),
        "curve_slope": pd.Series(np.random.normal(0.5, 0.3, n), index=dates),
        "vix": vix_series,
        "regime": "risk_on",
        "source": "live" if not _yfinance_broken else "synthetic",
    }

    _cache[cache_key] = result
    _cache_ts = now
    return result
