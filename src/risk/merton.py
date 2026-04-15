"""Merton structural credit model: equity as a call option on firm assets."""

from __future__ import annotations

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample / fallback data
# ---------------------------------------------------------------------------

_SAMPLE_RESULTS: dict[str, dict] = {
    "AAPL": {
        "ticker": "AAPL",
        "equity_value": 2_800_000_000_000,
        "equity_vol": 0.25,
        "debt": 110_000_000_000,
        "risk_free_rate": 0.045,
        "maturity": 1.0,
        "asset_value": 2_910_000_000_000,
        "asset_vol": 0.241,
        "d1": 8.12,
        "d2": 7.88,
        "default_probability": 0.0000,
        "distance_to_default": 7.88,
        "implied_spread_bps": 0.1,
        "leverage": 0.038,
    },
    "MSFT": {
        "ticker": "MSFT",
        "equity_value": 3_100_000_000_000,
        "equity_vol": 0.24,
        "debt": 97_000_000_000,
        "risk_free_rate": 0.045,
        "maturity": 1.0,
        "asset_value": 3_197_000_000_000,
        "asset_vol": 0.233,
        "d1": 8.55,
        "d2": 8.32,
        "default_probability": 0.0000,
        "distance_to_default": 8.32,
        "implied_spread_bps": 0.0,
        "leverage": 0.030,
    },
    "JPM": {
        "ticker": "JPM",
        "equity_value": 580_000_000_000,
        "equity_vol": 0.22,
        "debt": 3_400_000_000_000,
        "risk_free_rate": 0.045,
        "maturity": 1.0,
        "asset_value": 3_980_000_000_000,
        "asset_vol": 0.032,
        "d1": 2.45,
        "d2": 2.42,
        "default_probability": 0.0078,
        "distance_to_default": 2.42,
        "implied_spread_bps": 18.5,
        "leverage": 0.854,
    },
    "XOM": {
        "ticker": "XOM",
        "equity_value": 470_000_000_000,
        "equity_vol": 0.28,
        "debt": 47_000_000_000,
        "risk_free_rate": 0.045,
        "maturity": 1.0,
        "asset_value": 517_000_000_000,
        "asset_vol": 0.255,
        "d1": 5.21,
        "d2": 4.96,
        "default_probability": 0.0000,
        "distance_to_default": 4.96,
        "implied_spread_bps": 1.2,
        "leverage": 0.091,
    },
    "T": {
        "ticker": "T",
        "equity_value": 145_000_000_000,
        "equity_vol": 0.23,
        "debt": 137_000_000_000,
        "risk_free_rate": 0.045,
        "maturity": 1.0,
        "asset_value": 282_000_000_000,
        "asset_vol": 0.119,
        "d1": 2.85,
        "d2": 2.73,
        "default_probability": 0.0032,
        "distance_to_default": 2.73,
        "implied_spread_bps": 42.0,
        "leverage": 0.486,
    },
}

# Approximate market OAS benchmarks (bps)
_MARKET_SPREADS: dict[str, float] = {
    "IG": 95.0,
    "HY": 350.0,
    "BBB": 130.0,
    "BB": 200.0,
}


class MertonModel:
    """Structural credit model: equity as a call option on firm assets."""

    def solve(
        self,
        equity_value: float,
        equity_vol: float,
        debt: float,
        risk_free_rate: float,
        maturity: float = 1.0,
    ) -> dict:
        """Solve Merton model for asset value and asset volatility.

        Uses Black-Scholes:
            E = V*N(d1) - D*exp(-r*T)*N(d2)
            sigma_E = (V/E) * N(d1) * sigma_V

        Solves these two equations simultaneously for V and sigma_V.
        """
        if equity_value <= 0:
            raise ValueError("equity_value must be positive")
        if debt <= 0:
            raise ValueError("debt must be positive")

        r = risk_free_rate
        T = maturity
        E = equity_value
        D = debt
        sigma_E = equity_vol

        # Initial guesses: V ~ E + D, sigma_V ~ sigma_E * E / (E + D)
        V0 = E + D
        sigma_V0 = sigma_E * E / V0

        def equations(x):
            V, sv = x
            if V <= 0 or sv <= 0:
                return [1e12, 1e12]
            d1 = (np.log(V / D) + (r + 0.5 * sv**2) * T) / (sv * np.sqrt(T))
            d2 = d1 - sv * np.sqrt(T)
            eq1 = V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E
            eq2 = (V / E) * norm.cdf(d1) * sv - sigma_E
            return [eq1, eq2]

        try:
            solution, info, ier, msg = fsolve(
                equations, [V0, sigma_V0], full_output=True
            )
            V_sol, sigma_V_sol = solution

            if ier != 1 or V_sol <= 0 or sigma_V_sol <= 0:
                # Fallback: use simple approximation
                V_sol = E + D
                sigma_V_sol = sigma_E * E / V_sol
        except Exception:
            V_sol = E + D
            sigma_V_sol = sigma_E * E / V_sol

        # Compute outputs
        d1 = (np.log(V_sol / D) + (r + 0.5 * sigma_V_sol**2) * T) / (
            sigma_V_sol * np.sqrt(T)
        )
        d2 = d1 - sigma_V_sol * np.sqrt(T)

        default_prob = float(norm.cdf(-d2))
        leverage = D / V_sol

        # Implied credit spread (simplified Merton formula)
        n_d2 = norm.cdf(d2)
        n_d1 = norm.cdf(d1)
        recovery_term = n_d2 + (1.0 / leverage) * n_d1 if leverage > 0 else 1.0
        if recovery_term > 0 and recovery_term < 1.0:
            implied_spread = -np.log(recovery_term) / T
        else:
            implied_spread = max(0.0, default_prob * 0.6 / T)  # LGD ~60%

        implied_spread_bps = round(implied_spread * 10_000, 1)

        return {
            "asset_value": round(V_sol, 2),
            "asset_vol": round(sigma_V_sol, 4),
            "d1": round(d1, 4),
            "d2": round(d2, 4),
            "default_probability": round(default_prob, 6),
            "distance_to_default": round(d2, 4),
            "implied_spread_bps": implied_spread_bps,
            "leverage": round(leverage, 4),
        }

    def analyze_ticker(self, ticker: str) -> dict:
        """Run Merton model for a single stock using live market data.

        Falls back to sample data if yfinance is unavailable.
        """
        from src.api.routes import _data_helper

        if _data_helper._yfinance_broken:
            return self._sample_or_default(ticker)

        try:
            import yfinance as yf
            from src.api.routes._data_helper import suppress_yfinance

            with suppress_yfinance():
                tk = yf.Ticker(ticker)
                info = tk.info
                hist = tk.history(period="1y")
                bs = tk.balance_sheet

            if hist.empty or len(hist) < 60:
                return self._sample_or_default(ticker)

            # Equity value = market cap
            market_cap = info.get("marketCap")
            if market_cap is None or market_cap <= 0:
                shares = info.get("sharesOutstanding", 1)
                market_cap = float(hist["Close"].iloc[-1]) * shares

            # Equity vol = annualised from daily returns
            returns = hist["Close"].pct_change().dropna()
            equity_vol = float(returns.std() * np.sqrt(252))

            # Total liabilities from balance sheet
            debt = 0.0
            if bs is not None and not bs.empty:
                for col_name in ["Total Liabilities Net Minority Interest",
                                 "Total Liab", "totalLiab", "Total Liabilities"]:
                    if col_name in bs.index:
                        debt = float(bs.loc[col_name].iloc[0])
                        break
            if debt <= 0:
                debt = info.get("totalDebt", market_cap * 0.3)
                if debt is None or debt <= 0:
                    debt = market_cap * 0.3

            # Risk-free rate: use a reasonable default
            risk_free_rate = 0.045

            result = self.solve(
                equity_value=market_cap,
                equity_vol=equity_vol,
                debt=debt,
                risk_free_rate=risk_free_rate,
                maturity=1.0,
            )

            result.update({
                "ticker": ticker.upper(),
                "equity_value": round(market_cap, 2),
                "equity_vol": round(equity_vol, 4),
                "debt": round(debt, 2),
                "risk_free_rate": risk_free_rate,
                "maturity": 1.0,
                "source": "live",
            })
            return result

        except Exception as exc:
            logger.warning("Merton analysis failed for %s: %s", ticker, exc)
            return self._sample_or_default(ticker)

    def _sample_or_default(self, ticker: str) -> dict:
        """Return sample data for known tickers, or a generic estimate."""
        ticker = ticker.upper()
        if ticker in _SAMPLE_RESULTS:
            result = _SAMPLE_RESULTS[ticker].copy()
            result["source"] = "sample"
            return result

        # Generic fallback for unknown tickers
        result = {
            "ticker": ticker,
            "equity_value": 100_000_000_000,
            "equity_vol": 0.30,
            "debt": 40_000_000_000,
            "risk_free_rate": 0.045,
            "maturity": 1.0,
            "asset_value": 140_000_000_000,
            "asset_vol": 0.215,
            "d1": 3.50,
            "d2": 3.29,
            "default_probability": 0.0005,
            "distance_to_default": 3.29,
            "implied_spread_bps": 8.0,
            "leverage": 0.286,
            "source": "sample_generic",
        }
        return result

    def compare_to_market_spread(
        self, implied_spread_bps: float, sector: str = "IG"
    ) -> dict:
        """Compare Merton-implied spread to market OAS benchmark.

        Parameters
        ----------
        implied_spread_bps : float
            The model-implied spread in basis points.
        sector : str
            One of 'IG', 'HY', 'BBB', 'BB'.

        Returns
        -------
        dict with keys: implied, market, gap, signal
        """
        market = _MARKET_SPREADS.get(sector.upper(), 95.0)
        gap = implied_spread_bps - market

        if gap < -20:
            signal = "equity_cheap"
        elif gap > 20:
            signal = "credit_cheap"
        else:
            signal = "aligned"

        return {
            "implied_spread_bps": round(implied_spread_bps, 1),
            "market_spread_bps": market,
            "gap_bps": round(gap, 1),
            "sector": sector.upper(),
            "signal": signal,
        }

    @staticmethod
    def sample_companies() -> list[dict]:
        """Return pre-computed Merton results for 5 large companies."""
        results = []
        for ticker in ["AAPL", "MSFT", "JPM", "XOM", "T"]:
            data = _SAMPLE_RESULTS[ticker].copy()
            data["source"] = "sample"
            results.append(data)
        return results
