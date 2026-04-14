"""Cross-asset divergence signals — the core alpha of AlphaNet."""

import pandas as pd
import numpy as np


class CrossAssetSignals:
    """Detect divergences between equity and credit markets."""

    def credit_equity_divergence(
        self,
        equity_returns: pd.Series,
        hy_spread: pd.Series,
        lookback: int = 21,
    ) -> pd.Series:
        """Detect when credit and equity markets disagree.

        Credit widening + equity rallying = bearish signal (credit leads)
        Credit tightening + equity falling = bullish signal (credit leads)

        Historically, credit is right ~65% of the time.
        """
        equity_trend = equity_returns.rolling(lookback).mean()
        spread_trend = hy_spread.pct_change(lookback)

        divergence = pd.Series(0.0, index=equity_returns.index)
        # Credit bearish, equity bullish → bearish signal
        divergence[(spread_trend > 0) & (equity_trend > 0)] = -1
        # Credit bullish, equity bearish → bullish signal
        divergence[(spread_trend < 0) & (equity_trend < 0)] = 1
        return divergence

    def curve_equity_signal(
        self,
        equity_prices: pd.Series,
        curve_slope: pd.Series,
    ) -> pd.Series:
        """Yield curve inversion as equity risk signal.

        Inverted curve → recession risk → reduce equity exposure
        Steepening from inversion → recovery → increase equity
        """
        signals = pd.Series(0.0, index=curve_slope.index)
        inverted = curve_slope < 0
        signals[inverted] = -0.5  # reduce equity
        # Re-steepening after inversion is historically very bullish
        un_inverting = (~inverted) & inverted.shift(21)
        signals[un_inverting] = 1.0
        return signals

    def vix_term_structure(
        self,
        vix_spot: pd.Series,
        vix_futures: pd.Series,
        threshold: float = 0.0,
    ) -> pd.Series:
        """Signal based on VIX futures term structure (contango vs backwardation).

        In normal markets VIX futures trade above spot (contango), reflecting
        the insurance premium. When futures fall below spot (backwardation),
        the market is pricing near-term fear higher than future fear — a
        strong risk-off indicator.

        Args:
            vix_spot: VIX spot index level.
            vix_futures: Front-month VIX futures price (or VX1 continuous).
            threshold: Minimum spread to consider meaningful (default 0).

        Returns:
            Series of signals: -1 (risk-off / backwardation),
            0 (neutral), +1 (risk-on / contango).
        """
        term_spread = vix_futures - vix_spot
        signals = pd.Series(0.0, index=vix_spot.index)
        signals[term_spread < -threshold] = -1.0   # backwardation → risk-off
        signals[term_spread > threshold] = 1.0      # contango → risk-on
        return signals

    def flight_to_quality(
        self,
        spy_returns: pd.Series,
        tlt_returns: pd.Series,
        lookback: int = 5,
        equity_drop_threshold: float = -0.01,
        bond_rise_threshold: float = 0.005,
    ) -> pd.Series:
        """Detect simultaneous equity selling and Treasury buying.

        A classic flight-to-quality pattern: SPY falls while TLT rises over
        a short window. When this occurs, it signals institutional de-risking
        and the move often has momentum.

        Args:
            spy_returns: Daily returns for SPY (or broad equity proxy).
            tlt_returns: Daily returns for TLT (long-duration Treasuries).
            lookback: Rolling window (trading days) for cumulative returns.
            equity_drop_threshold: Cumulative equity return below which
                selling is considered meaningful.
            bond_rise_threshold: Cumulative bond return above which buying
                is considered meaningful.

        Returns:
            Series of signals: -1 (flight-to-quality detected, risk-off),
            +1 (reverse flight / risk-on rotation), 0 (no signal).
        """
        cum_spy = spy_returns.rolling(lookback).sum()
        cum_tlt = tlt_returns.rolling(lookback).sum()

        signals = pd.Series(0.0, index=spy_returns.index)

        # Flight to quality: equities down, bonds up
        ftq_mask = (cum_spy < equity_drop_threshold) & (cum_tlt > bond_rise_threshold)
        signals[ftq_mask] = -1.0

        # Reverse rotation: equities up, bonds down (risk-on)
        reverse_mask = (cum_spy > abs(equity_drop_threshold)) & (cum_tlt < -bond_rise_threshold)
        signals[reverse_mask] = 1.0

        return signals

    def risk_on_off_composite(
        self,
        equity_returns: pd.Series,
        hy_spread: pd.Series,
        vix: pd.Series,
        curve_slope: pd.Series,
    ) -> pd.Series:
        """Composite risk-on/risk-off score from all cross-asset signals."""
        score = pd.Series(0.0, index=equity_returns.index)

        # Equity momentum component (shift to avoid look-ahead)
        eq_mom = equity_returns.rolling(21).mean().shift(1)
        score += np.where(eq_mom > 0, 1, -1) * 0.25

        # Credit spread component (shift rolling stats)
        spread_mean = hy_spread.rolling(252).mean().shift(1)
        spread_std = hy_spread.rolling(252).std().shift(1)
        spread_z = (hy_spread - spread_mean) / spread_std
        score += np.where(spread_z < 0, 1, -1) * 0.30

        # VIX component (shift rolling stats)
        vix_mean = vix.rolling(252).mean().shift(1)
        vix_std = vix.rolling(252).std().shift(1)
        vix_z = (vix - vix_mean) / vix_std
        score += np.where(vix_z < 0, 1, -1) * 0.25

        # Curve component (use previous day's slope)
        score += np.where(curve_slope.shift(1) > 0, 1, -1) * 0.20

        return score
