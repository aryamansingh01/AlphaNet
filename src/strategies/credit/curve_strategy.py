"""Yield curve trading strategies."""

import pandas as pd
import numpy as np


class CurveStrategy:
    """Strategies based on yield curve shape and movements."""

    def flattener_steepener(self, curve_slope: pd.Series, threshold: float = 0.5) -> pd.Series:
        """Trade curve flattening/steepening.

        When slope is high (steep curve) → expect flattening → short TLT, long SHY
        When slope is low/inverted → expect steepening → long TLT, short SHY
        """
        # Shift by 1 to avoid look-ahead: stats are from [t-252, t-1]
        z_score = (curve_slope - curve_slope.rolling(252).mean().shift(1)) / curve_slope.rolling(252).std().shift(1)
        signals = pd.Series(0.0, index=curve_slope.index)
        signals[z_score > threshold] = -1   # expect flattening
        signals[z_score < -threshold] = 1   # expect steepening
        return signals

    def duration_timing(
        self, curve_slope: pd.Series, regime: pd.Series
    ) -> pd.Series:
        """Adjust duration exposure based on curve and regime.

        Risk-on + steep curve → short duration (SHY)
        Risk-off + flat/inverted → long duration (TLT)
        """
        signals = pd.Series(0.0, index=curve_slope.index)
        signals[(regime == "risk_on") & (curve_slope > 0)] = -1   # short duration
        signals[(regime == "risk_off") | (curve_slope < 0)] = 1    # long duration
        return signals

    def butterfly_trade(
        self,
        yield_2y: pd.Series,
        yield_5y: pd.Series,
        yield_10y: pd.Series,
        lookback: int = 63,
        threshold: float = 1.5,
    ) -> dict[str, pd.Series]:
        """Trade the curvature via a 2s5s10s butterfly spread.

        Butterfly = 2 * yield_5y - yield_2y - yield_10y.
        A high butterfly means the belly is cheap relative to the wings;
        a low butterfly means the belly is rich.

        When the butterfly z-score is elevated, sell the belly (short IEF)
        and buy the wings (long SHY + TLT). Reverse when depressed.

        Args:
            yield_2y: 2-year Treasury yield series.
            yield_5y: 5-year Treasury yield series.
            yield_10y: 10-year Treasury yield series.
            lookback: Rolling window for z-score calculation.
            threshold: Z-score threshold to trigger a trade.

        Returns:
            Dict mapping tickers (SHY, IEF, TLT) to signal Series in [-1, 1].
        """
        butterfly = 2.0 * yield_5y - yield_2y - yield_10y
        # Shift by 1 to avoid look-ahead
        rolling_mean = butterfly.rolling(lookback).mean().shift(1)
        rolling_std = butterfly.rolling(lookback).std().shift(1).replace(0, np.nan)
        z_score = (butterfly - rolling_mean) / rolling_std

        shy_signal = pd.Series(0.0, index=butterfly.index)
        ief_signal = pd.Series(0.0, index=butterfly.index)
        tlt_signal = pd.Series(0.0, index=butterfly.index)

        # Belly is cheap (high butterfly) → sell belly, buy wings
        rich_belly = z_score > threshold
        shy_signal[rich_belly] = 0.5
        ief_signal[rich_belly] = -1.0
        tlt_signal[rich_belly] = 0.5

        # Belly is rich (low butterfly) → buy belly, sell wings
        cheap_belly = z_score < -threshold
        shy_signal[cheap_belly] = -0.5
        ief_signal[cheap_belly] = 1.0
        tlt_signal[cheap_belly] = -0.5

        return {"SHY": shy_signal, "IEF": ief_signal, "TLT": tlt_signal}

    def recession_hedge(
        self,
        recession_prob: pd.Series,
        high_threshold: float = 0.6,
        low_threshold: float = 0.3,
    ) -> dict[str, float | pd.Series]:
        """Shift to long duration when recession probability is high.

        Uses a two-threshold system to avoid whipsawing:
        - Above ``high_threshold``: full long-duration allocation (TLT heavy).
        - Below ``low_threshold``: standard allocation (IEF/SHY mix).
        - In between: linearly interpolate.

        Args:
            recession_prob: Series of recession probability estimates (0-1).
            high_threshold: Probability above which full TLT allocation triggers.
            low_threshold: Probability below which normal allocation resumes.

        Returns:
            Dict mapping tickers (TLT, IEF, SHY) to signal Series.
        """
        tlt = pd.Series(0.0, index=recession_prob.index)
        ief = pd.Series(0.0, index=recession_prob.index)
        shy = pd.Series(0.0, index=recession_prob.index)

        high_mask = recession_prob >= high_threshold
        low_mask = recession_prob <= low_threshold
        mid_mask = ~high_mask & ~low_mask

        # Full recession hedge: heavy long duration
        tlt[high_mask] = 0.7
        ief[high_mask] = 0.2
        shy[high_mask] = 0.1

        # Normal environment: balanced duration
        tlt[low_mask] = 0.2
        ief[low_mask] = 0.4
        shy[low_mask] = 0.4

        # Transition band: linear interpolation
        if mid_mask.any():
            interp = (recession_prob[mid_mask] - low_threshold) / (high_threshold - low_threshold)
            tlt[mid_mask] = 0.2 + interp * 0.5
            ief[mid_mask] = 0.4 - interp * 0.2
            shy[mid_mask] = 0.4 - interp * 0.3

        return {"TLT": tlt, "IEF": ief, "SHY": shy}


class CreditRotationStrategy:
    """Rotate between IG and HY based on spread dynamics."""

    def ig_hy_rotation(
        self, ig_spread: pd.Series, hy_spread: pd.Series, lookback: int = 63
    ) -> dict[str, pd.Series]:
        """Rotate between LQD (IG) and HYG (HY) based on spread momentum.

        Tightening HY spreads → overweight HYG (risk-on)
        Widening HY spreads → overweight LQD (flight to quality)
        """
        hy_momentum = hy_spread.pct_change(lookback)
        signals_hyg = pd.Series(0.0, index=hy_spread.index)
        signals_lqd = pd.Series(0.0, index=hy_spread.index)

        signals_hyg[hy_momentum < 0] = 0.7   # spreads tightening → HY
        signals_lqd[hy_momentum < 0] = 0.3

        signals_hyg[hy_momentum > 0] = 0.2   # spreads widening → IG
        signals_lqd[hy_momentum > 0] = 0.8

        return {"HYG": signals_hyg, "LQD": signals_lqd}

    def spread_mean_reversion(
        self,
        ig_spread: pd.Series,
        hy_spread: pd.Series,
        lookback: int = 252,
        z_threshold: float = 1.5,
    ) -> dict[str, pd.Series]:
        """Trade convergence when the IG-HY spread ratio deviates from its mean.

        The ratio hy_spread / ig_spread captures relative value between
        investment-grade and high-yield credit. When the ratio is abnormally
        high (HY cheap vs IG), overweight HYG expecting convergence.
        When abnormally low (HY rich vs IG), overweight LQD.

        Args:
            ig_spread: Investment-grade OAS spread series.
            hy_spread: High-yield OAS spread series.
            lookback: Rolling window for mean/std of the ratio.
            z_threshold: Z-score level to trigger convergence trade.

        Returns:
            Dict mapping HYG/LQD to signal Series.
        """
        ratio = hy_spread / ig_spread.replace(0, np.nan)
        # Shift by 1 to avoid look-ahead
        rolling_mean = ratio.rolling(lookback).mean().shift(1)
        rolling_std = ratio.rolling(lookback).std().shift(1).replace(0, np.nan)
        z_score = (ratio - rolling_mean) / rolling_std

        hyg_signal = pd.Series(0.0, index=ig_spread.index)
        lqd_signal = pd.Series(0.0, index=ig_spread.index)

        # HY spreads relatively wide vs IG → HY cheap → buy HYG
        hy_cheap = z_score > z_threshold
        hyg_signal[hy_cheap] = 0.8
        lqd_signal[hy_cheap] = 0.2

        # HY spreads relatively tight vs IG → HY rich → buy LQD
        hy_rich = z_score < -z_threshold
        hyg_signal[hy_rich] = 0.2
        lqd_signal[hy_rich] = 0.8

        # Neutral zone: equal weight
        neutral = ~hy_cheap & ~hy_rich
        hyg_signal[neutral] = 0.5
        lqd_signal[neutral] = 0.5

        return {"HYG": hyg_signal, "LQD": lqd_signal}
