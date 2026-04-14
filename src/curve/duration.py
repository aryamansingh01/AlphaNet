"""Bond duration, convexity, and DV01 calculations."""

from __future__ import annotations

import numpy as np


def bond_price(face: float, coupon_rate: float, ytm: float, periods: int) -> float:
    """Calculate bond price given yield to maturity."""
    c = face * coupon_rate / 2  # semi-annual coupon
    y = ytm / 2
    n = periods * 2

    if y == 0:
        return c * n + face

    pv_coupons = c * (1 - (1 + y) ** -n) / y
    pv_face = face / (1 + y) ** n
    return pv_coupons + pv_face


def modified_duration(face: float, coupon_rate: float, ytm: float, periods: int) -> float:
    """Calculate modified duration."""
    price = bond_price(face, coupon_rate, ytm, periods)
    dy = 0.0001  # 1 basis point
    price_up = bond_price(face, coupon_rate, ytm + dy, periods)
    price_down = bond_price(face, coupon_rate, ytm - dy, periods)
    return -(price_up - price_down) / (2 * dy * price)


def convexity(face: float, coupon_rate: float, ytm: float, periods: int) -> float:
    """Calculate bond convexity."""
    price = bond_price(face, coupon_rate, ytm, periods)
    dy = 0.0001
    price_up = bond_price(face, coupon_rate, ytm + dy, periods)
    price_down = bond_price(face, coupon_rate, ytm - dy, periods)
    return (price_up + price_down - 2 * price) / (dy**2 * price)


def dv01(face: float, coupon_rate: float, ytm: float, periods: int) -> float:
    """Calculate dollar value of a basis point (DV01)."""
    price = bond_price(face, coupon_rate, ytm, periods)
    dur = modified_duration(face, coupon_rate, ytm, periods)
    return price * dur * 0.0001


# ------------------------------------------------------------------
# Portfolio-level analytics
# ------------------------------------------------------------------


def portfolio_duration(
    bonds: list[dict],
) -> float:
    """Calculate the market-value-weighted duration of a bond portfolio.

    Parameters
    ----------
    bonds : list[dict]
        Each dict must contain the keys ``face``, ``coupon_rate``, ``ytm``,
        ``periods``, and ``quantity`` (number of bonds held).

    Returns
    -------
    float
        Weighted-average modified duration of the portfolio.
    """
    total_mv = 0.0
    weighted_dur = 0.0

    for b in bonds:
        px = bond_price(b["face"], b["coupon_rate"], b["ytm"], b["periods"])
        mv = px * b["quantity"]
        dur = modified_duration(b["face"], b["coupon_rate"], b["ytm"], b["periods"])
        weighted_dur += dur * mv
        total_mv += mv

    if total_mv == 0:
        return 0.0

    return weighted_dur / total_mv


def price_change_estimate(
    mod_duration: float,
    convex: float,
    yield_change_bps: float,
) -> float:
    """Estimate percentage price change using duration and convexity.

    Uses the second-order Taylor expansion:
        dP/P ~ -D * dy + 0.5 * C * dy^2

    Parameters
    ----------
    mod_duration : float
        Modified duration of the bond.
    convex : float
        Convexity of the bond.
    yield_change_bps : float
        Yield change in basis points (e.g. +50 means yields rise 50 bps).

    Returns
    -------
    float
        Estimated percentage price change (e.g. -0.03 means a 3% decline).
    """
    dy = yield_change_bps / 10_000.0
    return -mod_duration * dy + 0.5 * convex * dy ** 2
