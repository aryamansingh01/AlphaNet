"""Tests for bond duration, convexity, and DV01 calculations."""

import pytest
from src.curve.duration import bond_price, modified_duration, convexity, dv01


# ------------------------------------------------------------------
# Bond price tests
# ------------------------------------------------------------------

def test_par_bond_price():
    """A bond at par should have price approximately equal to face value."""
    price = bond_price(face=1000, coupon_rate=0.05, ytm=0.05, periods=10)
    assert abs(price - 1000) < 0.01


def test_discount_bond():
    """Higher yield than coupon should produce a price below par."""
    price = bond_price(face=1000, coupon_rate=0.05, ytm=0.07, periods=10)
    assert price < 1000


def test_premium_bond():
    """Lower yield than coupon should produce a price above par."""
    price = bond_price(face=1000, coupon_rate=0.05, ytm=0.03, periods=10)
    assert price > 1000


def test_zero_coupon_bond():
    """A zero-coupon bond price should equal the present value of face alone."""
    price = bond_price(face=1000, coupon_rate=0.0, ytm=0.05, periods=10)
    # PV of face = 1000 / (1 + 0.025)^20
    expected = 1000 / (1.025 ** 20)
    assert abs(price - expected) < 0.01


def test_zero_coupon_bond_zero_yield():
    """Zero coupon, zero yield should equal face value."""
    price = bond_price(face=1000, coupon_rate=0.0, ytm=0.0, periods=10)
    assert abs(price - 1000) < 0.01


# ------------------------------------------------------------------
# Duration tests
# ------------------------------------------------------------------

def test_duration_positive():
    """Duration should always be positive for a standard bond."""
    dur = modified_duration(face=1000, coupon_rate=0.05, ytm=0.05, periods=10)
    assert dur > 0


def test_longer_maturity_higher_duration():
    """Longer maturity bonds should have higher duration."""
    dur_5y = modified_duration(face=1000, coupon_rate=0.05, ytm=0.05, periods=5)
    dur_30y = modified_duration(face=1000, coupon_rate=0.05, ytm=0.05, periods=30)
    assert dur_30y > dur_5y


def test_higher_coupon_lower_duration():
    """A higher coupon rate should produce lower modified duration (more cash flows up front)."""
    dur_low_coupon = modified_duration(face=1000, coupon_rate=0.02, ytm=0.05, periods=10)
    dur_high_coupon = modified_duration(face=1000, coupon_rate=0.08, ytm=0.05, periods=10)
    assert dur_high_coupon < dur_low_coupon


def test_zero_coupon_highest_duration():
    """A zero-coupon bond should have the highest duration for its maturity."""
    dur_zero = modified_duration(face=1000, coupon_rate=0.0, ytm=0.05, periods=10)
    dur_coupon = modified_duration(face=1000, coupon_rate=0.05, ytm=0.05, periods=10)
    assert dur_zero > dur_coupon


def test_higher_yield_lower_duration():
    """Higher yield should decrease modified duration."""
    dur_low_yield = modified_duration(face=1000, coupon_rate=0.05, ytm=0.02, periods=10)
    dur_high_yield = modified_duration(face=1000, coupon_rate=0.05, ytm=0.10, periods=10)
    assert dur_high_yield < dur_low_yield


# ------------------------------------------------------------------
# Convexity tests
# ------------------------------------------------------------------

def test_convexity_positive():
    """Convexity should be positive for vanilla bonds."""
    conv = convexity(face=1000, coupon_rate=0.05, ytm=0.05, periods=10)
    assert conv > 0


def test_longer_maturity_higher_convexity():
    """Longer maturity should produce higher convexity."""
    conv_5y = convexity(face=1000, coupon_rate=0.05, ytm=0.05, periods=5)
    conv_30y = convexity(face=1000, coupon_rate=0.05, ytm=0.05, periods=30)
    assert conv_30y > conv_5y


# ------------------------------------------------------------------
# DV01 tests
# ------------------------------------------------------------------

def test_dv01_positive():
    """DV01 should be positive."""
    d = dv01(face=1000, coupon_rate=0.05, ytm=0.05, periods=10)
    assert d > 0


def test_dv01_approximates_duration_times_price():
    """DV01 should be approximately equal to modified_duration * price * 0.0001."""
    face, coupon, ytm, periods = 1000, 0.05, 0.05, 10
    d = dv01(face, coupon, ytm, periods)
    price_val = bond_price(face, coupon, ytm, periods)
    dur = modified_duration(face, coupon, ytm, periods)
    expected = dur * price_val * 0.0001
    # They should be very close (the dv01 function uses the same formula)
    assert abs(d - expected) < 0.001


def test_dv01_matches_price_sensitivity():
    """DV01 should closely approximate the actual price change for a 1bp yield move."""
    face, coupon, ytm, periods = 1000, 0.05, 0.05, 10
    d = dv01(face, coupon, ytm, periods)
    price_base = bond_price(face, coupon, ytm, periods)
    price_up = bond_price(face, coupon, ytm + 0.0001, periods)
    actual_change = abs(price_up - price_base)
    # DV01 should be a good approximation of the 1bp price change
    assert abs(d - actual_change) / actual_change < 0.01  # within 1%


def test_dv01_longer_maturity_higher():
    """Longer maturity bonds should have higher DV01."""
    dv01_5y = dv01(face=1000, coupon_rate=0.05, ytm=0.05, periods=5)
    dv01_30y = dv01(face=1000, coupon_rate=0.05, ytm=0.05, periods=30)
    assert dv01_30y > dv01_5y
