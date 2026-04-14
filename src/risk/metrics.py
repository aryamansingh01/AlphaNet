"""Risk metrics — VaR, CVaR, drawdown, correlation."""

import numpy as np
import pandas as pd


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR at given confidence level."""
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """CVaR (Expected Shortfall) — average loss beyond VaR."""
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from peak."""
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    dd = (cumulative - peak) / peak
    return float(dd.min())


def rolling_sharpe(returns: pd.Series, window: int = 63) -> pd.Series:
    """Rolling Sharpe ratio."""
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_std = returns.rolling(window).std() * np.sqrt(252)
    return rolling_mean / rolling_std


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Asset correlation matrix."""
    return returns.corr()
