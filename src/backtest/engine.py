"""Backtesting engine with walk-forward validation and transaction costs."""

from typing import Optional

import pandas as pd
import numpy as np


class BacktestEngine:
    """Event-driven backtester with regime-conditional analysis."""

    def __init__(self, transaction_cost_bps: float = 5.0):
        self.tc = transaction_cost_bps / 10_000

    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None,
    ) -> dict:
        """Run backtest from signals and prices.

        Args:
            signals: DataFrame of weights per asset (index=dates, cols=tickers)
            prices: DataFrame of prices (same shape)
            regime_labels: Optional series of regime labels for conditional analysis
        """
        returns = prices.pct_change().dropna()
        signals = signals.reindex(returns.index).fillna(0)

        # Shift signals: signal at t trades at t+1
        shifted_signals = signals.shift(1)

        # Calculate turnover from actual position changes
        turnover = shifted_signals.diff().abs().sum(axis=1)
        costs = turnover * self.tc

        # Portfolio returns
        portfolio_returns = (shifted_signals * returns).sum(axis=1) - costs
        portfolio_returns = portfolio_returns.dropna()

        # Cumulative
        cumulative = (1 + portfolio_returns).cumprod()

        result = {
            "returns": portfolio_returns,
            "cumulative": cumulative,
            "metrics": self._compute_metrics(portfolio_returns),
            "turnover": turnover,
        }

        if regime_labels is not None:
            result["regime_metrics"] = self._regime_conditional_metrics(
                portfolio_returns, regime_labels
            )

        return result

    def walk_forward(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        train_pct: float = 0.7,
    ) -> dict:
        """Walk-forward validation: train on first N%, test on rest."""
        n = len(signals)
        split = int(n * train_pct)

        in_sample = self.run(signals.iloc[:split], prices.iloc[:split])
        out_sample = self.run(signals.iloc[split:], prices.iloc[split:])

        return {
            "in_sample": in_sample,
            "out_of_sample": out_sample,
            "degradation": {
                k: out_sample["metrics"][k] - in_sample["metrics"][k]
                for k in ["sharpe", "annual_return"]
            },
        }

    # ------------------------------------------------------------------
    # New: multi-strategy comparison
    # ------------------------------------------------------------------

    def compare_strategies(
        self,
        strategy_signals: dict[str, pd.DataFrame],
        prices: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Run multiple strategies and return a side-by-side comparison table.

        Args:
            strategy_signals: Dict mapping strategy name to its signal DataFrame.
            prices: Common price DataFrame shared by all strategies.
            regime_labels: Optional regime labels forwarded to each ``run()``.

        Returns:
            DataFrame where each row is a strategy and columns are performance
            metrics (annual_return, sharpe, max_drawdown, etc.).
        """
        rows: list[dict] = []
        for name, signals in strategy_signals.items():
            result = self.run(signals, prices, regime_labels)
            row = {"strategy": name, **result["metrics"]}
            rows.append(row)

        comparison = pd.DataFrame(rows).set_index("strategy")
        comparison = comparison.sort_values("sharpe", ascending=False)
        return comparison

    # ------------------------------------------------------------------
    # New: comprehensive report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None,
    ) -> dict:
        """Generate a comprehensive backtest report.

        Returns a dict containing:
            - metrics: standard performance metrics dict
            - drawdown_series: full drawdown time-series
            - monthly_returns: pivot table of monthly returns (rows=year, cols=month)
            - rolling_sharpe: 63-day rolling Sharpe ratio
            - annual_returns: per-calendar-year return summary
            - regime_metrics: per-regime metrics (if regime_labels provided)

        Args:
            signals: Weight DataFrame.
            prices: Price DataFrame.
            regime_labels: Optional regime labels.
        """
        result = self.run(signals, prices, regime_labels)
        returns = result["returns"]
        cumulative = result["cumulative"]

        # Drawdown series
        peak = cumulative.cummax()
        drawdown_series = (cumulative - peak) / peak

        # Monthly returns pivot
        monthly_returns = self._monthly_return_table(returns)

        # Rolling Sharpe (63-day ~ 3 months)
        rolling_mean = returns.rolling(63).mean() * 252
        rolling_vol = returns.rolling(63).std() * np.sqrt(252)
        rolling_sharpe = (rolling_mean / rolling_vol.replace(0, np.nan)).dropna()

        # Annual returns
        annual_returns = self._annual_return_table(returns)

        report: dict = {
            "metrics": result["metrics"],
            "drawdown_series": drawdown_series,
            "monthly_returns": monthly_returns,
            "rolling_sharpe": rolling_sharpe,
            "annual_returns": annual_returns,
            "cumulative": cumulative,
            "turnover": result["turnover"],
        }

        if "regime_metrics" in result:
            report["regime_metrics"] = result["regime_metrics"]

        return report

    # ------------------------------------------------------------------
    # New: benchmark comparison
    # ------------------------------------------------------------------

    def benchmark_comparison(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        benchmark_ticker: str = "SPY",
        regime_labels: Optional[pd.Series] = None,
    ) -> dict:
        """Compare strategy performance vs a buy-and-hold benchmark.

        Args:
            signals: Strategy weight DataFrame.
            prices: Price DataFrame (must contain ``benchmark_ticker`` column).
            benchmark_ticker: Column in ``prices`` to use as benchmark.
            regime_labels: Optional regime labels.

        Returns:
            Dict with keys:
                - strategy: full run() result dict for the strategy
                - benchmark: full run() result dict for buy-and-hold
                - comparison: DataFrame comparing key metrics side-by-side
                - excess_returns: daily excess return series (strategy - benchmark)
                - information_ratio: annualised IR of excess returns
                - tracking_error: annualised tracking error
        """
        if benchmark_ticker not in prices.columns:
            raise ValueError(
                f"Benchmark ticker '{benchmark_ticker}' not found in prices columns"
            )

        # Strategy result
        strat_result = self.run(signals, prices, regime_labels)

        # Buy-and-hold benchmark: 100% in benchmark_ticker
        bm_signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        bm_signals[benchmark_ticker] = 1.0
        bm_result = self.run(bm_signals, prices, regime_labels)

        # Align return series
        strat_ret = strat_result["returns"]
        bm_ret = bm_result["returns"]
        common_idx = strat_ret.index.intersection(bm_ret.index)
        excess = strat_ret.loc[common_idx] - bm_ret.loc[common_idx]

        tracking_error = float(excess.std() * np.sqrt(252))
        information_ratio = (
            float(excess.mean() * 252 / tracking_error) if tracking_error > 0 else 0.0
        )

        comparison = pd.DataFrame({
            "strategy": strat_result["metrics"],
            "benchmark": bm_result["metrics"],
        }).T

        return {
            "strategy": strat_result,
            "benchmark": bm_result,
            "comparison": comparison,
            "excess_returns": excess,
            "information_ratio": round(information_ratio, 4),
            "tracking_error": round(tracking_error, 4),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_metrics(self, returns: pd.Series) -> dict:
        """Calculate performance metrics."""
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()

        return {
            "annual_return": round(float(annual_return), 4),
            "annual_vol": round(float(annual_vol), 4),
            "sharpe": round(float(sharpe), 4),
            "sortino": round(float(self._sortino(returns)), 4),
            "max_drawdown": round(float(max_dd), 4),
            "calmar": round(float(annual_return / abs(max_dd)) if max_dd != 0 else 0, 4),
            "win_rate": round(float((returns > 0).mean()), 4),
            "total_return": round(float(cumulative.iloc[-1] - 1), 4),
        }

    def _sortino(self, returns: pd.Series) -> float:
        downside = returns[returns < 0].std() * np.sqrt(252)
        return returns.mean() * 252 / downside if downside > 0 else 0

    def _regime_conditional_metrics(
        self, returns: pd.Series, regimes: pd.Series
    ) -> dict:
        """Compute metrics per regime."""
        regimes = regimes.reindex(returns.index).ffill()
        result = {}
        for regime in regimes.unique():
            mask = regimes == regime
            if mask.sum() > 10:
                result[str(regime)] = self._compute_metrics(returns[mask])
        return result

    def _monthly_return_table(self, returns: pd.Series) -> pd.DataFrame:
        """Build a year x month pivot table of monthly returns."""
        monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        table = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        })
        pivot = table.pivot_table(
            index="year", columns="month", values="return", aggfunc="sum"
        )
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ][: len(pivot.columns)]
        return pivot

    def _annual_return_table(self, returns: pd.Series) -> pd.DataFrame:
        """Compute per-calendar-year performance summary."""
        annual = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        annual_vol = returns.resample("YE").std() * np.sqrt(252)
        table = pd.DataFrame({
            "return": annual,
            "volatility": annual_vol,
            "sharpe": annual / annual_vol.replace(0, np.nan),
        })
        table.index = table.index.year
        table.index.name = "year"
        return table.round(4)
