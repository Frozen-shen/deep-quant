"""
Vectorized backtest engine for A-share strategies.

Handles:
    - T+1 settlement (can't sell on buy day)
    - Limit-up/down (can't buy at limit-up, can't sell at limit-down)
    - Proper cost deduction on each trade
    - Benchmark comparison (CSI1000 default)
    - Turnover tracking for cost analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from .costs import CostModel


@dataclass
class BacktestResult:
    """
    Complete results from a backtest run.

    Attributes:
        equity_curve: Daily portfolio value series (indexed by date).
        benchmark_curve: Benchmark value series (same index).
        excess_returns: Daily excess returns vs benchmark.
        trades: DataFrame of all executed trades (date, symbol, weight_change, cost).
        turnover_series: Daily one-way turnover series.
        metrics: Dictionary of computed performance metrics.
    """

    equity_curve: pd.Series
    benchmark_curve: pd.Series
    excess_returns: pd.Series
    trades: pd.DataFrame
    turnover_series: pd.Series
    metrics: Dict = field(default_factory=dict)


class BacktestEngine:
    """
    Vectorized backtest engine.

    Input:
        - weights_panel: DataFrame (date x symbol) of portfolio weights
        - returns_panel: DataFrame (date x symbol) of daily returns
        - cost_model: CostModel instance

    Output:
        - equity_curve: Series of portfolio values
        - trades: DataFrame of all transactions
        - metrics: dict of performance statistics

    Key features:
        - Handles T+1 settlement (A-shares: can't sell on buy day)
        - Handles limit-up/down (can't buy at limit-up, can't sell at limit-down)
        - Proper cost deduction on each trade
        - Benchmark comparison (CSI1000 default)

    Args:
        cost_model: Transaction cost model instance.
        initial_capital: Starting portfolio value in yuan.
        benchmark_returns: Daily benchmark returns series for comparison.
            If None, no benchmark comparison is computed.
        limit_up_threshold: Return threshold for limit-up detection (default 9.8%).
        limit_down_threshold: Return threshold for limit-down detection (default -9.8%).
        enable_t_plus_1: Whether to enforce T+1 settlement rules.
    """

    def __init__(
        self,
        cost_model: CostModel,
        initial_capital: float = 1_000_000,
        benchmark_returns: Optional[pd.Series] = None,
        limit_up_threshold: float = 0.098,
        limit_down_threshold: float = -0.098,
        enable_t_plus_1: bool = True,
    ):
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

        self.cost_model = cost_model
        self.initial_capital = initial_capital
        self.benchmark_returns = benchmark_returns
        self.limit_up_threshold = limit_up_threshold
        self.limit_down_threshold = limit_down_threshold
        self.enable_t_plus_1 = enable_t_plus_1

    def run(
        self,
        weights_panel: pd.DataFrame,
        returns_panel: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run the backtest.

        The engine iterates over each date, computing portfolio returns
        based on weights and applying transaction costs when weights change.

        Args:
            weights_panel: DataFrame with DatetimeIndex (rows=dates,
                columns=symbols) containing target portfolio weights.
                Weights should be non-negative and sum to <= 1.0.
            returns_panel: DataFrame with same shape/index as weights_panel
                containing daily returns for each symbol.

        Returns:
            BacktestResult with equity curve, trades, and metrics.
        """
        # Align panels on common dates and symbols
        common_dates = weights_panel.index.intersection(returns_panel.index)
        common_symbols = weights_panel.columns.intersection(returns_panel.columns)

        if len(common_dates) == 0:
            raise ValueError("No common dates between weights and returns panels")
        if len(common_symbols) == 0:
            raise ValueError("No common symbols between weights and returns panels")

        weights = weights_panel.loc[common_dates, common_symbols].fillna(0.0)
        returns = returns_panel.loc[common_dates, common_symbols].fillna(0.0)

        n_dates = len(common_dates)

        # State tracking
        current_weights = pd.Series(0.0, index=common_symbols)
        portfolio_value = self.initial_capital

        # Output arrays
        equity_values = np.empty(n_dates, dtype=np.float64)
        turnover_values = np.empty(n_dates, dtype=np.float64)
        daily_returns = np.empty(n_dates, dtype=np.float64)

        # Trade log
        trade_records: List[Dict] = []

        # T+1 tracking: symbols bought today cannot be sold today
        bought_today: Set[str] = set()

        for i, dt in enumerate(common_dates):
            target_weights = weights.iloc[i]
            day_returns = returns.iloc[i]

            # Detect limit-up/down for this day
            limit_up_symbols = self._check_limits(
                day_returns, direction="up"
            )
            limit_down_symbols = self._check_limits(
                day_returns, direction="down"
            )

            # Compute desired trades
            desired_trades = self._compute_trades(current_weights, target_weights)

            # Apply T+1 constraint: can't sell what was bought yesterday
            # (In our daily loop, "bought_today" from previous iteration
            #  represents yesterday's buys that are now sellable.
            #  We track current-day buys to block same-day sells.)
            if self.enable_t_plus_1 and bought_today:
                for sym in bought_today:
                    if sym in desired_trades.index and desired_trades[sym] < 0:
                        desired_trades[sym] = 0.0

            # Apply limit constraints
            adjusted_trades = self._apply_limit_constraints(
                desired_trades, limit_up_symbols, limit_down_symbols
            )

            # Compute and apply costs
            trade_costs = self._apply_costs(adjusted_trades, portfolio_value)
            total_cost = trade_costs.sum()

            # Compute turnover (one-way)
            day_turnover = np.abs(adjusted_trades).sum() / 2.0
            turnover_values[i] = day_turnover

            # Log trades
            nonzero_trades = adjusted_trades[adjusted_trades.abs() > 1e-10]
            for sym, trade_w in nonzero_trades.items():
                trade_notional = abs(trade_w) * portfolio_value
                trade_cost = trade_costs.get(sym, 0.0)
                trade_records.append({
                    "date": dt,
                    "symbol": sym,
                    "weight_change": trade_w,
                    "notional": trade_notional,
                    "cost": trade_cost,
                    "direction": "buy" if trade_w > 0 else "sell",
                })

            # Update weights
            current_weights = current_weights + adjusted_trades

            # Compute portfolio return for this day
            # Return = sum(weight * return) - cost_drag
            portfolio_return = np.dot(current_weights.values, day_returns.values)
            cost_drag = total_cost / portfolio_value if portfolio_value > 0 else 0.0
            net_return = portfolio_return - cost_drag

            daily_returns[i] = net_return
            portfolio_value *= (1.0 + net_return)
            equity_values[i] = portfolio_value

            # Update T+1 tracking: today's buys
            bought_today = set(
                sym for sym, t in adjusted_trades.items() if t > 1e-10
            )

        # Build output series
        equity_curve = pd.Series(
            equity_values, index=common_dates, name="equity"
        )
        turnover_series = pd.Series(
            turnover_values, index=common_dates, name="turnover"
        )

        # Benchmark curve
        if self.benchmark_returns is not None:
            bench_ret = self.benchmark_returns.reindex(common_dates).fillna(0.0)
            benchmark_curve = pd.Series(
                self.initial_capital * (1.0 + bench_ret).cumprod(),
                index=common_dates,
                name="benchmark",
            )
            excess_returns = equity_curve.pct_change().fillna(0.0) - bench_ret
        else:
            benchmark_curve = pd.Series(
                np.full(n_dates, self.initial_capital),
                index=common_dates,
                name="benchmark",
            )
            excess_returns = equity_curve.pct_change().fillna(0.0)

        excess_returns.name = "excess_return"

        # Trades DataFrame
        if trade_records:
            trades_df = pd.DataFrame(trade_records)
        else:
            trades_df = pd.DataFrame(
                columns=["date", "symbol", "weight_change", "notional", "cost", "direction"]
            )

        # Compute metrics
        from .metrics import compute_metrics

        metrics = compute_metrics(
            equity_curve=equity_curve,
            benchmark=benchmark_curve,
            risk_free_rate=0.02,
        )
        # Add turnover info
        metrics["avg_daily_turnover"] = float(turnover_series.mean())
        metrics["total_turnover"] = float(turnover_series.sum())
        # Annualize turnover (approx 244 trading days/year)
        metrics["annual_turnover"] = float(turnover_series.mean() * 244)

        return BacktestResult(
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            excess_returns=excess_returns,
            trades=trades_df,
            turnover_series=turnover_series,
            metrics=metrics,
        )

    def _compute_trades(
        self,
        old_weights: pd.Series,
        new_weights: pd.Series,
    ) -> pd.Series:
        """
        Compute trade weights needed to move from old to new weights.

        Args:
            old_weights: Current portfolio weights.
            new_weights: Target portfolio weights.

        Returns:
            Series of trade weights (positive=buy, negative=sell).
        """
        trades = new_weights - old_weights
        # Zero out dust trades
        trades[trades.abs() < 1e-10] = 0.0
        return trades

    def _apply_costs(
        self,
        trades: pd.Series,
        portfolio_value: float,
    ) -> pd.Series:
        """
        Compute transaction costs for each trade.

        Args:
            trades: Trade weights (positive=buy, negative=sell).
            portfolio_value: Current portfolio value for notional calculation.

        Returns:
            Series of costs (in yuan) per symbol.
        """
        costs = pd.Series(0.0, index=trades.index)

        for sym, trade_w in trades.items():
            if abs(trade_w) < 1e-10:
                continue
            notional = abs(trade_w) * portfolio_value
            if trade_w > 0:
                costs[sym] = self.cost_model.buy_cost(notional)
            else:
                costs[sym] = self.cost_model.sell_cost(notional)

        return costs

    def _check_limits(
        self,
        returns: pd.Series,
        direction: str,
    ) -> Set[str]:
        """
        Detect limit-up or limit-down symbols.

        A-share stocks have ±10% daily price limits (±20% for
        ChiNext/STAR Market). We use a conservative threshold.

        Args:
            returns: Daily returns for all symbols.
            direction: "up" for limit-up, "down" for limit-down.

        Returns:
            Set of symbol names that hit the limit.
        """
        if direction == "up":
            mask = returns >= self.limit_up_threshold
        else:
            mask = returns <= self.limit_down_threshold

        return set(returns.index[mask])

    def _apply_limit_constraints(
        self,
        trades: pd.Series,
        limit_up_symbols: Set[str],
        limit_down_symbols: Set[str],
    ) -> pd.Series:
        """
        Adjust trades to respect limit-up/down constraints.

        Rules:
            - Can't BUY a stock at limit-up (no sellers available)
            - Can't SELL a stock at limit-down (no buyers available)

        Args:
            trades: Desired trades.
            limit_up_symbols: Symbols that hit limit-up today.
            limit_down_symbols: Symbols that hit limit-down today.

        Returns:
            Adjusted trades with blocked trades zeroed out.
        """
        adjusted = trades.copy()

        # Can't buy at limit-up
        for sym in limit_up_symbols:
            if sym in adjusted.index and adjusted[sym] > 0:
                adjusted[sym] = 0.0

        # Can't sell at limit-down
        for sym in limit_down_symbols:
            if sym in adjusted.index and adjusted[sym] < 0:
                adjusted[sym] = 0.0

        return adjusted
