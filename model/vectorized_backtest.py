"""Vectorized backtest engine — fast parameter sweeps via numpy panel ops."""
import numpy as np
import pandas as pd
from typing import Dict


class VectorizedBacktest:
    """Fast cross-sectional backtest. No lot-size rounding, proportional costs."""

    def __init__(self, top_k: int = 30, cost_bps: int = 50, rebalance_every: int = 5):
        self.top_k = top_k
        self.cost_rate = cost_bps / 10000
        self.rebalance_every = rebalance_every

    def run(self, scores_panel: pd.DataFrame, prices: pd.DataFrame) -> Dict:
        """
        Args:
            scores_panel: DataFrame(index=date, columns=symbol) model scores
            prices: DataFrame(index=date, columns=symbol) close prices
        """
        dates = scores_panel.index
        n_days = len(dates)
        returns = prices.pct_change().fillna(0).values
        scores = scores_panel.values

        nav = 1.0
        equity = np.ones(n_days)
        holdings = []
        total_turnover = 0.0
        n_rebalances = 0

        for i in range(n_days):
            if i % self.rebalance_every == 0:
                day_scores = scores[i].copy()
                valid = ~np.isnan(day_scores)
                if valid.sum() >= self.top_k:
                    day_scores[~valid] = -np.inf
                    top_idx = np.argsort(day_scores)[-self.top_k:]
                    new_holdings = sorted(top_idx.tolist())

                    if holdings:
                        old_set = set(holdings)
                        new_set = set(new_holdings)
                        n_changed = len(old_set.symmetric_difference(new_set))
                        turnover = n_changed / (2 * self.top_k)
                        nav *= (1 - turnover * 2 * self.cost_rate)
                        total_turnover += turnover
                    else:
                        nav *= (1 - self.cost_rate)
                        total_turnover += 1.0
                    holdings = new_holdings
                    n_rebalances += 1

            if holdings and i > 0:
                nav *= (1 + np.mean(returns[i, holdings]))
            equity[i] = nav

        return {
            'equity_curve': pd.Series(equity, index=dates, name='equity'),
            'total_turnover': total_turnover,
            'avg_turnover_per_rebal': total_turnover / max(n_rebalances, 1),
            'n_rebalances': n_rebalances,
            'final_nav': nav,
        }
