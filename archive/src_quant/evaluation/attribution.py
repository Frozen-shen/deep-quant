"""
Return attribution analysis.

Decompose portfolio returns into components:
    Total Return = Market Beta + Factor Exposure + Stock Selection + Transaction Costs

Uses regression-based attribution:
    R_p - R_f = alpha + beta * (R_m - R_f) + sum(gamma_i * F_i) + epsilon

This helps answer:
    - Is the strategy generating real alpha or just leveraged beta?
    - How much return is explained by known factors?
    - What's the true cost drag from trading?
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class AttributionResult:
    """
    Results of return attribution analysis.

    Attributes:
        total_return: Annualized total portfolio return.
        beta_return: Return attributable to market beta exposure.
        alpha_return: Residual return not explained by factors (Jensen's alpha).
        factor_returns: {factor_name: return_contribution} from factor exposures.
        cost_drag: Annual return lost to transaction costs.
        selection_return: Stock-specific return (alpha + unexplained).
        r_squared: Regression R-squared (how much is explained).
        beta: Market beta coefficient.
        factor_betas: {factor_name: beta} regression coefficients.
    """

    total_return: float
    beta_return: float
    alpha_return: float
    factor_returns: Dict[str, float] = field(default_factory=dict)
    cost_drag: float = 0.0
    selection_return: float = 0.0
    r_squared: float = 0.0
    beta: float = 0.0
    factor_betas: Dict[str, float] = field(default_factory=dict)


class ReturnAttribution:
    """
    Decompose portfolio returns into components.

    Uses OLS regression of portfolio excess returns on benchmark and
    factor excess returns to attribute performance.

    The decomposition:
        Total = Beta_component + Factor_components + Alpha (residual)

    Where:
        Beta_component = beta * mean(benchmark_excess_return) * 244
        Factor_component_i = gamma_i * mean(factor_i_excess_return) * 244
        Alpha = intercept * 244
    """

    TRADING_DAYS_PER_YEAR = 244

    def decompose(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        factor_returns: Optional[Dict[str, pd.Series]] = None,
        risk_free_rate: float = 0.02,
        cost_series: Optional[pd.Series] = None,
    ) -> AttributionResult:
        """
        Perform return attribution via regression.

        Args:
            portfolio_returns: Daily portfolio returns series.
            benchmark_returns: Daily benchmark returns series.
            factor_returns: Optional {factor_name: daily_factor_returns} for
                multi-factor attribution. If None, single-factor (CAPM) model.
            risk_free_rate: Annual risk-free rate.
            cost_series: Optional daily cost drag series (fraction of portfolio
                value lost to costs each day).

        Returns:
            AttributionResult with decomposed return components.
        """
        daily_rf = risk_free_rate / self.TRADING_DAYS_PER_YEAR

        # Align all series on common dates
        port_ret = portfolio_returns.dropna()
        bench_ret = benchmark_returns.reindex(port_ret.index).fillna(0.0)

        # Compute excess returns
        port_excess = port_ret - daily_rf
        bench_excess = bench_ret - daily_rf

        # Build regression matrix
        if factor_returns is not None and len(factor_returns) > 0:
            # Multi-factor model
            X_cols = {"market": bench_excess.values}
            factor_names = []
            for fname, fret in factor_returns.items():
                f_aligned = fret.reindex(port_ret.index).fillna(0.0)
                f_excess = f_aligned - daily_rf
                X_cols[fname] = f_excess.values
                factor_names.append(fname)

            X = np.column_stack(list(X_cols.values()))
            col_names = list(X_cols.keys())
        else:
            # Single-factor CAPM
            X = bench_excess.values.reshape(-1, 1)
            col_names = ["market"]
            factor_names = []

        y = port_excess.values
        n_obs = len(y)

        if n_obs < 10:
            return AttributionResult(
                total_return=float(port_ret.mean() * self.TRADING_DAYS_PER_YEAR),
                beta_return=0.0,
                alpha_return=0.0,
            )

        # OLS regression: y = alpha + X @ beta + epsilon
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(n_obs), X])
        try:
            # Use least squares
            coeffs, residuals, rank, sv = np.linalg.lstsq(
                X_with_intercept, y, rcond=None
            )
        except np.linalg.LinAlgError:
            return AttributionResult(
                total_return=float(port_ret.mean() * self.TRADING_DAYS_PER_YEAR),
                beta_return=0.0,
                alpha_return=0.0,
            )

        intercept = coeffs[0]
        betas = coeffs[1:]

        # Compute R-squared
        y_pred = X_with_intercept @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Annualize components
        # Alpha (intercept * trading days)
        alpha_annual = float(intercept * self.TRADING_DAYS_PER_YEAR)

        # Beta return contribution (market beta * mean market excess * 244)
        market_beta = float(betas[0]) if len(betas) > 0 else 0.0
        beta_return_annual = float(
            market_beta * bench_excess.mean() * self.TRADING_DAYS_PER_YEAR
        )

        # Factor return contributions
        factor_return_contrib = {}
        factor_beta_map = {}
        for i, fname in enumerate(factor_names):
            factor_beta = float(betas[i + 1])  # +1 for market beta
            factor_mean = X_cols[fname].mean()
            factor_contrib = float(
                factor_beta * factor_mean * self.TRADING_DAYS_PER_YEAR
            )
            factor_return_contrib[fname] = factor_contrib
            factor_beta_map[fname] = factor_beta

        # Total return (annualized)
        total_return_annual = float(port_ret.mean() * self.TRADING_DAYS_PER_YEAR)

        # Cost drag
        if cost_series is not None:
            cost_aligned = cost_series.reindex(port_ret.index).fillna(0.0)
            cost_drag_annual = float(cost_aligned.mean() * self.TRADING_DAYS_PER_YEAR)
        else:
            cost_drag_annual = 0.0

        # Selection return = total - beta - factors + cost_drag
        # (This is the stock-specific return including alpha)
        explained_by_factors = sum(factor_return_contrib.values())
        selection_return = (
            total_return_annual - beta_return_annual - explained_by_factors
        )

        return AttributionResult(
            total_return=total_return_annual,
            beta_return=beta_return_annual,
            alpha_return=alpha_annual,
            factor_returns=factor_return_contrib,
            cost_drag=cost_drag_annual,
            selection_return=float(selection_return),
            r_squared=float(r_squared),
            beta=market_beta,
            factor_betas=factor_beta_map,
        )

    def rolling_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 60,
        risk_free_rate: float = 0.02,
    ) -> pd.DataFrame:
        """
        Compute rolling attribution to see how alpha evolves over time.

        Args:
            portfolio_returns: Daily portfolio returns.
            benchmark_returns: Daily benchmark returns.
            window: Rolling window size in trading days.
            risk_free_rate: Annual risk-free rate.

        Returns:
            DataFrame with columns [rolling_alpha, rolling_beta, rolling_r2]
            indexed by date.
        """
        daily_rf = risk_free_rate / self.TRADING_DAYS_PER_YEAR

        port_ret = portfolio_returns.dropna()
        bench_ret = benchmark_returns.reindex(port_ret.index).fillna(0.0)

        port_excess = port_ret - daily_rf
        bench_excess = bench_ret - daily_rf

        dates = port_ret.index
        n = len(dates)

        alphas = np.full(n, np.nan)
        betas = np.full(n, np.nan)
        r2s = np.full(n, np.nan)

        for i in range(window, n):
            y = port_excess.iloc[i - window: i].values
            x = bench_excess.iloc[i - window: i].values

            X = np.column_stack([np.ones(window), x])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                alphas[i] = coeffs[0] * self.TRADING_DAYS_PER_YEAR
                betas[i] = coeffs[1]

                y_pred = X @ coeffs
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2s[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            except np.linalg.LinAlgError:
                continue

        return pd.DataFrame(
            {
                "rolling_alpha": alphas,
                "rolling_beta": betas,
                "rolling_r2": r2s,
            },
            index=dates,
        )

    def format_attribution(self, result: AttributionResult) -> str:
        """
        Format attribution results as a readable report.

        Args:
            result: AttributionResult from decompose().

        Returns:
            Multi-line formatted string.
        """
        lines = []
        lines.append("=" * 50)
        lines.append("RETURN ATTRIBUTION")
        lines.append("=" * 50)
        lines.append("")
        lines.append(f"  Total Return (ann.):    {result.total_return:>8.2%}")
        lines.append(f"  Market Beta Return:     {result.beta_return:>8.2%}")
        lines.append(f"  Jensen's Alpha:         {result.alpha_return:>8.2%}")
        lines.append(f"  Selection Return:       {result.selection_return:>8.2%}")
        lines.append(f"  Cost Drag:              {result.cost_drag:>8.2%}")
        lines.append("")
        lines.append(f"  Market Beta:            {result.beta:>8.3f}")
        lines.append(f"  R-squared:              {result.r_squared:>8.3f}")

        if result.factor_returns:
            lines.append("")
            lines.append("  Factor Contributions:")
            for fname, contrib in result.factor_returns.items():
                beta = result.factor_betas.get(fname, 0.0)
                lines.append(
                    f"    {fname:<20s}: {contrib:>7.2%} (beta={beta:.3f})"
                )

        lines.append("")
        lines.append("=" * 50)
        return "\n".join(lines)
