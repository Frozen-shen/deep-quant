"""
Performance metrics computation for backtest results.

Computes comprehensive risk-adjusted performance statistics including:
    - Return metrics: CAGR, cumulative return
    - Risk metrics: Sharpe, Sortino, max drawdown, Calmar
    - Trade metrics: win rate, profit factor
    - Benchmark-relative: information ratio, alpha, beta, tracking error
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


def compute_metrics(
    equity_curve: pd.Series,
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02,
) -> Dict:
    """
    Compute comprehensive performance metrics.

    Args:
        equity_curve: Series of portfolio values (indexed by date).
        benchmark: Series of benchmark values (same index). If None,
            benchmark-relative metrics are not computed.
        risk_free_rate: Annual risk-free rate (default 2% for China).

    Returns:
        Dictionary with all computed metrics:
            - cagr: Compound Annual Growth Rate
            - sharpe: Annualized Sharpe Ratio
            - sortino: Sortino Ratio (downside deviation)
            - max_drawdown: Maximum drawdown (negative number)
            - max_drawdown_duration: Days in max drawdown
            - calmar: CAGR / |MaxDD|
            - win_rate: Fraction of positive return days
            - profit_factor: gross profit / gross loss
            - annual_turnover: average annual turnover (if available)
            - information_ratio: excess return / tracking error
            - alpha: Jensen's alpha (annualized)
            - beta: market beta
            - tracking_error: std of excess returns (annualized)
    """
    metrics: Dict = {}

    # Compute daily returns
    returns = equity_curve.pct_change().dropna()
    n_days = len(returns)

    if n_days == 0:
        return {"error": "No returns data"}

    # Trading days per year (A-share market)
    trading_days_per_year = 244
    daily_rf = risk_free_rate / trading_days_per_year

    # --- Return Metrics ---
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    n_years = n_days / trading_days_per_year
    if n_years > 0 and (1 + total_return) > 0:
        cagr = (1 + total_return) ** (1.0 / n_years) - 1.0
    else:
        cagr = -1.0
    metrics["total_return"] = float(total_return)
    metrics["cagr"] = float(cagr)
    metrics["n_years"] = float(n_years)

    # --- Risk Metrics ---
    excess_returns = returns - daily_rf
    mean_excess = excess_returns.mean()
    std_returns = returns.std(ddof=1)

    # Sharpe Ratio (annualized)
    if std_returns > 0:
        sharpe = (mean_excess / std_returns) * np.sqrt(trading_days_per_year)
    else:
        sharpe = 0.0
    metrics["sharpe"] = float(sharpe)

    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_std = np.sqrt(np.mean(downside_returns**2))
        if downside_std > 0:
            sortino = (mean_excess / downside_std) * np.sqrt(trading_days_per_year)
        else:
            sortino = 0.0
    else:
        sortino = float("inf") if mean_excess > 0 else 0.0
    metrics["sortino"] = float(sortino)

    # Maximum Drawdown
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_drawdown = drawdown.min()
    metrics["max_drawdown"] = float(max_drawdown)

    # Max Drawdown Duration
    metrics["max_drawdown_duration"] = int(_max_drawdown_duration(drawdown))

    # Calmar Ratio
    if abs(max_drawdown) > 1e-10:
        calmar = cagr / abs(max_drawdown)
    else:
        calmar = float("inf") if cagr > 0 else 0.0
    metrics["calmar"] = float(calmar)

    # Volatility (annualized)
    annual_vol = std_returns * np.sqrt(trading_days_per_year)
    metrics["annual_volatility"] = float(annual_vol)

    # --- Trade Metrics ---
    positive_days = (returns > 0).sum()
    negative_days = (returns < 0).sum()
    total_days = positive_days + negative_days

    if total_days > 0:
        win_rate = positive_days / total_days
    else:
        win_rate = 0.0
    metrics["win_rate"] = float(win_rate)

    # Profit Factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0
    metrics["profit_factor"] = float(profit_factor)

    # --- Benchmark-Relative Metrics ---
    if benchmark is not None and len(benchmark) > 1:
        bench_returns = benchmark.pct_change().dropna()
        # Align on common dates
        common_idx = returns.index.intersection(bench_returns.index)
        if len(common_idx) > 10:
            aligned_returns = returns.loc[common_idx]
            aligned_bench = bench_returns.loc[common_idx]
            excess = aligned_returns - aligned_bench

            # Tracking Error (annualized)
            tracking_error = excess.std(ddof=1) * np.sqrt(trading_days_per_year)
            metrics["tracking_error"] = float(tracking_error)

            # Information Ratio
            mean_excess_bench = excess.mean() * trading_days_per_year
            if tracking_error > 0:
                information_ratio = mean_excess_bench / tracking_error
            else:
                information_ratio = 0.0
            metrics["information_ratio"] = float(information_ratio)

            # Beta and Alpha (CAPM regression)
            bench_excess = aligned_bench - daily_rf
            port_excess = aligned_returns - daily_rf

            if bench_excess.std() > 0:
                beta = np.cov(port_excess, bench_excess)[0, 1] / np.var(bench_excess, ddof=1)
                # Jensen's alpha (annualized)
                alpha_daily = port_excess.mean() - beta * bench_excess.mean()
                alpha = alpha_daily * trading_days_per_year
            else:
                beta = 0.0
                alpha = 0.0

            metrics["beta"] = float(beta)
            metrics["alpha"] = float(alpha)
        else:
            metrics["tracking_error"] = 0.0
            metrics["information_ratio"] = 0.0
            metrics["beta"] = 0.0
            metrics["alpha"] = 0.0
    else:
        metrics["tracking_error"] = 0.0
        metrics["information_ratio"] = 0.0
        metrics["beta"] = 0.0
        metrics["alpha"] = 0.0

    # Skewness and Kurtosis
    metrics["skewness"] = float(returns.skew())
    metrics["kurtosis"] = float(returns.kurtosis())

    return metrics


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    """
    Compute the maximum drawdown duration in trading days.

    Duration is measured from the peak before the drawdown to the
    recovery point (or end of series if not recovered).
    """
    in_drawdown = drawdown < 0
    if not in_drawdown.any():
        return 0

    max_duration = 0
    current_duration = 0

    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return max_duration


def format_report(metrics: Dict) -> str:
    """
    Format metrics as a readable table.

    Args:
        metrics: Dictionary from compute_metrics().

    Returns:
        Multi-line formatted string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("PERFORMANCE REPORT")
    lines.append("=" * 60)

    # Return section
    lines.append("")
    lines.append("--- Returns ---")
    lines.append(f"  Total Return:      {metrics.get('total_return', 0):>10.2%}")
    lines.append(f"  CAGR:              {metrics.get('cagr', 0):>10.2%}")
    lines.append(f"  Period (years):    {metrics.get('n_years', 0):>10.1f}")

    # Risk section
    lines.append("")
    lines.append("--- Risk ---")
    lines.append(f"  Annual Volatility: {metrics.get('annual_volatility', 0):>10.2%}")
    lines.append(f"  Max Drawdown:      {metrics.get('max_drawdown', 0):>10.2%}")
    lines.append(f"  Max DD Duration:   {metrics.get('max_drawdown_duration', 0):>10d} days")

    # Risk-adjusted section
    lines.append("")
    lines.append("--- Risk-Adjusted ---")
    sharpe = metrics.get("sharpe", 0)
    sortino = metrics.get("sortino", 0)
    calmar = metrics.get("calmar", 0)
    lines.append(f"  Sharpe Ratio:      {sharpe:>10.3f}")
    lines.append(f"  Sortino Ratio:     {sortino:>10.3f}" if sortino != float("inf") else f"  Sortino Ratio:     {'inf':>10s}")
    lines.append(f"  Calmar Ratio:      {calmar:>10.3f}" if calmar != float("inf") else f"  Calmar Ratio:      {'inf':>10s}")

    # Trade section
    lines.append("")
    lines.append("--- Trading ---")
    lines.append(f"  Win Rate:          {metrics.get('win_rate', 0):>10.2%}")
    pf = metrics.get("profit_factor", 0)
    lines.append(f"  Profit Factor:     {pf:>10.3f}" if pf != float("inf") else f"  Profit Factor:     {'inf':>10s}")
    lines.append(f"  Annual Turnover:   {metrics.get('annual_turnover', 0):>10.1%}")

    # Benchmark section
    lines.append("")
    lines.append("--- vs Benchmark ---")
    lines.append(f"  Information Ratio: {metrics.get('information_ratio', 0):>10.3f}")
    lines.append(f"  Alpha (annual):    {metrics.get('alpha', 0):>10.2%}")
    lines.append(f"  Beta:              {metrics.get('beta', 0):>10.3f}")
    lines.append(f"  Tracking Error:    {metrics.get('tracking_error', 0):>10.2%}")

    # Distribution
    lines.append("")
    lines.append("--- Distribution ---")
    lines.append(f"  Skewness:          {metrics.get('skewness', 0):>10.3f}")
    lines.append(f"  Kurtosis:          {metrics.get('kurtosis', 0):>10.3f}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def compare_strategies(results: Dict[str, "BacktestResult"]) -> pd.DataFrame:
    """
    Side-by-side comparison of multiple strategies.

    Args:
        results: {strategy_name: BacktestResult} dictionary.

    Returns:
        DataFrame with strategies as columns and metrics as rows.
    """
    from .engine import BacktestResult

    comparison = {}
    for name, result in results.items():
        m = result.metrics
        comparison[name] = {
            "CAGR": f"{m.get('cagr', 0):.2%}",
            "Sharpe": f"{m.get('sharpe', 0):.3f}",
            "Sortino": f"{m.get('sortino', 0):.3f}",
            "Max DD": f"{m.get('max_drawdown', 0):.2%}",
            "Calmar": f"{m.get('calmar', 0):.3f}",
            "Win Rate": f"{m.get('win_rate', 0):.1%}",
            "IR": f"{m.get('information_ratio', 0):.3f}",
            "Alpha": f"{m.get('alpha', 0):.2%}",
            "Beta": f"{m.get('beta', 0):.3f}",
            "Ann. Turnover": f"{m.get('annual_turnover', 0):.0%}",
            "Ann. Vol": f"{m.get('annual_volatility', 0):.2%}",
        }

    df = pd.DataFrame(comparison)
    df.index.name = "Metric"
    return df
