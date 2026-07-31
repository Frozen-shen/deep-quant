"""Performance metrics from equity curves."""
import numpy as np
import pandas as pd
from typing import Dict


def compute_metrics(equity_curve: pd.Series, benchmark: pd.Series = None,
                    risk_free_rate: float = 0.02) -> Dict:
    """Compute comprehensive performance metrics from a daily equity curve."""
    if len(equity_curve) < 20:
        return {'error': 'insufficient data'}

    daily_ret = equity_curve.pct_change().dropna()
    n_days = len(daily_ret)
    n_years = n_days / 252

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    daily_rf = risk_free_rate / 252
    excess_daily = daily_ret - daily_rf
    vol_annual = daily_ret.std() * np.sqrt(252)
    sharpe = (excess_daily.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    win_rate = (daily_ret > 0).mean()
    gains = daily_ret[daily_ret > 0].sum()
    losses = abs(daily_ret[daily_ret < 0].sum())
    profit_factor = gains / losses if losses > 0 else float('inf')

    monthly_ret = equity_curve.resample('ME').last().pct_change().dropna()
    monthly_win = (monthly_ret > 0).mean() if len(monthly_ret) > 0 else 0

    result = {
        'total_return_pct': total_return * 100,
        'cagr_pct': cagr * 100,
        'volatility_pct': vol_annual * 100,
        'sharpe': sharpe,
        'max_drawdown_pct': max_dd * 100,
        'calmar': calmar,
        'win_rate_daily': win_rate,
        'win_rate_monthly': monthly_win,
        'profit_factor': profit_factor,
        'skewness': float(daily_ret.skew()),
        'kurtosis': float(daily_ret.kurtosis()),
        'n_days': n_days,
        'n_years': round(n_years, 2),
    }

    if benchmark is not None and len(benchmark) > 1:
        bench_ret = benchmark.pct_change().dropna()
        common = daily_ret.index.intersection(bench_ret.index)
        active_ret = daily_ret[common] - bench_ret[common]
        bench_total = benchmark.iloc[-1] / benchmark.iloc[0] - 1
        tracking_error = active_ret.std() * np.sqrt(252)
        ir = (active_ret.mean() / active_ret.std() * np.sqrt(252)) if active_ret.std() > 0 else 0
        result.update({
            'benchmark_return_pct': bench_total * 100,
            'excess_return_pct': (total_return - bench_total) * 100,
            'tracking_error_pct': tracking_error * 100,
            'information_ratio': ir,
        })

    return result


def format_report(metrics: Dict, title: str = "Performance Report") -> str:
    """Format metrics as a readable report string."""
    lines = [f"\n{'='*60}", f"  {title}", f"{'='*60}"]
    fmt = [
        ('total_return_pct', 'Total Return', '%.2f%%'),
        ('cagr_pct', 'CAGR', '%.2f%%'),
        ('volatility_pct', 'Volatility', '%.2f%%'),
        ('sharpe', 'Sharpe', '%.3f'),
        ('max_drawdown_pct', 'Max Drawdown', '%.2f%%'),
        ('calmar', 'Calmar', '%.3f'),
        ('win_rate_daily', 'Daily Win Rate', '%.1f%%'),
        ('win_rate_monthly', 'Monthly Win Rate', '%.1f%%'),
        ('excess_return_pct', 'Excess Return', '%.2f%%'),
        ('information_ratio', 'IR', '%.3f'),
    ]
    for key, label, _ in fmt:
        if key in metrics:
            lines.append(f"  {label:20s}: {metrics[key]:+.2f}")
    lines.append(f"  {'Period':20s}: {metrics.get('n_years', 0):.1f}y ({metrics.get('n_days', 0)}d)")
    lines.append(f"{'='*60}\n")
    return '\n'.join(lines)
