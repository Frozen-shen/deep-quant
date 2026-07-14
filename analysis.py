"""
绩效分析模块 — 计算量化指标、统计检验、可视化

核心指标: 年化收益、夏普/Sortino、最大回撤、信息比率、VaR/CVaR、Omega
统计检验: t-test、Bootstrap 置信区间、Sharpe 渐近标准误
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from typing import Dict, Optional, Tuple

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class PerformanceAnalyzer:
    """
    绩效分析器。

    参数
    ----
    risk_free_rate : float
        无风险利率（默认 0.02 即 2%）
    """

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate

    # ================================================================
    #  核心：analyze() — 返回纯数字 dict
    # ================================================================
    def analyze(self, df: pd.DataFrame, initial_capital: float = 100_000,
                benchmark_col: str = "close") -> Dict[str, float]:
        """
        计算所有绩效指标，返回纯数字（浮点数）字典。

        参数
        ----
        df : pd.DataFrame
            必须包含 equity, daily_returns, trade；benchmark_col 可选
        initial_capital : float

        返回
        ----
        dict[str, float] : 所有指标为原始浮点数，供下游计算和聚合
        """
        equity = df["equity"]
        returns = df["daily_returns"]

        n_days = len(df)
        n_years = max(n_days / 252, 0.01)

        # ---- 基础 ----
        final_equity = equity.iloc[-1]
        total_return = final_equity / initial_capital - 1
        annual_return = (final_equity / initial_capital) ** (1 / n_years) - 1

        # ---- 波动率 ----
        annual_vol = returns.std() * np.sqrt(252)

        # ---- 下行波动率（用于 Sortino）----
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0

        # ---- 超额收益（相对无风险利率）----
        excess_daily = returns - self.risk_free_rate / 252
        excess_annual = excess_daily.mean() * 252

        # ---- Sharpe ----
        sharpe = (excess_daily.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0

        # ---- Sortino ----
        sortino = excess_annual / downside_std if downside_std > 0 else 0.0

        # ---- 最大回撤 ----
        cummax = equity.cummax()
        drawdown_series = (equity - cummax) / cummax
        max_drawdown = drawdown_series.min()
        max_dd_idx = drawdown_series.idxmin()
        max_dd_date = df.loc[max_dd_idx, "date"] if "date" in df.columns else None

        # ---- Calmar ----
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        # ---- 胜率 & 盈亏比 ----
        win_mask = returns > 0
        lose_mask = returns < 0
        win_days = win_mask.sum()
        lose_days = lose_mask.sum()
        win_rate = win_days / (win_days + lose_days) if (win_days + lose_days) > 0 else 0.0
        avg_win = returns[win_mask].mean() if win_days > 0 else 0.0
        avg_loss = abs(returns[lose_mask].mean()) if lose_days > 0 else 0.0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

        # ---- Profit Factor ----
        gross_profit = returns[win_mask].sum() if win_days > 0 else 0.0
        gross_loss = abs(returns[lose_mask].sum()) if lose_days > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # ---- 交易统计 ----
        total_trades = df["trade"].notna().sum()
        buy_trades = (df["trade"] == "buy").sum()
        sell_trades = (df["trade"] == "sell").sum()

        # ---- VaR / CVaR ----
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

        # ---- Omega Ratio (threshold = 0) ----
        gains = np.maximum(returns, 0).sum()
        losses = abs(np.minimum(returns, 0).sum())
        omega = gains / losses if losses > 0 else float("inf")

        # ---- 基准对比 ----
        benchmark_return = 0.0
        benchmark_annual = 0.0
        information_ratio = 0.0
        tracking_error = 0.0
        alpha = 0.0
        beta = 0.0

        if benchmark_col in df.columns and not df[benchmark_col].isna().all():
            bench_close = df[benchmark_col]
            benchmark_return = bench_close.iloc[-1] / bench_close.iloc[0] - 1
            benchmark_annual = (bench_close.iloc[-1] / bench_close.iloc[0]) ** (1 / n_years) - 1

            # 基准日收益
            bench_returns = bench_close.pct_change().fillna(0)

            # 跟踪误差 & 信息比率
            active_returns = returns - bench_returns
            tracking_error = active_returns.std() * np.sqrt(252)
            ir = (active_returns.mean() / active_returns.std()) * np.sqrt(252) if active_returns.std() > 0 else 0.0
            information_ratio = ir

            # Alpha / Beta (简单线性回归)
            if len(returns) > 10 and bench_returns.std() > 0:
                cov = np.cov(returns, bench_returns)
                if cov.shape == (2, 2):
                    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
                    alpha = (returns.mean() - beta * bench_returns.mean()) * 252

        # ---- 超额收益 ----
        excess_vs_benchmark = total_return - benchmark_return

        # ---- 构建返回字典（纯数字）----
        metrics = {
            # 基础
            "n_days": n_days,
            "n_years": round(n_years, 2),
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            # 风险调整
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "information_ratio": information_ratio,
            # 回撤
            "max_drawdown": max_drawdown,
            # 收益分解
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio if profit_loss_ratio != float("inf") else 999.0,
            "profit_factor": profit_factor if profit_factor != float("inf") else 999.0,
            # 风险度量
            "var_95": var_95,
            "cvar_95": cvar_95,
            "omega_ratio": omega if omega != float("inf") else 999.0,
            # 交易
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            # 基准
            "benchmark_return": benchmark_return,
            "benchmark_annual": benchmark_annual,
            "excess_vs_benchmark": excess_vs_benchmark,
            "tracking_error": tracking_error,
            "alpha": alpha,
            "beta": beta,
            # 辅助
            "max_dd_date": max_dd_date,
            "downside_std": downside_std,
        }

        return metrics

    # ================================================================
    #  统计显著性检验
    # ================================================================
    def test_significance(self, df: pd.DataFrame, benchmark_col: str = "close",
                          n_bootstrap: int = 10000) -> Dict:
        """
        对回测结果做统计显著性检验。

        返回
        ----
        dict: {
            "mean_return_annual": float,
            "t_statistic": float,
            "t_pvalue": float,
            "is_significant_5pct": bool,
            "sharpe_asymptotic_se": float,
            "sharpe_ci_95_lower": float,
            "sharpe_ci_95_upper": float,
            "bootstrap_ci_90": (float, float),
            "bootstrap_ci_95": (float, float),
        }
        """
        returns = df["daily_returns"].values
        n = len(returns)
        sharpe = self.analyze(df)["sharpe_ratio"]

        # ---- t-test: H0: mean(returns) = 0 ----
        t_stat, t_pvalue = stats.ttest_1samp(returns, 0.0)
        mean_return_annual = returns.mean() * 252

        # ---- Sharpe 渐近标准误 ----
        # SE(Sharpe) ≈ sqrt((1 + 0.5 * SR^2) / n)  per year
        sharpe_se = np.sqrt((1 + 0.5 * sharpe**2) / n) * np.sqrt(252)
        sharpe_ci_lower = sharpe - 1.96 * sharpe_se
        sharpe_ci_upper = sharpe + 1.96 * sharpe_se

        # ---- Bootstrap Sharpe ----
        rng = np.random.RandomState(42)
        bootstrap_sharpes = []
        for _ in range(n_bootstrap):
            sample = rng.choice(returns, size=n, replace=True)
            bs_mean = sample.mean()
            bs_std = sample.std()
            if bs_std > 0:
                bs_sr = (bs_mean - self.risk_free_rate / 252) / bs_std * np.sqrt(252)
            else:
                bs_sr = 0.0
            bootstrap_sharpes.append(bs_sr)

        bootstrap_sharpes = np.array(bootstrap_sharpes)
        bs_ci_90 = (np.percentile(bootstrap_sharpes, 5),
                     np.percentile(bootstrap_sharpes, 95))
        bs_ci_95 = (np.percentile(bootstrap_sharpes, 2.5),
                     np.percentile(bootstrap_sharpes, 97.5))

        return {
            "mean_return_annual": mean_return_annual,
            "t_statistic": t_stat,
            "t_pvalue": t_pvalue,
            "is_significant_5pct": t_pvalue < 0.05,
            "sharpe_asymptotic_se": sharpe_se,
            "sharpe_ci_95_lower": sharpe_ci_lower,
            "sharpe_ci_95_upper": sharpe_ci_upper,
            "bootstrap_sharpe_ci_90": bs_ci_90,
            "bootstrap_sharpe_ci_95": bs_ci_95,
        }

    # ================================================================
    #  报告输出
    # ================================================================
    @staticmethod
    def print_report(metrics: Dict, sig_tests: Optional[Dict] = None):
        """格式化打印绩效报告 + 统计检验"""

        # ---- 格式化辅助 ----
        def pct(v): return f"{v*100:+.2f}%"
        def num(v, d=2): return f"{v:,.{d}f}"
        def pval(p): return f"{p:.4f}" + (" *" if p < 0.05 else "") + (" **" if p < 0.01 else "")

        print("\n" + "=" * 62)
        print("                      绩 效 报 告")
        print("=" * 62)

        rows = [
            ("回测天数", f"{metrics['n_days']}"),
            ("回测年数", f"{metrics['n_years']}"),
            ("初始资金", f"{metrics['initial_capital']:,.0f}"),
            ("最终权益", f"{metrics['final_equity']:,.2f}"),
            ("", ""),
            ("总收益率", pct(metrics['total_return'])),
            ("年化收益率", pct(metrics['annual_return'])),
            ("年化波动率", pct(metrics['annual_volatility'])),
            ("", ""),
            ("夏普比率", f"{metrics['sharpe_ratio']:.3f}"),
            ("Sortino比率", f"{metrics['sortino_ratio']:.3f}"),
            ("卡玛比率", f"{metrics['calmar_ratio']:.3f}"),
            ("信息比率", f"{metrics['information_ratio']:.3f}"),
            ("", ""),
            ("最大回撤", pct(metrics['max_drawdown'])),
            ("VaR 95% (日)", pct(metrics['var_95'])),
            ("CVaR 95% (日)", pct(metrics['cvar_95'])),
            ("", ""),
            ("日胜率", pct(metrics['win_rate'])),
            ("盈亏比", f"{metrics['profit_loss_ratio']:.3f}"),
            ("Profit Factor", f"{metrics['profit_factor']:.3f}"),
            ("Omega Ratio", f"{metrics['omega_ratio']:.3f}"),
            ("", ""),
            ("总交易次数", f"{metrics['total_trades']}"),
            ("买入次数", f"{metrics['buy_trades']}"),
            ("卖出次数", f"{metrics['sell_trades']}"),
            ("", ""),
            ("Alpha (年化)", pct(metrics['alpha'])),
            ("Beta", f"{metrics['beta']:.3f}"),
            ("基准总收益", pct(metrics['benchmark_return'])),
            ("超额收益(vs基准)", pct(metrics['excess_vs_benchmark'])),
        ]

        for label, value in rows:
            if label == "":
                print()
            else:
                print(f"  {label:.<32} {value:>10}")

        # ---- 统计检验 ----
        if sig_tests:
            print("\n" + "-" * 62)
            print("                    统 计 检 验")
            print("-" * 62)
            print(f"  H0: 日均收益 = 0")
            print(f"    t = {sig_tests['t_statistic']:.3f},  p = {pval(sig_tests['t_pvalue'])}")
            print(f"    5%水平显著: {'是 ✅' if sig_tests['is_significant_5pct'] else '否 ❌'}")
            print()
            print(f"  Sharpe 渐近 95% CI:")
            print(f"    [{sig_tests['sharpe_ci_95_lower']:.3f}, {sig_tests['sharpe_ci_95_upper']:.3f}]")
            print(f"    (SE = {sig_tests['sharpe_asymptotic_se']:.3f})")
            print()
            print(f"  Bootstrap Sharpe 90% CI:  [{sig_tests['bootstrap_sharpe_ci_90'][0]:.3f}, {sig_tests['bootstrap_sharpe_ci_90'][1]:.3f}]")
            print(f"  Bootstrap Sharpe 95% CI:  [{sig_tests['bootstrap_sharpe_ci_95'][0]:.3f}, {sig_tests['bootstrap_sharpe_ci_95'][1]:.3f}]")

        print("=" * 62 + "\n")

    # ================================================================
    #  可视化（不变，保持兼容）
    # ================================================================
    @staticmethod
    def plot(df: pd.DataFrame, symbol: str = "", save_path: str = None,
             title_suffix: str = ""):
        """生成三合一绩效图表。"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        full_title = f"双均线交叉策略回测结果 — {symbol}"
        if title_suffix:
            full_title += f" ({title_suffix})"
        fig.suptitle(full_title, fontsize=16, fontweight="bold")

        # ---- 子图1: 价格与信号 ----
        ax1 = axes[0]
        ax1.plot(df["date"], df["close"], label="收盘价", color="black", alpha=0.6, linewidth=0.8)
        if "ma_short" in df.columns:
            ax1.plot(df["date"], df["ma_short"], label="MA短", color="blue", linewidth=1)
        if "ma_long" in df.columns:
            ax1.plot(df["date"], df["ma_long"], label="MA长", color="orange", linewidth=1)

        buys = df[df.get("trade") == "buy"] if "trade" in df.columns else df[df["signal"] == 1]
        sells = df[df.get("trade") == "sell"] if "trade" in df.columns else df[df["signal"] == -1]
        ax1.scatter(buys["date"], buys["close"], marker="^", color="red",
                    s=80, zorder=5, label=f"买入 ({len(buys)}次)", alpha=0.9)
        ax1.scatter(sells["date"], sells["close"], marker="v", color="green",
                    s=80, zorder=5, label=f"卖出 ({len(sells)}次)", alpha=0.9)
        ax1.set_ylabel("价格 (元)")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ---- 子图2: 权益曲线 ----
        ax2 = axes[1]
        ax2.plot(df["date"], df["equity"], label="策略权益", color="blue", linewidth=1.5)
        benchmark = df["close"] / df["close"].iloc[0] * df["equity"].iloc[0]
        ax2.plot(df["date"], benchmark, label="买入持有基准", color="gray",
                 linestyle="--", linewidth=1)
        ax2.axhline(y=df["equity"].iloc[0], color="black", linestyle=":",
                    alpha=0.5, label="初始资金")
        ax2.set_ylabel("权益 (元)")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, alpha=0.3)

        # ---- 子图3: 回撤 ----
        ax3 = axes[2]
        cummax = df["equity"].cummax()
        drawdown = (df["equity"] - cummax) / cummax * 100
        ax3.fill_between(df["date"], drawdown, 0, color="red", alpha=0.3, label="回撤")
        ax3.plot(df["date"], drawdown, color="darkred", linewidth=0.8)
        ax3.axhline(y=drawdown.min(), color="red", linestyle="--", linewidth=1,
                    label=f"最大回撤: {drawdown.min():.2f}%")
        ax3.set_ylabel("回撤 (%)")
        ax3.set_xlabel("日期")
        ax3.legend(loc="lower left", fontsize=8)
        ax3.grid(True, alpha=0.3)

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[Analysis] 图表已保存至: {save_path}")

        if matplotlib.get_backend() != "Agg":
            plt.show()

        return fig
