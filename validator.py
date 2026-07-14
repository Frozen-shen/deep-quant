"""
滚动窗口验证引擎 — 时间序列向前验证 + 参数扫描 + 多股票横截面

核心原则:
- 训练期 (In-Sample): 调参/优化 → 测试期 (Out-of-Sample): 只评估一次
- Expanding Window: 训练期随时间累积扩大（用尽全部历史信息）
- Rolling Window: 训练期固定长度滑动（更适合市场风格切换时）
"""

import os
import sys
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import DataFetcher
from strategy import MACrossoverStrategy
from backtest import BacktestEngine
from analysis import PerformanceAnalyzer

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================================
#  数据结构
# ============================================================================

@dataclass
class WindowResult:
    """单次窗口回测结果"""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train_days: int
    n_test_days: int
    params: Dict[str, Any]           # 使用的参数
    metrics: Dict[str, float]        # 测试期绩效指标
    equity_curve: Optional[pd.DataFrame] = None  # 测试期权益曲线


@dataclass
class WFOResult:
    """滚动窗口验证汇总结果"""
    symbol: str
    mode: str                          # "expanding" / "rolling"
    windows: List[WindowResult] = field(default_factory=list)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """人类可读的汇总"""
        m = self.aggregate_metrics
        sig = m.get("significant_windows", 0)
        total = len(self.windows)
        return (
            f"OOS总收益: {m.get('mean_annual_return', 0)*100:+.2f}%  |  "
            f"OOS夏普: {m.get('mean_sharpe', 0):.2f}  |  "
            f"显著窗口: {sig}/{total}  |  "
            f"OOS胜率: {m.get('mean_win_rate', 0)*100:.0f}%"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """每个窗口一行"""
        rows = []
        for w in self.windows:
            rows.append({
                "window": w.window_id,
                "train": f"{w.train_start}~{w.train_end}",
                "test": f"{w.test_start}~{w.test_end}",
                "n_test_days": w.n_test_days,
                "annual_return": w.metrics.get("annual_return", 0),
                "sharpe": w.metrics.get("sharpe_ratio", 0),
                "sortino": w.metrics.get("sortino_ratio", 0),
                "max_dd": w.metrics.get("max_drawdown", 0),
                "win_rate": w.metrics.get("win_rate", 0),
                "trades": w.metrics.get("total_trades", 0),
                "params": str(w.params),
            })
        return pd.DataFrame(rows)

    def plot(self, save_path: Optional[str] = None):
        """绘制所有窗口的 OOS 权益曲线叠加（按百分比归一化）。"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.windows)))

        # ---- 上: 归一化权益曲线 ----
        ax1 = axes[0]
        for i, w in enumerate(self.windows):
            if w.equity_curve is not None and len(w.equity_curve) > 0:
                eq = w.equity_curve["equity"].values
                eq_norm = eq / eq[0] * 100  # 归一化到 100
                dates = w.equity_curve["date"].values
                label = f"W{w.window_id} ({w.test_start[:4]}~{w.test_end[:4]})"
                ax1.plot(dates, eq_norm, color=colors[i], linewidth=1.2, label=label, alpha=0.8)
        ax1.axhline(y=100, color="black", linestyle=":", alpha=0.5)
        ax1.set_ylabel("权益 (归一化 %)")
        ax1.set_title(f"OOS 权益曲线 (各窗口归一化起点=100)")
        ax1.legend(loc="best", fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)

        # ---- 下: 窗口夏普柱状图 ----
        ax2 = axes[1]
        window_ids = [w.window_id for w in self.windows]
        sharpes = [w.metrics.get("sharpe_ratio", 0) for w in self.windows]
        bars = ax2.bar(window_ids, sharpes, color=[
            "green" if s > 0 else "red" for s in sharpes
        ], alpha=0.7)
        ax2.axhline(y=0, color="black", linewidth=0.8)
        ax2.axhline(y=self.aggregate_metrics.get("mean_sharpe", 0), color="blue",
                    linestyle="--", linewidth=1.5,
                    label=f"均值 Sharpe: {self.aggregate_metrics.get('mean_sharpe', 0):.3f}")
        ax2.set_xlabel("窗口编号")
        ax2.set_ylabel("Sharpe Ratio")
        ax2.set_title("各窗口 OOS Sharpe Ratio")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[Validator] 图表已保存至: {save_path}")
        return fig


# ============================================================================
#  滚动窗口验证器
# ============================================================================

class WalkForwardValidator:
    """
    时间序列向前滚动验证。

    参数
    ----
    n_splits : int
        分割次数（生成 n_splits 个 train/test 对）
    train_years : float
        每段训练期年数（expanding 模式下为首段长度）
    test_years : float
        每段测试期年数
    mode : str
        "expanding": 训练期随时间累积增长
        "rolling":   训练期固定长度滑动
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_years: float = 3.0,
        test_years: float = 1.0,
        mode: str = "expanding",
        initial_capital: float = 100_000,
        commission: float = 0.0003,
    ):
        self.n_splits = n_splits
        self.train_years = train_years
        self.test_years = test_years
        self.mode = mode
        self.initial_capital = initial_capital
        self.commission = commission

    def run(
        self,
        symbol: str = "600519",
        param_grid: Optional[Dict] = None,
        start_date: str = "20180101",
        end_date: str = "20260710",
        adjust: str = "qfq",
        quiet: bool = False,
    ) -> WFOResult:
        """
        完整滚动窗口验证。

        参数
        ----
        symbol : str
        param_grid : dict or None
            若不为 None，每段训练期内做网格搜索最优参数
        start_date / end_date : str
            数据范围 (YYYYMMDD)
        adjust : str
            复权方式
        quiet : bool
            静默模式（减少打印）

        返回
        ----
        WFOResult
        """
        # 获取全量数据
        fetcher = DataFetcher()
        df_full = fetcher.fetch_daily(symbol, start_date, end_date, adjust)
        if not quiet:
            print(f"\n[Validator] 全量数据: {df_full['date'].min().date()} ~ "
                  f"{df_full['date'].max().date()}, {len(df_full)} 天")

        # 生成窗口
        windows_def = self._generate_windows(df_full)
        results: List[WindowResult] = []

        for wdef in windows_def:
            wid = wdef["window_id"]
            train_mask = (df_full["date"] >= wdef["train_start"]) & (df_full["date"] <= wdef["train_end"])
            test_mask = (df_full["date"] >= wdef["test_start"]) & (df_full["date"] <= wdef["test_end"])
            df_train = df_full[train_mask].copy()
            df_test = df_full[test_mask].copy()

            if len(df_train) < 50 or len(df_test) < 20:
                if not quiet:
                    print(f"  W{wid}: 数据不足 (train={len(df_train)}, test={len(df_test)}), 跳过")
                continue

            # 确定参数
            best_params = {"short_window": 5, "long_window": 20}  # 默认
            if param_grid:
                sweeper = _ParameterSweeper(param_grid, self.initial_capital, self.commission)
                best_params = sweeper.search(df_train, quiet=quiet)
                if not quiet:
                    print(f"  W{wid} 最优参数: {best_params}")

            # 训练期 → 重新生成信号（用最优参数）
            strategy = MACrossoverStrategy(
                short_window=best_params["short_window"],
                long_window=best_params["long_window"],
            )
            df_test_sig = strategy.generate_signals(df_test)

            # 回测
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                commission=self.commission,
            )
            df_test_bt = engine.run(df_test_sig)

            # 分析
            analyzer = PerformanceAnalyzer()
            metrics = analyzer.analyze(df_test_bt, self.initial_capital)

            if not quiet:
                print(f"  W{wid} OOS: 收益={metrics['annual_return']*100:+.2f}%, "
                      f"Sharpe={metrics['sharpe_ratio']:.3f}, "
                      f"MaxDD={metrics['max_drawdown']*100:.2f}%")

            results.append(WindowResult(
                window_id=wid,
                train_start=wdef["train_start"].strftime("%Y-%m-%d"),
                train_end=wdef["train_end"].strftime("%Y-%m-%d"),
                test_start=wdef["test_start"].strftime("%Y-%m-%d"),
                test_end=wdef["test_end"].strftime("%Y-%m-%d"),
                n_train_days=len(df_train),
                n_test_days=len(df_test),
                params=best_params,
                metrics=metrics,
                equity_curve=df_test_bt[["date", "equity"]].copy(),
            ))

        # 聚合 OOS 指标
        agg = self._aggregate(results)

        return WFOResult(
            symbol=symbol,
            mode=self.mode,
            windows=results,
            aggregate_metrics=agg,
        )

    # ---------- 内部 ----------
    def _generate_windows(self, df: pd.DataFrame) -> List[Dict]:
        """生成 train/test 日期分界"""
        dates = df["date"]
        t0 = dates.min()
        t_end = dates.max()
        windows = []

        test_days = int(self.test_years * 252)

        for i in range(self.n_splits):
            test_start = t0 + pd.DateOffset(years=int(self.train_years + i * self.test_years))
            test_end = test_start + pd.DateOffset(years=int(self.test_years))

            if test_end > t_end:
                break

            if self.mode == "expanding":
                train_start = t0
            else:  # rolling
                train_start = test_start - pd.DateOffset(years=int(self.train_years))

            train_end = test_start - pd.DateOffset(days=1)

            windows.append({
                "window_id": i + 1,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": min(test_end, t_end),
            })

        return windows

    @staticmethod
    def _aggregate(results: List[WindowResult]) -> Dict[str, float]:
        """聚合所有窗口的 OOS 指标"""
        if not results:
            return {}

        sharpes = [r.metrics["sharpe_ratio"] for r in results]
        annual_rets = [r.metrics["annual_return"] for r in results]
        max_dds = [r.metrics["max_drawdown"] for r in results]
        win_rates = [r.metrics["win_rate"] for r in results]
        sortinos = [r.metrics["sortino_ratio"] for r in results]

        n_sig = sum(1 for s in sharpes if s > 0.3)

        return {
            "n_windows": len(results),
            "significant_windows": n_sig,
            "mean_sharpe": np.mean(sharpes),
            "std_sharpe": np.std(sharpes),
            "min_sharpe": np.min(sharpes),
            "max_sharpe": np.max(sharpes),
            "mean_annual_return": np.mean(annual_rets),
            "mean_max_dd": np.mean(max_dds),
            "mean_win_rate": np.mean(win_rates),
            "mean_sortino": np.mean(sortinos),
            "sharpe_of_sharpes": np.mean(sharpes) / np.std(sharpes) if np.std(sharpes) > 0 else 0,
        }


# ============================================================================
#  内部网格搜索器
# ============================================================================

class _ParameterSweeper:
    """训练期内网格搜索最优参数（不暴露给外部）"""

    def __init__(self, param_space: Dict, initial_capital: float, commission: float):
        self.param_space = param_space
        self.initial_capital = initial_capital
        self.commission = commission

    def search(self, df_train: pd.DataFrame, quiet: bool = False) -> Dict[str, Any]:
        """
        在训练数据上暴力搜索最优参数组合。

        评估标准: Sharpe Ratio
        """
        keys = list(self.param_space.keys())
        values = list(self.param_space.values())
        best_sharpe = -999
        best_params = {}

        total_combos = 1
        for v in values:
            total_combos *= len(v)

        combo_count = 0
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            combo_count += 1

            # 校验
            if params.get("short_window", 5) >= params.get("long_window", 20):
                continue

            try:
                strategy = MACrossoverStrategy(**params)
                df_sig = strategy.generate_signals(df_train)

                engine = BacktestEngine(self.initial_capital, self.commission)
                df_bt = engine.run(df_sig)

                analyzer = PerformanceAnalyzer()
                metrics = analyzer.analyze(df_bt, self.initial_capital)
                sharpe = metrics["sharpe_ratio"]

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
            except Exception:
                continue

        if not best_params:
            best_params = {"short_window": 5, "long_window": 20}

        if not quiet:
            print(f"    [Sweep] {combo_count} 组合 → 最优 {best_params} (Sharpe={best_sharpe:.3f})")

        return best_params


# ============================================================================
#  参数扫描（公开API）
# ============================================================================

class ParameterSweep:
    """
    在滚动窗口框架内做参数扫描，评估参数稳定性。

    用法:
        sweep = ParameterSweep(
            param_space={"short_window": [3,5,10], "long_window": [15,20,30]}
        )
        result = sweep.run(validator, "600519")
    """

    def __init__(self, param_space: Dict):
        self.param_space = param_space

    def run(self, validator: WalkForwardValidator, symbol: str = "600519") -> Dict:
        """
        对每个验证窗口做参数网格搜索，汇总结果。
        """
        print(f"\n{'='*60}")
        print(f"  参数扫描: {self.param_space}")
        print(f"{'='*60}")

        result = validator.run(
            symbol=symbol,
            param_grid=self.param_space,
        )

        # 参数稳定性分析
        all_params = [w.params for w in result.windows]
        stability = self._analyze_stability(all_params)

        # 打印
        df = result.to_dataframe()
        print("\n窗口明细:")
        print(df[["window", "annual_return", "sharpe", "max_dd", "params"]].to_string(index=False))

        print(f"\n参数稳定性:")
        for k, v in stability.items():
            print(f"  {k}: {v}")

        return {
            "wfo_result": result,
            "stability": stability,
        }

    @staticmethod
    def _analyze_stability(params_list: List[Dict]) -> Dict:
        """分析参数在不同窗口间的一致性"""
        if not params_list:
            return {}

        keys = params_list[0].keys()
        stability = {}
        for k in keys:
            values = [p[k] for p in params_list]
            stability[k] = {
                "values": values,
                "unique": len(set(values)),
                "mode": max(set(values), key=values.count),
                "stable": len(set(values)) <= 2,  # 少于3个不同值算稳定
            }
        return stability


# ============================================================================
#  多股票横截面验证
# ============================================================================

class MultiStockValidator:
    """
    同一策略在 N 只股票上分别做滚动窗口验证，报告横截面统计。
    """

    def __init__(self, symbols: List[str], validator: WalkForwardValidator):
        self.symbols = symbols
        self.validator = validator

    def run_all(self) -> Dict:
        """对所有股票依次跑 WFO，汇总横截面结论。"""
        print(f"\n{'='*60}")
        print(f"  多股票验证: {len(self.symbols)} 只")
        print(f"{'='*60}")

        all_results: Dict[str, WFOResult] = {}
        cross_sectional = []

        for sym in self.symbols:
            print(f"\n--- {sym} ---")
            try:
                wfo = self.validator.run(symbol=sym, quiet=True)
                all_results[sym] = wfo
                cross_sectional.append({
                    "symbol": sym,
                    "mean_sharpe": wfo.aggregate_metrics.get("mean_sharpe", 0),
                    "mean_return": wfo.aggregate_metrics.get("mean_annual_return", 0),
                    "mean_max_dd": wfo.aggregate_metrics.get("mean_max_dd", 0),
                    "sig_windows": wfo.aggregate_metrics.get("significant_windows", 0),
                    "n_windows": wfo.aggregate_metrics.get("n_windows", 0),
                })
            except Exception as e:
                print(f"  ❌ {sym} 失败: {e}")

        df_cs = pd.DataFrame(cross_sectional)

        # 横截面统计
        summary = {
            "total_stocks": len(self.symbols),
            "tested_stocks": len(df_cs),
            "positive_sharpe_pct": (df_cs["mean_sharpe"] > 0).mean() if len(df_cs) > 0 else 0,
            "mean_of_sharpes": df_cs["mean_sharpe"].mean() if len(df_cs) > 0 else 0,
            "median_sharpe": df_cs["mean_sharpe"].median() if len(df_cs) > 0 else 0,
            "sharpe_std_cross": df_cs["mean_sharpe"].std() if len(df_cs) > 0 else 0,
            "best_symbol": df_cs.loc[df_cs["mean_sharpe"].idxmax(), "symbol"] if len(df_cs) > 0 else "",
            "worst_symbol": df_cs.loc[df_cs["mean_sharpe"].idxmin(), "symbol"] if len(df_cs) > 0 else "",
        }

        print(f"\n{'='*60}")
        print(f"  横截面汇总")
        print(f"{'='*60}")
        print(f"  测试股票: {summary['tested_stocks']}/{summary['total_stocks']}")
        print(f"  正Sharpe比例: {summary['positive_sharpe_pct']*100:.0f}%")
        print(f"  平均Sharpe: {summary['mean_of_sharpes']:.3f}")
        print(f"  中位数Sharpe: {summary['median_sharpe']:.3f}")
        print(f"  Sharpe标准差: {summary['sharpe_std_cross']:.3f}")
        print(f"  最佳: {summary['best_symbol']}, 最差: {summary['worst_symbol']}")

        if len(df_cs) > 0:
            print(f"\n{df_cs.to_string(index=False)}")

        return {
            "all_results": all_results,
            "cross_sectional_df": df_cs,
            "summary": summary,
        }
