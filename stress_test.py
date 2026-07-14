"""
压力测试模块 — 崩盘回放、手续费/滑点敏感性、波动率分时段

不修改现有模块，直接组合调用 DataFetcher + Strategy + BacktestEngine。
"""

import os
import sys
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import DataFetcher
from strategy import MACrossoverStrategy
from backtest import BacktestEngine
from analysis import PerformanceAnalyzer

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================================
#  已知崩盘区间 (A股)
# ============================================================================

CRASH_PERIODS = {
    "2015股灾":     ("20150612", "20150826"),   # 上证 5178→2850
    "2016熔断":     ("20160104", "20160128"),   # 熔断潮
    "2018熊市":     ("20180129", "20190104"),   # 全年阴跌
    "2020新冠":     ("20200114", "20200319"),   # 全球流动性危机
    "2021春节后":    ("20210218", "20210309"),   # 抱团瓦解
    "2022俄乌":     ("20220224", "20220427"),   # 地缘冲击
    "2024救市反弹":  ("20240205", "20240318"),   # 极端低点反弹
}


# ============================================================================
#  StressTester
# ============================================================================

class StressTester:
    """
    压力测试器。

    参数
    ----
    short_window, long_window : int
        MA 参数
    initial_capital : float
    commission : float
        基准手续费
    """

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
        initial_capital: float = 100_000,
        commission: float = 0.0003,
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.initial_capital = initial_capital
        self.commission = commission

    # ---------- 内部：跑一次回测 ----------
    def _run_one(self, df: pd.DataFrame, commission: Optional[float] = None,
                 slippage: float = 0.0) -> Dict:
        """对一段数据跑回测，返回指标 dict"""
        comm = commission if commission is not None else self.commission
        strategy = MACrossoverStrategy(self.short_window, self.long_window)
        df_sig = strategy.generate_signals(df)

        # 模拟滑点：在信号日调整 close 价格
        if slippage > 0:
            df_sig_close = df_sig["close"].copy()
            buy_mask = df_sig["signal"] == 1
            sell_mask = df_sig["signal"] == -1
            df_sig.loc[buy_mask, "close"] *= (1 + slippage)   # 买入时更贵
            df_sig.loc[sell_mask, "close"] *= (1 - slippage)  # 卖出时更便宜

        engine = BacktestEngine(self.initial_capital, comm)
        df_bt = engine.run(df_sig)

        analyzer = PerformanceAnalyzer()
        return analyzer.analyze(df_bt, self.initial_capital)

    # ================================================================
    #  1. 历史崩盘回放
    # ================================================================
    def test_crashes(self, symbol: str = "600519") -> Dict:
        """
        在已知崩盘区间上跑策略，输出各区间表现。
        """
        print(f"\n{'='*60}")
        print(f"  压力测试: 历史崩盘回放 — {symbol}")
        print(f"{'='*60}")

        results = []
        for name, (start, end) in CRASH_PERIODS.items():
            try:
                fetcher = DataFetcher()
                df = fetcher.fetch_daily(symbol, start, end, adjust="qfq")

                if len(df) < 10:
                    print(f"  {name}: 数据不足 ({len(df)}天), 跳过")
                    continue

                metrics = self._run_one(df)
                benchmark = df["close"].iloc[-1] / df["close"].iloc[0] - 1

                results.append({
                    "period": name,
                    "dates": f"{start[:4]}-{start[4:6]} ~ {end[:4]}-{end[4:6]}",
                    "strategy_return": metrics["total_return"],
                    "benchmark_return": benchmark,
                    "sharpe": metrics["sharpe_ratio"],
                    "max_dd": metrics["max_drawdown"],
                    "trades": metrics["total_trades"],
                    "excess": metrics["total_return"] - benchmark,
                })

                print(f"  {name:12s}: 策略={metrics['total_return']*100:+.2f}%, "
                      f"基准={benchmark*100:+.2f}%, "
                      f"超额={metrics['total_return']-benchmark:+.4%}, "
                      f"Sharpe={metrics['sharpe_ratio']:.3f}")
            except Exception as e:
                print(f"  {name}: ❌ {e}")

        df_result = pd.DataFrame(results)

        summary = {}
        if not df_result.empty:
            summary = {
                "n_periods": len(df_result),
                "positive_excess_pct": (df_result["excess"] > 0).mean(),
                "mean_excess": df_result["excess"].mean(),
                "worst_period": df_result.loc[df_result["strategy_return"].idxmin(), "period"],
                "worst_return": df_result["strategy_return"].min(),
                "avg_sharpe_crash": df_result["sharpe"].mean(),
            }

            print(f"\n  汇总: {summary['n_periods']} 个崩盘区间, "
                  f"超额正收益 {summary['positive_excess_pct']*100:.0f}%, "
                  f"平均超额 {summary['mean_excess']*100:+.2f}%")

        return {"details": df_result, "summary": summary}

    # ================================================================
    #  2. 手续费敏感性扫描
    # ================================================================
    def test_commission_sensitivity(
        self,
        symbol: str = "600519",
        start_date: str = "20180101",
        end_date: str = "20260710",
        commission_rates: Optional[List[float]] = None,
    ) -> Dict:
        """
        在不同手续费率下回测，观察收益衰减曲线。
        """
        if commission_rates is None:
            commission_rates = [0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005]

        print(f"\n{'='*60}")
        print(f"  压力测试: 手续费敏感性 — {symbol}")
        print(f"{'='*60}")

        fetcher = DataFetcher()
        df = fetcher.fetch_daily(symbol, start_date, end_date, adjust="qfq")

        results = []
        for rate in commission_rates:
            metrics = self._run_one(df, commission=rate)
            results.append({
                "commission": rate,
                "commission_bps": int(rate * 10000),
                "annual_return": metrics["annual_return"],
                "sharpe": metrics["sharpe_ratio"],
                "total_trades": metrics["total_trades"],
                "final_equity": metrics["final_equity"],
            })
            print(f"  费率 {rate*10000:4.0f}bp: 年化={metrics['annual_return']*100:+.2f}%, "
                  f"Sharpe={metrics['sharpe_ratio']:.3f}, "
                  f"权益={metrics['final_equity']:,.0f}")

        df_res = pd.DataFrame(results)

        # 盈亏平衡费率
        breakeven = None
        for i in range(len(df_res) - 1):
            if df_res["annual_return"].iloc[i] > 0 and df_res["annual_return"].iloc[i+1] <= 0:
                breakeven = df_res["commission"].iloc[i]
                break

        summary = {
            "base_return": df_res["annual_return"].iloc[1] if len(df_res) > 1 else 0,
            "return_at_10bp": df_res["annual_return"].iloc[0] if len(df_res) > 0 else 0,
            "return_at_30bp": df_res["annual_return"].iloc[2] if len(df_res) > 2 else 0,
            "breakeven_rate": breakeven,
            "return_decay_per_10bp": (
                (df_res["annual_return"].iloc[0] - df_res["annual_return"].iloc[-1])
                / (df_res["commission"].iloc[-1] - df_res["commission"].iloc[0]) * 0.001
            ) if len(df_res) > 1 else 0,
        }

        if breakeven:
            print(f"\n  ⚠️ 盈亏平衡费率: {breakeven*10000:.0f}bp")
        print(f"  每+10bp手续费, 年化收益衰减约 {summary['return_decay_per_10bp']*100:.2f}%")

        return {"details": df_res, "summary": summary}

    # ================================================================
    #  3. 波动率分时段测试
    # ================================================================
    def test_volatility_regimes(
        self,
        symbol: str = "600519",
        start_date: str = "20180101",
        end_date: str = "20260710",
    ) -> Dict:
        """
        按历史波动率高低分组，测试策略在不同市场环境下的表现。
        """
        print(f"\n{'='*60}")
        print(f"  压力测试: 波动率分时段 — {symbol}")
        print(f"{'='*60}")

        fetcher = DataFetcher()
        df = fetcher.fetch_daily(symbol, start_date, end_date, adjust="qfq")

        # 计算滚动波动率 (20日)
        df["ret"] = df["close"].pct_change()
        df["vol_20d"] = df["ret"].rolling(20).std() * np.sqrt(252)

        # 去掉 NaN
        df_valid = df.dropna(subset=["vol_20d"]).copy()

        # 按波动率分三组
        vol_33 = df_valid["vol_20d"].quantile(0.33)
        vol_67 = df_valid["vol_20d"].quantile(0.67)

        labels = ["低波动", "中波动", "高波动"]
        df_valid["regime"] = pd.cut(df_valid["vol_20d"],
                                     bins=[-1, vol_33, vol_67, 999],
                                     labels=labels)

        results = []
        for regime_name in labels:
            df_regime = df_valid[df_valid["regime"] == regime_name]
            if len(df_regime) < 30:
                continue

            metrics = self._run_one(df_regime)
            results.append({
                "regime": regime_name,
                "n_days": len(df_regime),
                "annual_return": metrics["annual_return"],
                "sharpe": metrics["sharpe_ratio"],
                "max_dd": metrics["max_drawdown"],
                "sortino": metrics["sortino_ratio"],
            })
            print(f"  {regime_name} ({len(df_regime)}天): "
                  f"年化={metrics['annual_return']*100:+.2f}%, "
                  f"Sharpe={metrics['sharpe_ratio']:.3f}, "
                  f"Sortino={metrics['sortino_ratio']:.3f}")

        df_res = pd.DataFrame(results)

        summary = {}
        if len(df_res) >= 2:
            high = df_res[df_res["regime"] == "高波动"]
            low = df_res[df_res["regime"] == "低波动"]
            summary = {
                "high_vol_sharpe": high["sharpe"].values[0] if len(high) > 0 else 0,
                "low_vol_sharpe": low["sharpe"].values[0] if len(low) > 0 else 0,
                "sharpe_ratio_high_vs_low": (
                    high["sharpe"].values[0] / low["sharpe"].values[0]
                    if len(high) > 0 and len(low) > 0 and low["sharpe"].values[0] != 0
                    else 0
                ),
            }
            print(f"\n  高波动Sharpe / 低波动Sharpe = {summary['sharpe_ratio_high_vs_low']:.2f}")

        return {"details": df_res, "summary": summary}
