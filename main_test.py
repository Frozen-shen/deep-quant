"""
三策略对比测试 — MA技术面 vs LLM事件 vs LLM混合

用法:
  python main_test.py                          # mock模式全对比
  LLM_BACKEND=openai python main_test.py       # DeepSeek真实决策
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import DataFetcher
from strategy import MACrossoverStrategy, LLMEventStrategy, apply_llm_filter
from backtest import BacktestEngine
from analysis import PerformanceAnalyzer
from news_fetcher import NewsFetcher
from event_fetcher import EventFetcher
from llm_factor import LLMFactorEngine
from validator import WalkForwardValidator
from stress_test import StressTester


def run_backtest(df, strategy_fn, label, **kwargs):
    """统一回测运行器"""
    df = strategy_fn(df, **kwargs)
    engine = BacktestEngine(100_000, 0.0003)
    df = engine.run(df)
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze(df, 100_000)
    return metrics, df


def strategy_tech(df, **kw):
    """纯技术面"""
    s = MACrossoverStrategy(short_window=5, long_window=20)
    return s.generate_signals(df)


def strategy_llm_event(df, **kw):
    """LLM事件驱动（LLM为主角）"""
    events_df = kw["events_df"]
    strategy = LLMEventStrategy(confidence_threshold=0.6)
    return strategy.generate_signals(df, events_df)


def strategy_llm_filter(df, **kw):
    """MA + LLM过滤（LLM为配角）"""
    daily_factors = kw["daily_factors"]
    s = MACrossoverStrategy(short_window=5, long_window=20)
    df_sig = s.generate_signals(df)
    return apply_llm_filter(df_sig, daily_factors, sentiment_threshold=0.0)


def main():
    SYMBOL = "600519"
    LLM_BACKEND = os.environ.get("LLM_BACKEND", "mock")
    LLM_API_KEY = os.environ.get("OPENAI_API_KEY", None)
    LLM_MODEL = os.environ.get("LLM_MODEL", None)
    LLM_BASE_URL = os.environ.get("LLM_BASE_URL", None)

    print("\n" + "█" * 70)
    print("█  三 策 略 对 比 测 试:  MA技术面 vs LLM事件 vs LLM混合")
    print(f"█  LLM后端: {LLM_BACKEND}")
    print("█" * 70)

    # ---- 获取行情 ----
    fetcher = DataFetcher()
    df_price = fetcher.fetch_daily(SYMBOL, "20180101", "20260710", "qfq")

    # ---- 准备 LLM 引擎 ----
    llm = LLMFactorEngine(
        backend=LLM_BACKEND, api_key=LLM_API_KEY,
        model=LLM_MODEL, base_url=LLM_BASE_URL,
        cache_dir=os.path.join(os.path.dirname(__file__), ".llm_cache"),
    )

    # ---- 准备事件数据 (LLM事件模式用) ----
    event_fetcher = EventFetcher()
    events_df = event_fetcher.fetch_all_events(
        symbol=SYMBOL, begin_date="2018-01-01", end_date="2026-07-10", mode="mock")
    scored_events = llm.batch_score_events(events_df)

    # ---- 准备新闻数据 (LLM过滤模式用) ----
    news_fetcher = NewsFetcher()
    news_df = news_fetcher.fetch_stock_news(SYMBOL, lookback_days=30, mode="mock")
    scored_news = llm.batch_score(news_df)
    daily_factors = llm.aggregate_to_daily(scored_news)

    # ================================================================
    #  跑三种策略
    # ================================================================
    results = {}

    print("\n" + "=" * 70)
    print("  策略 1/3: 纯技术面 (MA5×MA20)")
    print("=" * 70)
    m1, _ = run_backtest(df_price.copy(), strategy_tech, "技术面")
    results["MA技术面"] = m1

    print("\n" + "=" * 70)
    print("  策略 2/3: LLM 事件驱动 (LLM → 事件 → 买卖信号)")
    print("=" * 70)
    m2, _ = run_backtest(df_price.copy(), strategy_llm_event, "LLM事件驱动",
                         events_df=scored_events)
    results["LLM事件驱动"] = m2

    print("\n" + "=" * 70)
    print("  策略 3/3: MA + LLM 情感过滤 (MA信号 → LLM二次确认)")
    print("=" * 70)
    m3, _ = run_backtest(df_price.copy(), strategy_llm_filter, "LLM过滤",
                         daily_factors=daily_factors)
    results["MA+LLM过滤"] = m3

    # ================================================================
    #  对比报告
    # ================================================================
    print("\n\n" + "█" * 70)
    print("█  三 策 略 对 比 报 告")
    print("█" * 70)

    header = f"{'指标':<20} {'MA技术面':>12} {'LLM事件驱动':>12} {'MA+LLM过滤':>12}"
    print(header)
    print("-" * 70)

    compare_keys = [
        ("total_return",       "总收益率",       ".1%"),
        ("annual_return",      "年化收益率",      ".2%"),
        ("sharpe_ratio",       "夏普比率",       ".3f"),
        ("sortino_ratio",      "Sortino比率",   ".3f"),
        ("max_drawdown",       "最大回撤",       ".1%"),
        ("win_rate",           "日胜率",        ".1%"),
        ("total_trades",       "交易次数",       ".0f"),
        ("final_equity",       "最终权益",       ".0f"),
    ]

    best_counts = {"MA技术面": 0, "LLM事件驱动": 0, "MA+LLM过滤": 0}

    for key, label, fmt in compare_keys:
        vals = {name: m[key] for name, m in results.items()}

        # 判断最佳（总收益率和交易次数越大越好，回撤越小越好）
        if key in ("max_drawdown",):
            best_name = min(vals, key=vals.get)
        else:
            best_name = max(vals, key=vals.get)
        best_counts[best_name] += 1

        # 格式化
        display_vals = {}
        for name, v in vals.items():
            if key in ("total_return", "annual_return", "max_drawdown", "win_rate"):
                display_vals[name] = f"{v*100:+.2f}%"
            elif key == "final_equity":
                display_vals[name] = f"{v:,.0f}"
            else:
                display_vals[name] = f"{v:.3f}"

        marker = " ←最佳"
        row = f"{label:<20}"
        for name in ["MA技术面", "LLM事件驱动", "MA+LLM过滤"]:
            m = f"{marker if name == best_name else '':>6}"
            row += f" {display_vals[name]:>12}{m}"
        print(row)

    print("-" * 70)

    # ---- 综合评判 ----
    print(f"\n  最佳指标数: ")
    for name in ["MA技术面", "LLM事件驱动", "MA+LLM过滤"]:
        print(f"    {name}: {best_counts[name]}/8")
    winner = max(best_counts, key=best_counts.get)
    print(f"\n  🏆 综合最优: {winner}")

    # ---- LLM角色定位分析 ----
    print(f"\n  LLM 角色分析:")
    for name, m in results.items():
        trades = m["total_trades"]
        ret = m["annual_return"]
        print(f"    {name:<12}: 交易{trades:.0f}次, 年化{ret*100:+.2f}%")

    print("\n█" * 70)
    print(f"  对比完成。LLM后端: {LLM_BACKEND}")
    print("█" * 70 + "\n")


if __name__ == "__main__":
    main()
