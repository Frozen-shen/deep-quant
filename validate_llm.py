"""
LLM 事件策略真实验证 — 用真实公司公告测试 DeepSeek 预测准确率

用法:
    python validate_llm.py                           # mock模式快速验证
    LLM_BACKEND=openai python validate_llm.py        # DeepSeek真实评分

验证方法:
    1. 拉取真实公告 (stock_individual_notice_report)
    2. LLM 评分 (action + confidence + horizon)
    3. 查公告日后 horizon_days 的实际股价变化
    4. 统计: LLM方向判断准确率 vs 随机基准
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data_fetcher import DataFetcher
from event_fetcher import EventFetcher
from llm_factor import LLMFactorEngine


def validate_llm(symbol: str = "600519", market: str = "a",
                 start: str = "2018-01-01", end: str = "2026-07-10",
                 max_events: int = 100):
    """
    LLM 事件策略验证主函数。

    返回: dict with accuracy, confusion_matrix, per_event_details
    """
    print(f"\n{'='*60}")
    print(f"  LLM 事件策略验证: {symbol}")
    print(f"{'='*60}")

    llm_backend = os.environ.get("LLM_BACKEND", "mock")
    print(f"  LLM后端: {llm_backend}")

    # ---- 1. 拉取真实公告 ----
    print(f"\n  [1/4] 拉取真实公告...")
    fetcher = EventFetcher()
    events = fetcher.fetch_notices(symbol=symbol, begin_date=start, end_date=end,
                                   mode="live" if llm_backend != "mock" else "hybrid")
    actual_count = len(events)
    if actual_count == 0:
        print("  无公告数据")
        return None

    # 截取
    if len(events) > max_events:
        events = events.head(max_events)
    print(f"  获取 {actual_count} 条公告, 使用 {len(events)} 条")

    # ---- 2. LLM 评分 ----
    print(f"\n  [2/4] LLM 评分...")
    llm = LLMFactorEngine(
        backend=llm_backend,
        api_key=os.environ.get("OPENAI_API_KEY"),
        model=os.environ.get("LLM_MODEL"),
        base_url=os.environ.get("LLM_BASE_URL"),
        cache_dir=os.path.join(os.path.dirname(__file__), ".llm_cache"),
    )

    # 适配 event format → LLM 需要的 columns
    if "notice_date" in events.columns:
        events["event_date"] = events["notice_date"]

    scored = llm.batch_score_events(events)

    # ---- 3. 价格验证 ----
    print(f"\n  [3/4] 拉取价格数据 & 计算实际收益...")
    fetcher_price = DataFetcher()
    df_price = fetcher_price.fetch(symbol, "20180101", "20260710", "qfq", market=market)
    df_price["date"] = pd.to_datetime(df_price["date"])

    results = []
    for _, row in scored.iterrows():
        action = row.get("llm_action", "hold")
        confidence = row.get("llm_confidence", 0)
        horizon = int(row.get("llm_horizon_days", 5))
        event_date = pd.to_datetime(row.get("event_date"))

        # 只统计有信心的 actionable 事件
        if action == "hold" or confidence < 0.5:
            continue

        # 找事件日后的价格
        # 事件日: 公告发布日
        # 买入日: 事件日后第一个交易日
        # 卖出日: 买入日后 horizon 个交易日
        event_mask = df_price["date"] >= event_date
        if not event_mask.any():
            continue

        entry_idx = df_price[event_mask].index[0]
        if entry_idx + horizon >= len(df_price):
            continue

        entry_price = df_price.loc[entry_idx, "open"]
        exit_idx = entry_idx + horizon
        exit_price = df_price.loc[exit_idx, "close"]

        actual_return = (exit_price / entry_price - 1) * 100
        direction = "up" if actual_return > 0 else "down"

        # 预测正确性
        if action == "buy" and actual_return > 0:
            correct = True
        elif action == "sell" and actual_return < 0:
            correct = True
        else:
            correct = False

        results.append({
            "event_date": event_date,
            "action": action,
            "confidence": confidence,
            "horizon": horizon,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "actual_return": actual_return,
            "direction": direction,
            "correct": correct,
            "title": row.get("title", "")[:50],
        })

    df_results = pd.DataFrame(results)

    # ---- 4. 统计 ----
    print(f"\n  [4/4] 统计结果...")
    n = len(df_results)

    if n == 0:
        print("  无可评估的事件 (所有事件置信度<0.5)")
        return None

    accuracy = df_results["correct"].mean() * 100

    # 按 action 分组
    buy_events = df_results[df_results["action"] == "buy"]
    sell_events = df_results[df_results["action"] == "sell"]

    buy_acc = buy_events["correct"].mean() * 100 if len(buy_events) > 0 else 0
    sell_acc = sell_events["correct"].mean() * 100 if len(sell_events) > 0 else 0

    # 超额收益
    avg_return = df_results["actual_return"].mean()
    avg_buy_return = buy_events["actual_return"].mean() if len(buy_events) > 0 else 0

    # 基准: 市场同期平均收益
    benchmark_return = (df_price["close"].iloc[-1] / df_price["close"].iloc[0] - 1) * 100

    # 输出
    print(f"\n  {'='*50}")
    print(f"  LLM 事件策略验证报告")
    print(f"  {'='*50}")
    print(f"  标的: {symbol}  |  LLM后端: {llm_backend}")
    print(f"  总事件: {actual_count} 条, 可评估: {n} 条")
    print(f"  整体准确率: {accuracy:.1f}%")
    print(f"  BUY准确率: {buy_acc:.1f}% ({len(buy_events)}次)")
    print(f"  SELL准确率: {sell_acc:.1f}% ({len(sell_events)}次)")
    print(f"  平均持仓收益: {avg_return:+.2f}%")
    print(f"  BUY平均收益: {avg_buy_return:+.2f}%")
    print(f"  市场基准收益: {benchmark_return:+.2f}%")
    print(f"  {'='*50}")

    # 判断
    if accuracy > 55:
        print(f"  ✅ LLM预测显著优于随机 (50%), 准确率={accuracy:.1f}%")
    elif accuracy > 50:
        print(f"  ⚠️ LLM预测略优于随机")
    else:
        print(f"  ❌ LLM预测未优于随机")

    return {
        "accuracy": accuracy,
        "buy_accuracy": buy_acc,
        "sell_accuracy": sell_acc,
        "n_events": n,
        "avg_return": avg_return,
        "details": df_results,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="600519")
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--market", default="a")
    args = parser.parse_args()

    validate_llm(
        symbol=args.symbol,
        market=args.market,
        max_events=args.max_events,
    )
