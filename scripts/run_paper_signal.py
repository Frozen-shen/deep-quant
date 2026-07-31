"""
Paper Trading 日频信号生成器

每个交易日收盘后运行:
  python scripts/run_paper_signal.py

输出: data/paper_signals.jsonl (追加写, 带时间戳, 不可修改)
"""
import sys, os, json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

SIGNAL_LOG = os.path.join(BASE_DIR, "data", "paper_signals.jsonl")


def generate_signal(date_str: str = None):
    """生成当天交易信号。date_str='2026-07-31' 或 None=今天"""

    if date_str is None:
        today = pd.Timestamp.now().normalize()
    else:
        today = pd.Timestamp(date_str)

    print(f"[PaperTrading] 生成 {today.date()} 信号...")

    from model.pipeline import load_config
    from model.pipeline import QuantPipeline

    config = load_config()

    # ── 1. 加载数据 ──
    pipeline = QuantPipeline(config, mode="blind")  # blind mode 使用全量数据
    pipeline._load_universe()
    pipeline._load_data()
    pipeline._precompute_factors()

    # ── 2. 用最近N天训练模型 ──
    train_end = today - timedelta(days=1)  # 训练到昨天
    train_start = train_end - pd.DateOffset(months=config["rolling"]["train_months"])

    # 构建训练窗口
    w = {
        "train_start": train_start,
        "train_end": train_end,
        "test_start": today,
        "test_end": today,  # 只预测今天
    }

    # 生成特征和标签
    xt, yt, gt = [], [], []
    all_dates = sorted(set().union(*[set(df["date"].tolist()) for df in pipeline._all_data.values()]))
    train_days = [d for d in all_dates if train_start <= d <= train_end - timedelta(days=config["rolling"]["embargo_days"])]

    for d in train_days[::2]:  # 每2天采样
        fn, labels, syms = pipeline._build_cs(d)
        if fn is not None:
            xt.extend(fn.tolist())
            yt.extend(labels.tolist())
            gt.extend([str(d)] * len(labels))

    if len(xt) < 100:
        print("  ❌ 训练数据不足")
        return None

    # ── 3. 训练模型 ──
    model = pipeline._train_model(np.array(xt), np.array(yt, dtype=int), gt, train_end)

    # ── 4. 生成今日信号 ──
    from portfolio_ranker import PortfolioRanker
    from trading_rules import TradingRules
    from model.engine import SimpleBacktest

    # dummy backtest engine for position tracking
    bt = SimpleBacktest(initial_capital=config["execution"]["initial_capital"],
                       top_k=pipeline.top_k, lot_size=pipeline.lot_size)
    ranker = PortfolioRanker(top_k=pipeline.top_k, n_drop=pipeline.n_drop,
                             hold_thresh=pipeline.hold_thresh,
                             sell_rank_buffer=pipeline.sell_rank_buffer,
                             buy_confirm_days=pipeline.buy_confirm_days,
                             cost_threshold=pipeline.cost_threshold)
    rules = TradingRules()

    signal = pipeline._generate_signal(model, today, rules, bt, ranker)

    # ── 5. 落盘 (追加写, 不可修改) ──
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "signal_date": str(today.date()),
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12],
        "buy": signal.get("buy", []) if signal else [],
        "sell": signal.get("sell", []) if signal else [],
        "hold": signal.get("hold", []) if signal else [],
        "top_k_scores": dict(list((signal or {}).get("top_k_scores", {}).items())[:10]) if signal else {},
    }

    with open(SIGNAL_LOG, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  信号已写入 {SIGNAL_LOG}")
    if signal:
        print(f"  买入: {entry['buy']}")
        print(f"  卖出: {entry['sell']}")
    else:
        print("  ⚠️ 今日无信号 (可能数据不足或市场休市)")

    return entry


if __name__ == "__main__":
    import hashlib
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    generate_signal(date_arg)
