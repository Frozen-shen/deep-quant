"""
scripts/active/run_factor_causality_test.py — 因子因果性检验 (Expanding Window)

检验因子IC是否随时间稳定 (非数据挖掘产物):
  - 用 2018-2019 训练窗口计算 IC
  - 用 2020-2022 验证窗口检验 IC 方向一致性
  - 如果 IC 方向一致且衰减 < 50%, 认为因子有因果基础

用法:
  py scripts/active/run_factor_causality_test.py

输出:
  data/ic_validation/factor_causality.json
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from data_cache import get_cached_symbols, load
from factor_scorer import FactorScorer
from factor_cache import FactorCache

IC_DIR = os.path.join(BASE_DIR, "data", "ic_validation")
OUTPUT_PATH = os.path.join(IC_DIR, "factor_causality.json")

# 时间窗口
TRAIN_START = "2018-01-01"
TRAIN_END = "2019-12-31"
VALID_START = "2020-01-01"
VALID_END = "2022-12-31"
HORIZON = 20  # 前瞻天数
MIN_CROSS_SECTION = 30


def compute_period_ic(factor_cache, all_data, factor_names, start, end):
    """计算指定期间的日均截面 IC。"""
    rs, re_ = pd.Timestamp(start), pd.Timestamp(end)

    # 收集交易日
    all_dates = set()
    for df in list(all_data.values())[:200]:
        mask = (df["date"] >= rs) & (df["date"] <= re_)
        all_dates.update(df.loc[mask, "date"].tolist())
    dates = sorted(all_dates)

    ic_by_factor = {name: [] for name in factor_names}

    for d in dates:
        d_ts = pd.Timestamp(d)
        # 获取前瞻收益
        fwd_returns = {}
        for sym, df in all_data.items():
            past = df[df["date"] <= d_ts]
            if len(past) == 0:
                continue
            future = df[df["date"] > d_ts].head(HORIZON)
            if len(future) >= HORIZON:
                fwd_ret = future["close"].iloc[-1] / past["close"].iloc[-1] - 1
                fwd_returns[sym] = fwd_ret

        if len(fwd_returns) < MIN_CROSS_SECTION:
            continue

        syms = sorted(fwd_returns.keys())
        fwd_arr = np.array([fwd_returns[s] for s in syms])

        for name in factor_names:
            vals = []
            valid_mask = []
            for s in syms:
                feats = factor_cache.get(s, d_ts)
                if feats and name in feats and not np.isnan(feats[name]):
                    vals.append(feats[name])
                    valid_mask.append(True)
                else:
                    vals.append(0)
                    valid_mask.append(False)

            valid_mask = np.array(valid_mask)
            if valid_mask.sum() < MIN_CROSS_SECTION:
                continue

            vals_arr = np.array(vals)
            ic, _ = spearmanr(vals_arr[valid_mask], fwd_arr[valid_mask])
            if not np.isnan(ic):
                ic_by_factor[name].append(ic)

    # 汇总
    result = {}
    for name, ics in ic_by_factor.items():
        if len(ics) >= 20:
            result[name] = {
                "mean_ic": float(np.mean(ics)),
                "std_ic": float(np.std(ics)),
                "icir": float(np.mean(ics) / np.std(ics)) if np.std(ics) > 0 else 0,
                "n_days": len(ics),
                "positive_ratio": float(np.mean(np.array(ics) > 0)),
            }
    return result


def main():
    print("=" * 60)
    print("  因子因果性检验 (Expanding Window Invariance)")
    print(f"  训练窗口: {TRAIN_START} ~ {TRAIN_END}")
    print(f"  验证窗口: {VALID_START} ~ {VALID_END}")
    print("=" * 60)

    # 加载数据
    print("  加载数据...")
    syms = get_cached_symbols()
    all_data = {}
    for sym in syms:
        df = load(sym)
        if df is not None and len(df) >= 500:
            all_data[sym] = df
    print(f"  有效: {len(all_data)} 只")

    # 预计算因子
    print("  预计算因子...")
    scorer = FactorScorer.from_preset("full_auto")
    factor_names = sorted(scorer.factor_weights.keys())
    factor_cache = FactorCache(scorer, factor_names)

    symbols = sorted(all_data.keys())
    for i in range(0, len(symbols), 200):
        batch = {s: all_data[s] for s in symbols[i:i + 200]}
        factor_cache.precompute(batch)
        if (i + 200) % 1000 == 0:
            print(f"    {min(i+200, len(symbols))}/{len(symbols)}")

    # 计算两个窗口的 IC
    print(f"\n  计算训练期 IC ({TRAIN_START}~{TRAIN_END})...")
    train_ic = compute_period_ic(factor_cache, all_data, factor_names,
                                  TRAIN_START, TRAIN_END)
    print(f"    有效因子: {len(train_ic)}")

    print(f"  计算验证期 IC ({VALID_START}~{VALID_END})...")
    valid_ic = compute_period_ic(factor_cache, all_data, factor_names,
                                  VALID_START, VALID_END)
    print(f"    有效因子: {len(valid_ic)}")

    # 对比
    print(f"\n  {'因子':<20} {'训练IC':>8} {'验证IC':>8} {'方向一致':>8} {'衰减':>8} {'判定':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    results = {}
    n_pass = 0
    n_total = 0

    for name in sorted(train_ic.keys()):
        if name not in valid_ic:
            continue
        n_total += 1

        t_icir = train_ic[name]["icir"]
        v_icir = valid_ic[name]["icir"]
        t_mean = train_ic[name]["mean_ic"]
        v_mean = valid_ic[name]["mean_ic"]

        # 方向一致性
        same_direction = (t_mean > 0 and v_mean > 0) or (t_mean < 0 and v_mean < 0)

        # 衰减
        if abs(t_icir) > 0.01:
            decay = 1 - abs(v_icir) / abs(t_icir)
        else:
            decay = 0

        # 判定: 方向一致 + 衰减 < 50%
        passed = same_direction and decay < 0.5
        if passed:
            n_pass += 1

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:<20} {t_icir:>+7.3f} {v_icir:>+7.3f} "
              f"{'YES' if same_direction else 'NO':>8} {decay:>7.0%} {status:>8}")

        results[name] = {
            "train_icir": round(t_icir, 4),
            "valid_icir": round(v_icir, 4),
            "train_mean_ic": round(t_mean, 5),
            "valid_mean_ic": round(v_mean, 5),
            "same_direction": same_direction,
            "decay_pct": round(decay * 100, 1),
            "passed": passed,
        }

    print(f"\n  通过率: {n_pass}/{n_total} ({n_pass/max(n_total,1)*100:.0f}%)")

    # 保存
    output = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "因子因果性检验 (Expanding Window Invariance)",
            "train_period": f"{TRAIN_START} ~ {TRAIN_END}",
            "valid_period": f"{VALID_START} ~ {VALID_END}",
            "horizon_days": HORIZON,
        },
        "summary": {
            "n_total": n_total,
            "n_pass": n_pass,
            "pass_rate": round(n_pass / max(n_total, 1), 3),
        },
        "factors": results,
    }

    os.makedirs(IC_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  结果: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()