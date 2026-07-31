"""
IC 衰减监控 — 每周重算因子IC, 检测alpha衰减信号

用法:
  python scripts/run_ic_monitor.py                      # 单次运行
  python scripts/run_ic_monitor.py --lookback 60        # 自定义回溯天数
  python scripts/run_ic_monitor.py --schedule           # 注册为调度器任务

输出: data/ic_monitor.json (历史记录, 每行一条)
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data_cache import get_cached_symbols, load_all
from factor_scorer import FactorScorer
from factor_cache import FactorCache
from data.calendar import get_trading_days

MONITOR_FILE = os.path.join(BASE_DIR, "data", "ic_monitor.json")
BASELINE_IC_FILE = os.path.join(BASE_DIR, "data", "ic_results.json")

# 默认参数
DEFAULT_LOOKBACK = 60     # 最近60个交易日
DEFAULT_LABEL_HORIZON = 20  # 与 config 一致
DECAY_THRESHOLD = 0.50    # IC低于基线50% → 衰减告警
DECAY_WEEKS = 2           # 连续衰减周数 → 告警


def load_baseline_ic() -> Dict[str, float]:
    """加载研究期的基线IC (从 ic_results.json)。"""
    if not os.path.exists(BASELINE_IC_FILE):
        print("⚠️ 基线IC文件不存在, 请先运行 scripts/run_ic_analysis.py")
        return {}

    with open(BASELINE_IC_FILE, encoding="utf-8") as f:
        results = json.load(f)

    baseline = {}
    for r in results:
        baseline[r["factor"]] = r["ic_mean"]
    return baseline


def compute_current_ic(lookback_days: int = DEFAULT_LOOKBACK,
                       label_horizon: int = DEFAULT_LABEL_HORIZON) -> List[dict]:
    """
    计算最近 N 个交易日的因子IC。

    Returns:
      [{factor, ic_mean, ic_std, icir, pos_ratio, n_days}, ...]
    """
    syms = get_cached_symbols()
    all_data = load_all(syms)
    all_data = {s: df for s, df in all_data.items() if len(df) >= 100}
    print(f"  有效股票: {len(all_data)}")

    # 预计算因子
    scorer = FactorScorer.from_preset("ic_auto")
    factor_names = sorted(scorer.factor_weights.keys())
    cache = FactorCache(scorer, factor_names)
    cache.precompute(all_data)
    print(f"  因子: {len(factor_names)} 个")

    # 确定分析日期范围
    trading_days = get_trading_days("2018-01-01", datetime.now().strftime("%Y-%m-%d"))
    if not trading_days:
        all_dates = sorted(set().union(*[set(df["date"].tolist())
                                         for df in all_data.values()]))
        trading_days = all_dates

    # 取最近 N 个交易日
    if len(trading_days) > lookback_days:
        recent_days = trading_days[-lookback_days:]
    else:
        recent_days = trading_days

    print(f"  分析日期: {min(recent_days).date()} ~ {max(recent_days).date()} "
          f"({len(recent_days)}个交易日)")

    # 每天采样
    ic_data = {f: {"values": [], "daily_count": 0} for f in factor_names}
    for di, today in enumerate(recent_days[::3]):  # 每3天采样加速
        rets = {}
        for sym in all_data:
            df = all_data[sym]
            mask = df["date"] == today
            if not mask.any():
                continue
            ip = df.index.get_loc(df.index[mask][0])
            if ip + label_horizon >= len(df):
                continue
            fwd = df.iloc[ip + label_horizon]["close"] / df.iloc[ip]["close"] - 1
            rets[sym] = fwd
        if len(rets) < 10:
            continue

        fvals_all = {f: [] for f in factor_names}
        valid_syms = []
        for sym in rets:
            feats = cache.get_features(sym, today)
            if feats is None:
                continue
            valid_syms.append(sym)
            for fi, fname in enumerate(factor_names):
                fvals_all[fname].append(feats[fi])
        if len(valid_syms) < 10:
            continue

        ret_arr = np.array([rets[s] for s in valid_syms])
        for fname in factor_names:
            fv = np.array(fvals_all[fname])
            if np.std(fv) < 1e-9:
                continue
            try:
                ic, _ = spearmanr(fv, ret_arr)
                ic_data[fname]["values"].append(ic)
                ic_data[fname]["daily_count"] += 1
            except Exception:
                pass

    # 汇总
    results = []
    for fname in factor_names:
        ics = ic_data[fname]["values"]
        if len(ics) < 10:
            continue
        mean_ic = np.mean(ics)
        std_ic = np.std(ics)
        icir = mean_ic / std_ic if std_ic > 0 else 0
        results.append({
            "factor": fname,
            "n_days": len(ics),
            "ic_mean": round(float(mean_ic), 6),
            "ic_std": round(float(std_ic), 6),
            "icir": round(float(icir), 4),
            "pos_ratio": round(float(sum(1 for x in ics if x > 0) / len(ics)), 4),
        })

    results.sort(key=lambda x: -abs(x["icir"]))
    return results


def check_decay(current_ic: List[dict],
                baseline_ic: Dict[str, float] = None) -> dict:
    """
    对比当前IC与基线IC, 检测衰减信号。

    Returns:
      {"decay_warning": bool, "decayed_factors": [...], "weeks_decaying": int}
    """
    if baseline_ic is None:
        baseline_ic = load_baseline_ic()

    if not baseline_ic:
        return {"decay_warning": False, "decayed_factors": [], "weeks_decaying": 0}

    decayed = []
    for r in current_ic:
        fname = r["factor"]
        if fname in baseline_ic:
            base_ic = baseline_ic[fname]
            if base_ic != 0:
                ratio = abs(r["ic_mean"]) / abs(base_ic)
                if ratio < DECAY_THRESHOLD:
                    decayed.append({
                        "factor": fname,
                        "current_ic": r["ic_mean"],
                        "baseline_ic": base_ic,
                        "ratio": round(ratio, 4),
                    })

    # 加载历史监控记录, 检查连续衰减周数
    weeks_decaying = 0
    if os.path.exists(MONITOR_FILE):
        try:
            with open(MONITOR_FILE, encoding="utf-8") as f:
                history = json.load(f)
            if history and history[-1].get("decay_warning"):
                weeks_decaying = history[-1].get("weeks_decaying", 0) + 1
        except Exception:
            pass
    if decayed and weeks_decaying == 0:
        weeks_decaying = 1

    decay_warning = weeks_decaying >= DECAY_WEEKS

    return {
        "decay_warning": decay_warning,
        "decayed_factors": decayed,
        "weeks_decaying": weeks_decaying,
    }


def run_ic_monitor(lookback: int = DEFAULT_LOOKBACK) -> dict:
    """运行一次IC监控并保存结果。"""
    print(f"\n{'='*50}")
    print(f"  IC衰减监控 (回溯 {lookback} 天)")
    print(f"{'='*50}")

    # 加载基线
    baseline_ic = load_baseline_ic()
    print(f"  基线因子数: {len(baseline_ic)}")

    # 计算当前IC
    current_ic = compute_current_ic(lookback_days=lookback)

    # 检测衰减
    decay_info = check_decay(current_ic, baseline_ic)

    # 构建输出
    entry = {
        "timestamp": datetime.now().isoformat(),
        "lookback_days": lookback,
        "top_factors": current_ic[:10],
        "decay_warning": decay_info["decay_warning"],
        "decayed_factors": decay_info["decayed_factors"],
        "weeks_decaying": decay_info["weeks_decaying"],
    }

    # 保存
    # 读取现有历史
    history = []
    if os.path.exists(MONITOR_FILE):
        try:
            with open(MONITOR_FILE, encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass

    history.append(entry)
    # 只保留最近 52 周
    history = history[-52:]

    os.makedirs(os.path.dirname(MONITOR_FILE), exist_ok=True)
    with open(MONITOR_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # ── 输出报告 ──
    print(f"\n  当前 Top-10 IC (绝对ICIR):")
    for r in current_ic[:10]:
        sgn = "+" if r["ic_mean"] > 0 else ""
        print(f"    {r['factor']:<25} IC={sgn}{r['ic_mean']:+.4f}  "
              f"ICIR={sgn}{r['icir']:+.2f}  pos={r['pos_ratio']:.0%}")

    if decay_info["decayed_factors"]:
        print(f"\n  ⚠️ 检测到衰减因子 ({len(decay_info['decayed_factors'])}个):")
        for d in decay_info["decayed_factors"][:10]:
            print(f"    {d['factor']}: {d['current_ic']:+.4f} "
                  f"(基线{d['baseline_ic']:+.4f}, 降至{d['ratio']:.0%})")

    if decay_info["decay_warning"]:
        print(f"\n  🚨 IC衰减告警! 连续 {decay_info['weeks_decaying']} 周衰减")
        # 触发告警
        try:
            from alerter import AlertManager
            am = AlertManager()
            am.send("ic_decay", {
                "factor": ", ".join(d["factor"] for d in decay_info["decayed_factors"][:5]),
                "current_ic": decay_info["decayed_factors"][0]["current_ic"],
                "baseline_ic": decay_info["decayed_factors"][0]["baseline_ic"],
                "weeks": decay_info["weeks_decaying"],
            })
        except Exception:
            pass

    print(f"\n  输出: {MONITOR_FILE}")
    return entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IC衰减监控")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK,
                       help=f"回溯交易日数 (默认{DEFAULT_LOOKBACK})")
    parser.add_argument("--schedule", action="store_true",
                       help="作为调度任务运行 (与scheduler.py配合)")
    args = parser.parse_args()

    run_ic_monitor(lookback=args.lookback)
