"""
run_full_ic_validation.py — 全量因子 IC 验证 (P3 重算)

用 3021 只股票 + 161 个因子, 在 research 期 (2018-2022) 计算截面 Spearman IC。
输出兼容 P5 组合验证脚本的格式。

用法:
  py scripts/run_full_ic_validation.py
  py scripts/run_full_ic_validation.py --sample 500   # 抽样500只 (快速测试)

输出:
  data/ic_validation/p3_full_ic.json
"""

import os
import sys
import json
import time
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from gate import load_config
from data_cache import get_cached_symbols, load
from factor_scorer import FactorScorer
from factor_cache import FactorCache

# ── 配置 ──
IC_DIR = os.path.join(BASE_DIR, "data", "ic_validation")
OUTPUT_PATH = os.path.join(IC_DIR, "p3_full_ic.json")

config = load_config(os.path.join(BASE_DIR, "config.yaml"))
RESEARCH_START = config["data_partition"]["research"]["start"]
RESEARCH_END = config["data_partition"]["research"]["end"]
HORIZON = config["label"]["horizon_days"]  # 20

MIN_CROSS_SECTION = 30   # 每日最少股票数
MIN_VALID_DAYS = 50      # 因子最少有效天数


def log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="全量因子IC验证")
    parser.add_argument("--sample", type=int, default=None,
                        help="抽样股票数 (默认全量)")
    parser.add_argument("--horizon", type=int, default=HORIZON,
                        help=f"前瞻收益天数 (默认 {HORIZON})")
    args = parser.parse_args()

    horizon = args.horizon
    log("=" * 60)
    log(f"  全量因子 IC 验证")
    log(f"  研究期: {RESEARCH_START} ~ {RESEARCH_END}")
    log(f"  Horizon: {horizon} 天")
    log("=" * 60)

    # ── 1. 加载数据 ──
    log("加载股票列表...")
    all_syms = get_cached_symbols()
    if args.sample and args.sample < len(all_syms):
        # 分层抽样: 按前缀均匀采样
        import random
        random.seed(42)
        all_syms = random.sample(all_syms, args.sample)
    log(f"  股票池: {len(all_syms)} 只")

    log("加载行情数据...")
    t0 = time.time()
    all_data = {}
    for i, sym in enumerate(all_syms):
        df = load(sym)
        if df is not None and len(df) >= 250:
            all_data[sym] = df
        if (i + 1) % 500 == 0:
            log(f"  已加载 {i+1}/{len(all_syms)}, 有效 {len(all_data)}")
    log(f"  有效数据: {len(all_data)} 只 ({time.time()-t0:.1f}s)")

    symbols = sorted(all_data.keys())

    # ── 2. 预计算因子 ──
    log("初始化因子引擎...")
    scorer = FactorScorer.from_preset("full_auto")
    factor_names = sorted(scorer.factor_weights.keys())
    log(f"  因子数: {len(factor_names)}")

    factor_cache = FactorCache(scorer, factor_names)

    log("预计算因子 (这可能需要几分钟)...")
    t0 = time.time()
    batch_size = 200
    for i in range(0, len(symbols), batch_size):
        batch = {s: all_data[s] for s in symbols[i:i+batch_size]}
        factor_cache.precompute(batch)
        elapsed = time.time() - t0
        done = min(i + batch_size, len(symbols))
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(symbols) - done) / rate if rate > 0 else 0
        log(f"  预计算: {done}/{len(symbols)} "
            f"({elapsed:.0f}s, ~{eta:.0f}s remaining)")
    log(f"  预计算完成: {time.time()-t0:.1f}s")

    # ── 3. 构建面板 ──
    log("构建因子面板...")
    rs = pd.Timestamp(RESEARCH_START)
    re_ = pd.Timestamp(RESEARCH_END)

    # 收集所有研究期内的交易日
    all_dates = set()
    for sym in symbols[:100]:  # 用前100只确定交易日
        df = all_data[sym]
        mask = (df["date"] >= rs) & (df["date"] <= re_)
        all_dates.update(df.loc[mask, "date"].tolist())
    trade_dates = sorted(all_dates)
    log(f"  研究期交易日: {len(trade_dates)} 天")

    # 构建前瞻收益: ret[sym][date] = close[date+horizon] / close[date] - 1
    log("构建前瞻收益...")
    fwd_ret = {}  # {sym: Series(date -> ret)}
    for sym in symbols:
        df = all_data[sym].set_index("date").sort_index()
        close = df["close"]
        ret = close.shift(-horizon) / close - 1
        # 只保留研究期
        ret = ret[(ret.index >= rs) & (ret.index <= re_)]
        fwd_ret[sym] = ret.dropna()

    # ── 4. 逐日截面 IC 计算 ──
    log(f"计算截面 IC (horizon={horizon})...")
    t0 = time.time()

    # 每5天采样一次 (减少计算量, 对ICIR影响极小)
    sample_interval = 5
    sample_dates = trade_dates[::sample_interval]
    log(f"  采样日期: {len(sample_dates)} 天 (每{sample_interval}天)")

    # 结果存储: {factor_name: [ic_values]}
    ic_series = {name: [] for name in factor_names}
    valid_dates = {name: 0 for name in factor_names}

    for di, today in enumerate(sample_dates):
        # 收集当日所有股票的因子值和前瞻收益
        day_factors = {name: {} for name in factor_names}
        day_returns = {}

        for sym in symbols:
            # 前瞻收益
            if today not in fwd_ret[sym].index:
                continue
            ret_val = fwd_ret[sym][today]
            if np.isnan(ret_val):
                continue
            day_returns[sym] = ret_val

            # 因子值
            feats = factor_cache.get(sym, today)
            if feats is None:
                continue
            for name in factor_names:
                val = feats.get(name, np.nan)
                if not np.isnan(val):
                    day_factors[name][sym] = val

        # 对每个因子计算截面 Spearman IC
        n_cross = len(day_returns)
        if n_cross < MIN_CROSS_SECTION:
            continue

        ret_syms = set(day_returns.keys())
        for name in factor_names:
            common = ret_syms & set(day_factors[name].keys())
            if len(common) < MIN_CROSS_SECTION:
                continue

            common = sorted(common)
            f_vals = np.array([day_factors[name][s] for s in common])
            r_vals = np.array([day_returns[s] for s in common])

            # 去除常量
            if np.std(f_vals) < 1e-12:
                continue

            corr, _ = spearmanr(f_vals, r_vals)
            if not np.isnan(corr):
                ic_series[name].append(corr)
                valid_dates[name] += 1

        if (di + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (di + 1) * (len(sample_dates) - di - 1)
            log(f"  IC进度: {di+1}/{len(sample_dates)} "
                f"({elapsed:.0f}s, ~{eta:.0f}s remaining)")

    log(f"  IC计算完成: {time.time()-t0:.1f}s")

    # ── 5. 汇总统计 ──
    log("汇总 IC 统计...")
    results = []
    for name in factor_names:
        ics = ic_series[name]
        n_days = len(ics)
        if n_days < MIN_VALID_DAYS:
            continue

        ic_arr = np.array(ics)
        ic_mean = float(np.mean(ic_arr))
        ic_std = float(np.std(ic_arr))
        icir = ic_mean / ic_std if ic_std > 1e-9 else 0.0
        pos_ratio = float(np.mean(ic_arr > 0))
        abs_ic_mean = float(np.mean(np.abs(ic_arr)))

        results.append({
            "factor": name,
            "horizon": horizon,
            "ic_mean": round(ic_mean, 6),
            "ic_std": round(ic_std, 6),
            "icir": round(icir, 4),
            "abs_ic_mean": round(abs_ic_mean, 6),
            "pos_ratio": round(pos_ratio, 4),
            "n_days": n_days,
        })

    # 按 |ICIR| 降序
    results.sort(key=lambda x: -abs(x["icir"]))

    # ── 6. 输出 ──
    os.makedirs(IC_DIR, exist_ok=True)
    output = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "全量因子IC验证 (161因子 × 3000+股票)",
            "research_period": f"{RESEARCH_START} ~ {RESEARCH_END}",
            "horizon": horizon,
            "n_stocks": len(symbols),
            "n_factors": len(factor_names),
            "n_sample_dates": len(sample_dates),
            "sample_interval": sample_interval,
            "min_cross_section": MIN_CROSS_SECTION,
        },
        "results": results,
        "summary": {
            "total_factors": len(results),
            "strong_factors": sum(1 for r in results if abs(r["icir"]) > 0.3),
            "moderate_factors": sum(1 for r in results if 0.2 < abs(r["icir"]) <= 0.3),
            "top5": [{"factor": r["factor"], "icir": r["icir"]} for r in results[:5]],
        },
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log(f"\n{'='*60}")
    log(f"  结果已保存: {OUTPUT_PATH}")
    log(f"  有效因子: {len(results)} / {len(factor_names)}")
    log(f"  |ICIR| > 0.3: {output['summary']['strong_factors']} 个")
    log(f"  Top-5:")
    for r in results[:5]:
        log(f"    {r['factor']:<25} ICIR={r['icir']:+.4f} "
            f"IC={r['ic_mean']:+.5f} n={r['n_days']}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
