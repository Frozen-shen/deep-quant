"""
盈利动量因子 IC 验证 — 评估5个因子的预测能力

计算 Spearman Rank IC @ 5/10/20天, 分前后半段检验衰减。
研究期: 2018-2024, 每5个交易日采样一次。

输出:
  data/ic_validation/p1_earnings_ic.json

用法:
  python scripts/validate_earnings_factors.py
  python scripts/validate_earnings_factors.py --start 2020-01-01 --end 2023-12-31
  python scripts/validate_earnings_factors.py --horizons 5 10 20
"""

import os
import sys
import json
import time
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from factors.earnings_momentum import EarningsMomentum, EM_FACTOR_NAMES

# 数据目录
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
DATA_STORE_DIR = os.path.join(BASE_DIR, "data_store")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "ic_validation")

# 默认参数
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_HORIZONS = [5, 10, 20]
SAMPLE_FREQ = 5  # 每5个交易日采样一次


def load_price_data(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """
    加载价格数据。优先从 data_store (全量), 回退到 data_cache。
    """
    result = {}

    for sym in symbols:
        # 优先 data_store
        path = os.path.join(DATA_STORE_DIR, f"{sym}.parquet")
        if not os.path.exists(path):
            path = os.path.join(DATA_CACHE_DIR, f"{sym}.parquet")
        if not os.path.exists(path):
            continue

        try:
            df = pd.read_parquet(path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            if len(df) > 100:
                result[sym] = df
        except Exception:
            continue

    return result


def compute_forward_returns(all_data: Dict[str, pd.DataFrame],
                            trade_date: pd.Timestamp,
                            horizon: int) -> Dict[str, float]:
    """
    计算从 trade_date 起 horizon 天后的前瞻收益。
    """
    rets = {}
    for sym, df in all_data.items():
        mask = df['date'] == trade_date
        if not mask.any():
            continue
        idx = df.index.get_loc(df.index[mask][0])
        if idx + horizon >= len(df):
            continue
        fwd_close = df.iloc[idx + horizon]['close']
        cur_close = df.iloc[idx]['close']
        if cur_close > 0:
            rets[sym] = fwd_close / cur_close - 1
    return rets


def compute_daily_ic(factor_values: pd.DataFrame,
                     forward_rets: Dict[str, float],
                     factor_col: str) -> Optional[float]:
    """
    计算单日单因子的 Spearman Rank IC。
    """
    # 对齐: 因子值和前瞻收益都有效的股票
    common_syms = []
    f_vals = []
    r_vals = []

    for _, row in factor_values.iterrows():
        sym = row['symbol']
        if sym in forward_rets:
            fv = row.get(factor_col)
            if pd.notna(fv):
                common_syms.append(sym)
                f_vals.append(fv)
                r_vals.append(forward_rets[sym])

    if len(common_syms) < 10:
        return None

    f_arr = np.array(f_vals)
    r_arr = np.array(r_vals)

    # 检查方差
    if np.std(f_arr) < 1e-9 or np.std(r_arr) < 1e-9:
        return None

    try:
        ic, _ = spearmanr(f_arr, r_arr)
        if np.isnan(ic):
            return None
        return float(ic)
    except Exception:
        return None


def run_validation(start: str, end: str, horizons: List[int],
                   sample_freq: int = SAMPLE_FREQ):
    """
    主验证逻辑。
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    print("=" * 60, flush=True)
    print("  盈利动量因子 IC 验证", flush=True)
    print("=" * 60, flush=True)
    print(f"  研究期: {start} ~ {end}", flush=True)
    print(f"  预测窗口: {horizons}", flush=True)
    print(f"  采样频率: 每{sample_freq}天", flush=True)

    # 加载价格数据
    em = EarningsMomentum()
    fund_symbols = em._get_cached_symbols()
    print(f"\n  基本面缓存: {len(fund_symbols)} 只", flush=True)

    all_data = load_price_data(fund_symbols)
    print(f"  价格数据有效: {len(all_data)} 只", flush=True)

    if len(all_data) < 30:
        print("  错误: 有效股票不足30只, 无法进行IC分析", flush=True)
        sys.exit(1)

    # 收集研究期交易日
    all_dates = set()
    for df in all_data.values():
        all_dates.update(df['date'].tolist())
    research_days = sorted([d for d in all_dates if start_dt <= d <= end_dt])
    research_days = research_days[::sample_freq]
    print(f"  研究日: {len(research_days)} 天", flush=True)

    # 分前后半段
    mid_date = start_dt + (end_dt - start_dt) / 2
    print(f"  分割点: {mid_date.date()}", flush=True)

    # ── 逐日计算IC ──
    # 结构: {factor: {horizon: {'front': [ics], 'back': [ics]}}}
    ic_results = {
        f: {h: {'front': [], 'back': []} for h in horizons}
        for f in EM_FACTOR_NAMES
    }

    t0 = time.time()
    valid_days = 0

    for di, today in enumerate(research_days):
        # 计算因子值
        factor_df = em.compute_all(today.strftime('%Y-%m-%d'), list(all_data.keys()))

        if len(factor_df) == 0:
            continue

        day_has_ic = False

        for h in horizons:
            # 前瞻收益
            fwd_rets = compute_forward_returns(all_data, today, h)
            if len(fwd_rets) < 10:
                continue

            # 分前后半段
            period = 'front' if today < mid_date else 'back'

            for f in EM_FACTOR_NAMES:
                ic = compute_daily_ic(factor_df, fwd_rets, f)
                if ic is not None:
                    ic_results[f][h][period].append(ic)
                    day_has_ic = True

        if day_has_ic:
            valid_days += 1

        if (di + 1) % 50 == 0:
            elapsed = time.time() - t0
            pct = (di + 1) / len(research_days) * 100
            print(f"  [{di+1}/{len(research_days)}] {pct:.0f}% "
                  f"[{elapsed:.0f}s, 有效{valid_days}天]", flush=True)

    # ── 汇总统计 ──
    print(f"\n{'='*60}", flush=True)
    print(f"  IC 验证完成: {valid_days} 有效天, "
          f"{time.time()-t0:.0f}s", flush=True)
    print(f"{'='*60}\n", flush=True)

    summary = []

    print(f"{'因子':<20} {'窗口':>4} {'期间':>6} "
          f"{'IC均值':>8} {'ICIR':>8} {'IC>0%':>7} {'N天':>5}", flush=True)
    print("-" * 70, flush=True)

    for f in EM_FACTOR_NAMES:
        for h in horizons:
            for period in ['front', 'back']:
                ics = ic_results[f][h][period]
                if len(ics) < 10:
                    continue

                mean_ic = np.mean(ics)
                std_ic = np.std(ics)
                icir = mean_ic / std_ic if std_ic > 0 else 0
                pos_ratio = sum(1 for x in ics if x > 0) / len(ics)

                period_label = '前半' if period == 'front' else '后半'
                sgn = "+" if mean_ic > 0 else ""
                print(f"  {f:<18} {h:>3}d {period_label:>4} "
                      f"{sgn}{mean_ic:>+7.4f} {sgn}{icir:>+7.3f} "
                      f"{pos_ratio:>6.1%} {len(ics):>5}", flush=True)

                summary.append({
                    'factor': f,
                    'horizon': h,
                    'period': period,
                    'ic_mean': round(float(mean_ic), 6),
                    'ic_std': round(float(std_ic), 6),
                    'icir': round(float(icir), 4),
                    'pos_ratio': round(float(pos_ratio), 4),
                    'n_days': len(ics),
                })

    # 全期汇总
    print(f"\n{'─'*70}", flush=True)
    print(f"{'因子':<20} {'窗口':>4} "
          f"{'全期IC':>8} {'全期ICIR':>8} {'衰减':>8}", flush=True)
    print("-" * 70, flush=True)

    for f in EM_FACTOR_NAMES:
        for h in horizons:
            front_ics = ic_results[f][h]['front']
            back_ics = ic_results[f][h]['back']
            all_ics = front_ics + back_ics

            if len(all_ics) < 10:
                continue

            mean_all = np.mean(all_ics)
            std_all = np.std(all_ics)
            icir_all = mean_all / std_all if std_all > 0 else 0

            # 衰减: 后半ICIR / 前半ICIR
            decay = ""
            if len(front_ics) >= 10 and len(back_ics) >= 10:
                icir_front = np.mean(front_ics) / np.std(front_ics) if np.std(front_ics) > 0 else 0
                icir_back = np.mean(back_ics) / np.std(back_ics) if np.std(back_ics) > 0 else 0
                if abs(icir_front) > 0.01:
                    decay_ratio = icir_back / icir_front
                    decay = f"{decay_ratio:.2f}x"

            sgn = "+" if mean_all > 0 else ""
            print(f"  {f:<18} {h:>3}d "
                  f"{sgn}{mean_all:>+7.4f} {sgn}{icir_all:>+7.3f} "
                  f"{decay:>8}", flush=True)

    # ── 保存结果 ──
    output = {
        'meta': {
            'start': start,
            'end': end,
            'horizons': horizons,
            'sample_freq': sample_freq,
            'n_symbols': len(all_data),
            'n_valid_days': valid_days,
            'mid_date': mid_date.strftime('%Y-%m-%d'),
            'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        },
        'results': summary,
    }

    out_path = os.path.join(OUTPUT_DIR, "p1_earnings_ic.json")
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(output, fp, indent=2, ensure_ascii=False)

    print(f"\n  结果已保存: {out_path}", flush=True)

    # ── 结论 ──
    print(f"\n{'='*60}", flush=True)
    print("  结论:", flush=True)
    good_factors = []
    for f in EM_FACTOR_NAMES:
        all_ics = []
        for h in horizons:
            all_ics.extend(ic_results[f][h]['front'])
            all_ics.extend(ic_results[f][h]['back'])
        if len(all_ics) >= 20:
            icir = np.mean(all_ics) / np.std(all_ics) if np.std(all_ics) > 0 else 0
            if abs(icir) > 0.15:
                good_factors.append((f, icir))

    if good_factors:
        good_factors.sort(key=lambda x: -abs(x[1]))
        print("  有效因子 (|ICIR| > 0.15):", flush=True)
        for f, icir in good_factors:
            print(f"    {f}: ICIR={icir:+.3f}", flush=True)
    else:
        print("  未发现 |ICIR| > 0.15 的因子 (可能数据不足)", flush=True)
    print(f"{'='*60}", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="盈利动量因子IC验证")
    parser.add_argument("--start", type=str, default=DEFAULT_START,
                        help=f"起始日期 (默认 {DEFAULT_START})")
    parser.add_argument("--end", type=str, default=DEFAULT_END,
                        help=f"结束日期 (默认 {DEFAULT_END})")
    parser.add_argument("--horizons", type=int, nargs='+',
                        default=DEFAULT_HORIZONS,
                        help="预测窗口天数 (默认 5 10 20)")
    parser.add_argument("--freq", type=int, default=SAMPLE_FREQ,
                        help=f"采样频率/天 (默认 {SAMPLE_FREQ})")
    args = parser.parse_args()

    run_validation(args.start, args.end, args.horizons, args.freq)


if __name__ == "__main__":
    main()
