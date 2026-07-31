"""
PEAD v2 — 预告类型分组 × CAR(+1,+20) × 多期 × 全量股票

修复:
  1. surprise=预告类型(预增/扭亏=好, 预减/首亏=坏) 替代同比数值
  2. CAR窗口: (+1,+20) 剔除公告日即时反应
  3. 多报告期: 年报(1231)+一季报(0331)+半年报(0630)+三季报(0930)
  4. 全量947只股票匹配
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import ttest_1samp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "pead_cache")

# 预告类型分组
GOOD_TYPES = {'预增', '扭亏', '略增', '续盈'}
BAD_TYPES = {'预减', '首亏', '增亏', '略减', '续亏'}


def fetch_all_forecasts():
    """拉取多期预告: 年报+季报"""
    import akshare as ak
    import warnings
    warnings.filterwarnings('ignore')

    all_dfs = []
    periods = ['1231', '0331', '0630', '0930']
    years = [2020, 2021, 2022, 2023]

    for y in years:
        for p in periods:
            cache_path = os.path.join(CACHE_DIR, f"forecast_{y}{p}.parquet")
            if os.path.exists(cache_path):
                df = pd.read_parquet(cache_path)
            else:
                try:
                    df = ak.stock_yjyg_em(date=f"{y}{p}")
                    if df is not None and len(df) > 0:
                        df.to_parquet(cache_path, index=False)
                except:
                    continue
            if df is not None and len(df) > 0:
                all_dfs.append(df)

    if not all_dfs:
        return None
    df = pd.concat(all_dfs, ignore_index=True)
    df['公告日期'] = pd.to_datetime(df['公告日期'])
    return df


def run_pead_v2():
    print("=" * 60)
    print("  PEAD v2 — 预告类型分组")
    print("=" * 60)

    # 1. 加载预告数据
    forecasts = fetch_all_forecasts()
    print(f"  总预告: {len(forecasts)}条")

    # 分组
    forecasts['group'] = 'neutral'
    forecasts.loc[forecasts['预告类型'].isin(GOOD_TYPES), 'group'] = 'good'
    forecasts.loc[forecasts['预告类型'].isin(BAD_TYPES), 'group'] = 'bad'

    good = forecasts[forecasts['group'] == 'good']
    bad = forecasts[forecasts['group'] == 'bad']
    print(f"  好消息(预增/扭亏/略增/续盈): {len(good)}条")
    print(f"  坏消息(预减/首亏/增亏/略减/续亏): {len(bad)}条")

    # 2. 加载股价
    from data_cache import get_cached_symbols, load_all
    syms = get_cached_symbols()  # 全量
    all_data = load_all(syms)
    all_data = {s: df for s, df in all_data.items() if len(df) >= 200}
    for s in all_data:
        all_data[s]['date'] = pd.to_datetime(all_data[s]['date'])
    print(f"  股价数据: {len(all_data)}只")

    # 3. 计算 CAR(+1,+20) — 优化版: 预计算基准
    # 对所有公告日期去重, 预计算每个日期的基准收益
    all_ann_dates = sorted(forecasts['公告日期'].dropna().unique())
    print(f"  唯一公告日: {len(all_ann_dates)}个")

    # 预计算每个公告日的基准收益 (★ day+1开盘价起算)
    bench_by_date = {}
    for ann_date in all_ann_dates:
        bench_rets = []
        for bs, bdf in all_data.items():
            bdf = bdf.sort_values('date')
            post = bdf[bdf['date'] > ann_date]
            if len(post) < 21: continue
            # ★ day+1开盘价作为买入价
            day1_open = post.iloc[0]['open'] if 'open' in post.columns else post.iloc[0]['close']
            bench_rets.append(post.iloc[20]['close'] / day1_open - 1)
        bench_by_date[ann_date] = np.mean(bench_rets) if bench_rets else 0

    print(f"  基准预计算完成")

    # 计算每个事件的CAR (★ day+1开盘价 + 涨停剔除)
    results = []
    limit_up_excluded = 0
    for _, event in forecasts.iterrows():
        sym = str(event.get('股票代码', '')).zfill(6)
        if sym not in all_data: continue
        ann_date = event['公告日期']
        group = event['group']
        if group == 'neutral': continue

        pdf = all_data[sym].sort_values('date')
        post = pdf[pdf['date'] > ann_date]
        if len(post) < 21: continue

        ann_close_row = pdf[pdf['date'] <= ann_date]
        if len(ann_close_row) == 0: continue
        ann_close = ann_close_row['close'].iloc[-1]

        # ★ day+1开盘价 (实际可成交价)
        day1_open = post.iloc[0]['open'] if 'open' in post.columns else post.iloc[0]['close']

        # ★ 涨停剔除: day+1开盘涨幅≥9.8% → 买不进, 排除
        if day1_open >= ann_close * 1.098:
            limit_up_excluded += 1
            continue

        # CAR: day+1开盘 → day+20收盘
        stock_ret = post.iloc[20]['close'] / day1_open - 1
        car = stock_ret - bench_by_date.get(ann_date, 0)

        results.append({
            'symbol': sym, 'ann_date': str(ann_date.date()),
            'type': event.get('预告类型', ''), 'group': group,
            'car': float(car),
        })

    rdf = pd.DataFrame(results)
    print(f"  有效CAR事件: {len(rdf)}条 (涨停剔除: {limit_up_excluded})")
    print(f"  好消息: {len(rdf[rdf['group']=='good'])}条")
    print(f"  坏消息: {len(rdf[rdf['group']=='bad'])}条")

    # 4. 统计检验 + ★ p值落盘
    output = {"n_total": len(rdf), "limit_up_excluded": limit_up_excluded}
    print(f"\n  === CAR(day+1开盘→+20收盘) 分组检验 ===")
    for g, label in [('good', '好消息(预增/扭亏/略增)'), ('bad', '坏消息(预减/首亏/增亏)')]:
        cars = rdf[rdf['group'] == g]['car']
        n = len(cars)
        tag = 'good' if g == 'good' else 'bad'
        output[f'{tag}_n'] = n
        if n < 30:
            print(f"  {label}: n={n} 不足30, 跳过")
            continue
        mean_car = cars.mean()
        std_car = cars.std(ddof=1)
        t, p = ttest_1samp(cars, 0)
        ci95 = 1.96 * std_car / np.sqrt(n)
        print(f"  {label}:")
        print(f"    n={n}  CAR均值={mean_car:+.4f} ({mean_car*100:+.2f}%)")
        print(f"    std={std_car:.4f}  95%CI=[{mean_car-ci95:+.4f},{mean_car+ci95:+.4f}]")
        print(f"    t={t:.3f}  p={p:.6f}  {'✅显著' if p<0.05 else '❌不显著'}")
        output[f'{tag}_car_mean'] = float(mean_car)
        output[f'{tag}_car_std'] = float(std_car)
        output[f'{tag}_t'] = float(t)
        output[f'{tag}_p'] = float(p)
        output[f'{tag}_ci95_low'] = float(mean_car - ci95)
        output[f'{tag}_ci95_high'] = float(mean_car + ci95)
        output[f'{tag}_significant'] = bool(p < 0.05)

    # 5. 按具体类型细分
    print(f"\n  === 按预告类型细分 ===")
    for ftype in sorted(rdf['type'].unique()):
        sub = rdf[rdf['type'] == ftype]
        if len(sub) < 20: continue
        m = sub['car'].mean()
        t, p = ttest_1samp(sub['car'], 0)
        print(f"  {ftype:6s}: n={len(sub):4d}  CAR={m*100:+.2f}%  t={t:+.2f}  p={p:.3f}")

    # 6. 保存 ★ p值 + 子类型
    output["by_type"] = {}
    for ftype in sorted(rdf['type'].unique()):
        sub = rdf[rdf['type'] == ftype]
        if len(sub) < 20: continue
        m = sub['car'].mean()
        t, p = ttest_1samp(sub['car'], 0)
        output["by_type"][ftype] = {"n": len(sub), "car_mean": float(m), "t": float(t), "p": float(p)}

    out_path = os.path.join(BASE_DIR, "data", "pead_results_v2.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  已保存: {out_path} (含p值+逐类型+95%CI)")

    return output


if __name__ == "__main__":
    run_pead_v2()
