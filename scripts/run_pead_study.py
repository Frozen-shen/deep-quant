"""
PEAD (Post-Earnings Announcement Drift) 事件研究

验证: A股业绩预告超预期 → 公告后漂移效应是否存在

数据: akshare stock_yjyg_em (业绩预告)
方法: 按公告日期对齐, 计算 CAR(0,+20) = 累计异常收益
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import ttest_1samp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "pead_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def fetch_forecasts(year: int, period: str = "1231") -> pd.DataFrame:
    """获取某年业绩预告数据"""
    import akshare as ak
    import warnings
    warnings.filterwarnings('ignore')

    cache_path = os.path.join(CACHE_DIR, f"forecast_{year}{period}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    try:
        df = ak.stock_yjyg_em(date=f"{year}{period}")
        if df is not None and len(df) > 0:
            df.to_parquet(cache_path, index=False)
            return df
    except Exception as e:
        print(f"  fetch {year}{period}: {e}")
    return None


def compute_surprise(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算业绩惊喜度。

    surprise = (预测数值 - 上年同期值) / |上年同期值|
    正惊喜 → 业绩超预期
    """
    df = df.copy()
    df['公告日期'] = pd.to_datetime(df['公告日期'])
    df['预测数值'] = pd.to_numeric(df['预测数值'], errors='coerce')
    df['上年同期值'] = pd.to_numeric(df['上年同期值'], errors='coerce')

    # 只保留有数值的行
    valid = df['预测数值'].notna() & df['上年同期值'].notna() & (df['上年同期值'] != 0)
    df = df[valid]

    df['surprise'] = (df['预测数值'] - df['上年同期值']) / df['上年同期值'].abs()

    # 分类: 正惊喜(超预期) / 负惊喜(低于预期)
    df['surprise_sign'] = np.where(df['surprise'] > 0, 'positive', 'negative')

    return df


def run_pead_study():
    """主 PEAD 事件研究"""
    from data_cache import get_cached_symbols, load_all

    print("=" * 60)
    print("  PEAD 事件研究")
    print("=" * 60)

    # ── 1. 加载多期业绩预告 ──
    all_forecasts = []
    for year in [2020, 2021, 2022, 2023]:
        df = fetch_forecasts(year, "1231")
        if df is not None:
            all_forecasts.append(df)
            print(f"  {year}年报: {len(df)}条预告")

    if not all_forecasts:
        print("  ❌ 无数据")
        return

    forecasts = pd.concat(all_forecasts, ignore_index=True)
    forecasts = compute_surprise(forecasts)
    print(f"  有效事件: {len(forecasts)}条 (有预测+上年数据)")
    print(f"  正惊喜: {len(forecasts[forecasts['surprise_sign']=='positive'])}条")
    print(f"  负惊喜: {len(forecasts[forecasts['surprise_sign']=='negative'])}条")

    # ── 2. 加载股价数据 ──
    syms = get_cached_symbols()
    all_data = load_all(syms[:300])  # 300只股票做快速验证
    all_data = {s: df for s, df in all_data.items() if len(df) >= 200}
    for s in all_data:
        all_data[s]['date'] = pd.to_datetime(all_data[s]['date'])
    print(f"  股价数据: {len(all_data)}只")

    # ── 3. 计算 CAR ──
    results = []
    for _, event in forecasts.iterrows():
        sym = str(event.get('股票代码', '')).zfill(6)
        if sym not in all_data:
            continue

        ann_date = event['公告日期']
        surprise = event['surprise']
        sign = event['surprise_sign']

        pdf = all_data[sym]
        pdf = pdf.sort_values('date')

        # 找公告日当天或之后第一个交易日
        post = pdf[pdf['date'] >= ann_date]
        if len(post) < 21:  # 需要至少20个交易日后数据
            continue

        # CAR(0, +20): 公告日后20个交易日累计收益 vs 同期CSI300
        # 简化: CAR = 股票累计收益 - 同期全市场等权收益
        ann_idx = post.index[0]
        car_end_idx = min(ann_idx + 20, len(pdf) - 1)
        if car_end_idx <= ann_idx:
            continue

        stock_ret = pdf.iloc[car_end_idx]['close'] / pdf.iloc[ann_idx]['close'] - 1

        # 同期基准收益 (用全部缓存股票等权)
        bench_rets = []
        for bs, bdf in all_data.items():
            bdf = bdf.sort_values('date')
            bpost = bdf[bdf['date'] >= ann_date]
            if len(bpost) >= 21:
                bi = bpost.index[0]
                be = min(bi + 20, len(bdf) - 1)
                if be > bi:
                    bench_rets.append(bdf.iloc[be]['close'] / bdf.iloc[bi]['close'] - 1)

        if bench_rets:
            bench_ret = np.mean(bench_rets)
            car = stock_ret - bench_ret
        else:
            car = stock_ret

        results.append({
            'symbol': sym,
            'ann_date': str(ann_date.date()),
            'surprise': float(surprise),
            'sign': sign,
            'car_20d': float(car),
        })

    # ── 4. 统计分析 ──
    rdf = pd.DataFrame(results)
    print(f"\n  有效CAR事件: {len(rdf)}条")

    pos_cars = rdf[rdf['sign'] == 'positive']['car_20d']
    neg_cars = rdf[rdf['sign'] == 'negative']['car_20d']

    print(f"\n  === CAR(0,+20) 分析 ===")
    if len(pos_cars) >= 30:
        t_pos, p_pos = ttest_1samp(pos_cars, 0)
        print(f"  正惊喜组: 均值 {pos_cars.mean():+.4f} ({pos_cars.mean()*100:+.2f}%)")
        print(f"            t={t_pos:.3f} p={p_pos:.4f} n={len(pos_cars)}")
        print(f"            显著{'✅' if p_pos < 0.05 else '❌'}")

    if len(neg_cars) >= 30:
        t_neg, p_neg = ttest_1samp(neg_cars, 0)
        print(f"  负惊喜组: 均值 {neg_cars.mean():+.4f} ({neg_cars.mean()*100:+.2f}%)")
        print(f"            t={t_neg:.3f} p={p_neg:.4f} n={len(neg_cars)}")
        print(f"            显著{'✅' if p_neg < 0.05 else '❌'}")

    # ── 5. 保存 ──
    output = {
        "n_events": len(rdf),
        "n_positive": int(len(pos_cars)),
        "n_negative": int(len(neg_cars)),
        "positive_car_mean": float(pos_cars.mean()) if len(pos_cars) > 0 else None,
        "negative_car_mean": float(neg_cars.mean()) if len(neg_cars) > 0 else None,
        "positive_t": float(t_pos) if len(pos_cars) >= 30 else None,
        "positive_p": float(p_pos) if len(pos_cars) >= 30 else None,
        "negative_t": float(t_neg) if len(neg_cars) >= 30 else None,
        "negative_p": float(p_neg) if len(neg_cars) >= 30 else None,
    }
    out_path = os.path.join(BASE_DIR, "data", "pead_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  已保存: {out_path}")

    return output


if __name__ == "__main__":
    run_pead_study()
