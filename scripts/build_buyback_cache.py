"""Build buyback events cache from akshare."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

def main():
    import akshare as ak

    print("拉取回购数据...", flush=True)
    df = ak.stock_repurchase_em()
    df = df[df['实施进度'] == '完成实施'].copy()
    df['event_date'] = pd.to_datetime(df['回购起始时间'])
    df['symbol'] = df['股票代码'].astype(str).str.zfill(6)
    df = df[df['event_date'] >= '2015-01-01'].copy()
    df['amount'] = pd.to_numeric(df['计划回购金额区间-下限'], errors='coerce')
    df = df.dropna(subset=['amount'])
    print(f"有效回购: {len(df)}条", flush=True)

    # Size scoring: small=3, medium=2, large=1
    q33 = df['amount'].quantile(0.33)
    q67 = df['amount'].quantile(0.67)
    df['size_score'] = np.where(df['amount'] <= q33, 3.0,
                       np.where(df['amount'] <= q67, 2.0, 1.0))
    df['size_group'] = np.where(df['amount'] <= q33, 'small',
                       np.where(df['amount'] <= q67, 'medium', 'large'))

    out = df[['symbol', 'event_date', 'amount', 'size_group', 'size_score']].copy()

    os.makedirs('data/event_cache', exist_ok=True)
    out.to_parquet('data/event_cache/buyback_events.parquet', index=False)
    print(f"已保存: {len(out)}条 → data/event_cache/buyback_events.parquet", flush=True)
    print(f"分布: {out['size_group'].value_counts().to_dict()}")

if __name__ == '__main__':
    main()
