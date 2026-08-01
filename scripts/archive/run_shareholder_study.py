"""股东户数集中事件研究"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import akshare as ak
import pandas as pd
import numpy as np
from scripts.event_study import EventStudy
from data_cache import load as load_stock, get_cached_symbols

def main():
    # 1. 拉取多期股东户数 (6期够用)
    dates = ['20230331','20230630','20230930','20231231','20240331','20240630',
             '20240930','20241231','20250331']
    all_rows = []
    for d in dates:
        try:
            df = ak.stock_zh_a_gdhs(symbol=d)
            all_rows.append(df)
            print(f'  {d}: {len(df)}条', flush=True)
        except Exception as e:
            print(f'  {d}: FAIL {e}', flush=True)

    if not all_rows:
        print("无数据"); return

    combined = pd.concat(all_rows, ignore_index=True)
    print(f'合计: {len(combined)}条', flush=True)

    # 2. 筛选筹码集中事件: 户数减少 > 5%
    combined['symbol'] = combined['代码'].astype(str).str.zfill(6)
    combined['event_date'] = pd.to_datetime(combined['公告日期'])
    concentrated = combined[combined['股东户数-增减比例'] < -5].copy()
    concentrated['group'] = 'concentrated'
    print(f'筹码集中事件(减少>5%): {len(concentrated)}条', flush=True)

    # 3. 加载股价 (用前500只加速)
    syms = get_cached_symbols()[:500]
    all_data = {}
    for s in syms:
        try:
            d = load_stock(s)
            if d is not None and len(d) > 100:
                all_data[s] = d
        except:
            pass
    print(f'有效股价: {len(all_data)}只', flush=True)

    # 4. 匹配
    events = concentrated[['symbol', 'event_date', 'group']].copy()
    events = events[events['symbol'].isin(all_data.keys())]
    events = events.dropna(subset=['event_date'])
    print(f'匹配事件: {len(events)}条', flush=True)

    # 5. 事件研究
    es = EventStudy(all_data, n_benchmark_stocks=200)
    rdf = es.compute_car(events, 'group')
    print(f'有效CAR: {len(rdf)}条', flush=True)

    if len(rdf) > 0:
        result = es.analyze(rdf, 'group')
        es.report(result)
        with open('data/event_study_shareholder.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print('已保存 data/event_study_shareholder.json', flush=True)
    else:
        print('无有效CAR', flush=True)

if __name__ == '__main__':
    main()
