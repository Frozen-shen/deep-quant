"""
通用事件研究框架 — 18轮学费换来的严格检验流程

输入: 事件表 {symbol, event_date, group}
输出: CAR(day+1开盘,+20) + t/p/CI + 分年稳健性 → 全部落盘JSON

规则(不可绕过):
  1. CAR从day+1开盘价起算(含跳空=幻觉)
  2. day+1涨停≥9.8%的事件剔除(买不进)
  3. 基准=全部缓存股票等权同期收益
  4. 分组t检验 + 子类型细分 + 分年稳健性
  5. 所有输出强制写JSON, print不算数
"""
import os, sys, json
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import ttest_1samp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class EventStudy:
    """通用事件研究引擎"""

    def __init__(self, all_data: dict, n_benchmark_stocks: int = 300):
        """
        Args:
          all_data: {symbol: DataFrame(date, open, close, ...)}
          n_benchmark_stocks: 基准计算用多少只股票
        """
        self.all_data = all_data
        self._bench_stocks = list(all_data.keys())[:n_benchmark_stocks]

    def _get_benchmark_ret(self, ann_date):
        """计算某日的基准收益: day+1开盘→day+20收盘"""
        brets = []
        for s in self._bench_stocks:
            bdf = self.all_data[s].sort_values('date')
            post = bdf[bdf['date'] > ann_date]
            if len(post) < 21:
                continue
            base = bdf[bdf['date'] <= ann_date]
            if len(base) == 0:
                continue
            day1_open = post.iloc[0]['open'] if 'open' in post.columns else post.iloc[0]['close']
            brets.append(post.iloc[20]['close'] / day1_open - 1)
        return np.mean(brets) if brets else 0.0

    def compute_car(self, events: pd.DataFrame, group_col: str = 'group') -> pd.DataFrame:
        """
        计算每个事件的CAR。

        events DataFrame 必须包含: symbol, event_date, {group_col}
        """
        results = []
        excluded_limit_up = 0
        excluded_no_data = 0

        for _, e in events.iterrows():
            sym = str(e.get('symbol', '')).zfill(6)
            ann_date = pd.Timestamp(e['event_date'])
            group = e.get(group_col, 'unknown')

            if sym not in self.all_data:
                excluded_no_data += 1
                continue

            pdf = self.all_data[sym].sort_values('date')
            post = pdf[pdf['date'] > ann_date]
            if len(post) < 21:
                excluded_no_data += 1
                continue

            ann_row = pdf[pdf['date'] <= ann_date]
            if len(ann_row) == 0:
                excluded_no_data += 1
                continue

            ann_close = ann_row['close'].iloc[-1]
            day1_open = post.iloc[0]['open'] if 'open' in post.columns else post.iloc[0]['close']

            # ★ 涨停剔除
            if day1_open >= ann_close * 1.098:
                excluded_limit_up += 1
                continue

            bench = self._get_benchmark_ret(ann_date)
            stock_ret = post.iloc[20]['close'] / day1_open - 1
            car = stock_ret - bench

            results.append({
                'symbol': sym,
                'event_date': str(ann_date.date()),
                'year': ann_date.year,
                'group': str(group),
                'car': float(car),
                'stock_ret': float(stock_ret),
                'bench_ret': float(bench),
            })

        print(f"  CAR计算: {len(results)}有效, 涨停剔除{excluded_limit_up}, 无数据{excluded_no_data}")
        return pd.DataFrame(results)

    @staticmethod
    def analyze(rdf: pd.DataFrame, group_col: str = 'group') -> dict:
        """分组统计检验 + 子类型 + 分年 → dict"""
        output = {
            'n_total': len(rdf),
            'by_group': {},
            'by_subtype': {},
            'by_year_group': {},
        }

        # ── 分组 ──
        for g in sorted(rdf[group_col].unique()):
            sub = rdf[rdf[group_col] == g]
            n = len(sub)
            if n < 20:
                continue
            mean_car = sub['car'].mean()
            std_car = sub['car'].std(ddof=1)
            t, p = ttest_1samp(sub['car'], 0)
            ci95 = 1.96 * std_car / np.sqrt(n)
            output['by_group'][str(g)] = {
                'n': n, 'car_mean': round(float(mean_car), 6),
                'car_std': round(float(std_car), 6),
                't': round(float(t), 4), 'p': round(float(p), 6),
                'ci95_low': round(float(mean_car - ci95), 6),
                'ci95_high': round(float(mean_car + ci95), 6),
                'significant': bool(p < 0.05),
            }

        # ── 子类型 (如果group_col不是'group'就用原列) ──
        subtype_col = group_col if group_col != 'group' else group_col
        for st in sorted(rdf[subtype_col].unique()):
            sub = rdf[rdf[subtype_col] == st]
            n = len(sub)
            if n < 20:
                continue
            mean_car = sub['car'].mean()
            t, p = ttest_1samp(sub['car'], 0)
            output['by_subtype'][str(st)] = {
                'n': n, 'car_mean': round(float(mean_car), 6),
                't': round(float(t), 4), 'p': round(float(p), 6),
                'significant': bool(p < 0.05),
            }

        # ── 分年 × 分组 ──
        if 'year' in rdf.columns:
            for g in sorted(rdf[group_col].unique()):
                gkey = str(g)
                output['by_year_group'][gkey] = {}
                for y in sorted(rdf['year'].unique()):
                    sub = rdf[(rdf[group_col] == g) & (rdf['year'] == y)]
                    n = len(sub)
                    if n < 10:
                        continue
                    mean_car = sub['car'].mean()
                    t, p = ttest_1samp(sub['car'], 0)
                    output['by_year_group'][gkey][int(y)] = {
                        'n': n, 'car_mean': round(float(mean_car), 6),
                        't': round(float(t), 4), 'p': round(float(p), 6),
                        'significant': bool(p < 0.05),
                    }

        return output

    @staticmethod
    def report(output: dict):
        """打印可读报告"""
        print("\n" + "=" * 60)
        print(f"  事件研究结果 (n={output['n_total']})")
        print("=" * 60)

        for g, stats in output.get('by_group', {}).items():
            sig = '✅' if stats['significant'] else '❌'
            print(f"\n  [{g}] n={stats['n']} CAR={stats['car_mean']*100:+.2f}% "
                  f"t={stats['t']:.2f} p={stats['p']:.4f} {sig}")

            if g in output.get('by_year_group', {}):
                print(f"    分年:")
                for y, ys in sorted(output['by_year_group'][g].items()):
                    ys_sig = '✅' if ys['significant'] else '❌'
                    print(f"      {y}: n={ys['n']:4d} CAR={ys['car_mean']*100:+6.2f}% "
                          f"t={ys['t']:+5.2f} p={ys['p']:.4f} {ys_sig}")

        if output.get('by_subtype'):
            print(f"\n  子类型:")
            for st, ss in sorted(output['by_subtype'].items(), key=lambda x: -x[1]['n']):
                sig = '✅' if ss['significant'] else '❌'
                print(f"    {st:8s} n={ss['n']:4d} CAR={ss['car_mean']*100:+6.2f}% "
                      f"t={ss['t']:+5.2f} p={ss['p']:.4f} {sig}")


def load_price_data(symbols: list = None, min_rows: int = 200):
    """加载股价数据"""
    from data_cache import get_cached_symbols, load_all
    if symbols is None:
        symbols = get_cached_symbols()
    data = load_all(symbols)
    data = {s: df for s, df in data.items() if len(df) >= min_rows}
    for s in data:
        data[s]['date'] = pd.to_datetime(data[s]['date'])
    return data


if __name__ == "__main__":
    # 演示: 用PEAD数据跑一次
    sys.path.insert(0, BASE_DIR)
    from scripts.run_pead_study import fetch_all_forecasts, GOOD_TYPES

    forecasts = fetch_all_forecasts()
    forecasts['公告日期'] = pd.to_datetime(forecasts['公告日期'])
    forecasts = forecasts[forecasts['预告类型'].notna()]

    # 构建事件表
    events = forecasts[['股票代码', '公告日期', '预告类型']].copy()
    events.columns = ['symbol', 'event_date', 'group']
    events['group'] = events['group'].apply(
        lambda t: 'good' if t in GOOD_TYPES else ('bad' if t not in {'不确定'} else 'neutral'))
    events = events[events['group'] != 'neutral']

    print(f"事件总数: {len(events)}")

    # 加载股价
    data = load_price_data(list(events['symbol'].unique())[:500])
    print(f"股价数据: {len(data)}只")

    # 计算CAR
    study = EventStudy(data)
    rdf = study.compute_car(events, group_col='group')

    # 分析
    output = study.analyze(rdf, group_col='group')

    # 把原始group替换为预告类型做子类型分析
    events_detail = forecasts[['股票代码', '公告日期', '预告类型']].copy()
    events_detail.columns = ['symbol', 'event_date', 'group']
    rdf_detail = study.compute_car(events_detail, group_col='group')
    output_detail = study.analyze(rdf_detail, group_col='group')
    output['by_subtype'] = output_detail['by_subtype']
    # 分年也细化
    for g in output.get('by_year_group', {}):
        pass  # 保留分组级别的分年

    study.report(output)

    # 保存
    out_path = os.path.join(BASE_DIR, "data", "event_study_pead.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  已保存: {out_path}")
