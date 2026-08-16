"""
执行质量指标 (TCA) — 成交价 vs 参考价偏差分布。

2026-08-15 (L1): 把"成交价假设是否保守/诚实"变成可度量指标:
  - vwap_dev: 成交价 vs 当日全天 VWAP
  - arrival_dev: 成交价 vs 到达价 (当日首根 5m bar 收盘)
  - perfect_gain: 完美择时上限 (BUY=当日最低价, SELL=当日最高价)

符号约定: BUY 的偏差 = fill/ref - 1, SELL 的偏差 = ref/fill - 1。
即"正 = 比参考价吃亏" (买贵了/卖便宜了), 单位 bps。
"""
from typing import Dict, List, Optional

import pandas as pd


def _signed_dev(fill: float, ref: float, action: str) -> float:
    """符号化偏差 (bps): 正 = 比参考价吃亏。"""
    if action == "BUY":
        return (fill / ref - 1) * 1e4
    return (ref / fill - 1) * 1e4


def _stats(values: List[float]) -> Dict:
    """mean/median/p95 (bps), 空序列返回 None。"""
    if not values:
        return {"mean": None, "median": None, "p95": None}
    s = pd.Series(values)
    return {"mean": round(float(s.mean()), 1),
            "median": round(float(s.median()), 1),
            "p95": round(float(s.quantile(0.95)), 1)}


def fill_quality(trades: List[dict], mf=None) -> Dict:
    """计算一组成交的执行质量指标。

    Args:
      trades: [{"date": "YYYY-MM-DD", "symbol": ..., "action": "BUY"/"SELL",
                "price": float, "qty": int}, ...]
      mf: 带 fetch(symbol, days=..., end_date=...) 的对象
          (MinuteFetcher 或依赖注入的测试替身); None 则默认 MinuteFetcher。

    Returns:
      {"n": 有效笔数,
       "by_action": {action: {"vwap_dev_bps": {...}, "arrival_dev_bps": {...},
                              "perfect_gain_bps": {...}}},
       "overall": {...同结构, 汇总...}}
    """
    if mf is None:
        from data.minute_fetcher import MinuteFetcher
        mf = MinuteFetcher(allow_network=False)

    # 按 symbol 分组, 每只股票只 fetch 一次 (覆盖该股最晚成交日), 再本地切片
    by_sym: Dict[str, List[dict]] = {}
    for t in trades:
        if t.get("action") in ("BUY", "SELL"):
            by_sym.setdefault(t["symbol"], []).append(t)
    sym_data: Dict[str, pd.DataFrame] = {}
    for sym, ts in by_sym.items():
        latest = max(str(t.get("date", ""))[:10] for t in ts)
        try:
            df = mf.fetch(sym, days=10, end_date=latest)
        except Exception:
            df = None
        if df is not None and len(df) > 0:
            sym_data[sym] = df

    per_action = {"BUY": [], "SELL": []}
    for sym, ts in by_sym.items():
        df = sym_data.get(sym)
        if df is None:
            continue
        for t in ts:
            action = t.get("action", "BUY")
            date_str = str(t.get("date", ""))[:10]
            date_dt = pd.Timestamp(date_str)
            day = df[df["时间"].dt.date == date_dt.date()]
            if len(day) == 0:
                continue
            vol = day["成交量"].astype(float)
            closes = day["收盘"].astype(float)
            vwap = float((closes * vol).sum() / vol.sum()) if vol.sum() > 0 \
                else float(closes.mean())
            arrival = float(closes.iloc[0])
            perfect = float(day["最低"].min()) if action == "BUY" \
                else float(day["最高"].max())
            fill = float(t["price"])
            per_action[action].append({
                "vwap": _signed_dev(fill, vwap, action),
                "arrival": _signed_dev(fill, arrival, action),
                "perfect": _signed_dev(fill, perfect, action),
            })

    def _agg(entries):
        return {
            "vwap_dev_bps": _stats([e["vwap"] for e in entries]),
            "arrival_dev_bps": _stats([e["arrival"] for e in entries]),
            "perfect_gain_bps": _stats([e["perfect"] for e in entries]),
        }

    out = {"n": sum(len(v) for v in per_action.values()),
           "by_action": {a: _agg(v) for a, v in per_action.items()},
           "overall": _agg([e for v in per_action.values() for e in v])}
    return out
