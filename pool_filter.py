"""
pool_filter.py — 波动率分层 × regime 乘数 (股票池分域, v10, 2026-08-09)

设计文档: docs/superpowers/specs/2026-08-09-vol-regime-pool-filter-design.md

选股层软偏好:
  - vol_bucket: 候选股按自身 lookback 日波动率跨截面三分位 (low/mid/high)
  - apply_pool_filter: 按市场波动率状态 (vol_pct) 选择乘数表, 对分数加权
  - 波动率数据不足的股票按 mid (×1.0) 处理, 静默跳过
"""

import numpy as np
import pandas as pd


def vol_bucket(scores: dict, all_data: dict, today,
               lookback: int = 60) -> dict:
    """候选股波动率三分位分档。

    Args:
        scores: {sym: score} 候选股票分数 (仅键有意义)
        all_data: {sym: DataFrame(date, open, close, ...)} 日线
        today: 调仓日 (pd.Timestamp), 只用 <= today 数据 (PIT)
        lookback: 波动率回看天数 (日收益 std)

    Returns:
        {sym: 'low'|'mid'|'high'}, 数据不足的股票返回 'mid'
    """
    vols = {}
    for sym in scores:
        if sym not in all_data:
            continue
        df = all_data[sym][all_data[sym]["date"] <= today]
        if len(df) < 20:
            continue
        rets = df["close"].pct_change().dropna().tail(lookback)
        if len(rets) < 10:
            continue
        v = float(rets.std())
        if v > 0 and not np.isnan(v):
            vols[sym] = v
    if not vols:
        return {s: "mid" for s in scores}

    # 跨截面分位 (波动率排序, 无聚类)
    arr = np.array(sorted(vols.values()))
    q30 = np.percentile(arr, 30)
    q70 = np.percentile(arr, 70)

    buckets = {}
    for sym in scores:
        v = vols.get(sym)
        if v is None:
            buckets[sym] = "mid"
        elif v < q30:
            buckets[sym] = "low"
        elif v > q70:
            buckets[sym] = "high"
        else:
            buckets[sym] = "mid"
    return buckets


def apply_pool_filter(scores: dict, buckets: dict, vol_pct: float,
                      mults: dict) -> dict:
    """按市场波动率状态对选股分施加档位乘数 (软偏好)。

    Args:
        scores: {sym: score} 原始选股分
        buckets: {sym: 'low'|'mid'|'high'} (vol_bucket 输出)
        vol_pct: 市场波动率百分位 (0-1, regime_detector.detect_v2 输出)
        mults: 乘数表 {"low": x, "mid": x, "high": x} (当前市场状态下)

    Returns:
        新分数 dict (不原地修改)
    """
    # 选择乘数表: 高波动市场用 mults 原表; 低波动市场由调用方传入对应表
    out = dict(scores)
    for sym in out:
        tier = buckets.get(sym, "mid")
        out[sym] = out[sym] * float(mults.get(tier, 1.0))
    return out
