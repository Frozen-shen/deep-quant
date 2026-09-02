"""
因子缓存 — 预计算所有因子值，避免训练循环中重复调用 compute_factors

将 compute_factors 从 O(N_days × N_stocks) 降到 O(N_stocks × 1)

用法:
  cache = FactorCache(scorer, factor_names)
  cache.precompute(ALL_DATA, all_days)
  values = cache.get(symbol, date)  # dict of factor values
"""

import pandas as pd
import numpy as np
from factor_scorer import FactorScorer


class FactorCache:
    """预计算全量因子值，按 (symbol, date) 快速查询。"""

    def __init__(self, scorer: FactorScorer, factor_names: list = None):
        self.scorer = scorer
        self.factor_names = factor_names or sorted(scorer.factor_weights.keys())
        # {symbol: DataFrame(date × factor)}
        self._cache = {}

    def precompute(self, all_data: dict, dates: list = None):
        """
        为所有股票预计算因子值。

        Args:
          all_data: {symbol: DataFrame(OHLCV)}
          dates: 需要计算的日期列表，默认全部
        """
        for sym, df in all_data.items():
            try:
                full_factors = self.scorer.compute_factors(df)
                if "date" not in full_factors.columns:
                    full_factors["date"] = df["date"].values
                # ★ 性能优化: 用date做索引, O(1)查找
                full_factors["date"] = pd.to_datetime(full_factors["date"])
                full_factors = full_factors.set_index("date", drop=False)
                self._cache[sym] = full_factors
            except Exception as e:
                self._cache[sym] = None

    def get(self, symbol: str, date) -> dict:
        """
        获取某只股票在某一天的因子值。O(1) via index lookup.
        """
        if symbol not in self._cache or self._cache[symbol] is None:
            return None
        df = self._cache[symbol]
        ts = pd.Timestamp(date)
        try:
            row = df.loc[ts]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
        except KeyError:
            return None
        result = {}
        for fn in self.factor_names:
            if fn in df.columns:
                val = row[fn]
                result[fn] = float(val) if not (isinstance(val, float) and np.isnan(val)) else 0.0
        return result

    def get_features(self, symbol: str, date) -> list:
        """
        获取特征向量 [f_0, f_1, ...]。
        """
        vals = self.get(symbol, date)
        if vals is None:
            return None
        return [vals.get(fn, 0.0) for fn in self.factor_names]


def demo():
    """演示因子缓存。"""
    from data_cache import load_all
    from factor_scorer import FactorScorer

    SYMBOLS = ["600519", "000858", "601318"]
    data = load_all(SYMBOLS)
    scorer = FactorScorer.from_preset("full_auto")  # 原 ic_optimized 硬编码权重预设已删除

    cache = FactorCache(scorer)
    cache.precompute(data)

    # 查询
    d = pd.Timestamp("2024-06-15")
    for sym in SYMBOLS:
        feats = cache.get_features(sym, d)
        if feats:
            print(f"  {sym} @ {d.date()}: {len(feats)} factors, first 3: {feats[:3]}")
        else:
            print(f"  {sym} @ {d.date()}: no data")

    print("✅ FactorCache works!")


if __name__ == "__main__":
    demo()
