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
            print(f"  Precomputing factors for {sym}...", end=" ")
            try:
                full_factors = self.scorer.compute_factors(df)
                if "date" not in full_factors.columns:
                    full_factors["date"] = df["date"].values
                self._cache[sym] = full_factors
                print(f"{len(full_factors)} rows")
            except Exception as e:
                print(f"ERROR: {e}")
                self._cache[sym] = None

    def get(self, symbol: str, date) -> dict:
        """
        获取某只股票在某一天的因子值。

        Returns:
          {factor_name: float} 或 None (如果数据不足)
        """
        if symbol not in self._cache or self._cache[symbol] is None:
            return None
        df = self._cache[symbol]
        # date 列的日期匹配
        mask = df["date"] == pd.Timestamp(date)
        if not mask.any():
            return None
        row = df[mask].iloc[-1]
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
    scorer = FactorScorer.from_preset("ic_optimized")

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
