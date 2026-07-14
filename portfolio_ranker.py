"""
组合排名器 — Top-K 选股 + 持仓轮换 (参考 Qlib TopkDropoutStrategy)

核心逻辑:
  每天: 所有股票打分 → 排名 → Top-K持有,其余不持有
  - 跌出Top-K → 卖出
  - 新进入Top-K → 等权重买入
  - 已持有且仍在Top-K → 不动

用法:
  ranker = PortfolioRanker(top_k=3)
  decisions = ranker.rank(scores, current_holdings)
  → {buy: ["01810"], sell: ["00700"], hold: ["09988"]}
"""

import numpy as np
from typing import Dict, List


class PortfolioRanker:
    """
    Top-K 排名选股器。

    参数:
      top_k: 持有股票数量 (默认3)
      n_drop: 每次最多替换数量 (默认1,避免过度换手)
      hold_thresh: 最小持有天数 (默认3,防止频繁进出)
    """

    def __init__(self, top_k: int = 3, n_drop: int = 1, hold_thresh: int = 3):
        self.top_k = top_k
        self.n_drop = n_drop
        self.hold_thresh = hold_thresh
        self._hold_since: Dict[str, int] = {}
        self._first_day = True  # 首日标识  # symbol → 持有天数

    def rank(self, scores: Dict[str, float],
             current_holdings: List[str]) -> dict:
        """
        根据分数排名,决定买卖。

        参数:
          scores: {symbol: score}
          current_holdings: 当前持有的股票列表

        返回:
          {buy: [symbols], sell: [symbols], hold: [symbols]}
        """
        # 1. 按分数降序排名
        ranked = sorted(scores.keys(), key=lambda s: scores[s], reverse=True)
        top_k = ranked[:self.top_k]

        # 2. 更新持有天数
        for sym in current_holdings:
            self._hold_since[sym] = self._hold_since.get(sym, 0) + 1
        for sym in top_k:
            if sym not in current_holdings:
                self._hold_since[sym] = 0

        # 3. 决定卖出: 跌出Top-K 且 持有超过hold_thresh
        to_sell = []
        for sym in current_holdings:
            if sym not in top_k:
                if self._hold_since.get(sym, 0) >= self.hold_thresh:
                    to_sell.append(sym)

        # 4. 决定买入: 在Top-K中但未持有
        to_buy = [s for s in top_k if s not in current_holdings]

        # 首日: 允许一次买满 top_k
        if self._first_day and not current_holdings:
            to_buy = top_k[:self.top_k]
            self._first_day = False

        # 5. 限制替换数量 (避免单日大换仓)
        if len(to_sell) > self.n_drop:
            # 卖出分数最低的 n_drop 只
            sell_scores = {s: scores[s] for s in to_sell}
            to_sell = sorted(sell_scores, key=sell_scores.get)[:self.n_drop]

        if len(to_buy) > self.n_drop:
            # 买入分数最高的 n_drop 只
            buy_scores = {s: scores[s] for s in to_buy}
            to_buy = sorted(buy_scores, key=buy_scores.get, reverse=True)[:self.n_drop]

        # 6. 持仓不变
        to_hold = [s for s in current_holdings if s not in to_sell]

        return {
            "buy": to_buy,
            "sell": to_sell,
            "hold": to_hold,
            "top_k": top_k,
            "ranked": ranked,
        }
