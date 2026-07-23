"""
组合排名器 — Top-K 选股 + 持仓轮换 (参考 Qlib TopkDropoutStrategy)

核心逻辑:
  每天: 所有股票打分 → 排名 → Top-K持有,其余不持有
  - 跌出Top-K → 卖出 (带缓冲区: 排名下降不多时继续持有)
  - 新进入Top-K → 等权重买入 (带确认期: 连续N天出现才买入)
  - 已持有且仍在Top-K → 不动
  - 成本惩罚: 新买入分数必须超过被卖出分数一定幅度才换仓

降低换手率的关键:
  1. 换手缓冲 (sell_rank_buffer): 持有股票排名跌到 top_k + buffer 才卖
  2. 买入确认 (buy_confirm_days): 新股票连续N天进入top_k才买
  3. 成本门槛 (cost_threshold): 分数差必须 > threshold 才换仓
  4. 最短持有期 (hold_thresh): 持有不满N天不卖

用法:
  ranker = PortfolioRanker(top_k=4, hold_thresh=5, sell_rank_buffer=2,
                            buy_confirm_days=2, cost_threshold=0.1)
  decisions = ranker.rank(scores, current_holdings)
  → {buy: ["01810"], sell: ["00700"], hold: ["09988"]}
"""

import numpy as np
from typing import Dict, List


class PortfolioRanker:
    """
    Top-K 排名选股器 (支持换手优化 + 板块中性化)。

    参数:
      top_k: 持有数量
      n_drop: 每次最多替换数
      hold_thresh: 最小持有天数 (持有不满此天数不卖)
      sell_rank_buffer: 卖出缓冲 — 持有股排名跌出 top_k+buffer 才触发卖出
      buy_confirm_days: 买入确认天数 — 新股票连续N天出现在 top_k 才买入
      cost_threshold: 成本门槛 — 新股分数必须超过旧股分数 * (1 + threshold) 才换仓
      sector_neutral: 是否板块中性化 (先板块内排名,再跨板块选)
    """

    def __init__(self, top_k: int = 3, n_drop: int = 1, hold_thresh: int = 5,
                 sell_rank_buffer: int = 2,
                 buy_confirm_days: int = 2,
                 cost_threshold: float = 0.1,
                 sector_neutral: bool = False):
        self.top_k = top_k
        self.n_drop = n_drop
        self.hold_thresh = hold_thresh
        self.sell_rank_buffer = sell_rank_buffer
        self.buy_confirm_days = buy_confirm_days
        self.cost_threshold = cost_threshold
        self.sector_neutral = sector_neutral
        self._hold_since: Dict[str, int] = {}
        self._topk_streak: Dict[str, int] = {}  # symbol → 连续出现在top_k的天数
        self._first_day = True

    def set_regime(self, regime: str):
        """
        根据市场状态动态调整参数 (Phase 3.2: 修正方向)。

        Args:
          regime: "trend_up" | "trend_down" | "range"
        """
        if regime == "trend_up":
            # 牛市: 积极换仓追涨
            self.hold_thresh = 5
            self.sell_rank_buffer = 2
            self.cost_threshold = 0.08
            self.n_drop = 2
        elif regime == "trend_down":
            # 熊市: 减仓少动防御
            self.hold_thresh = 7
            self.sell_rank_buffer = 3
            self.cost_threshold = 0.15
            self.n_drop = 1
        else:  # range (default)
            self.hold_thresh = 5
            self.sell_rank_buffer = 2
            self.cost_threshold = 0.12
            self.n_drop = 2

    def rank(self, scores: Dict[str, float],
             current_holdings: List[str],
             sectors: Dict[str, str] = None) -> dict:
        """
        排名决策 (换手优化版)。

        参数:
          scores: {symbol: score}
          current_holdings: 当前持仓
          sectors: {symbol: sector_name} 板块映射 (可选)

        返回: {buy, sell, hold, top_k, ranked}
        """
        # 板块中性化: 先板块内排名 → 再跨板块选
        if self.sector_neutral and sectors:
            scores = self._sector_neutralize(scores, sectors)

        # 1. 按分数降序排名
        ranked = sorted(scores.keys(), key=lambda s: scores[s], reverse=True)
        top_k = ranked[:self.top_k]

        # 2. 更新持有天数
        for sym in current_holdings:
            self._hold_since[sym] = self._hold_since.get(sym, 0) + 1

        # 3. 更新 top_k 连续出现天数 (买入确认用)
        top_k_set = set(top_k)
        for sym in list(self._topk_streak.keys()):
            if sym in top_k_set:
                self._topk_streak[sym] += 1
            else:
                self._topk_streak[sym] = 0  # 重置
        for sym in top_k:
            if sym not in self._topk_streak:
                self._topk_streak[sym] = 1

        # 4. 卖出决策 — 带缓冲区
        #    持有股票排名跌出 top_k + sell_rank_buffer 才考虑卖出
        sell_threshold_rank = self.top_k + self.sell_rank_buffer
        to_sell = []
        for sym in current_holdings:
            # 最短持有期保护
            if self._hold_since.get(sym, 0) < self.hold_thresh:
                continue

            sym_rank = ranked.index(sym) + 1 if sym in ranked else 999
            if sym_rank > sell_threshold_rank:
                to_sell.append(sym)

        # 5. 买入决策 — 带确认期 + 成本门槛
        to_buy = []
        for s in top_k:
            if s in current_holdings:
                continue
            # 买入确认: 必须连续 buy_confirm_days 天出现在 top_k
            if self._topk_streak.get(s, 0) < self.buy_confirm_days:
                continue
            to_buy.append(s)

        # 首日: 允许一次买满 top_k (跳过确认期)
        if self._first_day and not current_holdings:
            to_buy = top_k[:self.top_k]
            self._first_day = False

        # 6. 成本门槛: 只有新股分数显著超过旧股才换仓
        if to_sell and to_buy and self.cost_threshold > 0:
            filtered_sell = []
            filtered_buy = []
            # 只考虑在 scores 中存在的 (已被 filter_tradeable 移除的不参与比较)
            sell_sorted = sorted([s for s in to_sell if s in scores],
                               key=lambda s: scores.get(s, -999))
            buy_sorted = sorted([s for s in to_buy if s in scores],
                               key=lambda s: scores.get(s, -999), reverse=True)

            for i in range(min(len(sell_sorted), len(buy_sorted))):
                old_score = scores.get(sell_sorted[i], -999)
                new_score = scores.get(buy_sorted[i], -999)
                # 新股分数必须超过旧股分数 * (1 + cost_threshold)
                if old_score > 0 and new_score > old_score * (1 + self.cost_threshold):
                    filtered_sell.append(sell_sorted[i])
                    filtered_buy.append(buy_sorted[i])
                elif old_score <= 0 and new_score > 0:
                    # 旧股分数为负，直接替换
                    filtered_sell.append(sell_sorted[i])
                    filtered_buy.append(buy_sorted[i])

            to_sell = filtered_sell
            to_buy = filtered_buy

        # 7. 限制替换数量
        if len(to_sell) > self.n_drop:
            sell_scores = {s: scores.get(s, -999) for s in to_sell}
            to_sell = sorted(sell_scores, key=sell_scores.get)[:self.n_drop]

        if len(to_buy) > self.n_drop:
            buy_scores = {s: scores.get(s, -999) for s in to_buy}
            to_buy = sorted(buy_scores, key=buy_scores.get, reverse=True)[:self.n_drop]

        # 8. 清理已卖出股票的持有记录
        for sym in to_sell:
            self._hold_since.pop(sym, None)

        # 9. 持仓不变
        to_hold = [s for s in current_holdings if s not in to_sell]

        return {
            "buy": to_buy,
            "sell": to_sell,
            "hold": to_hold,
            "top_k": top_k,
            "ranked": ranked,
        }

    def _sector_neutralize(self, scores: Dict[str, float],
                           sectors: Dict[str, str]) -> Dict[str, float]:
        """
        板块中性化: 每个板块内 z-score → 消除板块偏差。

        半导体板块平均涨5%，白酒板块跌2% → 中性化后
        半导体里的股票不会因为"在好板块"而天然高分。
        """
        # 按板块分组
        sector_groups = {}
        for sym, score in scores.items():
            sec = sectors.get(sym, "其他")
            sector_groups.setdefault(sec, {})[sym] = score

        # 每个板块内 z-score
        neutralized = {}
        for sec, group in sector_groups.items():
            vals = np.array(list(group.values()))
            mean, std = vals.mean(), vals.std() if vals.std() > 0 else 1.0
            for sym, score in group.items():
                neutralized[sym] = (score - mean) / std

        return neutralized
