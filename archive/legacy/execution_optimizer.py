"""
执行层优化模块 — 信号衰减、冲击成本建模、换手率控制

解决的问题:
  1. 信号无衰减: 20天前的信号和今天的信号权重相同，导致过时信号产生无效交易
  2. 交易成本粗糙: 固定30bp滑点不区分大小单，大单冲击被严重低估
  3. 换手率失控: BLIND期21.2%换手率(正常TEST期11%)，信号频繁翻转浪费成本

用法:
    from execution_optimizer import (
        SignalDecayManager,
        TurnoverController,
        estimate_impact_cost,
        optimal_execution_schedule,
    )

    # 信号衰减
    decay = SignalDecayManager(half_life_days=10)
    decay.update("000001.SZ", 0.85, date=pd.Timestamp("2024-01-05"))
    score = decay.get_current_score("000001.SZ", date=pd.Timestamp("2024-01-15"))

    # 冲击成本
    cost = estimate_impact_cost(
        order_size=50000, avg_daily_volume=1e6,
        volatility=0.025, price=15.0
    )

    # 换手率控制
    ctrl = TurnoverController(max_monthly_turnover=0.3)
    adjusted = ctrl.compute_optimal_turnover(current_weights, target_weights, costs)
"""

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ================================================================
#  1. 信号衰减 (Signal Decay)
# ================================================================

class SignalDecayManager:
    """指数衰减信号管理器。

    核心思想: 信号随时间失去信息量，用指数衰减建模:
        score(t) = new_score * exp(-lambda * days_since_update)
        lambda = ln(2) / half_life_days

    半衰期越短，信号衰减越快，对"新鲜度"要求越高。

    Attributes:
        half_life_days: 信号半衰期(天)，默认10天
        _lambda: 衰减系数
        _store: {symbol: (raw_score, last_update_date)} 内部存储
    """

    def __init__(self, half_life_days: int = 10) -> None:
        """初始化衰减管理器。

        Args:
            half_life_days: 信号半衰期(天)。
                - 5天: 激进，适合高频信号
                - 10天: 默认，适合周频调仓
                - 20天: 保守，适合月频调仓
        """
        if half_life_days <= 0:
            raise ValueError(f"half_life_days must be positive, got {half_life_days}")
        self.half_life_days = half_life_days
        self._lambda = math.log(2) / half_life_days
        # {symbol: (raw_score, last_update_date)}
        self._store: Dict[str, Tuple[float, pd.Timestamp]] = {}

    def update(self, symbol: str, new_score: float, date) -> None:
        """收到新信号时更新。

        Args:
            symbol: 股票代码 (如 "000001.SZ")
            new_score: 新的原始分数
            date: 信号生成日期 (str 或 pd.Timestamp)
        """
        ts = pd.Timestamp(date)
        self._store[symbol] = (new_score, ts)

    def get_current_score(self, symbol: str, date) -> float:
        """获取衰减后的当前分数。

        score(t) = raw_score * exp(-lambda * days_since_update)

        如果该股票从未更新过信号，返回 0.0。

        Args:
            symbol: 股票代码
            date: 当前日期

        Returns:
            衰减后的分数，范围取决于原始分数
        """
        if symbol not in self._store:
            return 0.0
        raw_score, last_update = self._store[symbol]
        ts = pd.Timestamp(date)
        days_elapsed = (ts - last_update).days
        if days_elapsed < 0:
            # 日期回退，不做衰减(视为同一天)
            days_elapsed = 0
        decay_factor = math.exp(-self._lambda * days_elapsed)
        return raw_score * decay_factor

    def get_all_scores(self, date) -> Dict[str, float]:
        """获取所有股票的衰减分数。

        Args:
            date: 当前日期

        Returns:
            {symbol: decayed_score} 字典
        """
        ts = pd.Timestamp(date)
        result: Dict[str, float] = {}
        for symbol, (raw_score, last_update) in self._store.items():
            days_elapsed = (ts - last_update).days
            if days_elapsed < 0:
                days_elapsed = 0
            decay_factor = math.exp(-self._lambda * days_elapsed)
            result[symbol] = raw_score * decay_factor
        return result

    def get_active_symbols(self, min_score: float = 0.01) -> List[str]:
        """获取衰减后分数仍高于阈值的股票列表(辅助方法)。

        Args:
            min_score: 最低有效分数阈值

        Returns:
            有效股票代码列表
        """
        today = max(v[1] for v in self._store.values()) if self._store else pd.Timestamp.now()
        scores = self.get_all_scores(today)
        return [s for s, sc in scores.items() if abs(sc) >= min_score]


# ================================================================
#  2. 非线性冲击成本模型 (Almgren-Chriss 简化)
# ================================================================

def estimate_impact_cost(
    order_size: float,
    avg_daily_volume: float,
    volatility: float,
    price: float,
) -> float:
    """估算市场冲击成本 (Almgren-Chriss 简化临时冲击模型)。

    模型: temporary_impact = eta * sigma * sqrt(order_size / ADV)

    参数含义:
        - eta=0.1: 经验冲击系数，A股中小盘可上调至0.15-0.2
        - sigma: 日波动率(标准差)
        - order_size / ADV: 参与率，越大冲击越强

    适用场景:
        - 判断大单是否应该拆分执行
        - 回测中替代固定滑点，使成本估计更贴近真实

    Args:
        order_size: 订单股数
        avg_daily_volume: 日均成交量(股)
        volatility: 日波动率 (如0.025表示2.5%)
        price: 当前价格(用于上下文，模型本身用比例)

    Returns:
        冲击成本比例 (如0.003表示0.3%)

    Raises:
        ValueError: 参数不合法时
    """
    if order_size < 0:
        raise ValueError(f"order_size must be non-negative, got {order_size}")
    if avg_daily_volume <= 0:
        raise ValueError(f"avg_daily_volume must be positive, got {avg_daily_volume}")
    if volatility < 0:
        raise ValueError(f"volatility must be non-negative, got {volatility}")
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")

    eta = 0.1  # 经验冲击系数
    participation_rate = order_size / avg_daily_volume
    temporary_impact = eta * volatility * math.sqrt(participation_rate)
    return temporary_impact


def optimal_execution_schedule(total_shares: int, days: int = 5) -> List[int]:
    """TWAP (Time-Weighted Average Price) 均匀分单计划。

    将总订单均匀分配到多天执行，降低单日冲击。
    余数分配到前几天(避免最后一天集中)。

    Args:
        total_shares: 总股数
        days: 执行天数，默认5天

    Returns:
        每天的股数列表，长度等于days，总和等于total_shares

    Raises:
        ValueError: days <= 0 时
    """
    if days <= 0:
        raise ValueError(f"days must be positive, got {days}")
    if total_shares < 0:
        raise ValueError(f"total_shares must be non-negative, got {total_shares}")

    base = total_shares // days
    remainder = total_shares % days

    schedule: List[int] = []
    for i in range(days):
        # 余数分配到前 remainder 天
        schedule.append(base + (1 if i < remainder else 0))
    return schedule


# ================================================================
#  3. 换手率控制 (Turnover Control)
# ================================================================

class TurnoverController:
    """换手率控制器 — 在成本约束下优化调仓。

    核心逻辑:
        1. 小额调整不值得交易 (signal_threshold 过滤)
        2. 预期收益必须覆盖交易成本 (成本收益比)
        3. 总换手率不超过月度上限 (贪心分配)

    Attributes:
        max_monthly_turnover: 月度最大换手率 (如0.3表示30%)
        signal_threshold: 最小调仓幅度阈值
    """

    def __init__(
        self,
        max_monthly_turnover: float = 0.3,
        signal_threshold: float = 0.02,
    ) -> None:
        """初始化换手率控制器。

        Args:
            max_monthly_turnover: 月度最大换手率上限。
                0.3 = 每月最多换30%的仓位。
            signal_threshold: 最小调仓信号阈值。
                |target - current| 低于此值时不交易。
        """
        if max_monthly_turnover <= 0:
            raise ValueError(f"max_monthly_turnover must be positive, got {max_monthly_turnover}")
        if signal_threshold < 0:
            raise ValueError(f"signal_threshold must be non-negative, got {signal_threshold}")
        self.max_monthly_turnover = max_monthly_turnover
        self.signal_threshold = signal_threshold

    def should_trade(
        self,
        symbol: str,
        current_weight: float,
        target_weight: float,
        transaction_cost: float,
    ) -> bool:
        """判断是否应该对某只股票执行交易。

        条件(必须同时满足):
            1. |target - current| > signal_threshold
            2. 调仓带来的预期改善 > transaction_cost
               (用 |target - current| 作为预期改善的代理变量)

        Args:
            symbol: 股票代码
            current_weight: 当前权重
            target_weight: 目标权重
            transaction_cost: 单次交易成本比例 (如0.003)

        Returns:
            True 表示应该交易
        """
        delta = abs(target_weight - current_weight)
        # 条件1: 信号足够大
        if delta <= self.signal_threshold:
            return False
        # 条件2: 预期收益覆盖成本
        # 这里用 delta 作为"预期改善"的近似:
        # 如果调整幅度都不够覆盖成本，交易无意义
        if delta <= transaction_cost:
            return False
        return True

    def compute_optimal_turnover(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        costs: Dict[str, float],
    ) -> Dict[str, float]:
        """在换手约束下计算最优调仓方案。

        算法: 贪心 — 按"性价比"(调整幅度/成本)降序排列，
        依次分配换手预算，直到达到上限。

        对于预算不足的股票，按比例缩减调整幅度。

        Args:
            current_weights: {symbol: current_weight}
            target_weights: {symbol: target_weight}
            costs: {symbol: transaction_cost_ratio}

        Returns:
            {symbol: adjusted_weight} — 调整后的目标权重
        """
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        # 计算每只股票的调整需求
        candidates: List[Tuple[str, float, float]] = []  # (symbol, delta, cost)
        for symbol in all_symbols:
            cur = current_weights.get(symbol, 0.0)
            tgt = target_weights.get(symbol, 0.0)
            delta = tgt - cur  # 带方向
            abs_delta = abs(delta)
            cost = costs.get(symbol, 0.003)  # 默认3bp

            # 过滤: 低于阈值的不调整
            if abs_delta <= self.signal_threshold:
                continue
            candidates.append((symbol, delta, cost))

        # 按性价比排序: |delta| / cost 越大越优先
        # cost为0时视为无穷大优先级
        def sort_key(item: Tuple[str, float, float]) -> float:
            _, d, c = item
            if c <= 0:
                return float("inf")
            return abs(d) / c

        candidates.sort(key=sort_key, reverse=True)

        # 贪心分配换手预算
        remaining_budget = self.max_monthly_turnover
        result: Dict[str, float] = dict(current_weights)  # 从当前仓位开始

        for symbol, delta, cost in candidates:
            abs_delta = abs(delta)
            if remaining_budget <= 0:
                break

            if abs_delta <= remaining_budget:
                # 预算充足，全额调整
                result[symbol] = current_weights.get(symbol, 0.0) + delta
                remaining_budget -= abs_delta
            else:
                # 预算不足，按比例缩减
                fraction = remaining_budget / abs_delta
                partial_delta = delta * fraction
                result[symbol] = current_weights.get(symbol, 0.0) + partial_delta
                remaining_budget = 0.0

        # 确保不在target中的股票保持当前权重(已处理)
        # 确保target中但未被选中的股票保持当前权重(已处理)
        return result
