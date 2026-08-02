"""
持仓管理器 — 加载/更新/快照

基于 storage.py 的 SQLite 后端，管理:
- 当前现金和持仓
- 应用交易 → 更新持仓
- 每日权益快照
- 资金/仓位约束
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import storage


@dataclass
class PortfolioState:
    """持仓状态快照"""
    cash: float = 100_000.0
    positions: Dict[str, Dict] = field(default_factory=dict)  # symbol → {market, qty, avg_cost}
    initial_capital: float = 100_000.0
    last_date: str = ""

    @property
    def total_holdings_value(self) -> float:
        """持仓市值（需外部提供最新价格）。"""
        return 0.0  # 需要外部注入价格

    @property
    def total_equity(self) -> float:
        return self.cash + self.total_holdings_value


class PortfolioManager:
    """
    持仓管理器。

    用法:
        pm = PortfolioManager(market="hk", initial_capital=100_000)
        state = pm.load()
        pm.apply_buy("01810", 200, 35.20, reason="MA金叉")
        pm.apply_sell("01810", 200, 36.50, reason="MA死叉")
        pm.snapshot(date, close_prices={"01810": 36.50})
    """

    def __init__(self, market: str = "hk", initial_capital: float = 100_000):
        self.market = market
        self.initial_capital = initial_capital
        storage.init_db()

        # 初始化配置（仅首次）
        if not storage.get_config("initial_capital"):
            storage.set_config("initial_capital", str(initial_capital))
            storage.set_config("last_date", "")
            storage.set_config("market", market)

    # ================================================================
    #  加载 / 保存状态
    # ================================================================
    def load(self) -> PortfolioState:
        """从数据库加载当前持仓状态。"""
        positions_raw = storage.get_all_positions()
        positions = {}
        for p in positions_raw:
            positions[p["symbol"]] = {
                "market": p["market"],
                "qty": p["qty"],
                "avg_cost": p["avg_cost"],
            }

        initial = float(storage.get_config("initial_capital", str(self.initial_capital)))
        last_date = storage.get_config("last_date", "")

        # 计算现金：初始资金 - 已买入总成本 + 已卖出总收入
        cash = initial
        trades = storage.get_trades(limit=9999)
        for t in trades:
            if t["action"] == "BUY":
                cash -= t["qty"] * t["price"] + t["commission"]
            elif t["action"] == "SELL":
                cash += t["qty"] * t["price"] - t["commission"]

        return PortfolioState(
            cash=cash,
            positions=positions,
            initial_capital=initial,
            last_date=last_date,
        )

    # ================================================================
    #  交易操作
    # ================================================================
    def apply_buy(self, symbol: str, qty: int, price: float,
                  commission: float = 0.0, reason: str = "",
                  trade_date: str = None) -> int:
        """
        执行买入：扣现金、增持仓、记录交易。

        返回 trade_id
        """
        today = trade_date or datetime.now().strftime("%Y-%m-%d")

        # 获取当前持仓
        existing = storage.get_position(symbol)
        if existing and existing["qty"] > 0:
            # 加仓：更新平均成本
            old_qty = existing["qty"]
            old_cost = existing["avg_cost"]
            new_qty = old_qty + qty
            new_avg_cost = (old_qty * old_cost + qty * (price + commission / qty)) / new_qty
            storage.upsert_position(symbol, self.market, new_qty, new_avg_cost)
        else:
            # 新建仓
            avg_cost = price + commission / qty if qty > 0 else price
            storage.upsert_position(symbol, self.market, qty, avg_cost)

        tid = storage.record_trade(symbol, self.market, today, "BUY",
                                    qty, price, commission, reason)
        return tid

    def apply_sell(self, symbol: str, qty: int, price: float,
                   commission: float = 0.0, reason: str = "",
                   trade_date: str = None) -> int:
        """
        执行卖出：增现金、减持仓、记录交易。
        如果全部卖出，删除持仓记录。
        """
        today = trade_date or datetime.now().strftime("%Y-%m-%d")

        existing = storage.get_position(symbol)
        if not existing or existing["qty"] <= 0:
            raise ValueError(f"没有 {symbol} 的持仓，无法卖出")

        remaining = existing["qty"] - qty
        if remaining < 0:
            raise ValueError(f"{symbol} 持仓不足: 持有{existing['qty']}股, 尝试卖出{qty}股")

        if remaining == 0:
            storage.upsert_position(symbol, self.market, 0, 0.0)
        else:
            storage.upsert_position(symbol, self.market, remaining, existing["avg_cost"])

        tid = storage.record_trade(symbol, self.market, today, "SELL",
                                    qty, price, commission, reason)
        return tid

    # ================================================================
    #  每日快照
    # ================================================================
    def snapshot(self, date: str, close_prices: Dict[str, float],
                 daily_return: float = 0.0):
        """
        记录每日权益快照。

        参数
        ----
        date : str
            日期 YYYY-MM-DD
        close_prices : dict
            symbol → 收盘价
        daily_return : float
            当日收益率（相对前一日）
        """
        state = self.load()
        holdings_val = 0.0
        for sym, pos in state.positions.items():
            if sym in close_prices and pos["qty"] > 0:
                holdings_val += pos["qty"] * close_prices[sym]

        storage.log_equity(date, state.cash, holdings_val, daily_return)
        storage.set_config("last_date", date)

    # ================================================================
    #  汇总 & 检查
    # ================================================================
    def get_summary(self, close_prices: Optional[Dict[str, float]] = None) -> Dict:
        """获取当前账户汇总。"""
        state = self.load()
        holdings_val = 0.0
        positions_detail = []

        for sym, pos in state.positions.items():
            price = close_prices.get(sym, pos["avg_cost"]) if close_prices else pos["avg_cost"]
            market_val = pos["qty"] * price
            pnl = (price - pos["avg_cost"]) * pos["qty"]
            holdings_val += market_val
            positions_detail.append({
                "symbol": sym,
                "market": pos["market"],
                "qty": pos["qty"],
                "avg_cost": pos["avg_cost"],
                "current_price": price,
                "market_value": market_val,
                "unrealized_pnl": pnl,
                "pnl_pct": (price / pos["avg_cost"] - 1) * 100 if pos["avg_cost"] > 0 else 0,
            })

        return {
            "cash": state.cash,
            "holdings_value": holdings_val,
            "total_equity": state.cash + holdings_val,
            "initial_capital": state.initial_capital,
            "total_return_pct": (state.cash + holdings_val) / state.initial_capital - 1,
            "positions": positions_detail,
            "position_count": len(positions_detail),
        }

    def can_buy(self, symbol: str, qty: int, price: float,
                commission: float = 0.0) -> Tuple[bool, str]:
        """检查是否可以买入。"""
        state = self.load()
        cost = qty * price + commission
        if cost > state.cash:
            return False, f"现金不足 (需要 {cost:.0f}, 可用 {state.cash:.0f})"

        # 单票仓位上限 50%
        after_buy_val = sum(
            p["qty"] * p["avg_cost"] for p in state.positions.values()
        ) + cost
        if after_buy_val > state.cash * 2:  # 杠杆检查 (简化)
            return False, "仓位超限"

        return True, "OK"

    def can_sell(self, symbol: str, qty: int) -> Tuple[bool, str]:
        """检查是否可以卖出。"""
        existing = storage.get_position(symbol)
        if not existing or existing["qty"] < qty:
            return False, f"持仓不足 (持有{existing['qty'] if existing else 0}股)"
        return True, "OK"
