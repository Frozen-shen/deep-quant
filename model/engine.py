"""
简单透明回测引擎 — 替代 PortfolioManager 数据库层
纯 Python 变量，每步可验证。
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from trading_rules import calc_buy_commission, calc_sell_commission


class SimpleBacktest:
    """透明的回测引擎：cash + positions dict，无数据库依赖。"""

    def __init__(self, initial_capital=100000, top_k=5, lot_size=100):
        self.initial = initial_capital
        self.cash = float(initial_capital)
        self.positions = {}  # {symbol: {"qty": int, "entry_price": float, "entry_date": str}}
        self.top_k = top_k
        self.lot_size = lot_size

    @property
    def total_equity(self):
        """用给定的收盘价计算权益（需要外部传入cp_today）。"""
        return self.cash  # 实际权益由外部按收盘价计算

    def execute(self, decision: dict, today, all_data, rules):
        """
        执行买卖决定。T+1开盘价成交。
        返回: (buy_count, sell_count, trades_log)
        """
        buys, sells = 0, 0
        trades = []

        # ── 卖 ──
        for s in decision.get("sell", []):
            pos = self.positions.get(s, {})
            qty = pos.get("qty", 0)
            if qty <= 0 or s not in all_data:
                continue
            dt = all_data[s][all_data[s]["date"] <= today].tail(1)
            if len(dt) == 0:
                continue
            px = float(dt["open"].iloc[-1]) if "open" in dt.columns else float(dt["close"].iloc[-1])
            comm = calc_sell_commission(qty, px)
            proceeds = qty * px - comm
            self.cash += proceeds
            del self.positions[s]
            sells += 1
            trades.append({"date": str(today.date()), "symbol": s, "action": "SELL",
                           "price": px, "qty": qty, "commission": comm,
                           "proceeds": proceeds, "cash_after": self.cash})

        # ── 买 ──
        buy_list = decision.get("buy", [])
        if buy_list:
            cash_per = self.cash * 0.9 / max(1, len(buy_list))
            for s in buy_list:
                if s not in all_data:
                    continue
                dt = all_data[s][all_data[s]["date"] <= today].tail(1)
                if len(dt) == 0:
                    continue
                px = float(dt["open"].iloc[-1]) if "open" in dt.columns else float(dt["close"].iloc[-1])
                # ★ 涨跌停检查
                if not rules.can_buy(s, dt):
                    continue
                qty = int(cash_per / px / self.lot_size) * self.lot_size
                if qty < self.lot_size:
                    continue
                cost = qty * px
                if cost > self.cash * 0.95:  # 留5%余地
                    qty = int(self.cash * 0.95 / px / self.lot_size) * self.lot_size
                    cost = qty * px
                if qty < self.lot_size:
                    continue
                comm = calc_buy_commission(qty, px)
                total_cost = cost + comm
                if total_cost > self.cash:
                    continue
                self.cash -= total_cost
                self.positions[s] = {"qty": qty, "entry_price": px, "entry_date": str(today.date())}
                buys += 1
                trades.append({"date": str(today.date()), "symbol": s, "action": "BUY",
                               "price": px, "qty": qty, "commission": comm,
                               "cost": total_cost, "cash_after": self.cash})

        # 限制持仓数
        while len(self.positions) > self.top_k:
            oldest = min(self.positions.keys(), key=lambda s: self.positions[s].get("entry_date", ""))
            self._force_sell(oldest, today, all_data)

        return buys, sells, trades

    def _force_sell(self, symbol, today, all_data):
        """强制卖出最老的持仓。"""
        pos = self.positions.get(symbol)
        if pos is None:
            return
        dt = all_data[symbol][all_data[symbol]["date"] <= today].tail(1)
        if len(dt) == 0:
            return
        px = float(dt["open"].iloc[-1]) if "open" in dt.columns else float(dt["close"].iloc[-1])
        qty = pos["qty"]
        comm = calc_sell_commission(qty, px)
        self.cash += qty * px - comm
        del self.positions[symbol]

    def mark_to_market(self, close_prices: dict):
        """按收盘价计算总权益。"""
        holdings_val = sum(close_prices.get(s, 0) * p["qty"]
                          for s, p in self.positions.items())
        return self.cash + holdings_val
