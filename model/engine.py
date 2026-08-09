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

    def __init__(self, initial_capital=100000, top_k=5, lot_size=100,
                 slippage_bps=0, turnover_limit_pct=1.0):
        self.initial = initial_capital
        self.cash = float(initial_capital)
        self.positions = {}  # {symbol: {"qty": int, "entry_price": float, "entry_date": str}}
        self.top_k = top_k
        self.lot_size = lot_size
        self.slippage_bps = slippage_bps        # 滑点 (bps), 小盘建议30-50
        self.turnover_limit = turnover_limit_pct  # 月单边换手上限, 0.5=50%
        self._monthly_buy = {}   # {YYYY-MM: amount}
        self._monthly_sell = {}  # {YYYY-MM: amount}
        self._month_capital = initial_capital  # 月初净值, 用于计算换手率

    @property
    def total_equity(self):
        """用给定的收盘价计算权益（需要外部传入cp_today）。"""
        return self.cash  # 实际权益由外部按收盘价计算

    def _get_position_value(self, s, today, all_data):
        """计算持仓市值"""
        if s not in self.positions or s not in all_data:
            return 0.0
        dt = all_data[s][all_data[s]["date"] <= today].tail(1)
        if len(dt) == 0:
            return 0.0
        px = float(dt["close"].iloc[-1]) if "close" in dt.columns else float(dt["open"].iloc[-1])
        return self.positions[s]["qty"] * px

    def _apply_slippage(self, price, side):
        """滑点: 买入价上浮, 卖出价下沉"""
        if self.slippage_bps <= 0:
            return price
        slip = price * self.slippage_bps / 10000.0
        return price + slip if side == "BUY" else price - slip

    def execute(self, decision: dict, today, all_data, rules,
                unadjusted_data: dict = None):
        """
        执行买卖决定。T+1开盘价成交。
        
        Args:
          all_data: 后复权数据 (用于成交价计算)
          unadjusted_data: 未复权数据 (用于涨跌停判断, 可选)
        """
        limit_data = unadjusted_data if unadjusted_data else all_data
        buys, sells = 0, 0
        trades = []

        # ── 月初重置换手追踪 ──
        month_key = today.strftime("%Y-%m")
        if month_key not in self._monthly_buy:
            self._monthly_buy[month_key] = 0.0
            self._monthly_sell[month_key] = 0.0
            self._month_capital = self.cash + sum(
                self._get_position_value(s, today, all_data) for s in self.positions)

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
            px = self._apply_slippage(px, "SELL")  # ★ 卖出滑点
            comm = calc_sell_commission(qty, px)
            proceeds = qty * px - comm
            self.cash += proceeds
            self._monthly_sell[month_key] += proceeds  # ★ 换手追踪
            del self.positions[s]
            sells += 1
            trades.append({"date": str(today.date()), "symbol": s, "action": "SELL",
                           "price": px, "qty": qty, "commission": comm,
                           "proceeds": proceeds, "cash_after": self.cash})

        # ── 买 ──
        buy_list = decision.get("buy", [])
        if buy_list:
            weights = decision.get("weights")  # {sym: 目标权重} (可选, v9b 组合优化)
            wsum = 0.0
            if weights:
                wsum = sum(weights.get(s, 0.0) for s in buy_list)
            # ★ 波动率目标仓位 (P0): 可选总仓位缩放 (无 cash_scale 时恒为 1.0,
            # 行为与原来完全一致)
            cash_pool = self.cash * 0.99 * float(decision.get("cash_scale", 1.0))
            for s in buy_list:
                if s not in all_data:
                    continue
                if weights and wsum > 0:
                    cash_per = cash_pool * (weights.get(s, 0.0) / wsum)
                else:
                    cash_per = cash_pool / max(1, len(buy_list))  # 等权兜底
                if cash_per <= 0:
                    continue
                dt = all_data[s][all_data[s]["date"] <= today].tail(1)
                if len(dt) == 0:
                    continue
                px = float(dt["open"].iloc[-1]) if "open" in dt.columns else float(dt["close"].iloc[-1])
                px = self._apply_slippage(px, "BUY")  # ★ 买入滑点
                # ★ 涨停检查: 优先用未复权价, 无则后退后复权
                if s in limit_data:
                    dt2_limit = limit_data[s][limit_data[s]["date"] <= today].tail(2)
                else:
                    dt2_limit = all_data[s][all_data[s]["date"] <= today].tail(2)
                if len(dt2_limit) >= 2 and not rules.can_buy(s, dt2_limit):
                    continue
                qty = int(cash_per / px / self.lot_size) * self.lot_size
                if qty < self.lot_size:
                    continue
                cost = qty * px
                # ★ 月换手上限: 单边买入不能超过月初净值×limit
                mcap = max(self._month_capital, self.initial)
                if self._monthly_buy[month_key] + cost > mcap * self.turnover_limit:
                    continue
                if cost > self.cash * 0.99:  # 留1%余地
                    qty = int(self.cash * 0.99 / px / self.lot_size) * self.lot_size
                    cost = qty * px
                if qty < self.lot_size:
                    continue
                comm = calc_buy_commission(qty, px)
                total_cost = cost + comm
                if total_cost > self.cash:
                    continue
                self.cash -= total_cost
                self._monthly_buy[month_key] += cost  # ★ 换手追踪
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
