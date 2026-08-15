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
                 slippage_bps=0, turnover_limit_pct=1.0,
                 execution_price="open", vwap_residual_bps=0,
                 vwap_panel: dict = None):
        self.initial = initial_capital
        self.cash = float(initial_capital)
        self.positions = {}  # {symbol: {"qty": int, "entry_price": float, "entry_date": str}}
        self.top_k = top_k
        self.lot_size = lot_size
        self.slippage_bps = slippage_bps        # 滑点 (bps), 小盘建议30-50
        self.turnover_limit = turnover_limit_pct  # 月单边换手上限, 0.5=50%
        # ★ 执行价模式 (方案B v24, 2026-08-11):
        #   open = 次日开盘价 (原逻辑); vwap = 次日 VWAP (拆单执行模拟)
        self.execution_price = execution_price
        # VWAP 残差滑点 (bps): 真实拆单无法完美命中 VWAP, 默认 10bps
        self.vwap_residual_bps = vwap_residual_bps
        # vwap_panel: {symbol: DataFrame(date 索引 × vwap 列)}, 懒加载由外部注入
        self.vwap_panel = vwap_panel or {}
        self._monthly_buy = {}   # {YYYY-MM: amount}
        self._monthly_sell = {}  # {YYYY-MM: amount}
        self._month_capital = initial_capital  # 月初净值, 用于计算换手率

    def _exec_price(self, s, today, all_data, order_qty: float = 0.0,
                    fill_times: list | None = None):
        """成交价: open=次日开盘; vwap=次日VWAP; pov=成交量比例拆单 (POV)。

        残差滑点由 _apply_slippage 承担 (slippage_bps=vwap_residual_bps,
        买上浮/卖下沉, 方向正确); 此处只返回基准价。
        pov 模式: 用 5m 分钟数据按市场成交量节奏拆单 (2022+ 数据可用;
        无分钟数据 → 回退 vwap → open)。
        fill_times: 可选 list, POV 成交后写入逐时段时间 ["09:35", ...]
        """
        if s not in all_data:
            return None
        dt = all_data[s][all_data[s]["date"] <= today].tail(1)
        if len(dt) == 0:
            return None
        px = float(dt["open"].iloc[-1]) if "open" in dt.columns else float(dt["close"].iloc[-1])
        if self.execution_price == "pov":
            # POV: 需要订单量 (买入时由调用方先估 qty, 卖出用持仓数量)
            pov_px = self._pov_price(s, today, order_qty, fill_times=fill_times)
            if pov_px is not None and pov_px > 0:
                return float(pov_px)
            # fallthrough: POV 无分钟数据 → 回退 vwap
        if self.execution_price in ("vwap", "pov"):
            vf = self.vwap_panel.get(s)
            if vf is not None and len(vf) > 0:
                t = pd.Timestamp(today)
                sub = vf[vf.index <= t]
                if len(sub) > 0 and np.isfinite(sub["vwap"].iloc[-1]):
                    px = float(sub["vwap"].iloc[-1])
                    if px <= 0:
                        px = float(dt["open"].iloc[-1]) if "open" in dt.columns else px
        return px

    def _pov_price(self, s, today, order_qty: float, fill_times: list | None = None):
        """POV 拆单成交价 (5m 分钟数据, 2022+ 可用; 无数据返回 None)。"""
        if order_qty <= 0:
            return None
        try:
            from data.minute_fetcher import MinuteFetcher
            # 回测只用本地 minute_5m 全历史 (allow_network=False):
            # 2022 前本地无数据 → None → 上层回退 VWAP/开盘, 不碰网络。
            mf = MinuteFetcher(allow_network=False)
            date_str = str(pd.Timestamp(today).date())
            res = mf.get_pov_fills(s, date_str, order_qty)
            if res is None:
                return None
            if fill_times is not None:
                fill_times.extend(f["time"] for f in res["fills"])
            return res["price"]
        except Exception:
            return None

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
            fill_times: list = []
            px = self._exec_price(s, today, all_data, order_qty=qty,
                                  fill_times=fill_times)
            if px is None:
                continue
            px = self._apply_slippage(px, "SELL")  # ★ 卖出滑点
            comm = calc_sell_commission(qty, px)
            proceeds = qty * px - comm
            self.cash += proceeds
            self._monthly_sell[month_key] += proceeds  # ★ 换手追踪
            del self.positions[s]
            sells += 1
            trades.append({"date": str(today.date()), "symbol": s, "action": "SELL",
                           "price": px, "qty": qty, "commission": comm,
                           "proceeds": proceeds, "cash_after": self.cash,
                           "fill_times": fill_times})

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
                # 先估临时价 (vwap/open) 确定买入数量, POV 需要订单量
                prov_px = self._exec_price(s, today, all_data)
                if prov_px is None:
                    continue
                prov_qty = int(cash_per / prov_px / self.lot_size) * self.lot_size
                if prov_qty < self.lot_size:
                    continue
                fill_times: list = []
                px = self._exec_price(s, today, all_data, order_qty=prov_qty,
                                      fill_times=fill_times)
                if px is None:
                    continue
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
                               "cost": total_cost, "cash_after": self.cash,
                               "fill_times": fill_times})

        # 限制持仓数
        while len(self.positions) > self.top_k:
            oldest = min(self.positions.keys(), key=lambda s: self.positions[s].get("entry_date", ""))
            t = self._force_sell(oldest, today, all_data)
            if t:
                sells += 1
                trades.append(t)

        return buys, sells, trades

    def _force_sell(self, symbol, today, all_data):
        """强制卖出最老的持仓 (记录逐笔, v24c 2026-08-12: 补录 trades)。"""
        pos = self.positions.get(symbol)
        if pos is None:
            return None
        qty = pos["qty"]
        fill_times: list = []
        px = self._exec_price(symbol, today, all_data, order_qty=qty,
                              fill_times=fill_times)
        if px is None:
            return None
        comm = calc_sell_commission(qty, px)
        proceeds = qty * px - comm
        self.cash += proceeds
        del self.positions[symbol]
        return {"date": str(today.date()), "symbol": symbol, "action": "SELL",
                "price": px, "qty": qty, "commission": comm,
                "proceeds": proceeds, "cash_after": self.cash,
                "fill_times": fill_times}

    def mark_to_market(self, close_prices: dict):
        """按收盘价计算总权益。"""
        holdings_val = sum(close_prices.get(s, 0) * p["qty"]
                          for s, p in self.positions.items())
        return self.cash + holdings_val
