"""
风控模块 — 止损 / 仓位管理 / 预交易检查

用法:
  from risk_manager import RiskManager
  
  rm = RiskManager(max_position_pct=0.25, max_total_exposure=0.95)
  
  # 每日更新止损
  rm.update_stops(position_entry, cp_today)
  
  # 预交易检查
  decision = rm.check(decision, position_entry, portfolio_state, cp_today)
"""

import numpy as np
from typing import Dict, List, Any, Tuple


class RiskManager:
    """量化风控管理器 — 止损追踪 + 仓位控制 + 熔断机制。"""

    def __init__(self,
                 max_position_pct: float = 0.25,     # 单票最大仓位
                 max_total_exposure: float = 0.95,    # 最大总敞口
                 max_daily_loss_pct: float = 0.05,    # 日内亏损熔断 (5%)
                 max_drawdown_pct: float = 0.25,      # 回撤熔断 (25%)
                 stop_loss_pct: float = 0.08,         # 默认止损 8%
                 trail_stop_pct: float = 0.05,        # 追踪止损 5%
                 take_profit_pct: float = 0.30,       # 止盈 30%
                 atr_stop_multiple: float = 2.0,      # ATR止损倍数
                 max_positions: int = 6,              # 最大持仓数
                 ):
        self.max_position_pct = max_position_pct
        self.max_total_exposure = max_total_exposure
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.stop_loss_pct = stop_loss_pct
        self.trail_stop_pct = trail_stop_pct
        self.take_profit_pct = take_profit_pct
        self.atr_stop_multiple = atr_stop_multiple
        self.max_positions = max_positions

        # 状态
        self.day_start_equity = None
        self.peak_equity = None
        self.halted = False
        self.halt_reason = ""

    def init_day(self, current_equity: float):
        """每日开盘时初始化。"""
        if self.day_start_equity is None:
            self.day_start_equity = current_equity
        if self.peak_equity is None or current_equity > self.peak_equity:
            self.peak_equity = current_equity

    # ════════════════════════════════════
    #  止损追踪
    # ════════════════════════════════════

    def init_stop(self, symbol: str, entry_price: float, entry_date: str,
                  qty: int, close_price: float, high: float, low: float,
                  atr: float = None) -> dict:
        """
        初始化止损参数。

        Returns:
          更新后的 position_entry 字典
        """
        entry = {
            "entry_price": entry_price,
            "entry_date": entry_date,
            "qty": qty,
            "stop_price": entry_price * (1 - self.stop_loss_pct),
            "stop_type": "fixed",
            "trail_high": high if high > entry_price else entry_price,
            "trail_pct": self.trail_stop_pct,
            "take_profit_price": entry_price * (1 + self.take_profit_pct),
            "atr_entry_atr": atr,
            "atr_multiple": self.atr_stop_multiple,
        }
        if atr is not None and atr > 0:
            entry["stop_price"] = entry_price - atr * self.atr_stop_multiple
            entry["stop_type"] = "atr"
        return entry

    def update_stops(self, position_entry: dict, cp_today: dict,
                     highs: dict = None, lows: dict = None) -> List[str]:
        """
        更新所有持仓的止损价, 返回触发止损的股票列表。

        Args:
          position_entry: {symbol: {entry_price, stop_price, trail_high, ...}}
          cp_today: {symbol: close_price}
          highs: {symbol: high_price} for ATR-style trailing
          lows: {symbol: low_price}

        Returns:
          list of symbols that triggered stop-loss/take-profit
        """
        triggered = []
        highs = highs or {}
        lows = lows or {}

        for sym, entry in list(position_entry.items()):
            if sym not in cp_today:
                continue
            px = cp_today[sym]

            # --- 追踪止损: 随价格上涨上移 ---
            if entry.get("stop_type") in ("trail", "fixed"):
                # 更新最高价
                hi = highs.get(sym, px)
                if hi > entry.get("trail_high", entry["entry_price"]):
                    entry["trail_high"] = hi
                    # 上移止损
                    new_stop = hi * (1 - entry.get("trail_pct", self.trail_stop_pct))
                    if new_stop > entry.get("stop_price", 0):
                        entry["stop_price"] = new_stop
                        entry["stop_type"] = "trail"  # 转为追踪模式

            # --- ATR止损: 最高价 - N*ATR ---
            if entry.get("stop_type") == "atr":
                hi = highs.get(sym, px)
                atr = entry.get("atr_entry_atr", 0)
                if hi > entry.get("trail_high", entry["entry_price"]):
                    entry["trail_high"] = hi
                    new_stop = hi - atr * entry.get("atr_multiple", self.atr_stop_multiple)
                    if new_stop > entry.get("stop_price", 0):
                        entry["stop_price"] = new_stop

            # --- 检查触发 ---
            if px <= entry.get("stop_price", 0):
                triggered.append(sym)
                continue

            tp = entry.get("take_profit_price")
            if tp and px >= tp:
                triggered.append(sym)

        return triggered

    # ════════════════════════════════════
    #  仓位计算
    # ════════════════════════════════════

    @staticmethod
    def kelly_size(win_rate: float, avg_win: float, avg_loss: float,
                   equity: float, fraction: float = 0.5) -> float:
        """
        Kelly公式仓位计算 (默认半凯利)。

        Args:
          fraction: 凯利分数 (0.5 = 半凯利)
        """
        if avg_loss <= 0 or win_rate <= 0:
            return 0.0
        kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / (avg_win * avg_loss)
        kelly_pct = max(0.0, kelly_pct)
        return equity * kelly_pct * fraction

    @staticmethod
    def atr_size(equity: float, risk_per_trade: float,
                 atr: float, atr_multiple: float = 2.0) -> int:
        """
        ATR仓位: 每笔风险 = risk_per_trade% × 权益
        
        qty = (equity * risk_pct) / (atr * atr_multiple)
        """
        if atr <= 0:
            return 0
        risk_amount = equity * risk_per_trade
        return int(risk_amount / (atr * atr_multiple) / 100) * 100  # A股百股整数

    @staticmethod
    def equal_risk_size(equity: float, n_positions: int, close_price: float) -> int:
        """等风险: 总仓位等分到N个标的。"""
        if n_positions <= 0 or close_price <= 0:
            return 0
        per_position = equity * 0.9 / n_positions
        return int(per_position / close_price / 100) * 100

    # ════════════════════════════════════
    #  预交易检查
    # ════════════════════════════════════

    def check(self, decision: dict, position_entry: dict,
              portfolio_state: Any, cp_today: dict,
              highs: dict = None, lows: dict = None) -> dict:
        """
        预交易检查: 过滤不合规的买卖。

        Args:
          decision: {"buy": [...], "sell": [...], "hold": [...]}
          position_entry: 当前持仓
          portfolio_state: pm.load() 返回的状态
          cp_today: {symbol: close_price}
          highs, lows: 日内高低价

        Returns:
          过滤后的 decision
        """
        # ── 1. 熔断检查 ──
        total_equity = getattr(portfolio_state, 'total_equity',
                               getattr(portfolio_state, 'cash', 100000))
        self.init_day(total_equity)

        if self.halted:
            return {"buy": [], "sell": list(position_entry.keys()), "hold": []}

        # 日内亏损熔断
        if self.day_start_equity and total_equity > 0:
            daily_pnl = (total_equity - self.day_start_equity) / self.day_start_equity
            if daily_pnl < -self.max_daily_loss_pct:
                self.halted = True
                self.halt_reason = f"日内亏损熔断 ({daily_pnl*100:.1f}%)"
                return {"buy": [], "sell": list(position_entry.keys()), "hold": []}

        # 回撤熔断
        if self.peak_equity and total_equity > 0:
            dd = (total_equity - self.peak_equity) / self.peak_equity
            if dd < -self.max_drawdown_pct:
                self.halted = True
                self.halt_reason = f"回撤熔断 ({dd*100:.1f}%)"
                return {"buy": [], "sell": list(position_entry.keys()), "hold": []}

        # ── 2. 止损触发 ──
        stop_triggers = self.update_stops(position_entry, cp_today, highs, lows)
        sells = list(decision.get("sell", []))
        for s in stop_triggers:
            if s not in sells:
                sells.append(s)

        # ── 3. 持仓数限制 ──
        current_positions = len([p for p in position_entry if p not in sells])
        max_new = max(0, self.max_positions - current_positions)
        buys = list(decision.get("buy", []))[:max_new]

        # ── 4. 单票仓位限制 ──
        filtered_buys = []
        for s in buys:
            if s not in cp_today:
                continue
            proposed_value = cp_today[s] * 100  # 估算
            if total_equity > 0 and proposed_value / total_equity <= self.max_position_pct:
                filtered_buys.append(s)

        return {"buy": filtered_buys, "sell": sells,
                "hold": decision.get("hold", [])}

    # ════════════════════════════════════
    #  统计
    # ════════════════════════════════════

    def get_stats(self) -> dict:
        """获取风控统计。"""
        return {
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "peak_equity": self.peak_equity,
            "day_start_equity": self.day_start_equity,
        }
