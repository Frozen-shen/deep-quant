"""
模拟盘执行引擎 — 持久化版的 SimpleBacktest

将 model/engine.py 的 SimpleBacktest 执行逻辑与 storage.py 的 SQLite 持久化打通,
实现跨日持仓追踪。

核心差异 vs SimpleBacktest:
  - positions 存储在 SQLite 而非内存 dict
  - cash 从交易历史反算 (而非内存变量)
  - 每笔成交记录到 trades 表
  - 每日收盘记录到 equity_log 表
  - 完整的 ExecutionReport (信号→成交偏差追踪)

用法:
  from execution.paper_executor import PaperExecutor

  executor = PaperExecutor()
  state = executor.load_state()                       # 加载昨日持仓+现金

  report = executor.execute_orders(
      buy_list=["600519", "000858"],
      sell_list=["002594"],
      today=pd.Timestamp("2026-08-03"),
  )

  executor.snapshot("2026-08-03", close_prices={...})  # 收盘快照
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import storage
from trading_rules import (
    TradingRules, calc_buy_commission, calc_sell_commission
)


class RejectReason(Enum):
    """订单被拒原因。"""
    LIMIT_UP = "limit_up"              # 涨停
    LIMIT_DOWN = "limit_down"          # 跌停
    ONE_WORD_BOARD = "one_word_board"  # 一字板
    SUSPENDED = "suspended"            # 停牌
    INSUFFICIENT_CASH = "insufficient_cash"
    LOT_SIZE = "lot_size"              # 不足一手
    POSITION_LIMIT = "position_limit"  # 单票仓位上限
    LIQUIDITY_LOW = "liquidity_low"    # 流动性不足
    NO_DATA = "no_data"                # 无行情数据
    TURNOVER_LIMIT = "turnover_limit"  # 月换手超限


@dataclass
class OrderResult:
    """单笔订单结果。"""
    symbol: str
    action: str            # BUY or SELL
    status: str            # filled / rejected
    reject_reason: Optional[str] = None
    qty: int = 0
    price: float = 0.0
    commission: float = 0.0
    amount: float = 0.0    # 成交金额 (含手续费)


@dataclass
class ExecutionReport:
    """执行报告。"""
    date: str
    buy_signals: List[str] = field(default_factory=list)
    sell_signals: List[str] = field(default_factory=list)
    buy_filled: List[OrderResult] = field(default_factory=list)
    sell_filled: List[OrderResult] = field(default_factory=list)
    buy_rejected: List[OrderResult] = field(default_factory=list)
    sell_rejected: List[OrderResult] = field(default_factory=list)
    cash_before: float = 0.0
    cash_after: float = 0.0
    equity_before: float = 0.0
    equity_after: float = 0.0

    @property
    def fill_rate_buy(self) -> float:
        if not self.buy_signals:
            return 1.0
        return len(self.buy_filled) / len(self.buy_signals)

    @property
    def fill_rate_sell(self) -> float:
        if not self.sell_signals:
            return 1.0
        return len(self.sell_filled) / len(self.sell_signals)

    @property
    def total_commission(self) -> float:
        return sum(o.commission for o in self.buy_filled + self.sell_filled)

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "buy_signals": self.buy_signals,
            "sell_signals": self.sell_signals,
            "buy_filled": len(self.buy_filled),
            "sell_filled": len(self.sell_filled),
            "buy_rejected": [{"symbol": r.symbol, "reason": r.reject_reason}
                            for r in self.buy_rejected],
            "sell_rejected": [{"symbol": r.symbol, "reason": r.reject_reason}
                             for r in self.sell_rejected],
            "fill_rate_buy": round(self.fill_rate_buy, 4),
            "fill_rate_sell": round(self.fill_rate_sell, 4),
            "total_commission": round(self.total_commission, 2),
            "cash_before": round(self.cash_before, 2),
            "cash_after": round(self.cash_after, 2),
            "equity_before": round(self.equity_before, 2),
            "equity_after": round(self.equity_after, 2),
        }


@dataclass
class PaperState:
    """模拟盘状态快照。"""
    cash: float
    positions: Dict[str, dict]   # {symbol: {qty, avg_cost, market}}
    initial_capital: float
    last_date: str
    total_trades: int = 0


def _load_data_for_symbols(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """
    从 parquet 缓存加载指定股票的行情数据。

    Returns:
      {symbol: DataFrame with columns [date, open, high, low, close, volume, ...]}
    """
    from data_cache import load as load_single
    result = {}
    for sym in symbols:
        df = load_single(sym)
        if df is not None and len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            result[sym] = df
    return result


def _load_unadjusted_for_symbols(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """加载未复权数据 (用于涨跌停判断)。"""
    unadj_dir = os.path.join(BASE_DIR, "data_cache", "unadjusted")
    result = {}
    for sym in symbols:
        path = os.path.join(unadj_dir, f"{sym}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            result[sym] = df
    return result


def get_lot_size_for_market(market: str = "a") -> int:
    """获取每手股数。"""
    if market == "hk":
        return 200
    return 100


class PaperExecutor:
    """
    模拟盘执行引擎 — 持久化版。

    用法:
      executor = PaperExecutor()
      state = executor.load_state()
      report = executor.execute_orders(buy_list, sell_list, today)
      executor.snapshot(date_str, close_prices)
    """

    def __init__(self, market: str = "a", initial_capital: float = None,
                 top_k: int = 30, lot_size: int = 100,
                 slippage_bps: int = 30, turnover_limit_pct: float = 0.5,
                 max_single_pct: float = 0.25):
        """
        Args:
          market: 'a' or 'hk'
          initial_capital: 初始资金 (None=从 config 读取或默认100000)
          top_k: 最大持仓数量
          lot_size: 每手股数
          slippage_bps: 滑点 (bp)
          turnover_limit_pct: 月单边换手上限
          max_single_pct: 单票最大仓位比例
        """
        self.market = market
        self.top_k = top_k
        self.lot_size = lot_size
        self.slippage_bps = slippage_bps
        self.turnover_limit_pct = turnover_limit_pct
        self.max_single_pct = max_single_pct

        storage.init_db()

        # 初始化账户
        if initial_capital is None:
            saved = storage.get_config("initial_capital")
            if saved:
                initial_capital = float(saved)
            else:
                initial_capital = 100_000.0
        self.initial_capital = initial_capital

        if not storage.get_config("initial_capital"):
            storage.set_config("initial_capital", str(initial_capital))
            storage.set_config("market", market)
            storage.set_config("paper_start_date", "")

        # 交易规则
        self.rules = TradingRules()

        # 月换手追踪
        self._monthly_buy: Dict[str, float] = {}
        self._monthly_sell: Dict[str, float] = {}
        self._month_capital: float = initial_capital

    # ════════════════════════════════════════
    #  状态加载
    # ════════════════════════════════════════

    def load_state(self) -> PaperState:
        """从数据库加载当前持仓和现金。"""
        positions_raw = storage.get_all_positions()
        positions = {}
        for p in positions_raw:
            positions[p["symbol"]] = {
                "qty": p["qty"],
                "avg_cost": p["avg_cost"],
                "market": p["market"],
            }

        # 从交易历史反算现金
        cash = self.initial_capital
        trades = storage.get_trades(limit=99999)
        total_trades = len(trades)
        for t in trades:
            if t["action"] == "BUY":
                cash -= t["qty"] * t["price"] + t["commission"]
            elif t["action"] == "SELL":
                cash += t["qty"] * t["price"] - t["commission"]

        # 加载月换手追踪 (从本月已执行的交易)
        today = datetime.now()
        month_key = today.strftime("%Y-%m")
        month_trades = [t for t in trades if t["date"].startswith(month_key)]
        self._monthly_buy[month_key] = sum(
            t["qty"] * t["price"] for t in month_trades if t["action"] == "BUY")
        self._monthly_sell[month_key] = sum(
            t["qty"] * t["price"] for t in month_trades if t["action"] == "SELL")

        # 月初净值 (用上月末权益近似)
        last_equity = storage.get_equity_log(limit=1)
        if last_equity:
            self._month_capital = last_equity[0]["total_equity"]
        else:
            self._month_capital = self.initial_capital

        last_date = storage.get_config("last_date", "")

        return PaperState(
            cash=cash,
            positions=positions,
            initial_capital=self.initial_capital,
            last_date=last_date,
            total_trades=total_trades,
        )

    # ════════════════════════════════════════
    #  执行
    # ════════════════════════════════════════

    def execute_orders(self, buy_list: List[str], sell_list: List[str],
                      today, all_data: dict = None,
                      unadjusted_data: dict = None,
                      close_prices: dict = None) -> ExecutionReport:
        """
        执行买卖订单。T+1 开盘价成交。

        Args:
          buy_list: 买入信号股票列表
          sell_list: 卖出信号股票列表
          today: 交易日期 pd.Timestamp
          all_data: 后复权行情数据 {symbol: DataFrame}
          unadjusted_data: 未复权行情数据 (用于涨跌停判断)
          close_prices: 收盘价 (用于计算权益), None=从 all_data 获取

        Returns:
          ExecutionReport
        """
        date_str = str(today.date())
        state = self.load_state()

        # 加载行情 (如果未提供)
        if all_data is None:
            all_data = _load_data_for_symbols(
                list(set(buy_list + sell_list + list(state.positions.keys()))))
        if unadjusted_data is None:
            unadjusted_data = _load_unadjusted_for_symbols(
                list(set(buy_list + sell_list + list(state.positions.keys()))))

        limit_data = unadjusted_data if unadjusted_data else all_data
        month_key = today.strftime("%Y-%m")

        if month_key not in self._monthly_buy:
            self._monthly_buy[month_key] = 0.0
            self._monthly_sell[month_key] = 0.0
            # 月初净值: 用当前总权益
            if close_prices:
                holdings_val = sum(
                    close_prices.get(s, state.positions[s]["avg_cost"]) * state.positions[s]["qty"]
                    for s in state.positions)
            else:
                holdings_val = sum(
                    p["qty"] * p["avg_cost"] for p in state.positions.values())
            self._month_capital = state.cash + holdings_val

        report = ExecutionReport(
            date=date_str,
            buy_signals=list(buy_list),
            sell_signals=list(sell_list),
            cash_before=state.cash,
        )

        # ── 先计算权益 (用于仓位检查) ──
        if close_prices is None:
            close_prices = {}
            for sym in all_data:
                dt = all_data[sym][all_data[sym]["date"] <= today].tail(1)
                if len(dt) > 0:
                    close_prices[sym] = float(dt["close"].iloc[-1])

        current_equity = state.cash + sum(
            close_prices.get(s, state.positions[s]["avg_cost"]) * state.positions[s]["qty"]
            for s in state.positions)

        report.equity_before = current_equity

        # ── 卖 ──
        for sym in sell_list:
            pos = state.positions.get(sym, {})
            qty = pos.get("qty", 0)
            if qty <= 0:
                report.sell_rejected.append(OrderResult(
                    sym, "SELL", "rejected", RejectReason.NO_DATA.value))
                continue
            if sym not in all_data:
                report.sell_rejected.append(OrderResult(
                    sym, "SELL", "rejected", RejectReason.NO_DATA.value))
                continue

            # 跌停/停牌检查
            dt_limit = limit_data.get(sym, all_data.get(sym))
            if dt_limit is not None:
                dt_check = dt_limit[dt_limit["date"] <= today].tail(2)
                if len(dt_check) >= 2 and not self.rules.can_sell(sym, dt_check):
                    reason = RejectReason.LIMIT_DOWN.value
                    if self.rules.is_suspended(sym, dt_check):
                        reason = RejectReason.SUSPENDED.value
                    report.sell_rejected.append(OrderResult(
                        sym, "SELL", "rejected", reason))
                    continue

            dt = all_data[sym][all_data[sym]["date"] <= today].tail(1)
            if len(dt) == 0:
                report.sell_rejected.append(OrderResult(
                    sym, "SELL", "rejected", RejectReason.NO_DATA.value))
                continue

            px = float(dt["open"].iloc[-1]) if "open" in dt.columns else float(dt["close"].iloc[-1])
            px = self._apply_slippage(px, "SELL")
            comm = calc_sell_commission(qty, px)
            proceeds = qty * px - comm

            state.cash += proceeds
            self._monthly_sell[month_key] += proceeds

            # 持久化: 清空持仓
            storage.upsert_position(sym, self.market, 0, 0.0)
            storage.record_trade(sym, self.market, date_str, "SELL",
                                qty, px, comm, "paper_signal")

            report.sell_filled.append(OrderResult(
                sym, "SELL", "filled", qty=qty, price=px,
                commission=comm, amount=proceeds))

        # ── 买 ──
        if buy_list:
            cash_per = state.cash * 0.99 / max(1, len(buy_list))

            for sym in buy_list:
                if sym not in all_data:
                    report.buy_rejected.append(OrderResult(
                        sym, "BUY", "rejected", RejectReason.NO_DATA.value))
                    continue

                dt = all_data[sym][all_data[sym]["date"] <= today].tail(1)
                if len(dt) == 0:
                    report.buy_rejected.append(OrderResult(
                        sym, "BUY", "rejected", RejectReason.NO_DATA.value))
                    continue

                px = float(dt["open"].iloc[-1]) if "open" in dt.columns else float(dt["close"].iloc[-1])
                px = self._apply_slippage(px, "BUY")

                # 涨停/停牌检查
                dt_limit_sym = limit_data.get(sym, all_data[sym])
                if dt_limit_sym is not None:
                    dt_check = dt_limit_sym[dt_limit_sym["date"] <= today].tail(2)
                    if len(dt_check) >= 2 and not self.rules.can_buy(sym, dt_check):
                        is_up, is_down, is_word = self.rules.is_limit_hit(sym, dt_check)
                        reason = RejectReason.LIMIT_UP.value if is_up else RejectReason.SUSPENDED.value
                        if is_word:
                            reason = RejectReason.ONE_WORD_BOARD.value
                        report.buy_rejected.append(OrderResult(
                            sym, "BUY", "rejected", reason))
                        continue

                # 计算买入数量 (手数对齐)
                qty = int(cash_per / px / self.lot_size) * self.lot_size
                if qty < self.lot_size:
                    report.buy_rejected.append(OrderResult(
                        sym, "BUY", "rejected", RejectReason.LOT_SIZE.value))
                    continue

                cost = qty * px

                # 单票仓位上限 (Phase 3.1)
                existing_market_val = 0.0
                if sym in state.positions:
                    existing_market_val = state.positions[sym]["qty"] * close_prices.get(sym, px)
                after_position_val = existing_market_val + cost
                if after_position_val > current_equity * self.max_single_pct:
                    # 缩减买入数量
                    max_allowed_cost = current_equity * self.max_single_pct - existing_market_val
                    if max_allowed_cost <= 0:
                        report.buy_rejected.append(OrderResult(
                            sym, "BUY", "rejected", RejectReason.POSITION_LIMIT.value))
                        continue
                    qty = int(max_allowed_cost / px / self.lot_size) * self.lot_size
                    if qty < self.lot_size:
                        report.buy_rejected.append(OrderResult(
                            sym, "BUY", "rejected", RejectReason.POSITION_LIMIT.value))
                        continue
                    cost = qty * px

                # 月换手上限
                mcap = max(self._month_capital, self.initial_capital)
                if self._monthly_buy.get(month_key, 0) + cost > mcap * self.turnover_limit_pct:
                    report.buy_rejected.append(OrderResult(
                        sym, "BUY", "rejected", RejectReason.TURNOVER_LIMIT.value))
                    continue

                if cost > state.cash * 0.99:
                    qty = int(state.cash * 0.99 / px / self.lot_size) * self.lot_size
                    if qty < self.lot_size:
                        report.buy_rejected.append(OrderResult(
                            sym, "BUY", "rejected", RejectReason.INSUFFICIENT_CASH.value))
                        continue
                    cost = qty * px

                comm = calc_buy_commission(qty, px)
                total_cost = cost + comm
                if total_cost > state.cash:
                    report.buy_rejected.append(OrderResult(
                        sym, "BUY", "rejected", RejectReason.INSUFFICIENT_CASH.value))
                    continue

                state.cash -= total_cost
                self._monthly_buy[month_key] = self._monthly_buy.get(month_key, 0) + cost

                # 持久化: 更新持仓
                existing = storage.get_position(sym)
                if existing and existing["qty"] > 0:
                    old_qty = existing["qty"]
                    old_cost = existing["avg_cost"]
                    new_qty = old_qty + qty
                    new_avg_cost = (old_qty * old_cost + qty * (px + comm / qty)) / new_qty
                    storage.upsert_position(sym, self.market, new_qty, new_avg_cost)
                else:
                    avg_cost = px + comm / qty if qty > 0 else px
                    storage.upsert_position(sym, self.market, qty, avg_cost)

                storage.record_trade(sym, self.market, date_str, "BUY",
                                    qty, px, comm, "paper_signal")

                report.buy_filled.append(OrderResult(
                    sym, "BUY", "filled", qty=qty, price=px,
                    commission=comm, amount=total_cost))

        # ── 持仓数量限制 (超出 top_k 时强制卖出最早) ──
        current_positions = storage.get_all_positions()
        while len(current_positions) > self.top_k:
            # 找最早买入的持仓
            oldest_trade = None
            for pos in current_positions:
                trades = storage.get_trades(symbol=pos["symbol"], limit=1)
                if trades:
                    if oldest_trade is None or trades[0]["date"] < oldest_trade["date"]:
                        oldest_trade = trades[0]

            if oldest_trade:
                sym = oldest_trade["symbol"]
                pos_info = storage.get_position(sym)
                if pos_info and pos_info["qty"] > 0:
                    # 强制卖出 (不检查涨跌停)
                    dt = all_data.get(sym)
                    if dt is not None:
                        dt_row = dt[dt["date"] <= today].tail(1)
                        if len(dt_row) > 0:
                            px = float(dt_row["open"].iloc[-1]) if "open" in dt_row.columns else float(dt_row["close"].iloc[-1])
                            qty = pos_info["qty"]
                            comm = calc_sell_commission(qty, px)
                            state.cash += qty * px - comm
                            storage.upsert_position(sym, self.market, 0, 0.0)
                            storage.record_trade(sym, self.market, date_str, "SELL",
                                                qty, px, comm, "position_limit_force_sell")
            current_positions = storage.get_all_positions()

        report.cash_after = state.cash

        # ── 计算执行后权益 ──
        after_holdings = sum(
            close_prices.get(s, 0) * storage.get_position(s)["qty"]
            for s in close_prices
            if storage.get_position(s) and storage.get_position(s)["qty"] > 0)
        report.equity_after = state.cash + after_holdings

        # ── 保存执行报告 ──
        self._save_report(report)

        return report

    # ════════════════════════════════════════
    #  快照
    # ════════════════════════════════════════

    def snapshot(self, date_str: str, close_prices: dict = None):
        """记录每日权益快照。"""
        state = self.load_state()

        if close_prices is None:
            # 从缓存加载收盘价
            symbols = list(state.positions.keys())
            all_data = _load_data_for_symbols(symbols)
            close_prices = {}
            for sym, df in all_data.items():
                dt = df[df["date"] <= pd.Timestamp(date_str)].tail(1)
                if len(dt) > 0:
                    close_prices[sym] = float(dt["close"].iloc[-1])

        holdings_val = 0.0
        for sym, pos in state.positions.items():
            px = close_prices.get(sym, pos["avg_cost"])
            holdings_val += pos["qty"] * px

        # 计算日收益率
        prev_log = storage.get_equity_log(limit=1)
        prev_equity = prev_log[0]["total_equity"] if prev_log else self.initial_capital
        current_equity = state.cash + holdings_val
        daily_return = (current_equity / prev_equity - 1) if prev_equity > 0 else 0.0

        storage.log_equity(date_str, state.cash, holdings_val, daily_return)
        storage.set_config("last_date", date_str)

    def mark_to_market(self, close_prices: dict) -> float:
        """按收盘价计算总权益。"""
        state = self.load_state()
        holdings_val = sum(
            close_prices.get(s, 0) * p["qty"]
            for s, p in state.positions.items())
        return state.cash + holdings_val

    # ════════════════════════════════════════
    #  工具方法
    # ════════════════════════════════════════

    def _apply_slippage(self, price: float, side: str) -> float:
        """滑点: 买入上浮, 卖出下沉。"""
        if self.slippage_bps <= 0:
            return price
        slip = price * self.slippage_bps / 10000.0
        return price + slip if side == "BUY" else price - slip

    def _save_report(self, report: ExecutionReport):
        """追加执行报告到 JSONL。"""
        report_path = os.path.join(BASE_DIR, "data", "paper_executions.jsonl")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(report.to_dict(), ensure_ascii=False) + "\n")

    def get_equity_curve(self) -> pd.DataFrame:
        """获取权益曲线。"""
        rows = storage.get_equity_log(limit=9999)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df["cumulative_return"] = df["total_equity"] / self.initial_capital - 1
        return df


# ════════════════════════════════════════
#  快速测试
# ════════════════════════════════════════

if __name__ == "__main__":
    print("PaperExecutor 模块加载成功")
    print(f"  storage.init_db()...")
    storage.init_db()
    executor = PaperExecutor()
    state = executor.load_state()
    print(f"  初始资金: {state.initial_capital:,.0f}")
    print(f"  当前现金: {state.cash:,.0f}")
    print(f"  持仓数量: {len(state.positions)}")
    print(f"  历史交易: {state.total_trades}")
    print(f"  最后日期: {state.last_date or '(无)'}")
