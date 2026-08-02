"""
Paper trading simulation with realistic execution.

Simulates trade execution for paper trading, tracking:
    - Virtual portfolio (positions + cash)
    - Trade history
    - Daily P&L
    - Tracking error vs backtest

Key features:
    - Executes at next-day open (realistic: signal today, execute tomorrow)
    - Applies realistic costs (commission + slippage + stamp tax)
    - Handles limit-up (can't buy) and limit-down (can't sell)
    - Tracks deviation from theoretical backtest

Storage: paper_trade/portfolio.json (persisted between runs)

Usage:
    from quant.production.paper_trader import PaperTrader

    trader = PaperTrader(initial_capital=1_000_000)
    report = trader.execute_orders(orders, market_data)
    trader.mark_to_market(market_data)
    perf = trader.get_performance()
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_STATE_FILE = _PROJECT_ROOT / "paper_trade" / "portfolio.json"

# A-share cost parameters
COMMISSION_RATE = 0.00025     # 万2.5 per side
SLIPPAGE_RATE = 0.0005        # 万5 market impact
STAMP_TAX_RATE = 0.0005       # 0.05% sell-side only (post 2023-08-28)
TRANSFER_FEE_RATE = 0.00001   # 0.001% both sides
MIN_COMMISSION = 5.0          # minimum 5 yuan per trade

# Limit-up/down thresholds
LIMIT_UP_PCT = 0.10           # 10% for main board
LIMIT_DOWN_PCT = -0.10


@dataclass
class Position:
    """A single stock position."""
    symbol: str
    qty: int
    avg_cost: float
    entry_date: str
    market_value: float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "avg_cost": self.avg_cost,
            "entry_date": self.entry_date,
            "market_value": self.market_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(
            symbol=d["symbol"],
            qty=d["qty"],
            avg_cost=d["avg_cost"],
            entry_date=d.get("entry_date", ""),
            market_value=d.get("market_value", 0.0),
        )


@dataclass
class TradeRecord:
    """Record of a single executed trade."""
    date: str
    symbol: str
    action: str           # "BUY" or "SELL"
    qty: int
    price: float
    commission: float
    amount: float         # total cost (buy) or proceeds (sell)
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExecutionResult:
    """Result of a single order execution attempt."""
    symbol: str
    action: str
    status: str           # "filled", "rejected"
    reject_reason: str = ""
    qty: int = 0
    price: float = 0.0
    commission: float = 0.0
    amount: float = 0.0


class PaperTrader:
    """
    Simulates trade execution for paper trading.

    Maintains a virtual portfolio with cash and positions, executing
    orders at realistic prices with full cost modeling. State is
    persisted to a JSON file between runs.

    Args:
        initial_capital: Starting capital in yuan.
        state_file: Path to the portfolio state JSON file.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        state_file: str | Path = _DEFAULT_STATE_FILE,
    ):
        self.initial_capital = initial_capital
        self.state_file = Path(state_file)

        # Portfolio state
        self.cash: float = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeRecord] = []
        self.daily_equity: List[dict] = []  # [{date, equity, cash, holdings_value}]
        self.inception_date: str = ""

        # Load existing state if available
        self.load_state()

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def execute_orders(
        self, orders: List[dict], market_data: Dict[str, pd.DataFrame]
    ) -> List[ExecutionResult]:
        """
        Execute orders at today's prices. Returns execution report.

        Orders are executed at the open price of the execution date.
        Sells are processed before buys (to free up cash).

        Args:
            orders: List of order dicts with keys:
                action (BUY/SELL), symbol, weight, reason
            market_data: Dict of symbol -> DataFrame with OHLCV data.
                Must contain the execution date's data.

        Returns:
            List of ExecutionResult for each order.
        """
        results = []
        execution_date = datetime.now().strftime("%Y-%m-%d")

        if not self.inception_date:
            self.inception_date = execution_date

        # Separate sells and buys
        sells = [o for o in orders if o["action"] == "SELL"]
        buys = [o for o in orders if o["action"] == "BUY"]

        # Execute sells first (free up cash)
        for order in sells:
            result = self._execute_sell(order, market_data, execution_date)
            results.append(result)

        # Then execute buys
        for order in buys:
            result = self._execute_buy(order, market_data, execution_date)
            results.append(result)

        # Save state after execution
        self.save_state()

        filled = sum(1 for r in results if r.status == "filled")
        rejected = sum(1 for r in results if r.status == "rejected")
        logger.info(
            "Executed %d orders: %d filled, %d rejected",
            len(results), filled, rejected,
        )

        return results

    def _execute_sell(
        self, order: dict, market_data: Dict[str, pd.DataFrame], exec_date: str
    ) -> ExecutionResult:
        """Execute a sell order."""
        symbol = order["symbol"]

        # Check if we hold this position
        pos = self.positions.get(symbol)
        if pos is None or pos.qty <= 0:
            return ExecutionResult(
                symbol=symbol, action="SELL", status="rejected",
                reject_reason="no_position",
            )

        # Get execution price (open of execution date)
        price = self._get_execution_price(symbol, market_data, exec_date)
        if price is None:
            return ExecutionResult(
                symbol=symbol, action="SELL", status="rejected",
                reject_reason="no_price_data",
            )

        # Check limit-down (can't sell if limit-down)
        if self._is_limit_down(symbol, market_data, exec_date):
            return ExecutionResult(
                symbol=symbol, action="SELL", status="rejected",
                reject_reason="limit_down",
            )

        # Apply slippage (sell at slightly lower price)
        exec_price = price * (1 - SLIPPAGE_RATE)

        # Compute costs
        qty = pos.qty
        notional = qty * exec_price
        commission = max(notional * COMMISSION_RATE, MIN_COMMISSION)
        stamp_tax = notional * STAMP_TAX_RATE
        transfer_fee = notional * TRANSFER_FEE_RATE
        total_cost = commission + stamp_tax + transfer_fee

        proceeds = notional - total_cost

        # Update state
        self.cash += proceeds
        del self.positions[symbol]

        # Record trade
        self.trade_history.append(TradeRecord(
            date=exec_date,
            symbol=symbol,
            action="SELL",
            qty=qty,
            price=exec_price,
            commission=total_cost,
            amount=proceeds,
            reason=order.get("reason", ""),
        ))

        return ExecutionResult(
            symbol=symbol, action="SELL", status="filled",
            qty=qty, price=exec_price, commission=total_cost, amount=proceeds,
        )

    def _execute_buy(
        self, order: dict, market_data: Dict[str, pd.DataFrame], exec_date: str
    ) -> ExecutionResult:
        """Execute a buy order."""
        symbol = order["symbol"]
        target_weight = order.get("weight", 1.0 / 30)

        # Get execution price (open of execution date)
        price = self._get_execution_price(symbol, market_data, exec_date)
        if price is None:
            return ExecutionResult(
                symbol=symbol, action="BUY", status="rejected",
                reject_reason="no_price_data",
            )

        # Check limit-up (can't buy if limit-up)
        if self._is_limit_up(symbol, market_data, exec_date):
            return ExecutionResult(
                symbol=symbol, action="BUY", status="rejected",
                reject_reason="limit_up",
            )

        # Apply slippage (buy at slightly higher price)
        exec_price = price * (1 + SLIPPAGE_RATE)

        # Compute target amount based on portfolio value
        total_equity = self._compute_equity(market_data, exec_date)
        target_amount = total_equity * target_weight * 0.95  # 5% cash buffer

        # Cap at available cash
        target_amount = min(target_amount, self.cash * 0.95)

        # Compute quantity (round down to lot size of 100)
        qty = int(target_amount / exec_price / 100) * 100
        if qty < 100:
            return ExecutionResult(
                symbol=symbol, action="BUY", status="rejected",
                reject_reason="insufficient_cash",
            )

        # Compute costs
        notional = qty * exec_price
        commission = max(notional * COMMISSION_RATE, MIN_COMMISSION)
        transfer_fee = notional * TRANSFER_FEE_RATE
        total_cost = notional + commission + transfer_fee

        if total_cost > self.cash:
            # Reduce quantity to fit
            qty = int(self.cash * 0.99 / exec_price / 100) * 100
            if qty < 100:
                return ExecutionResult(
                    symbol=symbol, action="BUY", status="rejected",
                    reject_reason="insufficient_cash",
                )
            notional = qty * exec_price
            commission = max(notional * COMMISSION_RATE, MIN_COMMISSION)
            transfer_fee = notional * TRANSFER_FEE_RATE
            total_cost = notional + commission + transfer_fee

        # Update state
        self.cash -= total_cost

        if symbol in self.positions:
            # Add to existing position
            existing = self.positions[symbol]
            new_qty = existing.qty + qty
            new_avg_cost = (
                (existing.qty * existing.avg_cost + qty * exec_price) / new_qty
            )
            self.positions[symbol] = Position(
                symbol=symbol,
                qty=new_qty,
                avg_cost=new_avg_cost,
                entry_date=existing.entry_date,
            )
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                qty=qty,
                avg_cost=exec_price,
                entry_date=exec_date,
            )

        # Record trade
        self.trade_history.append(TradeRecord(
            date=exec_date,
            symbol=symbol,
            action="BUY",
            qty=qty,
            price=exec_price,
            commission=commission + transfer_fee,
            amount=total_cost,
            reason=order.get("reason", ""),
        ))

        return ExecutionResult(
            symbol=symbol, action="BUY", status="filled",
            qty=qty, price=exec_price,
            commission=commission + transfer_fee, amount=total_cost,
        )

    # ------------------------------------------------------------------
    # Mark to market
    # ------------------------------------------------------------------

    def mark_to_market(self, market_data: Dict[str, pd.DataFrame]) -> dict:
        """
        Update portfolio value with latest prices.

        Args:
            market_data: Symbol -> DataFrame with latest OHLCV data.

        Returns:
            Dict with equity summary:
            {date, total_equity, cash, holdings_value, daily_return, positions}
        """
        today = datetime.now().strftime("%Y-%m-%d")

        holdings_value = 0.0
        position_details = []

        for symbol, pos in self.positions.items():
            price = self._get_latest_close(symbol, market_data)
            if price is not None:
                mv = pos.qty * price
                pos.market_value = mv
                holdings_value += mv
                position_details.append({
                    "symbol": symbol,
                    "qty": pos.qty,
                    "avg_cost": round(pos.avg_cost, 3),
                    "current_price": round(price, 3),
                    "market_value": round(mv, 2),
                    "pnl": round(mv - pos.qty * pos.avg_cost, 2),
                    "pnl_pct": round((price / pos.avg_cost - 1) * 100, 2),
                })
            else:
                # Use avg_cost as fallback
                mv = pos.qty * pos.avg_cost
                holdings_value += mv

        total_equity = self.cash + holdings_value

        # Compute daily return
        prev_equity = self._get_previous_equity()
        daily_return = (total_equity / prev_equity - 1) if prev_equity > 0 else 0.0

        # Record daily equity
        equity_record = {
            "date": today,
            "total_equity": round(total_equity, 2),
            "cash": round(self.cash, 2),
            "holdings_value": round(holdings_value, 2),
            "daily_return": round(daily_return, 6),
            "n_positions": len(self.positions),
        }
        self.daily_equity.append(equity_record)

        # Keep only last 500 records
        if len(self.daily_equity) > 500:
            self.daily_equity = self.daily_equity[-500:]

        self.save_state()

        return {
            "date": today,
            "total_equity": round(total_equity, 2),
            "cash": round(self.cash, 2),
            "holdings_value": round(holdings_value, 2),
            "daily_return": round(daily_return, 6),
            "cumulative_return": round(total_equity / self.initial_capital - 1, 4),
            "n_positions": len(self.positions),
            "positions": position_details,
        }

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------

    def get_performance(self) -> dict:
        """
        Compute performance metrics since inception.

        Returns:
            Dict with performance metrics:
            {total_return, annualized_return, max_drawdown, sharpe_ratio,
             win_rate, n_trades, avg_holding_days, total_commission}
        """
        if not self.daily_equity:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
                "n_trades": len(self.trade_history),
                "total_commission": 0.0,
            }

        equities = [r["total_equity"] for r in self.daily_equity]
        returns = [r["daily_return"] for r in self.daily_equity]

        # Total return
        total_return = equities[-1] / self.initial_capital - 1

        # Annualized return
        n_days = len(equities)
        if n_days > 1:
            annualized = (1 + total_return) ** (252 / n_days) - 1
        else:
            annualized = 0.0

        # Max drawdown
        max_dd = self._compute_max_drawdown(equities)

        # Sharpe ratio (annualized, assuming rf=0)
        if len(returns) > 1:
            returns_arr = np.array(returns)
            mean_ret = np.mean(returns_arr)
            std_ret = np.std(returns_arr, ddof=1)
            sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        # Win rate (from closed trades)
        win_rate = self._compute_win_rate()

        # Total commission
        total_commission = sum(t.commission for t in self.trade_history)

        return {
            "total_return": round(total_return, 4),
            "annualized_return": round(annualized, 4),
            "max_drawdown": round(max_dd, 4),
            "sharpe_ratio": round(sharpe, 3),
            "win_rate": round(win_rate, 3),
            "n_trades": len(self.trade_history),
            "n_positions": len(self.positions),
            "total_commission": round(total_commission, 2),
            "current_equity": round(equities[-1], 2),
            "inception_date": self.inception_date,
        }

    def get_tracking_error(self, backtest_returns: List[float]) -> float:
        """
        Compare paper trade returns vs backtest returns.

        Computes the annualized tracking error (standard deviation of
        the difference between paper and backtest daily returns).

        Args:
            backtest_returns: List of daily returns from the backtest.

        Returns:
            Annualized tracking error (e.g., 0.02 = 2%).
        """
        paper_returns = [r["daily_return"] for r in self.daily_equity]

        # Align lengths (use the shorter series)
        n = min(len(paper_returns), len(backtest_returns))
        if n < 5:
            return 0.0

        paper_arr = np.array(paper_returns[-n:])
        bt_arr = np.array(backtest_returns[-n:])

        diff = paper_arr - bt_arr
        te_daily = np.std(diff, ddof=1)
        te_annual = te_daily * np.sqrt(252)

        return float(te_annual)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist portfolio state to JSON file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "initial_capital": self.initial_capital,
            "cash": round(self.cash, 2),
            "inception_date": self.inception_date,
            "positions": {
                sym: pos.to_dict() for sym, pos in self.positions.items()
            },
            "trade_history": [t.to_dict() for t in self.trade_history[-200:]],
            "daily_equity": self.daily_equity[-500:],
            "last_updated": datetime.now().isoformat(),
        }

        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_state(self) -> None:
        """Load portfolio state from JSON file."""
        if not self.state_file.exists():
            logger.info("No existing state file, starting fresh")
            return

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            self.initial_capital = state.get("initial_capital", self.initial_capital)
            self.cash = state.get("cash", self.initial_capital)
            self.inception_date = state.get("inception_date", "")

            # Load positions
            self.positions = {}
            for sym, pos_dict in state.get("positions", {}).items():
                self.positions[sym] = Position.from_dict(pos_dict)

            # Load trade history
            self.trade_history = []
            for t in state.get("trade_history", []):
                self.trade_history.append(TradeRecord(**t))

            # Load daily equity
            self.daily_equity = state.get("daily_equity", [])

            logger.info(
                "Loaded state: cash=%.0f, %d positions, %d trades",
                self.cash, len(self.positions), len(self.trade_history),
            )

        except Exception as e:
            logger.error("Failed to load state: %s", e)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_execution_price(
        self, symbol: str, market_data: Dict[str, pd.DataFrame], exec_date: str
    ) -> Optional[float]:
        """Get the open price for the execution date."""
        df = market_data.get(symbol)
        if df is None or df.empty:
            return None

        target = pd.Timestamp(exec_date)
        row = df[df["date"] == target]
        if row.empty:
            # Try the latest available date
            available = df[df["date"] <= target]
            if available.empty:
                return None
            row = available.tail(1)

        if "open" in row.columns:
            price = float(row["open"].iloc[0])
        else:
            price = float(row["close"].iloc[0])

        return price if price > 0 else None

    def _get_latest_close(
        self, symbol: str, market_data: Dict[str, pd.DataFrame]
    ) -> Optional[float]:
        """Get the latest close price for a symbol."""
        df = market_data.get(symbol)
        if df is None or df.empty:
            return None
        if "close" in df.columns:
            return float(df["close"].iloc[-1])
        return None

    def _is_limit_up(
        self, symbol: str, market_data: Dict[str, pd.DataFrame], exec_date: str
    ) -> bool:
        """Check if a stock hit limit-up (can't buy)."""
        df = market_data.get(symbol)
        if df is None or df.empty:
            return False

        target = pd.Timestamp(exec_date)
        available = df[df["date"] <= target]
        if len(available) < 2:
            return False

        last_two = available.tail(2)
        prev_close = float(last_two["close"].iloc[0])
        curr_close = float(last_two["close"].iloc[1])

        if prev_close <= 0:
            return False

        pct_change = (curr_close - prev_close) / prev_close
        return pct_change >= LIMIT_UP_PCT - 0.001  # tolerance

    def _is_limit_down(
        self, symbol: str, market_data: Dict[str, pd.DataFrame], exec_date: str
    ) -> bool:
        """Check if a stock hit limit-down (can't sell)."""
        df = market_data.get(symbol)
        if df is None or df.empty:
            return False

        target = pd.Timestamp(exec_date)
        available = df[df["date"] <= target]
        if len(available) < 2:
            return False

        last_two = available.tail(2)
        prev_close = float(last_two["close"].iloc[0])
        curr_close = float(last_two["close"].iloc[1])

        if prev_close <= 0:
            return False

        pct_change = (curr_close - prev_close) / prev_close
        return pct_change <= LIMIT_DOWN_PCT + 0.001  # tolerance

    def _compute_equity(
        self, market_data: Dict[str, pd.DataFrame], exec_date: str
    ) -> float:
        """Compute total portfolio equity."""
        holdings_value = 0.0
        target = pd.Timestamp(exec_date)

        for symbol, pos in self.positions.items():
            df = market_data.get(symbol)
            if df is not None and not df.empty:
                available = df[df["date"] <= target]
                if not available.empty:
                    price = float(available["close"].iloc[-1])
                    holdings_value += pos.qty * price
                else:
                    holdings_value += pos.qty * pos.avg_cost
            else:
                holdings_value += pos.qty * pos.avg_cost

        return self.cash + holdings_value

    def _get_previous_equity(self) -> float:
        """Get the previous day's equity (for daily return calculation)."""
        if len(self.daily_equity) >= 2:
            return self.daily_equity[-2]["total_equity"]
        return self.initial_capital

    def _compute_max_drawdown(self, equities: List[float]) -> float:
        """Compute maximum drawdown from equity series."""
        if not equities:
            return 0.0
        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _compute_win_rate(self) -> float:
        """Compute win rate from closed round-trip trades."""
        # Match sells to buys by symbol
        buy_costs: Dict[str, List[float]] = {}
        wins = 0
        total_closed = 0

        for trade in self.trade_history:
            if trade.action == "BUY":
                buy_costs.setdefault(trade.symbol, []).append(trade.price)
            elif trade.action == "SELL":
                costs = buy_costs.get(trade.symbol, [])
                if costs:
                    avg_buy = np.mean(costs)
                    if trade.price > avg_buy:
                        wins += 1
                    total_closed += 1
                    # Remove matched buys
                    buy_costs[trade.symbol] = []

        if total_closed == 0:
            return 0.0
        return wins / total_closed
