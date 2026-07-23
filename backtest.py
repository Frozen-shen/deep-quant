"""
回测引擎 — 逐日模拟交易，支持 A股(T+1) + 港股(T+0)

基于收盘价执行，双边手续费。
支持真实A股费率 (印花税/过户费/最低5元) 通过 trading_rules 模块。
"""

import pandas as pd
import numpy as np


class BacktestEngine:
    """
    逐日回测引擎，支持多市场 + 买卖分别计费。

    参数
    ----
    initial_capital : float
    buy_commission : float
        买入手续费率 (仅在不使用 commission_fn 时生效)
    sell_commission : float
        卖出手续费率 (仅在不使用 commission_fn 时生效)
    t_plus : int
    lot_size : int
        (保留) 向后兼容
    commission : float
        (保留) 向后兼容 — 若设置则覆盖 buy/sell
    commission_fn : callable, optional
        自定义费率函数 (qty, price, side) -> fee_amount
        若提供则覆盖 buy_commission/sell_commission
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        buy_commission: float = 0.0003,
        sell_commission: float = 0.0003,
        t_plus: int = 1,
        lot_size: int = 0,
        stop_loss_atr: float = 2.0,
        trailing_stop_pct: float = 0.02,
        max_hold_days: int = 20,
        atr_position_sizing: bool = True,   # ★ ATR动态仓位
        risk_per_trade: float = 0.02,       # ★ 单笔风险2%
        commission_fn=None,                 # ★ 自定义费率函数
        **kwargs,
    ):
        self.initial_capital = initial_capital
        if "commission" in kwargs:
            self.buy_commission = kwargs["commission"]
            self.sell_commission = kwargs["commission"]
        else:
            self.buy_commission = buy_commission
            self.sell_commission = sell_commission
        self.t_plus = t_plus
        self.lot_size = lot_size
        self.stop_loss_atr = stop_loss_atr
        self.trailing_stop_pct = trailing_stop_pct
        self.max_hold_days = max_hold_days
        self.atr_position_sizing = atr_position_sizing
        self.risk_per_trade = risk_per_trade
        self.commission_fn = commission_fn

    def _calc_buy_fee(self, qty: float, price: float) -> float:
        """计算买入手续费。"""
        if self.commission_fn:
            return self.commission_fn(qty, price, "buy")
        return qty * price * self.buy_commission

    def _calc_sell_fee(self, qty: float, price: float) -> float:
        """计算卖出手续费 (含印花税等)。"""
        if self.commission_fn:
            return self.commission_fn(qty, price, "sell")
        return qty * price * self.sell_commission

    @classmethod
    def for_market(cls, market: str = "a", initial_capital: float = 100_000) -> "BacktestEngine":
        """根据市场创建预配置的回测引擎。"""
        from data_fetcher import MARKET_CONFIG
        cfg = MARKET_CONFIG.get(market, MARKET_CONFIG["a"])

        # A股使用真实费率函数 (印花税+过户费+最低5元)
        commission_fn = None
        if market == "a":
            try:
                from trading_rules import calc_commission
                commission_fn = calc_commission
            except ImportError:
                pass

        return cls(
            initial_capital=initial_capital,
            buy_commission=cfg.get("buy_commission", cfg["commission_default"]),
            sell_commission=cfg.get("sell_commission", cfg["commission_default"]),
            t_plus=cfg["t_plus"],
            lot_size=cfg["lot_size"],
            stop_loss_atr=2.0,
            trailing_stop_pct=0.02,
            max_hold_days=20,
            commission_fn=commission_fn,
        )

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行回测。

        执行规则:
        - 信号在 day T 收盘后确认 (基于 close[T] 计算)
        - 交易在 day T+1 开盘价执行 (open[T+1])
        - 每日权益按收盘价标记 (close[T])
        - 内置风控: ATR止损 / 移动止盈 / 最大持仓天数

        这消除了"未来函数"——回测不再假设你能在信号确认的同一秒成交。

        参数
        ----
        df : pd.DataFrame
            必须包含: date, open, close, signal, position

        返回
        ----
        pd.DataFrame
            新增列: cash, holdings, equity, daily_returns, trade
        """
        df = df.copy()

        # 信号前移 1 天
        df["exec_signal"] = df["signal"].shift(1).fillna(0).astype(int)

        # ATR(14) 用于止损
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1))
            )
        )
        df["atr14"] = df["tr"].rolling(14).mean()

        cash = self.initial_capital
        holdings = 0.0
        entry_price = 0.0       # 买入价(用于止损)
        highest_since_entry = 0.0  # 持仓期间最高价(用于移动止盈)
        hold_days = 0           # 持仓天数
        stop_count = 0          # 止损次数
        trail_count = 0         # 移动止盈次数

        cash_list, holdings_list, equity_list, trade_list = [], [], [], []

        last_buy_idx = -999

        for i, row in df.iterrows():
            exec_sig = row["exec_signal"]
            open_price = row.get("open", row["close"])
            close = row["close"]
            atr = row.get("atr14", 0)
            if pd.isna(atr):
                atr = 0
            trade = None

            can_buy = True
            can_sell = holdings > 0

            if self.t_plus >= 1 and (i - last_buy_idx) < self.t_plus:
                can_sell = False

            # ---- 风控检查 (持有中) ----
            risk_sell = False
            risk_reason = ""

            if holdings > 0 and entry_price > 0:
                hold_days = i - last_buy_idx
                highest_since_entry = max(highest_since_entry, close)

                # ATR 止损
                if self.stop_loss_atr > 0 and atr > 0:
                    stop_price = entry_price - self.stop_loss_atr * atr
                    if close <= stop_price:
                        risk_sell = True
                        risk_reason = f"ATR止损({close:.2f}<={stop_price:.2f})"
                        stop_count += 1

                # 移动止盈 (盈利>5%后启用)
                if not risk_sell and self.trailing_stop_pct > 0:
                    profit_pct = (highest_since_entry / entry_price - 1)
                    if profit_pct > 0.05:  # 盈利超过5%
                        trail_stop = highest_since_entry * (1 - self.trailing_stop_pct)
                        if close <= trail_stop:
                            risk_sell = True
                            risk_reason = f"移动止盈(高{highest_since_entry:.2f},回撤至{close:.2f})"
                            trail_count += 1

                # 最大持仓天数
                if not risk_sell and self.max_hold_days > 0:
                    if hold_days >= self.max_hold_days:
                        risk_sell = True
                        risk_reason = f"超期平仓({hold_days}天)"
                        stop_count += 1

            # ---- 买入信号 (在当日开盘价执行) ----
            if exec_sig == 1 and holdings == 0 and can_buy:
                # ATR 动态仓位: 波动越大仓位越小
                if self.atr_position_sizing and atr > 0 and open_price > 0:
                    risk_amount = cash * self.risk_per_trade
                    stop_distance = self.stop_loss_atr * atr
                    target_shares = risk_amount / stop_distance
                    cost = min(target_shares * open_price, cash)
                    if self.lot_size > 0:
                        lots = int(cost / (open_price * self.lot_size))
                        holdings = lots * self.lot_size
                    else:
                        holdings = cost / open_price
                    buy_fee = self._calc_buy_fee(holdings, open_price)
                    cash -= holdings * open_price + buy_fee
                else:
                    # 原逻辑: 全仓
                    if self.lot_size > 0:
                        lots = int(cash / (open_price * self.lot_size))
                        holdings = lots * self.lot_size
                    else:
                        holdings = cash / open_price
                    buy_fee = self._calc_buy_fee(holdings, open_price)
                    cash -= holdings * open_price + buy_fee
                entry_price = open_price
                highest_since_entry = close
                hold_days = 0
                last_buy_idx = i
                trade = "buy"

            # ---- 卖出信号 OR 风控卖出 ----
            elif (exec_sig == -1 or risk_sell) and can_sell:
                sell_fee = self._calc_sell_fee(holdings, open_price)
                cash += holdings * open_price - sell_fee
                holdings = 0.0
                entry_price = 0.0
                highest_since_entry = 0.0
                trade = f"stop:{risk_reason}" if risk_sell else "sell"

            # ---- 按收盘价标记当日权益 ----
            equity = cash + holdings * close

            cash_list.append(cash)
            holdings_list.append(holdings)
            equity_list.append(equity)
            trade_list.append(trade)

        # 填入结果
        df["cash"] = cash_list
        df["holdings"] = holdings_list
        df["equity"] = equity_list
        df["trade"] = trade_list
        df["daily_returns"] = df["equity"].pct_change().fillna(0)
        df = df.drop(columns=["exec_signal", "tr", "atr14"])

        total_trades = df["trade"].notna().sum()
        final_equity = df["equity"].iloc[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100
        print(f"[Backtest] 初始资金: {self.initial_capital:,.0f}  "
              f"最终权益: {final_equity:,.2f}  "
              f"总收益率: {total_return:+.2f}%  "
              f"交易次数: {total_trades}  "
              f"止损:{stop_count} 移动止盈:{trail_count}  "
              f"(T+{self.t_plus}, 买{self.buy_commission*10000:.0f}bp)")

        return df
