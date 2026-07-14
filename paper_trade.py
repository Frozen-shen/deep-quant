"""
纸面交易模拟器 — 在生产模块上历史回放，验证全链路

与 backtest.py 的区别:
- backtest.py:   研究层，假设全量数据一次性回测（含未来函数风险）
- paper_trade.py: 生产层，逐日模拟真实运行流程，用 SIGNAL HUB + PORTFOLIO + EXECUTOR

用法:
    python paper_trade.py                          # 默认: 港股小米, 2024全年
    python paper_trade.py --symbol 600519 --market a  # A股茅台
    python paper_trade.py --start 2025-01-01         # 指定起止日期
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import storage
from data_fetcher import DataFetcher, MARKET_CONFIG
from portfolio import PortfolioManager
from signal_hub import SignalHub, make_ma_signal_fn, make_llm_event_signal_fn
from executor import MockExecutor
from alerter import Alerter
from strategy import EnhancedMACrossoverStrategy, StrategyRouter, RSIMeanReversionStrategy
from factor_scorer import FactorScorer
from stock_filter import StockFilter
from fundamental_llm import FundamentalLLM
from macro_overlay import MacroOverlay
from sector_analyzer import SectorAnalyzer
from llm_weight_optimizer import LLMWeightOptimizer
from backtest import BacktestEngine


class PaperTrader:
    """
    纸面交易模拟器 (v2 — 增强版策略 + 风控)。
    """

    def __init__(self, symbol: str, market: str = "hk",
                 initial_capital: float = 100_000,
                 enhanced: bool = True):
        self.symbol = symbol
        self.market = market
        self.cfg = MARKET_CONFIG[market]
        self.initial_capital = initial_capital
        self.enhanced = enhanced

        # 初始化生产模块
        storage.init_db()
        self.pm = PortfolioManager(market=market, initial_capital=initial_capital)
        self.executor = MockExecutor()
        self.alerter = Alerter()

        # 信号中心 — 使用增强版策略
        self.hub = SignalHub(symbol, market)
        # ── 板块分析 ──
        self.sector_analyzer = SectorAnalyzer(market=market)
        sector_info = self.sector_analyzer.get_sector(symbol)
        print(f"  板块: {sector_info['sector']}")

        # ── LLM定制因子权重 ──
        optimizer = LLMWeightOptimizer(backend="mock")
        features = {
            "volatility": 0.3, "trend_adx": 20, "daily_range": 0.03,
            "sector": sector_info["sector"],
        }
        opt_result = optimizer.optimize(symbol, features)
        self.scorer = FactorScorer(
            factor_weights=opt_result["factor_weights"],
            buy_threshold=opt_result["buy_threshold"],
            sell_threshold=opt_result["sell_threshold"],
        )
        print(f"  LLM权重: 阈值(买>{opt_result['buy_threshold']:.2f},卖<{opt_result['sell_threshold']:.2f}), "
              f"因子数={len([w for w in opt_result['factor_weights'].values() if abs(w)>0.01])}")
        self.filter = StockFilter()
        self.fundamental_llm = FundamentalLLM(backend="mock")  # mock→后续接DeepSeek

        # ── 宏观叠加层 ──
        self.macro = MacroOverlay(market=market)
        self.macro.update()
        macro_score = self.macro.market_score
        print(f"  宏观评分: {macro_score:+.2f} ({self.macro.regime})")

        df_check = DataFetcher().fetch(symbol, "20230101", "20260710", "qfq", market=market)
        ok, reason = self.filter.check(df_check, symbol)
        self._tradeable = ok
        print(f"  选股检查: {'✅' if ok else '❌'} {reason}")
        if not ok:
            print(f"  ⚠️ {symbol} 不适合当前策略,将跳过交易")

        # ── 基本面评分 ──
        self._fundamental_score = self.fundamental_llm.evaluate(symbol)
        print(f"  基本面评分: {self._fundamental_score:+.2f}")

    def run(self, start_date: str, end_date: str, lookback_days: int = 120):
        """
        逐日模拟纸面交易。

        参数
        ----
        start_date, end_date : str  YYYY-MM-DD
        lookback_days : int  每天回看天数(确保MA计算需要足够历史)

        返回
        ----
        dict: {equity_log, trades, daily_summary}
        """
        print(f"\n{'='*60}")
        print(f"  纸面交易模拟")
        print(f"  标的: {self.symbol} ({self.cfg['name']})")
        print(f"  区间: {start_date} ~ {end_date}")
        print(f"  初始资金: {self.cfg['currency']} {self.initial_capital:,.0f}")
        print(f"  手续费: 买{self.cfg.get('buy_commission', self.cfg['commission_default'])*10000:.1f}bp"
              f"/卖{self.cfg.get('sell_commission', self.cfg['commission_default'])*10000:.1f}bp")
        print(f"{'='*60}\n")

        # 先拉全量数据（用于获取每日收盘价和基准对比）
        fetcher = DataFetcher()
        df_full = fetcher.fetch(self.symbol, "20180101", end_date.replace("-", ""),
                                "qfq", market=self.market)
        df_full["date"] = pd.to_datetime(df_full["date"])

        # 筛选交易日期范围
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        # 获取交易日列表
        trading_days = df_full[
            (df_full["date"] >= start_dt) & (df_full["date"] <= end_dt)
        ]["date"].tolist()

        print(f"  交易日数: {len(trading_days)}")

        # 记录
        equity_log = []
        trade_log = []
        daily_records = []

        for day_idx, today in enumerate(trading_days):
            today_str = today.strftime("%Y-%m-%d")

            # 当天可用的数据（截至 today）
            df_today = df_full[df_full["date"] <= today].tail(lookback_days).copy()

            if len(df_today) < 50:
                continue  # 数据不足

            last_close = df_today["close"].iloc[-1]
            bench_close = df_full[df_full["date"] == today]["close"]
            if len(bench_close) == 0:
                continue
            bench_close = bench_close.iloc[0]

            # ---- 1. 多因子打分 + 信号 ----
            try:
                df_sig = self.scorer.generate_signals(df_today)
                last = df_sig.iloc[-1]
                target_position = last.get("position", 0)
                last_close = last["close"]
                factor_score = last.get("factor_score", 0)
            except Exception as e:
                print(f"  {today_str}: 信号失败 ({e})")
                continue

            # ---- 判断是否需要交易 (考虑基本面和选股过滤) ----
            state = self.pm.load()
            current_holding = any(p["qty"] > 0 for p in state.positions.values())

            # 注入基本面分数 + 宏观叠加 (当日市场状态)
            fund_penalty = min(0, self._fundamental_score) * 0.1
            today_macro = self.macro.score_at(today)
            macro_boost = 1 + today_macro * 0.4  # ±40%
            adjusted_score = (factor_score + fund_penalty) * macro_boost

            decision_action = None
            if not self._tradeable:
                decision_action = "HOLD"  # 选股器判定不适合
            elif target_position == 1 and not current_holding and adjusted_score > 0:
                decision_action = "BUY"
            elif target_position == 0 and current_holding:
                decision_action = "SELL"
            else:
                decision_action = "HOLD"

            # ---- 2. 执行交易 (仓位风控) ----
            traded = False
            trade_detail = ""
            if decision_action == "BUY":
                # 单票上限20%, 总仓上限80%
                max_per_stock = self.initial_capital * 0.20
                total_equity = state.cash + sum(
                    p["qty"] * last_close for p in state.positions.values()
                )
                available_for_new = min(state.cash, self.initial_capital * 0.80 - total_equity + state.cash)

                buy_qty = int(min(max_per_stock, available_for_new * 0.9) / last_close / 
                              self.cfg["lot_size"]) * self.cfg["lot_size"]
                if buy_qty >= self.cfg["lot_size"]:
                    comm = buy_qty * last_close * self.cfg.get("buy_commission", self.cfg["commission_default"])
                    can, _ = self.pm.can_buy(self.symbol, buy_qty, last_close, comm)
                    if can:
                        self.pm.apply_buy(self.symbol, buy_qty, last_close, commission=comm)
                        self.executor.place(self.symbol, "BUY", buy_qty, last_close)
                        traded = True
                        trade_detail = f"BUY {buy_qty}@{last_close:.2f}"
            elif decision_action == "SELL":
                state = self.pm.load()
                pos = state.positions.get(self.symbol, {})
                sell_qty = pos.get("qty", 0)
                if sell_qty > 0:
                    comm = sell_qty * last_close * self.cfg.get("sell_commission", self.cfg["commission_default"])
                    can, _ = self.pm.can_sell(self.symbol, sell_qty)
                    if can:
                        self.pm.apply_sell(self.symbol, sell_qty, last_close, commission=comm)
                        self.executor.place(self.symbol, "SELL", sell_qty, last_close)
                        traded = True
                        trade_detail = f"SELL {sell_qty}@{last_close:.2f}"

            # ---- 3. 每日快照 ----
            self.pm.snapshot(today_str, {self.symbol: last_close})

            summary = self.pm.get_summary({self.symbol: last_close})
            equity_log.append({
                "date": today_str,
                "equity": summary["total_equity"],
                "cash": summary["cash"],
                "holdings_value": summary["holdings_value"],
            })

            daily_records.append({
                "date": today_str,
                "close": last_close,
                "equity": summary["total_equity"],
                "traded": traded,
                "trade": trade_detail or "",
                "signal": decision_action,
                "confidence": 0.7 if decision_action != "HOLD" else 0,
                "equity_change": summary["total_equity"] / self.initial_capital - 1,
            })

            if day_idx % 20 == 0:
                print(f"  {today_str}: 权益={summary['total_equity']:,.0f}, "
                      f"score={factor_score:+.3f}, "
                      f"{'🔴' if traded else '⏸️'} "
                      f"{trade_detail or decision_action}")

        # ---- 4. 最终报告 ----
        df_equity = pd.DataFrame(equity_log)
        df_daily = pd.DataFrame(daily_records)

        final_equity = df_equity["equity"].iloc[-1] if len(df_equity) > 0 else self.initial_capital
        total_return = (final_equity / self.initial_capital - 1) * 100

        # 基准
        bench_start_price = df_full[df_full["date"] >= start_dt]["close"].iloc[0]
        bench_end_price = df_full[df_full["date"] <= end_dt]["close"].iloc[-1]
        bench_return = (bench_end_price / bench_start_price - 1) * 100

        # 统计
        total_trades = df_daily["traded"].sum()
        win_days = (df_daily["equity_change"].diff() > 0).sum() if len(df_daily) > 1 else 0
        total_days = len(df_daily)

        print(f"\n{'='*60}")
        print(f"  纸面交易报告")
        print(f"{'='*60}")
        print(f"  最终权益: {self.cfg['currency']} {final_equity:,.0f}")
        print(f"  总收益率: {total_return:+.2f}%")
        print(f"  基准收益: {bench_return:+.2f}% (买入持有)")
        print(f"  超额收益: {total_return - bench_return:+.2f}%")
        print(f"  交易次数: {total_trades}")
        print(f"  胜率(日): {win_days/total_days*100:.1f}%" if total_days > 0 else "N/A")
        print(f"{'='*60}")

        return {
            "equity_log": df_equity,
            "daily": df_daily,
            "final_equity": final_equity,
            "total_return": total_return,
            "benchmark_return": bench_return,
            "total_trades": total_trades,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="纸面交易模拟")
    parser.add_argument("--symbol", default="01810")
    parser.add_argument("--market", default="hk")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-07-10")
    parser.add_argument("--capital", type=float, default=100000)
    args = parser.parse_args()

    trader = PaperTrader(
        symbol=args.symbol,
        market=args.market,
        initial_capital=args.capital,
    )
    trader.run(args.start, args.end)
