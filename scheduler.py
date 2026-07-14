"""
定时调度引擎 — 每个交易日盘后自动运行

用法:
    python scheduler.py                          # 前台运行
    python scheduler.py --once                   # 只跑一次(测试用)

部署 (Windows):
    nssm install QuantScheduler python scheduler.py
    nssm start QuantScheduler
"""

import os
import sys
import argparse
from datetime import datetime, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from data_fetcher import DataFetcher, MARKET_CONFIG
from strategy import MACrossoverStrategy, LLMEventStrategy
from backtest import BacktestEngine
from analysis import PerformanceAnalyzer
from signal_hub import SignalHub, make_ma_signal_fn
from portfolio import PortfolioManager
from alerter import Alerter


# ================================================================
#  交易日历
# ================================================================

def _is_trading_day(date, market: str = "a") -> bool:
    """检查是否为交易日 (A股使用新浪日历)。"""
    from datetime import date as _date
    if isinstance(date, _date):
        pass
    else:
        date = date.date() if hasattr(date, 'date') else date

    # 周末肯定不是交易日
    if date.weekday() >= 5:
        return False

    # A股日历
    if market == "a":
        try:
            import akshare as ak
            cal = ak.tool_trade_date_hist_sina()
            return date in cal["trade_date"].values
        except Exception:
            return True  # 日历获取失败时不阻塞，假设是交易日

    # 港股：简化为周末检查（真实日历需额外数据源）
    return True


# ================================================================
#  核心：每日任务
# ================================================================

def daily_job():
    """
    每个交易日盘后执行的完整流程。

    步骤:
    1. 拉取最近60天行情
    2. 多策略生成信号 → SignalHub 聚合
    3. 如有交易信号 → 检查风控 → 记录到 storage
    4. 更新持仓快照
    5. 发送通知
    """
    MARKET = os.environ.get("MARKET", "hk")
    SYMBOL = os.environ.get("SYMBOL", "01810")
    cfg = MARKET_CONFIG[MARKET]

    now = datetime.now()
    print(f"\n{'='*60}")
    print(f"  [{now:%Y-%m-%d %H:%M:%S}] 每日盘后流程")
    print(f"  标的: {SYMBOL} ({cfg['name']})")
    print(f"{'='*60}")

    alerter = Alerter()
    pm = PortfolioManager(market=MARKET)

    # ---- 0. 交易日检查 ----
    if not _is_trading_day(now.date(), MARKET):
        print(f"  今日 ({now.date()}) 非交易日，跳过")
        return

    try:
        # ---- 1. 拉数据 (带重试) ----
        fetcher = DataFetcher()
        df = fetcher.fetch(
            SYMBOL,
            start_date=(now - pd.DateOffset(days=90)).strftime("%Y%m%d") if 'pd' in dir() else "20260501",
            end_date=now.strftime("%Y%m%d"),
            adjust="qfq",
            market=MARKET,
        )
        import pandas as _pd
        # Re-fetch with proper date if needed
        df = fetcher.fetch(
            SYMBOL,
            (now - _pd.DateOffset(days=90)).strftime("%Y%m%d"),
            now.strftime("%Y%m%d"),
            "qfq", market=MARKET,
        )

        if df.empty:
            print("  今日无数据 (可能非交易日)")
            return

        last_close = df["close"].iloc[-1]
        last_date = df["date"].iloc[-1]
        print(f"  最新: {last_date.date()} close={cfg['currency']} {last_close:.2f}")

        # ---- 2. 信号生成 ----
        hub = SignalHub(SYMBOL, MARKET)
        hub.register("ma_cross", make_ma_signal_fn(5, 20), weight=1.0)

        # LLM事件策略 (如果有事件数据)
        try:
            from event_fetcher import EventFetcher
            from llm_factor import LLMFactorEngine
            ef = EventFetcher()
            events = ef.fetch_all_events(
                SYMBOL,
                begin_date=(now - _pd.DateOffset(days=30)).strftime("%Y-%m-%d"),
                end_date=now.strftime("%Y-%m-%d"),
                mode="mock",
            )
            if not events.empty:
                llm = LLMFactorEngine(
                    backend=os.environ.get("LLM_BACKEND", "mock"),
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    model=os.environ.get("LLM_MODEL"),
                    base_url=os.environ.get("LLM_BASE_URL"),
                )
                scored = llm.batch_score_events(events)
                hub.register("llm_event", make_llm_event_signal_fn(0.6), weight=1.5)
                decision = hub.generate(df, extra_kwargs={"events_df": scored})
            else:
                decision = hub.generate(df)
        except Exception:
            decision = hub.generate(df)

        print(f"  信号: {decision.action} (置信度: {decision.confidence:.2f})")
        print(f"  理由: {decision.reason}")

        # ---- 3. 风控检查 & 执行 ----
        if decision.should_trade and decision.action == "BUY":
            # 全仓买入 (简化: 用80%可用现金)
            state = pm.load()
            buy_qty = int(state.cash * 0.8 / last_close / cfg["lot_size"]) * cfg["lot_size"]
            if buy_qty >= cfg["lot_size"]:
                can, msg = pm.can_buy(SYMBOL, buy_qty, last_close,
                                      commission=buy_qty * last_close * cfg["commission_default"])
                if can:
                    tid = pm.apply_buy(SYMBOL, buy_qty, last_close,
                                       commission=buy_qty * last_close * cfg["commission_default"],
                                       reason=decision.reason)
                    print(f"  ✅ 买入: {buy_qty}股 @{last_close}, trade_id={tid}")
                    alerter.signal_alert(decision)
                else:
                    print(f"  ⚠️ 买入被拒: {msg}")

        elif decision.should_trade and decision.action == "SELL":
            state = pm.load()
            pos = state.positions.get(SYMBOL, {})
            sell_qty = pos.get("qty", 0)
            if sell_qty > 0:
                can, msg = pm.can_sell(SYMBOL, sell_qty)
                if can:
                    tid = pm.apply_sell(SYMBOL, sell_qty, last_close,
                                        commission=sell_qty * last_close * cfg["commission_default"],
                                        reason=decision.reason)
                    print(f"  ✅ 卖出: {sell_qty}股 @{last_close}, trade_id={tid}")
                    alerter.signal_alert(decision)
                else:
                    print(f"  ⚠️ 卖出被拒: {msg}")

        # ---- 4. 快照 ----
        pm.snapshot(last_date.strftime("%Y-%m-%d"), {SYMBOL: last_close})

        # ---- 5. 日报 ----
        summary = pm.get_summary({SYMBOL: last_close})
        alerter.daily_summary(summary, {SYMBOL: last_close})

        print(f"  权益: {cfg['currency']} {summary['total_equity']:,.0f}")
        print(f"  ✅ 流程完成")

    except Exception as e:
        print(f"  ❌ 流程异常: {e}")
        alerter.error_alert("daily_job", str(e))


# ================================================================
#  入口
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="只运行一次(测试)")
    args = parser.parse_args()

    if args.once:
        daily_job()
    else:
        scheduler = BlockingScheduler()
        # 每个交易日 15:30 (A股) 或 16:05 (港股)
        scheduler.add_job(
            daily_job,
            CronTrigger(hour=15, minute=30, day_of_week="mon-fri"),
            id="daily_job",
            replace_existing=True,
        )
        print("调度器已启动: 每个工作日 15:30 运行")
        print("按 Ctrl+C 停止")
        try:
            scheduler.start()
        except KeyboardInterrupt:
            print("\n调度器已停止")


if __name__ == "__main__":
    # 运行前将函数内 import 提升
    import pandas as pd
    from signal_hub import make_llm_event_signal_fn
    main()
