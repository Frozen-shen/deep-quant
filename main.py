"""
主入口 — 一键运行量化回测全流程 (A股 + 港股)

市场切换:
  MARKET=a  → A股 (默认, 如 600519 茅台)
  MARKET=hk → 港股 (如 01810 小米)

三种策略模式:
  STRATEGY_MODE=tech        → 纯技术面 (MA双均线交叉)
  STRATEGY_MODE=llm_filter  → MA + LLM情感过滤 (LLM为配角)
  STRATEGY_MODE=llm_event   → LLM事件驱动 (LLM为主角) ★

LLM配置:
  LLM_BACKEND=mock|openai|ollama|rule_based|none
  OPENAI_API_KEY=sk-...     (openai模式必填)
  LLM_MODEL=deepseek-chat   (模型名)
  LLM_BASE_URL=https://...  (自定义端点)

示例:
  MARKET=a  python main.py                    # A股茅台, 技术面
  MARKET=hk python main.py                    # 港股小米, 技术面
  MARKET=hk STRATEGY_MODE=llm_event LLM_BACKEND=openai python main.py  # 小米+LLM事件
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import DataFetcher, MARKET_CONFIG
from strategy import MACrossoverStrategy, apply_llm_filter, LLMEventStrategy
from backtest import BacktestEngine
from analysis import PerformanceAnalyzer
from news_fetcher import NewsFetcher
from event_fetcher import EventFetcher
from llm_factor import LLMFactorEngine


def main():
    # ==================== 参数配置 ====================
    MARKET = os.environ.get("MARKET", "a")         # "a" / "hk"
    MARKET_CFG = MARKET_CONFIG[MARKET]

    # 根据市场选默认标的
    DEFAULT_SYMBOLS = {"a": "600519", "hk": "01810"}
    SYMBOL = os.environ.get("SYMBOL", DEFAULT_SYMBOLS.get(MARKET, "600519"))

    START_DATE = "20180101"
    END_DATE = "20260710"
    ADJUST = "qfq"
    INITIAL_CAPITAL = 100_000

    # 策略模式
    STRATEGY_MODE = os.environ.get("STRATEGY_MODE", "tech")

    # LLM 配置
    LLM_BACKEND = os.environ.get("LLM_BACKEND", "mock")
    LLM_API_KEY = os.environ.get("OPENAI_API_KEY", None)
    LLM_MODEL = os.environ.get("LLM_MODEL", None)
    LLM_BASE_URL = os.environ.get("LLM_BASE_URL", None)

    # 事件驱动参数
    EVENT_CONFIDENCE_THRESHOLD = 0.6
    EVENT_MODE = "mock"  # 事件数据模式: mock/live/hybrid

    ENABLE_LLM = LLM_BACKEND != "none"

    # ==================== 1. 行情数据 ====================
    print("\n" + "=" * 60)
    print(f"  Step 1: 获取行情数据 (市场: {MARKET_CFG['name']}, "
          f"策略: {STRATEGY_MODE}, 标的: {SYMBOL})")
    print("=" * 60)
    fetcher = DataFetcher()
    df = fetcher.fetch(SYMBOL, START_DATE, END_DATE, ADJUST, market=MARKET)

    # ==================== 2. 策略信号 ====================
    if STRATEGY_MODE == "llm_event":
        # ---------- LLM 事件驱动 (LLM为主角) ----------
        print("\n" + "=" * 60)
        print("  Step 2: LLM 事件驱动策略 (LLM → 事件 → 信号)")
        print("=" * 60)

        # 2a. 获取公司事件
        event_fetcher = EventFetcher()
        events_df = event_fetcher.fetch_all_events(
            symbol=SYMBOL,
            begin_date=f"{START_DATE[:4]}-{START_DATE[4:6]}-{START_DATE[6:8]}",
            end_date=f"{END_DATE[:4]}-{END_DATE[4:6]}-{END_DATE[6:8]}",
            mode=EVENT_MODE,
        )

        # 2b. LLM 事件评分
        print(f"\n  LLM 事件评分 (backend={LLM_BACKEND})...")
        llm_engine = LLMFactorEngine(
            backend=LLM_BACKEND,
            api_key=LLM_API_KEY,
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            cache_dir=os.path.join(os.path.dirname(__file__), ".llm_cache"),
        )
        scored_events = llm_engine.batch_score_events(events_df)

        # 2c. 生成信号
        event_strategy = LLMEventStrategy(
            confidence_threshold=EVENT_CONFIDENCE_THRESHOLD,
        )
        df = event_strategy.generate_signals(df, scored_events)

    elif STRATEGY_MODE == "llm_filter":
        # ---------- MA + LLM过滤 (LLM为配角) ----------
        print("\n" + "=" * 60)
        print("  Step 2: 技术信号 + LLM情感过滤")
        print("=" * 60)

        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        df = strategy.generate_signals(df)

        if ENABLE_LLM:
            news_fetcher = NewsFetcher()
            news_df = news_fetcher.fetch_stock_news(SYMBOL, lookback_days=30, mode="mock")
            llm_engine = LLMFactorEngine(
                backend=LLM_BACKEND, api_key=LLM_API_KEY,
                model=LLM_MODEL, base_url=LLM_BASE_URL,
                cache_dir=os.path.join(os.path.dirname(__file__), ".llm_cache"),
            )
            scored_news = llm_engine.batch_score(news_df)
            daily_factors = llm_engine.aggregate_to_daily(scored_news)
            df = apply_llm_filter(df, daily_factors, sentiment_threshold=0.0)

    else:
        # ---------- 纯技术面 ----------
        print("\n" + "=" * 60)
        print("  Step 2: 纯技术面信号 (MA双均线交叉)")
        print("=" * 60)
        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        df = strategy.generate_signals(df)

    # ==================== 3. 回测 ====================
    print("\n" + "=" * 60)
    print(f"  Step 3: 执行回测 "
          f"(T+{MARKET_CFG['t_plus']}, "
          f"手续费{MARKET_CFG['commission_default']*10000:.0f}bp, "
          f"{MARKET_CFG['currency']})")
    print("=" * 60)
    engine = BacktestEngine.for_market(MARKET, INITIAL_CAPITAL)
    df = engine.run(df)

    # ==================== 4. 绩效分析 ====================
    print("\n" + "=" * 60)
    print("  Step 4: 绩效分析")
    print("=" * 60)
    analyzer = PerformanceAnalyzer(
        risk_free_rate=MARKET_CFG["risk_free_rate"]
    )
    metrics = analyzer.analyze(df, INITIAL_CAPITAL)
    sig_tests = analyzer.test_significance(df)
    analyzer.print_report(metrics, sig_tests)

    # ---- 保存到数据库 ----
    try:
        import storage
        storage.init_db()
        bid = storage.save_backtest(
            symbol=SYMBOL, market=MARKET, strategy=STRATEGY_MODE,
            start_date=START_DATE, end_date=END_DATE,
            params={"short_window": 5, "long_window": 20},
            metrics=metrics,
            notes=f"LLM={LLM_BACKEND}",
        )
        print(f"  [DB] 回测结果已保存 (id={bid})")
    except Exception as e:
        print(f"  [DB] 保存失败: {e}")

    # ==================== 5. 可视化 ====================
    mode_tag = {"tech": "tech", "llm_filter": "llm_filter",
                "llm_event": "llm_event"}.get(STRATEGY_MODE, STRATEGY_MODE)
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"backtest_{SYMBOL}_{mode_tag}.png")
    analyzer.plot(df, symbol=f"{SYMBOL} ({mode_tag})", save_path=save_path)

    print(f"\n✅ 全流程完成！ (策略: {mode_tag}, LLM: {LLM_BACKEND})")


if __name__ == "__main__":
    main()
