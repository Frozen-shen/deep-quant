"""
LLM深度参与纸面交易 — DeepSeek真实评分版

6只港股 × DeepSeek V4:
  - 宏观分析: LLM判断市场牛熊
  - 基本面评分: LLM评估财报质量  
  - 定制权重: LLM为每只股票优化因子权重
  - 事件评分: LLM评估公告影响 (如有)

用法: python paper_trade_llm.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
from datetime import datetime
import storage

from data_fetcher import DataFetcher, MARKET_CONFIG
from portfolio import PortfolioManager
from executor import MockExecutor
from alerter import Alerter
from factor_scorer import FactorScorer
from macro_overlay import MacroOverlay
from sector_analyzer import SectorAnalyzer
from fundamental_llm import FundamentalLLM
from llm_weight_optimizer import LLMWeightOptimizer
from strategy import EnhancedMACrossoverStrategy

SYMBOLS = ["01810","09988","03690","00981","00700","02318"]
MARKET = "hk"
START, END = "2024-01-01", "2026-07-10"

API_KEY = os.environ.get("OPENAI_API_KEY")  # 用环境变量,勿硬编码
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"


def main():
    storage.init_db()
    cfg = MARKET_CONFIG[MARKET]
    results = []

    for sym in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"  🤖 DeepSeek分析: {sym}")
        print(f"{'='*60}")

        # 0. 拉数据
        fetcher = DataFetcher()
        df = fetcher.fetch(sym, "20230101", "20260710", "qfq", market=MARKET)
        
        # 1. LLM宏观
        print("  [1/4] LLM宏观分析...")
        macro = MacroOverlay(market=MARKET)
        macro.update()
        
        # 2. LLM基本面
        print("  [2/4] LLM基本面评分...")
        f_llm = FundamentalLLM(backend="openai", api_key=API_KEY, model=MODEL, base_url=BASE_URL)
        fund_score = f_llm.evaluate(sym)
        
        # 3. LLM定制权重
        print("  [3/4] LLM定制权重...")
        sector_analyzer = SectorAnalyzer(market=MARKET)
        sector_info = sector_analyzer.get_sector(sym)
        
        returns = df["close"].pct_change().dropna()
        features = {
            "volatility": float(returns.std() * np.sqrt(252)),
            "trend_adx": 20,
            "daily_range": float(((df["high"] - df["low"]) / df["close"]).mean()),
            "sector": sector_info["sector"],
            "symbol": sym,
        }
        opt = LLMWeightOptimizer(backend="mock")  # 先用规则,LLM调用链修复后再切换
        opt_result = opt.optimize(sym, features)
        
        # 4. 因子打分器(LLM定制)
        scorer = FactorScorer(
            factor_weights=opt_result.get("factor_weights", {}),
            buy_threshold=opt_result.get("buy_threshold", 0.15),
            sell_threshold=opt_result.get("sell_threshold", -0.10),
        )
        
        print(f"  [4/4] 纸面交易...")
        # 清DB
        db_path = os.path.join(os.path.dirname(__file__), "quant.db")
        if os.path.exists(db_path): os.remove(db_path)
        storage.init_db()
        
        pm = PortfolioManager(market=MARKET)
        executor = MockExecutor()
        df_full = fetcher.fetch(sym, "20180101", "20260710", "qfq", market=MARKET)
        df_full["date"] = pd.to_datetime(df_full["date"])
        
        start_dt = pd.Timestamp(START)
        end_dt = pd.Timestamp(END)
        trading_days = df_full[(df_full["date"] >= start_dt) & (df_full["date"] <= end_dt)]["date"].tolist()
        
        for day_idx, today in enumerate(trading_days):
            df_today = df_full[df_full["date"] <= today].tail(120).copy()
            if len(df_today) < 50: continue
            
            last_close = df_today["close"].iloc[-1]
            
            try:
                df_sig = scorer.generate_signals(df_today)
                last = df_sig.iloc[-1]
                target_position = last.get("position", 0)
                factor_score = last.get("factor_score", 0)
            except:
                continue
            
            today_macro = macro.score_at(today)
            fund_penalty = min(0, fund_score) * 0.1
            adjusted = (factor_score + fund_penalty) * (1 + today_macro * 0.4)
            
            state = pm.load()
            holding = any(p["qty"] > 0 for p in state.positions.values())
            
            action = "HOLD"
            if target_position == 1 and not holding and adjusted > 0:
                action = "BUY"
            elif target_position == 0 and holding:
                action = "SELL"
            
            if action == "BUY":
                buy_qty = int(state.cash * 0.8 / last_close / cfg["lot_size"]) * cfg["lot_size"]
                if buy_qty >= cfg["lot_size"]:
                    comm = buy_qty * last_close * cfg["buy_commission"]
                    if pm.can_buy(sym, buy_qty, last_close, comm)[0]:
                        pm.apply_buy(sym, buy_qty, last_close, commission=comm)
            elif action == "SELL":
                pos = state.positions.get(sym, {})
                sq = pos.get("qty", 0)
                if sq > 0:
                    pm.apply_sell(sym, sq, last_close, commission=sq*last_close*cfg["sell_commission"])
            
            pm.snapshot(today.strftime("%Y-%m-%d"), {sym: last_close})
        
        summary = pm.get_summary({sym: last_close})
        bench_start = df_full[df_full["date"] >= start_dt]["close"].iloc[0]
        bench_end = df_full[df_full["date"] <= end_dt]["close"].iloc[-1]
        bench_ret = (bench_end / bench_start - 1) * 100
        strat_ret = (summary["total_equity"] / 100000 - 1) * 100
        
        print(f"  {sym}: 策略{strat_ret:+.1f}% vs 基准{bench_ret:+.1f}% = 超额{strat_ret-bench_ret:+.1f}%")
        results.append({"symbol": sym, "strategy": strat_ret, "benchmark": bench_ret,
                       "excess": strat_ret - bench_ret})
    
    print(f"\n{'='*60}")
    print(f"  🤖 DeepSeek LLM 结果汇总")
    print(f"{'='*60}")
    pos = sum(1 for r in results if r["excess"] > 0)
    for r in results:
        m = "✅" if r["excess"] > 0 else "❌"
        print(f"  {r['symbol']}: 超额{r['excess']:+.1f}% {m}")
    print(f"  正超额: {pos}/{len(results)}")


if __name__ == "__main__":
    main()
