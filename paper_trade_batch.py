"""
批量纸面交易 v3 — 增强策略 + 因子引擎 + DB持久化
"""

import os, sys, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd, numpy as np
from datetime import datetime
from data_fetcher import MARKET_CONFIG
from paper_trade import PaperTrader
import storage

TEST_STOCKS = [
    {"symbol": "01810", "market": "hk", "name": "小米集团", "style": "消费电子", "volatility": "中高"},
    {"symbol": "00700", "market": "hk", "name": "腾讯控股", "style": "互联网", "volatility": "中"},
    {"symbol": "09988", "market": "hk", "name": "阿里巴巴", "style": "电商", "volatility": "中"},
    {"symbol": "09618", "market": "hk", "name": "京东集团", "style": "电商", "volatility": "中"},
    {"symbol": "03690", "market": "hk", "name": "美团",     "style": "本地生活", "volatility": "高"},
    {"symbol": "09999", "market": "hk", "name": "网易",     "style": "游戏", "volatility": "中"},
    {"symbol": "02020", "market": "hk", "name": "安踏体育", "style": "消费", "volatility": "中"},
    {"symbol": "02318", "market": "hk", "name": "中国平安", "style": "金融", "volatility": "中"},
    {"symbol": "01211", "market": "hk", "name": "比亚迪股份","style": "新能源车", "volatility": "高"},
    {"symbol": "00981", "market": "hk", "name": "中芯国际", "style": "半导体", "volatility": "高"},
    {"symbol": "02269", "market": "hk", "name": "药明生物", "style": "医药", "volatility": "高"},
    {"symbol": "09888", "market": "hk", "name": "百度集团", "style": "AI科技", "volatility": "中"},
]

START_DATE, END_DATE = "2024-01-01", "2026-07-10"


def run_batch():
    storage.init_db()
    results = []

    for i, stock in enumerate(TEST_STOCKS):
        sym, market, name = stock["symbol"], stock["market"], stock["name"]
        cfg = MARKET_CONFIG[market]

        print(f"\n{'#'*60}")
        print(f"  [{i+1}/{len(TEST_STOCKS)}] {sym} {name} ({cfg['name']})")
        print(f"  增强策略 + 因子引擎 + ATR风控 + DB持久化")
        print(f"{'#'*60}")

        db_path = os.path.join(os.path.dirname(__file__), "quant.db")
        if os.path.exists(db_path): os.remove(db_path)
        storage.init_db()

        try:
            trader = PaperTrader(symbol=sym, market=market, initial_capital=100_000, enhanced=True)
            result = trader.run(START_DATE, END_DATE)

            r = {
                "symbol": sym, "name": name, "market": cfg['name'],
                "style": stock["style"], "volatility": stock["volatility"],
                "currency": cfg['currency'],
                "strategy_return": result["total_return"],
                "benchmark_return": result["benchmark_return"],
                "excess_return": result["total_return"] - result["benchmark_return"],
                "total_trades": result["total_trades"], "final_equity": result["final_equity"],
            }

            # 保存到DB
            storage.save_backtest(
                symbol=sym, market=market, strategy="enhanced_ma",
                start_date=START_DATE, end_date=END_DATE,
                params={"enhanced": True, "volume_filter": True, "atr_stop": True},
                metrics={
                    "total_return": r["strategy_return"] / 100,
                    "excess_vs_benchmark": r["excess_return"] / 100,
                    "total_trades": r["total_trades"],
                    "final_equity": r["final_equity"],
                },
                notes=f"批量测试 {datetime.now():%Y-%m-%d}",
            )
            results.append(r)
        except Exception as e:
            print(f"  ❌ {sym} 失败: {e}")
            import traceback; traceback.print_exc()
            results.append({"symbol": sym, "name": name, "market": cfg['name'],
                           "style": stock["style"], "volatility": stock["volatility"],
                           "strategy_return": None, "benchmark_return": None,
                           "excess_return": None, "total_trades": 0, "error": str(e)})

    # 报告
    df = pd.DataFrame(results)
    valid = df[df["strategy_return"].notna()]
    print(f"\n{'='*80}")
    print(f"  最终批量测试报告 ({START_DATE} ~ {END_DATE})")
    print(f"{'='*80}")
    print(f"{'代码':<8} {'名称':<10} {'市场':<5} {'策略收益':>10} {'基准收益':>10} {'超额':>10} {'交易':>5}")
    print("-" * 70)
    for _, r in df.iterrows():
        if r.get("error"): continue
        sr, br, er = f"{r['strategy_return']:+.2f}%", f"{r['benchmark_return']:+.2f}%", f"{r['excess_return']:+.2f}%"
        m = "🏆" if r['excess_return'] > 20 else ("✅" if r['excess_return'] > 0 else "❌")
        print(f"{r['symbol']:<8} {r['name']:<10} {r['market']:<5} {sr:>10} {br:>10} {er:>10} {int(r['total_trades']):>5} {m}")
    print("-" * 70)
    pos = (valid["excess_return"] > 0).sum()
    print(f"\n正超额: {pos}/{len(valid)} | 平均超额: {valid['excess_return'].mean():+.2f}%")
    print(f"A股均值: {valid[valid['market']=='A股']['excess_return'].mean():+.2f}% | 港股均值: {valid[valid['market']=='港股']['excess_return'].mean():+.2f}%")
    print(f"高波动均值: {valid[valid['volatility']=='高']['excess_return'].mean():+.2f}% | 中低波动均值: {valid[valid['volatility']!='高']['excess_return'].mean():+.2f}%")
    csv_path = os.path.join(os.path.dirname(__file__), "paper_trade_comparison.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n报告: {csv_path}")


if __name__ == "__main__":
    run_batch()
