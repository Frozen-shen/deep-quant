"""
A股策略 A/B 对比测试 — 一键验证所有配置组合

0 token, 纯本地计算

用法:
  python test_a_share.py          → 跑全部4组对比
  python test_a_share.py --quick  → 只跑前60天(2分钟)
"""

import os, sys, shutil
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd, numpy as np
from datetime import datetime
import storage
from data_fetcher import DataFetcher, MARKET_CONFIG
from portfolio import PortfolioManager
from factor_scorer import FactorScorer
from portfolio_ranker import PortfolioRanker
from macro_overlay import MacroOverlay
from alt_data import peer_relative_factor

# 从 paper_trade_a 导入配置
sys.path.insert(0, os.path.dirname(__file__))
from paper_trade_a import SYMBOLS, STOCK_NAMES, A_SECTORS

MARKET, START, END = "a", "2024-01-01", "2026-07-10"
TOP_K, INITIAL = 4, 100_000


def run_config(name: str, preset: str, sector_neutral: bool,
               quick: bool = False) -> dict:
    """跑一组配置,返回结果字典。"""
    db_path = os.path.join(os.path.dirname(__file__), "quant.db")
    if os.path.exists(db_path): os.remove(db_path)
    storage.init_db()

    cfg = MARKET_CONFIG[MARKET]
    pm = PortfolioManager(market=MARKET, initial_capital=INITIAL)
    ranker = PortfolioRanker(top_k=TOP_K, n_drop=1, hold_thresh=10,
                             sector_neutral=sector_neutral)
    scorer = FactorScorer.from_preset(preset)
    macro = MacroOverlay(market=MARKET)
    macro.update()

    fetcher = DataFetcher()
    all_data = {}
    for sym in SYMBOLS:
        try:
            df = fetcher.fetch(sym, "20180101", END.replace("-",""), "qfq", market=MARKET)
            df["date"] = pd.to_datetime(df["date"])
            all_data[sym] = df
        except: pass

    start_dt, end_dt = pd.Timestamp(START), pd.Timestamp(END)
    sample = list(all_data.values())[0]
    days = sample[(sample["date"]>=start_dt)&(sample["date"]<=end_dt)]["date"].tolist()
    if quick: days = days[:60]

    sector_map = {s: v[0] for s, v in A_SECTORS.items()}
    trade_count = 0

    for today in days:
        stock_data, close_prices = {}, {}
        for sym in SYMBOLS:
            if sym not in all_data: continue
            df_t = all_data[sym][all_data[sym]["date"]<=today].tail(120)
            if len(df_t) < 50: continue
            stock_data[sym] = df_t
            close_prices[sym] = df_t["close"].iloc[-1]
        if len(stock_data) < TOP_K: continue

        try:
            scores = scorer.cross_sectional_score(stock_data)
        except: continue
        peer = peer_relative_factor(SYMBOLS, stock_data)
        for sym in scores:
            scores[sym] = scores[sym]*0.7 + peer.get(sym,0)*0.3
        for sym in scores:
            scores[sym] *= (1 + macro.score_at(today)*0.3)

        state = pm.load()
        holdings = [s for s, p in state.positions.items() if p["qty"]>0]
        decision = ranker.rank(scores, holdings, sectors=sector_map)

        for sym in decision["sell"]:
            pos = state.positions.get(sym,{})
            qty = pos.get("qty",0)
            if qty>0 and sym in close_prices:
                pm.apply_sell(sym,qty,close_prices[sym],
                              commission=qty*close_prices[sym]*cfg.get("sell_commission",0.0008))
                trade_count += 1

        for sym in decision["buy"]:
            if sym in close_prices:
                state = pm.load()
                cash_per = state.cash*0.9/max(1,len(decision["buy"]))
                px = close_prices[sym]
                qty = int(cash_per/px/cfg["lot_size"])*cfg["lot_size"]
                if qty>=cfg["lot_size"]:
                    comm = qty*px*cfg.get("buy_commission",0.0003)
                    if pm.can_buy(sym,qty,px,comm)[0]:
                        pm.apply_buy(sym,qty,px,commission=comm)
                        trade_count += 1

        pm.snapshot(today.strftime("%Y-%m-%d"), close_prices)

    summary = pm.get_summary(close_prices)
    final_equity = summary["total_equity"]
    total_ret = (final_equity/INITIAL-1)*100

    bench_rets = []
    for sym in SYMBOLS:
        if sym in all_data:
            df = all_data[sym]
            bdf = df[(df["date"]>=start_dt)&(df["date"]<=end_dt)]
            if len(bdf)>0: bench_rets.append(bdf["close"].iloc[-1]/bdf["close"].iloc[0]-1)
    bench_avg = np.mean(bench_rets)*100

    # 保存到DB
    storage.save_backtest(
        symbol="A_SHARE_PORTFOLIO", market=MARKET, strategy=name,
        start_date=START, end_date=END,
        params={"preset": preset, "sector_neutral": sector_neutral, "top_k": TOP_K},
        metrics={"total_return": total_ret/100, "final_equity": final_equity,
                "excess_vs_benchmark": (total_ret-bench_avg)/100, "total_trades": trade_count},
        notes=f"test run {datetime.now():%Y-%m-%d %H:%M}")

    return {"name": name, "preset": preset, "sector_neutral": sector_neutral,
            "total_return": total_ret, "benchmark": bench_avg,
            "excess": total_ret - bench_avg, "trades": trade_count,
            "final_equity": final_equity}


def main():
    quick = "--quick" in sys.argv
    mode = "快速(60天)" if quick else "完整(603天)"
    print(f"\n{'='*70}")
    print(f"  A股策略 A/B 对比测试 — {mode}")
    print(f"  股票: {len(SYMBOLS)}只 | 区间: {START}~{END} | Token: 0")
    print(f"{'='*70}")

    configs = [
        ("① 基线: trend_momentum",  "trend_momentum", False),
        ("② IC优化: ic_optimized",   "ic_optimized",   False),
        ("③ 板块中性: trend_mom",    "trend_momentum", True),
        ("④ 全量: ic_opt+板块中性",  "ic_optimized",   True),
    ]

    results = []
    for name, preset, sn in configs:
        print(f"\n  {name}...")
        r = run_config(name, preset, sn, quick)
        results.append(r)
        print(f"    收益={r['total_return']:+.1f}% 超额={r['excess']:+.1f}% 交易={r['trades']}")

    # 对比表
    print(f"\n{'='*70}")
    print(f"  对 比 报 告")
    print(f"{'='*70}")
    print(f"{'配置':<30} {'策略收益':>10} {'基准收益':>10} {'超额':>10} {'交易':>6}")
    print("-"*65)
    for r in results:
        mark = "🏆" if r['excess'] > max(0, *[x['excess'] for x in results if x!=r]) else ""
        print(f"{r['name']:<30} {r['total_return']:>+9.1f}% {r['benchmark']:>+9.1f}% {r['excess']:>+9.1f}% {r['trades']:>6} {mark}")
    print("-"*65)

    # 最优
    best = max(results, key=lambda x: x['excess'])
    print(f"\n  最优配置: {best['name']} → 超额{best['excess']:+.1f}%")

    # 保存测试记录
    df = pd.DataFrame(results)
    path = os.path.join(os.path.dirname(__file__), f"test_results_{datetime.now():%Y%m%d_%H%M}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  记录已保存: {path}")
    print(f"  DB记录: {len(storage.get_backtests(limit=100))} 条")


if __name__ == "__main__":
    main()
