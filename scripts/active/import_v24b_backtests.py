"""
scripts/active/import_v24b_backtests.py — 将 v24b 最优实验写入 SQLite backtests 表

把 walkforward_results_v24b_vwap.json 的 fold_1~5 + extend_val 聚合指标
导入 storage.save_backtest (backtests 表, 专为回测记录设计, 0 行未用)。
模拟盘表 (trades/positions/equity_log) 不受影响。

幂等: 按 strategy 名 + start_date 去重 (已存在则跳过)。
用法:
  py scripts/active/import_v24b_backtests.py
  py scripts/active/import_v24b_backtests.py --dry-run
"""
import argparse
import json
import os
import sqlite3
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from storage import save_backtest, DB_PATH

SRC = os.path.join(BASE_DIR, "data", "ic_validation", "walkforward_results_v24b_vwap.json")


def metric_map(v: dict) -> dict:
    """v24b JSON 字段 → save_backtest metrics 字段"""
    return {
        "total_return": v.get("total_return"),
        "annual_return": v.get("annual_return"),
        "sharpe_ratio": v.get("sharpe"),
        "max_drawdown": v.get("max_drawdown"),
        "calmar_ratio": v.get("calmar"),
        "benchmark_return": v.get("benchmark_annual"),
        "excess_vs_benchmark": v.get("excess_annual"),
        "final_equity": None,  # 回测以比例计, 无绝对权益
        "total_trades": v.get("n_rebalances", 0) * 2,  # 近似: 调仓次数×双边
        "win_rate": None,
    }


def already_exists(conn, strategy: str, start: str, end: str) -> bool:
    n = conn.execute(
        "SELECT COUNT(*) FROM backtests WHERE strategy=? AND start_date=? AND end_date=?",
        (strategy, start, end)).fetchone()[0]
    return n > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    v24 = json.load(open(SRC, encoding="utf-8"))
    r = v24.get("results", {})
    meta = v24.get("meta", {})

    entries = []
    for fk in ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]:
        f = r.get(fk, {})
        if not f:
            continue
        entries.append((
            "v24b_walkforward", "all", fk,
            f.get("period", "").split(" ~ ")[0],
            f.get("period", "").split(" ~ ")[-1],
            {"n_selected_factors": f.get("n_selected_factors"),
             "n_days": f.get("n_days"),
             "source": "walkforward_results_v24b_vwap.json",
             "version": "v24b (VWAP执行+10bps残差)"},
            metric_map(f),
            f"fold {fk}: train={f.get('train')} val={f.get('val')}",
        ))
    ev = r.get("extend_val", {})
    if ev:
        entries.append((
            "v24b_walkforward", "all", "extend_val",
            ev.get("period", "").split(" ~ ")[0],
            ev.get("period", "").split(" ~ ")[-1],
            {"n_days": ev.get("n_days"), "avg_turnover": ev.get("avg_turnover"),
             "avg_pit_size": ev.get("avg_pit_size"),
             "avg_n_selected_factors": ev.get("avg_n_selected_factors"),
             "source": "walkforward_results_v24b_vwap.json",
             "version": "v24b (VWAP执行+10bps残差)"},
            metric_map(ev),
            "EXTEND 模拟考 (2025-01~2026-06, TEST①毕业段)",
        ))

    print(f"待导入: {len(entries)} 条 (fold_1~5 + extend_val)")
    for e in entries:
        # entries = (sym, mkt, strat, start, end, params, metrics, notes)
        m = e[6]
        print(f"  {e[2]:<12} {e[3]} ~ {e[4]} | excess={m.get('excess_vs_benchmark'):+.1f}% "
              f"sharpe={m.get('sharpe_ratio')} maxdd={m.get('max_drawdown')}%")

    if args.dry_run:
        print("\n[dry-run] 未写入")
        return 0

    conn = sqlite3.connect(DB_PATH)
    inserted = skipped = 0
    for sym, mkt, strat, start, end, params, metrics, notes in entries:
        if already_exists(conn, strat, start, end):
            skipped += 1
            continue
        bid = save_backtest(sym, mkt, strat, start, end, params, metrics, notes)
        inserted += 1
        print(f"  ✅ id={bid}: {strat} {start}~{end}")
    conn.close()
    print(f"\n完成: 新增 {inserted}, 跳过(已存在) {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
