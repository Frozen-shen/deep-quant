"""web/api/aggregators.py — 个股盈亏 FIFO 聚合 + 基准曲线 (纯函数, 回测/模拟盘共用)。"""
import sys
from pathlib import Path
from collections import defaultdict, deque

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))


def aggregate_stock_pnl(trades: list[dict]) -> list[dict]:
    """FIFO 配对买卖计算每股已实现盈亏。

    trades: [{date, symbol, action(BUY/SELL), price, qty, commission}]
    Returns: [{symbol, total_pnl, realized_pnl, n_round_trips, win_rate,
               buy_count, sell_count, open_qty}] (total_pnl=已实现, 已扣佣金)
    """
    # symbol -> deque of (price, qty)
    open_lots = defaultdict(deque)
    # symbol -> {pnl 累计, 已平仓次数, 盈利次数}
    stats = defaultdict(lambda: {"pnl": 0.0, "closed": 0, "wins": 0})
    counts = defaultdict(lambda: {"buy": 0, "sell": 0})

    for t in trades:
        sym = t["symbol"]
        action = (t.get("action") or "").upper()
        qty = float(t.get("qty") or 0)
        price = float(t.get("price") or 0)
        comm = float(t.get("commission") or 0)
        if qty <= 0 or price <= 0 or action not in ("BUY", "SELL"):
            continue
        if action == "BUY":
            counts[sym]["buy"] += 1
            # 买佣金均摊到每股
            open_lots[sym].append((price + comm / qty, qty))
        else:  # SELL
            counts[sym]["sell"] += 1
            sell_comm_per = comm / qty
            remaining = qty
            lot_pnls = []
            while remaining > 0 and open_lots[sym]:
                lot_price, lot_qty = open_lots[sym][0]
                take = min(remaining, lot_qty)
                pnl = (price - sell_comm_per - lot_price) * take
                lot_pnls.append(pnl)
                remaining -= take
                if take >= lot_qty:
                    open_lots[sym].popleft()
                else:
                    open_lots[sym][0] = (lot_price, lot_qty - take)
            if lot_pnls:
                round_pnl = sum(lot_pnls)
                stats[sym]["pnl"] += round_pnl
                stats[sym]["closed"] += 1
                if round_pnl > 0:
                    stats[sym]["wins"] += 1

    out = []
    for sym in sorted(set(counts) | set(stats)):
        s = stats[sym]
        open_qty = sum(lot[1] for lot in open_lots[sym])
        closed = s["closed"]
        out.append({
            "symbol": sym,
            "total_pnl": round(s["pnl"], 2),
            "realized_pnl": round(s["pnl"], 2),
            "n_round_trips": closed,
            "win_rate": round(s["wins"] / closed, 4) if closed else None,
            "buy_count": counts[sym]["buy"],
            "sell_count": counts[sym]["sell"],
            "open_qty": open_qty,
        })
    return out


def build_benchmark_curve(start: str, end: str) -> list[dict]:
    """中证1000 收盘序列 (data/cache/index_csi1000.parquet), 裁剪到 [start, end]."""
    path = BASE_DIR / "data" / "cache" / "index_csi1000.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    sub = df[mask]
    return [{"date": str(r["date"])[:10], "close": float(r["close"])}
            for _, r in sub.iterrows()]
