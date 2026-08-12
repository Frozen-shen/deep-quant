"""券商执行状态端点（默认 PaperAdapter 降级）。"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter  # noqa: E402
from execution.broker import get_adapter  # noqa: E402
import storage  # noqa: E402

router = APIRouter(prefix="/api/broker", tags=["broker"])
_adapter = get_adapter("paper")  # config.yaml broker.adapter 可切换 qmt

# v24e (2026-08-12): 最新生产实验 — POV 执行 + 修复后分钟数据 (volume单位统一股),
# 含真实逐笔成交 + 每笔 POV 时段成交时间 (fill_times)
V24B_RESULT = (Path(__file__).resolve().parents[3] / "data" / "ic_validation"
               / "walkforward_results_v24e_pov.json")


@router.get("/status")
def broker_status():
    return {
        "adapter": "paper",
        "connected": _adapter.connect(),
        "balance": _adapter.get_balance(),
        "positions": _adapter.get_positions(),
        "orders": _adapter.get_orders(""),
        "trades": _adapter.get_trades(""),
    }


@router.get("/trades")
def broker_trades(year: int | None = None, limit: int = 2000):
    """按年份查询成交记录（2021-2024 回测成交 / 2026 模拟盘实盘）。"""
    rows = storage.get_trades(year=year, limit=limit)
    return {"year": year, "count": len(rows), "trades": rows}


def _load_v24b_extend() -> dict | None:
    """加载 v24b EXTEND 模拟考数据: {positions_history, equity_curve, weights_evolution}。"""
    if not V24B_RESULT.exists():
        return None
    try:
        d = json.loads(V24B_RESULT.read_text(encoding="utf-8"))
        return d.get("results", {}).get("extend_val", {})
    except Exception:
        return None


def _reconstruct_v24b_trades(ev: dict) -> list[dict]:
    """按回测真实逻辑重建成交记录 (EXTEND 2025-01~2026-06, 18 次调仓)。

    模拟规则 (与 run_walkforward_backtest 一致):
      - 初始资金 100,000, top_k=30, 整手 100 股
      - 成交价 = 调仓日 VWAP (v24b 为 VWAP 执行)
      - 首次建仓等权分配; 之后 diff: 新增=买入, 退出=卖出 (数量=持仓)
      - 佣金: 买 0.00025 / 卖 0.00075
    说明: 回测 JSON 只存持仓列表 (无逐笔价格/数量), 本函数按上述规则
    重建, 价格/数量为近似值 (非逐笔真实成交, 前端已注明)。
    """
    import pandas as pd

    BASE = Path(__file__).resolve().parents[3]
    u_dir = BASE / "data_cache" / "unadjusted"
    adj_dir = BASE / "data_store"

    # 模块级 VWAP 缓存: {sym: DataFrame(date 索引 × vwap)} 或 None
    _cache = getattr(_reconstruct_v24b_trades, "_vwap_cache", {})

    def vwap(sym: str, date_str: str) -> float | None:
        """单只股票单日 VWAP (未复权 amount/volume × 复权因子, 单位自动检测)。"""
        nonlocal _cache
        if sym not in _cache:
            upath = u_dir / f"{sym}.parquet"
            apath = adj_dir / f"{sym}.parquet"
            if not (upath.exists() and apath.exists()):
                _cache[sym] = None
            else:
                try:
                    u = pd.read_parquet(upath, columns=["date", "close", "amount", "volume"])
                    a = pd.read_parquet(apath, columns=["date", "close"])
                    u["date"] = pd.to_datetime(u["date"])
                    a["date"] = pd.to_datetime(a["date"])
                    m = u.merge(a, on="date", suffixes=("_u", "_adj"), how="inner")
                    m = m[(m["volume"] > 0) & (m["amount"] > 0) & (m["close_u"] > 0)]
                    if len(m) == 0:
                        _cache[sym] = None
                    else:
                        per = (m["amount"] / m["volume"]).median()
                        med_close = m["close_u"].median()
                        vol_factor = 100.0 if med_close * 50 < per < med_close * 200 else 1.0
                        m["vwap"] = (m["amount"] / (m["volume"] * vol_factor)) * \
                                    (m["close_adj"] / m["close_u"])
                        m = m[m["vwap"].notna() & (m["vwap"] > 0)]
                        _cache[sym] = m[["date", "vwap"]].set_index("date").sort_index() \
                            if len(m) else None
                except Exception:
                    _cache[sym] = None
        _reconstruct_v24b_trades._vwap_cache = _cache
        vf = _cache.get(sym)
        if vf is None or len(vf) == 0:
            return None
        t = pd.Timestamp(date_str)
        sub = vf[vf.index <= t]
        return float(sub["vwap"].iloc[-1]) if len(sub) else None

    pts = [p for p in ev.get("positions_history", []) if p.get("positions")]
    trades = []
    cash = 100_000.0
    holdings: dict[str, int] = {}
    lot = 100
    prev: set[str] = set()

    for pt in pts:
        cur = set(pt["positions"])
        date = pt["date"]
        # 卖出: 退出的持仓 (数量=持仓, 价=VWAP, 卖佣 0.00075)
        for s in sorted(prev - cur):
            qty = holdings.get(s, 0)
            if qty <= 0:
                continue
            px = vwap(s, date) or 0.0
            commission = round(qty * px * 0.00075, 2) if px > 0 else 0.0
            cash += qty * px - commission
            trades.append({"date": date, "symbol": s, "action": "SELL",
                           "qty": qty, "price": round(px, 2), "commission": commission,
                           "reason": "v24b实验"})
        # 买入: 新增持仓, 等权分配剩余现金 (留 1% 余量)
        to_buy = sorted(cur - prev)
        if to_buy:
            per = cash * 0.99 / len(to_buy)
            for s in to_buy:
                px = vwap(s, date) or 0.0
                if px <= 0:
                    continue
                qty = int(per / px / lot) * lot
                if qty < lot:
                    continue
                cost = qty * px
                if cost > cash * 0.99:
                    qty = int(cash * 0.99 / px / lot) * lot
                    cost = qty * px
                if qty < lot:
                    continue
                commission = round(cost * 0.00025, 2)
                cash -= cost + commission
                holdings[s] = qty
                trades.append({"date": date, "symbol": s, "action": "BUY",
                               "qty": qty, "price": round(px, 2), "commission": commission,
                               "reason": "v24b实验"})
        prev = cur
    return trades


@router.get("/backtest-trades")
def backtest_trades():
    """v24b 最优实验 (EXTEND 模拟考) 的调仓记录。

    优先使用回测 JSON 中记录的**真实逐笔成交** (v24c 重跑后写入,
    含真实成交价/数量/佣金); 旧版 JSON 无 trades 字段时回退重建近似。
    """
    ev = _load_v24b_extend()
    if not ev:
        return {"available": False, "count": 0, "trades": [],
                "note": "walkforward_results_v24b_vwap.json 不存在"}
    # 真实逐笔成交 (v24c 重跑产物) vs 重建近似
    real_trades = ev.get("trades") or []
    if real_trades:
        trades = [dict(t) for t in real_trades]
        source = "real"
    else:
        trades = _reconstruct_v24b_trades(ev)
        source = "reconstructed"
    # 附带: 净值曲线 / 超额收益 / 调仓点 (供前端图表)
    curve = ev.get("equity_curve") or []
    active = ev.get("daily_active_returns") or []
    rebalances = [
        {"date": p["date"], "n_positions": len(p.get("positions", []))}
        for p in (ev.get("positions_history") or []) if p.get("positions")]
    return {
        "available": True,
        "version": "v24b (VWAP执行+10bps残差)",
        "period": ev.get("period"),
        "excess_annual": ev.get("excess_annual"),
        "sharpe": ev.get("sharpe"),
        "max_drawdown": ev.get("max_drawdown"),
        "n_rebalances": ev.get("n_rebalances"),
        "count": len(trades),
        "trades": trades,
        "source": source,               # real=回测真实逐笔 / reconstructed=近似重建
        "equity_curve": curve,          # [{date, equity}]
        "active_returns": active,       # [float] 日超额收益 (与 equity_curve 同序)
        "rebalances": rebalances,       # [{date, n_positions}]
    }
