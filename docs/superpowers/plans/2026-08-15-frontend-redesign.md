# 前端全面重构（实验为中心 + 可插拔）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 quant-starter 看板重构为"实验为中心 + 四区导航"架构：新实验 JSON 零代码接入前端、个股盈亏（回测+模拟盘）、交易详情、业界标准图表（净值三件套/月度热力图/盈亏贡献/买卖标记）。

**Architecture:** 后端加实验注册表 API（扫描 JSON 目录 → 统一 metrics/series/trades/stock_pnl schema）+ 个股盈亏 FIFO 聚合；前端加全局实验 Context（URL 同步）+ 通用渲染组件（按 schema 渲染，无实验专属代码）。spec: `docs/superpowers/specs/2026-08-15-frontend-redesign-design.md`

**Tech Stack:** FastAPI (后端, :8000)、React 19 + antd 6 + echarts 6 + react-query 5 + TS 6 + Vite 8 (前端, :5173)、pytest (后端测试)、Playwright (验收截图)

## Global Constraints

- 所有参数走 `config.yaml`，本计划不动任何回测参数
- Python 用 `py` 启动（`python` 命令在本机不可用）；命令在 `C:\Users\Frozen\ZCodeProject\quant-starter` 下执行（`web/ui` 下用 `npm`）
- 前端配色遵守 A 股习惯：**涨红 (#cf1322) / 跌绿 (#3f8600)**（现有 Trading.tsx 已如此）
- 后端路由前缀 `/api`，响应 JSON；坏 JSON 静默跳过（try/except continue）
- 每次改动后跑 `py -m pytest tests/ -q`（61 项）+ `py -m pytest web/api/tests/ -q`；前端每次改动后 `npm run build` 验证 TS
- 提交信息用中文 feat/fix 前缀（现有 git log 风格）
- 实验 id 约定：文件名去 `.json` 后缀（如 `walkforward_results_v24e_pov`）；registry 按 generated_at 倒序

---

## File Structure

**后端（新建/修改）：**
- Create: `web/api/aggregators.py` — 纯函数聚合：`aggregate_stock_pnl(trades)`（FIFO 配对）、`build_benchmark_curve(start, end)`（中证1000 收盘）
- Modify: `web/api/routers/experiments.py` — 重写：registry 扫描 + walkforward/exp schema 解析
- Modify: `web/api/routers/portfolio.py` — 补现价 + 盈亏率
- Create: `web/api/routers/paper.py` — 模拟盘个股盈亏端点（SQLite trades FIFO + 持仓浮盈亏）
- Modify: `web/api/main.py` — include paper router
- Test: `web/api/tests/test_experiment_registry.py`、`web/api/tests/test_aggregators.py`

**前端（新建/修改）：**
- Create: `web/ui/src/experiment-context.tsx` — 全局实验 Context + URL `?exp=` 同步
- Create: `web/ui/src/components/EquityTriptych.tsx` — 净值+基准+回撤阴影三件套
- Create: `web/ui/src/components/MonthlyHeatmap.tsx` — 月度收益热力图
- Create: `web/ui/src/components/PnlBarChart.tsx` — 盈亏贡献横向条形图
- Create: `web/ui/src/components/ExperimentReport.tsx` — 通用实验报告渲染器
- Modify: `web/ui/src/api.ts` — 新端点 + 类型
- Modify: `web/ui/src/App.tsx` — 四区导航 + Header 实验选择器
- Modify: `web/ui/src/pages/Overview.tsx` — 仪表盘（三件套+热力图+盈亏贡献）
- Modify: `web/ui/src/pages/Experiments.tsx` — 注册表卡片列表 + 报告视图 + 对比模式
- Modify: `web/ui/src/pages/Trading.tsx` — 双 Tab（回测实验/模拟盘）+ 下钻
- Modify: `web/ui/src/pages/Portfolio.tsx` — 现价列 + 盈亏贡献图
- Modify: `web/ui/src/pages/Stocks.tsx` — K 线买卖标记

---

### Task 1: 个股盈亏 FIFO 聚合 + 基准曲线（后端纯函数）

**Files:**
- Create: `web/api/aggregators.py`
- Test: `web/api/tests/test_aggregators.py`

**Interfaces:**
- Consumes: trades 列表 `[{"date","symbol","action","price","qty","commission"}]`（回测 JSON 与 SQLite 同构）
- Produces:
  - `aggregate_stock_pnl(trades: list[dict]) -> list[dict]` → `[{"symbol","total_pnl","n_round_trips","win_rate","realized_pnl","buy_count","sell_count"}]`（total_pnl 已扣佣金）
  - `build_benchmark_curve(start: str, end: str) -> list[dict]` → `[{"date","close"}]`

- [ ] **Step 1: 写失败测试**

```python
# web/api/tests/test_aggregators.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from web.api.aggregators import aggregate_stock_pnl, build_benchmark_curve


def test_fifo_pairing_single_round_trip():
    trades = [
        {"date": "2025-01-02", "symbol": "000001", "action": "BUY", "price": 10.0, "qty": 100, "commission": 5.0},
        {"date": "2025-02-03", "symbol": "000001", "action": "SELL", "price": 11.0, "qty": 100, "commission": 5.0},
    ]
    out = {r["symbol"]: r for r in aggregate_stock_pnl(trades)}
    r = out["000001"]
    # 盈亏 = (11-10)*100 - 佣金10 = 90
    assert r["total_pnl"] == 90.0
    assert r["n_round_trips"] == 1
    assert r["win_rate"] == 1.0
    assert r["realized_pnl"] == 90.0


def test_fifo_partial_sell():
    trades = [
        {"date": "2025-01-02", "symbol": "000001", "action": "BUY", "price": 10.0, "qty": 200, "commission": 5.0},
        {"date": "2025-02-03", "symbol": "000001", "action": "SELL", "price": 11.0, "qty": 100, "commission": 5.0},
    ]
    out = {r["symbol"]: r for r in aggregate_stock_pnl(trades)}
    r = out["000001"]
    # 只卖了一半: (11-10)*100 - 卖佣5 = 95
    assert r["total_pnl"] == 95.0
    assert r["n_round_trips"] == 1
    assert r["sell_count"] == 1 and r["buy_count"] == 1


def test_fifo_multiple_buys_avg_cost():
    trades = [
        {"date": "2025-01-02", "symbol": "000001", "action": "BUY", "price": 10.0, "qty": 100, "commission": 0.0},
        {"date": "2025-01-03", "symbol": "000001", "action": "BUY", "price": 12.0, "qty": 100, "commission": 0.0},
        {"date": "2025-02-03", "symbol": "000001", "action": "SELL", "price": 13.0, "qty": 100, "commission": 0.0},
    ]
    out = {r["symbol"]: r for r in aggregate_stock_pnl(trades)}
    # FIFO: 先卖 10 元买的 100 股 → 赚 300
    assert out["000001"]["total_pnl"] == 300.0


def test_losing_trade_win_rate():
    trades = [
        {"date": "2025-01-02", "symbol": "000001", "action": "BUY", "price": 10.0, "qty": 100, "commission": 0.0},
        {"date": "2025-02-03", "symbol": "000001", "action": "SELL", "price": 9.0, "qty": 100, "commission": 0.0},
        {"date": "2025-01-02", "symbol": "000002", "action": "BUY", "price": 10.0, "qty": 100, "commission": 0.0},
        {"date": "2025-02-03", "symbol": "000002", "action": "SELL", "price": 11.0, "qty": 100, "commission": 0.0},
    ]
    out = {r["symbol"]: r for r in aggregate_stock_pnl(trades)}
    assert out["000001"]["win_rate"] == 0.0
    assert out["000002"]["win_rate"] == 1.0


def test_benchmark_curve_bounds():
    curve = build_benchmark_curve("2025-01-01", "2025-06-30")
    assert len(curve) > 50
    assert curve[0]["date"] >= "2025-01-01"
    assert curve[-1]["date"] <= "2025-06-30"
    assert all(c["close"] > 0 for c in curve)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `py -m pytest web/api/tests/test_aggregators.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现**

```python
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
    # symbol -> deque of (price, qty, buy_commission_share)
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
            open_lots[sym].append((price + comm / qty, qty, qty))
        else:  # SELL
            counts[sym]["sell"] += 1
            sell_comm_per = comm / qty
            remaining = qty
            lot_pnls = []
            while remaining > 0 and open_lots[sym]:
                lot_price, lot_qty, lot_total = open_lots[sym][0]
                take = min(remaining, lot_qty)
                pnl = (price - sell_comm_per - lot_price) * take
                lot_pnls.append(pnl)
                remaining -= take
                if take >= lot_qty:
                    open_lots[sym].popleft()
                else:
                    open_lots[sym][0] = (lot_price, lot_qty - take, lot_total - take)
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `py -m pytest web/api/tests/test_aggregators.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add web/api/aggregators.py web/api/tests/test_aggregators.py
git commit -m "feat(web-api): 个股盈亏FIFO聚合 + 中证1000基准曲线纯函数"
```

---

### Task 2: 实验注册表 API（扫描 + 统一 schema）

**Files:**
- Modify: `web/api/routers/experiments.py`（整体重写）
- Test: `web/api/tests/test_experiment_registry.py`

**Interfaces:**
- Consumes: `aggregate_stock_pnl`, `build_benchmark_curve`（Task 1）
- Produces:
  - `GET /api/experiments/registry` → `{"count": n, "experiments": [{id, kind, name, generated_at, description, has_trades, summary: {excess_annual, sharpe, max_drawdown, total_return}}]}`
  - `GET /api/experiments/{exp_id}` → 完整 schema（见下方 Step 3 代码注释）
  - 兼容旧 `GET /api/experiments`（原行为，列表 exp_*.json）——前端 Signals/Factors 页可能引用，先保留

- [ ] **Step 1: 写失败测试**

```python
# web/api/tests/test_experiment_registry.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient
from web.api.main import app

client = TestClient(app)


def test_registry_lists_walkforward_results():
    resp = client.get("/api/experiments/registry")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] > 0
    ids = [e["id"] for e in data["experiments"]]
    # v24e/v25 等 walkforward 结果应在注册表中
    assert any("walkforward_results_v24e_pov" in i for i in ids)
    for e in data["experiments"]:
        assert set(["id", "kind", "name", "generated_at"]) <= set(e.keys())


def test_registry_sorted_by_generated_at_desc():
    resp = client.get("/api/experiments/registry")
    data = resp.json()
    exps = data["experiments"]
    gen = [e.get("generated_at", "") for e in exps]
    assert gen == sorted(gen, reverse=True)


def test_experiment_detail_schema():
    resp = client.get("/api/experiments/walkforward_results_v24e_pov")
    assert resp.status_code == 200
    d = resp.json()
    assert d["meta"]["id"] == "walkforward_results_v24e_pov"
    assert isinstance(d["metrics"], list) and len(d["metrics"]) >= 6
    for m in d["metrics"]:
        assert set(["key", "label", "value", "format", "better"]) <= set(m.keys())
    assert isinstance(d["series"], list) and len(d["series"]) >= 2
    assert isinstance(d["folds"], list)
    assert isinstance(d["stock_pnl"], list) and len(d["stock_pnl"]) > 0
    assert isinstance(d["trades"], list) and len(d["trades"]) > 0
    # stock_pnl 字段完整
    sp = d["stock_pnl"][0]
    assert set(["symbol", "total_pnl", "n_round_trips", "win_rate"]) <= set(sp.keys())


def test_experiment_detail_404():
    resp = client.get("/api/experiments/not_exist_xyz")
    assert resp.status_code == 404


def test_legacy_endpoint_kept():
    resp = client.get("/api/experiments")
    assert resp.status_code == 200
    assert "count" in resp.json()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `py -m pytest web/api/tests/test_experiment_registry.py -q`
Expected: FAIL（registry 路由不存在）

- [ ] **Step 3: 重写路由**

```python
"""实验记录端点: 注册表扫描 + 统一 schema (可插拔核心)。"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter, HTTPException  # noqa: E402
from web.api import config  # noqa: E402
from web.api.aggregators import aggregate_stock_pnl, build_benchmark_curve  # noqa: E402

router = APIRouter(prefix="/api", tags=["experiments"])


def _iter_walkforward_files():
    """data/ic_validation/walkforward_results_v*.json (跳过 *_bak_* 备份)。"""
    if not config.IC_DIR.exists():
        return
    for p in sorted(config.IC_DIR.glob("walkforward_results_v*.json")):
        if "_bak_" in p.stem or p.stem.endswith("_results"):
            continue
        yield p


def _load_walkforward(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _walkforward_meta(path: Path, d: dict) -> dict:
    meta = d.get("meta", {}) or {}
    results = d.get("results", {}) or {}
    ev = results.get("extend_val", {}) or {}
    return {
        "id": path.stem,
        "kind": "walkforward",
        "name": meta.get("description") or path.stem,
        "generated_at": meta.get("generated_at", ""),
        "has_trades": len(ev.get("trades", [])) > 0,
        "summary": {
            "excess_annual": ev.get("excess_annual"),
            "sharpe": ev.get("sharpe"),
            "max_drawdown": ev.get("max_drawdown"),
            "total_return": ev.get("total_return"),
        },
    }


def _walkforward_schema(path: Path, d: dict) -> dict:
    """walkforward 结果 → 统一 schema。extend_val 为展示主体, folds 作分段成绩。"""
    meta = _walkforward_meta(path, d)
    results = d.get("results", {}) or {}
    ev = results.get("extend_val", {}) or {}
    period = ev.get("period", "")
    p_start, _, p_end = (period.split(" ~ ") + ["", ""])[:3]

    def metric(key, label, value, fmt="pct", better="high"):
        return {"key": key, "label": label, "value": value, "format": fmt, "better": better}

    metrics = [
        metric("excess_annual", "年化超额", ev.get("excess_annual"), "pct", "high"),
        metric("total_return", "总收益", ev.get("total_return"), "pct", "high"),
        metric("annual_return", "年化收益", ev.get("annual_return"), "pct", "high"),
        metric("sharpe", "Sharpe", ev.get("sharpe"), "num", "high"),
        metric("max_drawdown", "最大回撤", ev.get("max_drawdown"), "pct", "low"),
        metric("calmar", "Calmar", ev.get("calmar"), "num", "high"),
        metric("ir", "IR", ev.get("ir"), "num", "high"),
        metric("avg_turnover", "平均换手", ev.get("avg_turnover"), "pct", "low"),
        metric("n_rebalances", "调仓次数", ev.get("n_rebalances"), "num", "high"),
    ]

    eq = ev.get("equity_curve", []) or []
    series = [
        {"name": "组合净值", "type": "line",
         "x": [p["date"] for p in eq], "y": [p["equity"] for p in eq]},
        {"name": "基准(中证1000归一)", "type": "line",
         "x": [], "y": []},
    ]
    # 基准曲线: 与净值起点同基 (中证1000收盘 / 首日 × 初始资金)
    bench = build_benchmark_curve(p_start or "2025-01-01", p_end or "2026-06-30")
    if bench and eq:
        base = bench[0]["close"]
        eq0 = eq[0]["equity"] if eq else 100000.0
        series[1]["x"] = [b["date"] for b in bench]
        series[1]["y"] = [round(b["close"] / base * eq0, 2) for b in bench]

    folds = []
    for k in sorted(results.keys()):
        if not k.startswith("fold_"):
            continue
        f = results[k]
        folds.append({
            "name": k, "train": f.get("train", ""), "val": f.get("val", ""),
            "excess_annual": f.get("excess_annual"),
            "sharpe": f.get("sharpe"), "max_drawdown": f.get("max_drawdown"),
            "ir": f.get("ir"), "avg_turnover": f.get("avg_turnover"),
        })

    trades = ev.get("trades", []) or []
    stock_pnl = aggregate_stock_pnl(trades)

    return {"meta": meta, "metrics": metrics, "series": series, "folds": folds,
            "stock_pnl": stock_pnl, "trades": trades,
            "equity_curve": eq, "benchmark_curve": bench}


def _exp_meta(path: Path, d: dict) -> dict:
    return {"id": path.stem, "kind": "experiment",
            "name": d.get("script", path.stem), "generated_at": d.get("timestamp", ""),
            "has_trades": False, "summary": {}}


def _exp_schema(path: Path, d: dict) -> dict:
    """旧 exp_*.json (KV 自由格式) → 兜底 schema: 仅 metrics, 由前端 KV 表格渲染。"""
    meta = _exp_meta(path, d)
    metrics = []
    results = d.get("results", {}) or {}
    if isinstance(results, dict):
        for k, v in results.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                metrics.append({"key": k, "label": k, "value": v,
                                "format": "num", "better": "high"})
            elif isinstance(v, str):
                metrics.append({"key": k, "label": k, "value": v,
                                "format": "str", "better": "high"})
    params = d.get("parameters", {}) or {}
    param_str = json.dumps(params, ensure_ascii=False)[:200] if params else ""
    return {"meta": {**meta, "description": param_str}, "metrics": metrics,
            "series": [], "folds": [], "stock_pnl": [], "trades": [],
            "equity_curve": [], "benchmark_curve": []}


def _registry() -> list[dict]:
    items = []
    for p in _iter_walkforward_files():
        d = _load_walkforward(p)
        if d:
            items.append(_walkforward_meta(p, d))
    if config.EXPERIMENTS_DIR.exists():
        for p in sorted(config.EXPERIMENTS_DIR.glob("exp_*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                items.append(_exp_meta(p, d))
            except Exception:
                continue
    items.sort(key=lambda x: x.get("generated_at", ""), reverse=True)
    return items


@router.get("/experiments/registry")
def get_registry():
    exps = _registry()
    return {"count": len(exps), "experiments": exps}


@router.get("/experiments/{exp_id}")
def get_experiment(exp_id: str):
    p = config.IC_DIR / f"{exp_id}.json"
    if p.exists():
        d = _load_walkforward(p)
        if d:
            return _walkforward_schema(p, d)
    p2 = config.EXPERIMENTS_DIR / f"{exp_id}.json"
    if config.EXPERIMENTS_DIR.exists() and p2.exists():
        try:
            d = json.loads(p2.read_text(encoding="utf-8"))
            return _exp_schema(p2, d)
        except Exception:
            pass
    raise HTTPException(status_code=404, detail=f"experiment {exp_id} not found")


# ── 旧端点保留 (兼容现有前端) ──
@config.ttl_cache(60)
def load_experiments():
    exps = []
    if config.EXPERIMENTS_DIR.exists():
        for p in sorted(config.EXPERIMENTS_DIR.glob("exp_*.json")):
            try:
                exps.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
    return exps


@router.get("/experiments")
def get_experiments():
    exps = load_experiments()
    by_script = {}
    by_partition = {}
    for e in exps:
        by_script[e.get("script", "?")] = by_script.get(e.get("script", "?"), 0) + 1
        by_partition[e.get("partition", "?")] = by_partition.get(e.get("partition", "?"), 0) + 1
    return {"count": len(exps), "experiments": exps[-100:],
            "by_script": by_script, "by_partition": by_partition}
```

注意：`_iter_walkforward_files` 的过滤条件需正确排除 `walkforward_results.json`（最新输出，无版本号）与 `_bak_` 备份——`glob("walkforward_results_v*.json")` 只匹配带版本号的，`walkforward_results.json` 不匹配（无 `_v`），天然排除。

- [ ] **Step 4: 跑测试确认通过**

Run: `py -m pytest web/api/tests/test_experiment_registry.py -q`
Expected: PASS (6 passed)。若 v24e JSON 的 extend_val 缺某些字段导致断言失败，检查 `walkforward_results_v24e_pov.json` 的 `results.extend_val` 键（已知含 trades/equity_curve/excess_annual 等）。

- [ ] **Step 5: 回归旧测试**

Run: `py -m pytest web/api/tests/ -q`
Expected: 全部 PASS（旧 experiments 端点行为不变）

- [ ] **Step 6: Commit**

```bash
git add web/api/routers/experiments.py web/api/tests/test_experiment_registry.py
git commit -m "feat(web-api): 实验注册表API — 扫描walkforward/exp JSON统一schema(可插拔)"
```

---

### Task 3: portfolio 现价 + 模拟盘个股盈亏端点

**Files:**
- Modify: `web/api/routers/portfolio.py`
- Create: `web/api/routers/paper.py`
- Modify: `web/api/main.py`
- Test: `web/api/tests/test_aggregators.py`（追加 2 个测试到现有文件）

**Interfaces:**
- Consumes: `storage.get_trades(year=None, limit=N)`, `storage.get_all_positions()`, `config.PAPER_PORTFOLIO`
- Produces:
  - `GET /api/portfolio` 的 positions 每项新增 `current_price`（data_store 日线最后 close）与 `pnl`, `pnl_pct`
  - `GET /api/paper/stock-pnl` → `{"items": [{symbol, name, realized_pnl, win_rate, n_round_trips, open_qty, current_price, unrealized_pnl, avg_cost}]}`（realized 用 Task 1 FIFO，unrealized = (现价-成本)×持仓）

- [ ] **Step 1: 追加失败测试**

```python
# 追加到 web/api/tests/test_aggregators.py 末尾
from fastapi.testclient import TestClient
from web.api.main import app

client = TestClient(app)


def test_paper_stock_pnl_endpoint():
    resp = client.get("/api/paper/stock-pnl")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    for it in data["items"]:
        assert set(["symbol", "realized_pnl", "n_round_trips"]) <= set(it.keys())


def test_portfolio_has_current_price():
    resp = client.get("/api/portfolio")
    assert resp.status_code == 200
    data = resp.json()
    for p in data.get("positions", []):
        # 有持仓时必含现价; 空仓时不要求
        assert "current_price" in p
```

- [ ] **Step 2: 跑测试确认失败**

Run: `py -m pytest web/api/tests/test_aggregators.py -q`
Expected: FAIL（/api/paper/stock-pnl 404）

- [ ] **Step 3: 实现 portfolio 现价**

```python
# web/api/routers/portfolio.py — 在 load_portfolio 后补现价
"""组合端点：读 paper_trade/portfolio.json + storage。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402
from fastapi import APIRouter  # noqa: E402
import storage  # noqa: E402
from web.api import config  # noqa: E402

router = APIRouter(prefix="/api", tags=["portfolio"])


@config.ttl_cache(30)
def load_portfolio():
    return config.read_json(config.PAPER_PORTFOLIO)


def _current_price(symbol: str) -> float | None:
    """data_store 日线最后一根 close (与模拟盘 market_value 同日口径)。"""
    path = config.DATA_STORE / f"{symbol}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, columns=["date", "close"])
        if len(df) == 0:
            return None
        df = df.sort_values("date")
        return float(df["close"].iloc[-1])
    except Exception:
        return None


@router.get("/portfolio")
def get_portfolio():
    pf = load_portfolio() or {}
    positions = []
    for sym, p in (pf.get("positions") or {}).items():
        cur = _current_price(sym)
        qty = p.get("qty") or 0
        avg = p.get("avg_cost") or 0
        mv = p.get("market_value") or 0
        cost = avg * qty
        pnl = mv - cost
        positions.append({
            "symbol": sym, "qty": qty, "avg_cost": avg,
            "market_value": mv, "entry_date": p.get("entry_date"),
            "current_price": cur,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / cost, 4) if cost else None,
        })
    trades = storage.get_trades(limit=50)
    return {"cash": pf.get("cash"), "initial_capital": pf.get("initial_capital"),
            "inception_date": pf.get("inception_date"), "positions": positions,
            "recent_trades": trades}
```

- [ ] **Step 4: 实现 paper 路由 + 注册**

```python
# web/api/routers/paper.py
"""模拟盘个股盈亏端点 (SQLite trades FIFO + 持仓浮盈亏)。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402
from fastapi import APIRouter  # noqa: E402
import storage  # noqa: E402
from web.api import config  # noqa: E402
from web.api.aggregators import aggregate_stock_pnl  # noqa: E402

router = APIRouter(prefix="/api/paper", tags=["paper"])


@router.get("/stock-pnl")
def paper_stock_pnl():
    rows = storage.get_trades(limit=10000)
    trades = [{"date": r["date"], "symbol": r["symbol"], "action": r["action"],
               "price": r["price"], "qty": r["qty"], "commission": r["commission"]}
              for r in rows]
    agg = {r["symbol"]: r for r in aggregate_stock_pnl(trades)}
    items = []
    for sym, a in agg.items():
        pos = storage.get_position(sym)
        open_qty = (pos or {}).get("qty") or a.get("open_qty") or 0
        avg_cost = (pos or {}).get("avg_cost") or 0
        path = config.DATA_STORE / f"{sym}.parquet"
        cur_price = None
        if path.exists():
            try:
                df = pd.read_parquet(path, columns=["close"])
                cur_price = float(df["close"].iloc[-1]) if len(df) else None
            except Exception:
                pass
        unreal = (round((cur_price - avg_cost) * open_qty, 2)
                  if cur_price and avg_cost and open_qty else None)
        items.append({**a, "open_qty": open_qty, "avg_cost": avg_cost,
                      "current_price": cur_price, "unrealized_pnl": unreal})
    items.sort(key=lambda x: -(x["total_pnl"] + (x["unrealized_pnl"] or 0)))
    return {"items": items, "count": len(items)}
```

`web/api/main.py` 修改（import 行 + include_router）：

```python
from web.api.routers import equity, portfolio, graduation  # noqa: E402
from web.api.routers import signals, experiments, factors, universe, stocks  # noqa: E402
from web.api.routers import broker, decisions, paper  # noqa: E402
# ...
app.include_router(paper.router)
```

- [ ] **Step 5: 跑测试确认通过 + 回归**

Run: `py -m pytest web/api/tests/test_aggregators.py -q && py -m pytest tests/ -q`
Expected: PASS（新 2 项 + 旧 61 项）

- [ ] **Step 6: Commit**

```bash
git add web/api/routers/portfolio.py web/api/routers/paper.py web/api/main.py web/api/tests/test_aggregators.py
git commit -m "feat(web-api): 组合现价/盈亏率 + 模拟盘个股盈亏端点(FIFO)"
```

---

### Task 4: 前端 API 层 + 全局实验 Context

**Files:**
- Modify: `web/ui/src/api.ts`（追加类型与函数）
- Create: `web/ui/src/experiment-context.tsx`

**Interfaces:**
- Consumes: 后端 `/api/experiments/registry`, `/api/experiments/{id}`, `/api/paper/stock-pnl`
- Produces:
  - `fetchExperimentRegistry(): Promise<RegistryResponse>`
  - `fetchExperimentDetail(id: string): Promise<ExperimentDetail>`
  - `fetchPaperStockPnl(): Promise<{items: PaperPnlItem[]}>`
  - `ExperimentProvider` / `useExperiment()` → `{expId, setExpId, registry, detail}`（URL `?exp=` 同步）

- [ ] **Step 1: api.ts 追加**

```typescript
// 追加到 web/ui/src/api.ts 末尾

export interface ExperimentRegistryItem {
  id: string
  kind: 'walkforward' | 'experiment'
  name: string
  generated_at: string
  has_trades: boolean
  summary: { excess_annual?: number | null; sharpe?: number | null; max_drawdown?: number | null; total_return?: number | null }
}
export interface MetricItem { key: string; label: string; value: number | string | null; format: 'pct' | 'num' | 'money' | 'str'; better: 'high' | 'low' }
export interface SeriesItem { name: string; type: 'line' | 'bar'; x: string[]; y: number[] }
export interface FoldResult { name: string; train?: string; val?: string; excess_annual?: number | null; sharpe?: number | null; max_drawdown?: number | null; ir?: number | null; avg_turnover?: number | null }
export interface StockPnlItem { symbol: string; total_pnl: number; realized_pnl: number; n_round_trips: number; win_rate: number | null; buy_count: number; sell_count: number; open_qty: number; current_price?: number | null; unrealized_pnl?: number | null; avg_cost?: number | null }
export interface ExperimentDetail {
  meta: { id: string; kind: string; name: string; generated_at: string; description?: string }
  metrics: MetricItem[]
  series: SeriesItem[]
  folds: FoldResult[]
  stock_pnl: StockPnlItem[]
  trades: any[]
  equity_curve: Array<{ date: string; equity: number }>
  benchmark_curve: Array<{ date: string; close: number }>
}

export async function fetchExperimentRegistry(): Promise<{ count: number; experiments: ExperimentRegistryItem[] }> {
  const { data } = await api.get('/experiments/registry')
  return data
}
export async function fetchExperimentDetail(id: string): Promise<ExperimentDetail> {
  const { data } = await api.get(`/experiments/${id}`)
  return data
}
export async function fetchPaperStockPnl(): Promise<{ items: StockPnlItem[]; count: number }> {
  const { data } = await api.get('/paper/stock-pnl')
  return data
}
```

- [ ] **Step 2: experiment-context.tsx**

```tsx
// web/ui/src/experiment-context.tsx
import { createContext, useContext, useEffect, useMemo, type ReactNode } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { fetchExperimentRegistry, fetchExperimentDetail } from './api'

interface ExperimentCtx {
  expId: string | null
  setExpId: (id: string | null) => void
  registry: { count: number; experiments: any[] } | undefined
  detail: any
  detailLoading: boolean
}

const Ctx = createContext<ExperimentCtx | null>(null)

export function ExperimentProvider({ children }: { children: ReactNode }) {
  const [params, setParams] = useSearchParams()
  const expId = params.get('exp')

  const registry = useQuery({ queryKey: ['exp-registry'], queryFn: fetchExperimentRegistry, staleTime: 60_000 })
  const detail = useQuery({
    queryKey: ['exp-detail', expId],
    queryFn: () => fetchExperimentDetail(expId as string),
    enabled: !!expId,
    staleTime: 60_000,
  })

  // 默认选中最新实验
  useEffect(() => {
    if (!expId && registry.data?.experiments?.length) {
      const latest = registry.data.experiments[0]
      setParams(prev => { const p = new URLSearchParams(prev); p.set('exp', latest.id); return p }, { replace: true })
    }
  }, [expId, registry.data, setParams])

  const value = useMemo(() => ({
    expId,
    setExpId: (id: string | null) => {
      setParams(prev => {
        const p = new URLSearchParams(prev)
        if (id) p.set('exp', id); else p.delete('exp')
        return p
      })
    },
    registry: registry.data,
    detail: detail.data,
    detailLoading: detail.isLoading,
  }), [expId, registry.data, detail.data, detail.isLoading, setParams])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useExperiment(): ExperimentCtx {
  const v = useContext(Ctx)
  if (!v) throw new Error('useExperiment must be used within ExperimentProvider')
  return v
}
```

- [ ] **Step 3: 构建验证**

Run: `cd web/ui && npm run build`
Expected: PASS（TS 编译通过；若 `setParams` 第二参类型报错，删去 `{ replace: true }`）

- [ ] **Step 4: Commit**

```bash
git add web/ui/src/api.ts web/ui/src/experiment-context.tsx
git commit -m "feat(web-ui): API层注册表/实验详情类型 + 全局实验Context(URL同步)"
```

---

### Task 5: 通用图表组件（三件套/热力图/盈亏图/报告渲染器）

**Files:**
- Create: `web/ui/src/components/EquityTriptych.tsx`
- Create: `web/ui/src/components/MonthlyHeatmap.tsx`
- Create: `web/ui/src/components/PnlBarChart.tsx`
- Create: `web/ui/src/components/ExperimentReport.tsx`

**Interfaces:**
- Consumes: `ExperimentDetail`（Task 4 类型）
- Produces（均为默认导出 React 组件）:
  - `EquityTriptych({ equity, benchmark }: { equity: {date,equity}[], benchmark?: {date,close}[] })` — 净值(左轴)+基准(左轴)+回撤阴影(右轴, areaStyle)
  - `MonthlyHeatmap({ dailyReturns }: { dailyReturns: {date:string, ret:number}[] })` — 年×月热力图（红正绿负白零）
  - `PnlBarChart({ items, nameOf }: { items: {symbol,total_pnl}[], nameOf: (s:string)=>string })` — 横向条形图按盈亏排序
  - `ExperimentReport({ detail, nameOf }: { detail: ExperimentDetail, nameOf: (s:string)=>string })` — metrics 卡行 + series 图 + folds 表 + stock_pnl 榜 + trades 表

- [ ] **Step 1: EquityTriptych.tsx**

```tsx
import ReactECharts from 'echarts-for-react'

interface Props { equity: Array<{ date: string; equity: number }>; benchmark?: Array<{ date: string; close: number }> }

export default function EquityTriptych({ equity, benchmark }: Props) {
  if (!equity.length) return null
  const dates = equity.map(p => p.date)
  const eq = equity.map(p => p.equity)
  // 回撤序列 (水下)
  let peak = -Infinity
  const dd = equity.map(p => {
    peak = Math.max(peak, p.equity)
    return Number(((p.equity / peak - 1) * 100).toFixed(2))
  })
  const series: any[] = [
    { name: '组合净值', type: 'line', data: eq, showSymbol: false, lineStyle: { width: 2 } },
    { name: '回撤 %', type: 'line', yAxisIndex: 1, data: dd, showSymbol: false,
      lineStyle: { width: 1, color: '#cf1322' },
      areaStyle: { color: 'rgba(207,19,34,0.12)' } },
  ]
  if (benchmark?.length) {
    const bDates = benchmark.map(p => p.date)
    const bVals = benchmark.map(p => p.close)
    // 仅在同日期窗口内画基准 (xAxis 用组合净值日期)
    const bMap = new Map(bDates.map((d, i) => [d, bVals[i]]))
    series.push({
      name: '基准(中证1000)', type: 'line', data: dates.map(d => bMap.get(d) ?? null),
      showSymbol: false, lineStyle: { width: 1.5, type: 'dashed', color: '#888' },
    })
  }
  return <ReactECharts option={{
    tooltip: { trigger: 'axis' },
    legend: { data: series.map(s => s.name) },
    grid: { left: 60, right: 60, top: 40, bottom: 30 },
    xAxis: { type: 'category', data: dates, boundaryGap: false },
    yAxis: [
      { type: 'value', name: '净值', scale: true },
      { type: 'value', name: '回撤%', max: 0, axisLabel: { formatter: '{value}%' } },
    ],
    series,
  }} style={{ height: 360 }} />
}
```

- [ ] **Step 2: MonthlyHeatmap.tsx**

```tsx
import ReactECharts from 'echarts-for-react'

interface Props { dailyReturns: Array<{ date: string; ret: number }> }

/** 月度收益热力图: 年(行) × 月(列), 红正绿负 (QuantStats 风格)。 */
export default function MonthlyHeatmap({ dailyReturns }: Props) {
  if (!dailyReturns.length) return null
  const byYm = new Map<string, number>()
  for (const d of dailyReturns) {
    const ym = d.date.slice(0, 7)
    byYm.set(ym, (byYm.get(ym) ?? 1) * (1 + d.ret) - 1)
  }
  const years = [...new Set([...byYm.keys()].map(k => k.slice(0, 4)))].sort()
  const months = Array.from({ length: 12 }, (_, i) => String(i + 1).padStart(2, '0'))
  const data: Array<[number, number, number]> = []
  years.forEach((y, yi) => {
    months.forEach((m, mi) => {
      const v = byYm.get(`${y}-${m}`)
      if (v !== undefined) data.push([mi, yi, Number((v * 100).toFixed(2))])
    })
  })
  const maxAbs = Math.max(1, ...data.map(d => Math.abs(d[2])))
  return <ReactECharts option={{
    tooltip: { formatter: (p: any) => `${years[p.value[1]]}-${months[p.value[0]]}: ${p.value[2]}%` },
    grid: { left: 50, right: 20, top: 10, bottom: 40 },
    xAxis: { type: 'category', data: months.map(m => `${Number(m)}月`) },
    yAxis: { type: 'category', data: years },
    visualMap: { min: -maxAbs, max: maxAbs, calculable: true, orient: 'horizontal',
      left: 'center', bottom: 0, inRange: { color: ['#3f8600', '#f0f0f0', '#cf1322'] } },
    series: [{ type: 'heatmap', data,
      label: { show: true, formatter: (p: any) => `${p.value[2] > 0 ? '+' : ''}${p.value[2]}%` } }],
  }} style={{ height: 220 }} />
}
```

- [ ] **Step 3: PnlBarChart.tsx**

```tsx
import ReactECharts from 'echarts-for-react'

interface Props { items: Array<{ symbol: string; total_pnl: number }>; nameOf: (s: string) => string }

/** 盈亏贡献横向条形图: 按盈亏额排序, 红正绿负。 */
export default function PnlBarChart({ items, nameOf }: Props) {
  if (!items.length) return null
  const sorted = [...items].sort((a, b) => a.total_pnl - b.total_pnl)
  const labels = sorted.map(i => `${nameOf(i.symbol)} ${i.symbol}`)
  const vals = sorted.map(i => Number(i.total_pnl.toFixed(0)))
  return <ReactECharts option={{
    tooltip: { trigger: 'axis' },
    grid: { left: 110, right: 30, top: 10, bottom: 30 },
    xAxis: { type: 'value' },
    yAxis: { type: 'category', data: labels },
    series: [{ type: 'bar', data: vals,
      itemStyle: { color: (p: any) => (p.value >= 0 ? '#cf1322' : '#3f8600') } }],
  }} style={{ height: Math.max(200, labels.length * 26) }} />
}
```

- [ ] **Step 4: ExperimentReport.tsx**

```tsx
import { Card, Col, Row, Table, Typography, Empty, Tag } from 'antd'
import ReactECharts from 'echarts-for-react'
import type { ExperimentDetail, MetricItem } from '../api'
import { fmtNum, fmtPct } from '../lib/format'
import EquityTriptych from './EquityTriptych'
import PnlBarChart from './PnlBarChart'

interface Props { detail: ExperimentDetail; nameOf: (s: string) => string }

function metricText(m: MetricItem): string {
  if (m.value === null || m.value === undefined) return '—'
  if (m.format === 'pct') return fmtPct(m.value as number)
  if (m.format === 'str') return String(m.value)
  return fmtNum(m.value as number, 2)
}

/** 通用实验报告渲染器: metrics 卡 → 净值三件套 → folds 表 → 个股盈亏 → 逐笔成交。 */
export default function ExperimentReport({ detail, nameOf }: Props) {
  const { metrics, series, folds, stock_pnl, trades, equity_curve } = detail
  const eqSeries = series.find(s => s.name === '组合净值')
  const benchSeries = series.find(s => s.name.includes('基准'))
  const equity = eqSeries && eqSeries.x.length
    ? eqSeries.x.map((d, i) => ({ date: d, equity: eqSeries.y[i] }))
    : equity_curve
  const benchmark = benchSeries && benchSeries.x.length
    ? benchSeries.x.map((d, i) => ({ date: d, close: benchSeries.y[i] }))
    : undefined

  const pnlColor = (v: number) => (v > 0 ? '#cf1322' : v < 0 ? '#3f8600' : undefined)

  return (
    <div>
      <Row gutter={[12, 12]}>
        {metrics.map(m => (
          <Col key={m.key} xs={12} md={6} xl={4}>
            <Card size="small" title={m.label}>
              <div style={{ fontSize: 18, fontWeight: 600, color: m.better === 'low'
                ? (typeof m.value === 'number' && m.value < 0 ? '#cf1322' : undefined)
                : undefined }}>
                {metricText(m)}
              </div>
            </Card>
          </Col>
        ))}
      </Row>

      <Card title="净值与回撤" style={{ marginTop: 16 }}>
        <EquityTriptych equity={equity} benchmark={benchmark} />
      </Card>

      {folds.length > 0 && (
        <Card title="Walk-Forward 各折成绩" style={{ marginTop: 16 }}>
          <Table size="small" rowKey="name" pagination={false}
            dataSource={folds}
            columns={[
              { title: 'Fold', dataIndex: 'name' },
              { title: '验证期', dataIndex: 'val' },
              { title: '年化超额', dataIndex: 'excess_annual', render: (v) => v == null ? '—' : fmtPct(v) },
              { title: 'Sharpe', dataIndex: 'sharpe', render: (v) => v == null ? '—' : fmtNum(v, 2) },
              { title: '最大回撤', dataIndex: 'max_drawdown', render: (v) => v == null ? '—' : fmtPct(v) },
              { title: 'IR', dataIndex: 'ir', render: (v) => v == null ? '—' : fmtNum(v, 2) },
            ]} />
        </Card>
      )}

      {stock_pnl.length > 0 && (
        <Card title={`个股盈亏（${stock_pnl.length} 只）`} style={{ marginTop: 16 }}>
          <PnlBarChart items={stock_pnl.map(s => ({ symbol: s.symbol, total_pnl: s.total_pnl }))} nameOf={nameOf} />
          <Table size="small" rowKey="symbol" style={{ marginTop: 8 }} pagination={{ pageSize: 15 }}
            dataSource={stock_pnl}
            columns={[
              { title: '代码', dataIndex: 'symbol', render: (v) => `${nameOf(v)} ${v}` },
              { title: '已实现盈亏', dataIndex: 'total_pnl', sorter: (a: any, b: any) => a.total_pnl - b.total_pnl,
                render: (v: number) => <span style={{ color: pnlColor(v) }}>{fmtNum(v, 0)}</span> },
              { title: '回合数', dataIndex: 'n_round_trips' },
              { title: '胜率', dataIndex: 'win_rate', render: (v) => v == null ? '—' : fmtPct(v) },
            ]} />
        </Card>
      )}

      {trades.length > 0 && (
        <Card title={`逐笔成交（${trades.length} 笔）`} style={{ marginTop: 16 }}>
          <Table size="small" rowKey={(r) => `${r.date}-${r.symbol}-${r.action}-${r.price}`}
            pagination={{ pageSize: 20 }} dataSource={trades}
            columns={[
              { title: '日期', dataIndex: 'date' },
              { title: '代码', dataIndex: 'symbol', render: (v) => `${nameOf(v)} ${v}` },
              { title: '方向', dataIndex: 'action', render: (v) => <Tag color={v === 'BUY' ? 'red' : 'green'}>{v === 'BUY' ? '买入' : '卖出'}</Tag> },
              { title: '价格', dataIndex: 'price', render: (v) => fmtNum(v, 2) },
              { title: '数量', dataIndex: 'qty' },
              { title: '佣金', dataIndex: 'commission', render: (v) => fmtNum(v, 2) },
            ]} />
        </Card>
      )}

      {!metrics.length && !trades.length && (
        <Empty description="该实验无结构化结果（旧格式实验，展示参数）" />
      )}
    </div>
  )
}
```

- [ ] **Step 5: 构建验证**

Run: `cd web/ui && npm run build`
Expected: PASS（TS 编译通过；`fmtNum`/`fmtPct` 已在 lib/format.ts 导出）

- [ ] **Step 6: Commit**

```bash
git add web/ui/src/components/EquityTriptych.tsx web/ui/src/components/MonthlyHeatmap.tsx web/ui/src/components/PnlBarChart.tsx web/ui/src/components/ExperimentReport.tsx
git commit -m "feat(web-ui): 通用图表组件 — 净值三件套/月度热力图/盈亏条形图/实验报告渲染器"
```

---

### Task 6: App 四区导航 + 全局实验选择器

**Files:**
- Modify: `web/ui/src/App.tsx`

**Interfaces:**
- Consumes: `ExperimentProvider`, `useExperiment`（Task 4）
- Produces: 四区导航（仪表盘/研究/交易/数据）+ Header 实验选择器 Dropdown

- [ ] **Step 1: 重写 App.tsx**

```tsx
import { Component, type ReactNode } from 'react'
import { Layout, Menu, Alert, Button, Select, Typography } from 'antd'
import { Routes, Route, Link, useLocation } from 'react-router-dom'
import Overview from './pages/Overview'
import Portfolio from './pages/Portfolio'
import Signals from './pages/Signals'
import Factors from './pages/Factors'
import Experiments from './pages/Experiments'
import Stocks from './pages/Stocks'
import DataStatus from './pages/DataStatus'
import Trading from './pages/Trading'
import { ExperimentProvider, useExperiment } from './experiment-context'

// 四区导航: 仪表盘 / 研究(因子+实验) / 交易(组合+信号+成交) / 数据(个股+数据状态)
const items = [
  { key: '/', label: <Link to="/">仪表盘</Link> },
  { key: '/research', label: '研究', children: [
    { key: '/factors', label: <Link to="/factors">因子</Link> },
    { key: '/experiments', label: <Link to="/experiments">实验</Link> },
  ]},
  { key: '/trading', label: '交易', children: [
    { key: '/portfolio', label: <Link to="/portfolio">组合</Link> },
    { key: '/signals', label: <Link to="/signals">信号</Link> },
    { key: '/trades', label: <Link to="/trades">成交明细</Link> },
  ]},
  { key: '/data', label: '数据', children: [
    { key: '/stocks', label: <Link to="/stocks">个股</Link> },
    { key: '/datastatus', label: <Link to="/datastatus">数据状态</Link> },
  ]},
]

class ErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean; message: string }> {
  constructor(props: { children: ReactNode }) {
    super(props)
    this.state = { hasError: false, message: '' }
  }
  static getDerivedStateFromError(error: unknown) {
    return { hasError: true, message: error instanceof Error ? error.message : String(error) }
  }
  render() {
    if (this.state.hasError) {
      return (
        <Alert type="error" showIcon message="页面渲染出错"
          description={`${this.state.message}。请检查后端是否已启动 (uvicorn :8000)，或刷新重试`}
          action={<Button size="small" onClick={() => { this.setState({ hasError: false }); window.location.reload() }}>刷新</Button>} />
      )
    }
    return this.props.children
  }
}

/** Header 实验选择器（全局上下文，URL ?exp= 同步）。 */
function ExperimentPicker() {
  const { expId, setExpId, registry } = useExperiment()
  const exps = registry?.experiments ?? []
  return (
    <Select
      style={{ width: 340 }}
      placeholder="选择实验"
      value={expId ?? undefined}
      onChange={(v) => setExpId(v)}
      options={exps.map(e => ({ value: e.id, label: `${e.name}${e.kind === 'walkforward' ? ' (回测)' : ' (实验)'}` }))}
    />
  )
}

function Shell() {
  const location = useLocation()
  const selectedKey = location.pathname.startsWith('/trades') ? '/trades' : location.pathname
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Layout.Header style={{ display: 'flex', alignItems: 'center', gap: 16, padding: '0 24px' }}>
        <Typography.Text strong style={{ color: '#fff', fontSize: 16 }}>quant-starter</Typography.Text>
        <ExperimentPicker />
      </Layout.Header>
      <Layout>
        <Layout.Sider theme="dark" width={180}>
          <Menu theme="dark" mode="inline" items={items} selectedKeys={[selectedKey]} defaultOpenKeys={['/research', '/trading', '/data']} />
        </Layout.Sider>
        <Layout.Content style={{ padding: 24 }}>
          <ErrorBoundary>
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/factors" element={<Factors />} />
              <Route path="/experiments" element={<Experiments />} />
              <Route path="/portfolio" element={<Portfolio />} />
              <Route path="/signals" element={<Signals />} />
              <Route path="/trades" element={<Trading />} />
              <Route path="/stocks" element={<Stocks />} />
              <Route path="/datastatus" element={<DataStatus />} />
            </Routes>
          </ErrorBoundary>
        </Layout.Content>
      </Layout>
    </Layout>
  )
}

export default function App() {
  return (
    <ExperimentProvider>
      <Shell />
    </ExperimentProvider>
  )
}
```

注意：`defaultOpenKeys` 只是初始值（非受控），页面刷新后子菜单收起是 antd 正常行为，可接受。

- [ ] **Step 2: 构建验证**

Run: `cd web/ui && npm run build`
Expected: PASS。若 `defaultOpenKeys`/`selectedKeys` 类型告警，保持现状（antd Menu 接受 string[]）。

- [ ] **Step 3: Commit**

```bash
git add web/ui/src/App.tsx
git commit -m "feat(web-ui): 四区导航重构 + Header全局实验选择器(URL同步)"
```

---

### Task 7: 仪表盘重构（Overview）

**Files:**
- Modify: `web/ui/src/pages/Overview.tsx`

**Interfaces:**
- Consumes: `EquityTriptych`, `MonthlyHeatmap`, `PnlBarChart`（Task 5）、`fetchEquity`, `fetchGraduation`, `fetchPaperStockPnl`, `fetchUniverse`（api.ts）
- Produces: 仪表盘 = 毕业指标卡 + 净值三件套（模拟盘实盘） + 月度热力图 + 模拟盘盈亏贡献

- [ ] **Step 1: 重写 Overview.tsx**

```tsx
import { Card, Col, Row, Tag, Spin, Alert, Empty } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { useMemo } from 'react'
import { fetchGraduation, fetchEquity, fetchPaperStockPnl, fetchUniverse } from '../api'
import type { GraduationMetric } from '../api'
import { fmtNum, fmtPct } from '../lib/format'
import { statusMap } from '../lib/labels'
import StatCard from '../components/StatCard'
import EquityTriptych from '../components/EquityTriptych'
import MonthlyHeatmap from '../components/MonthlyHeatmap'
import PnlBarChart from '../components/PnlBarChart'

const pctKeys = new Set(['excess_return', 'max_drawdown', 'fill_rate', 'monthly_win_rate'])

function valueText(m: GraduationMetric): string {
  if (m.value === null || m.value === undefined) return '—'
  return pctKeys.has(m.key) ? fmtPct(m.value) : fmtNum(m.value, m.key === 'runtime_days' ? 0 : 2)
}
function thresholdText(m: GraduationMetric): string {
  if (m.threshold === null || m.threshold === undefined) return '—'
  if (typeof m.threshold === 'string') return m.threshold
  return pctKeys.has(m.key) ? fmtPct(m.threshold, 0) : String(m.threshold)
}

export default function Overview() {
  const g = useQuery({ queryKey: ['graduation'], queryFn: fetchGraduation, refetchInterval: 60_000 })
  const eq = useQuery({ queryKey: ['equity'], queryFn: fetchEquity, refetchInterval: 60_000 })
  const pnl = useQuery({ queryKey: ['paper-stock-pnl'], queryFn: fetchPaperStockPnl, refetchInterval: 60_000 })
  const uni = useQuery({ queryKey: ['universe'], queryFn: fetchUniverse, staleTime: 300_000 })

  const nameOf = useMemo(() => {
    const m = new Map<string, string>()
    for (const s of uni.data?.stocks ?? []) m.set(s.symbol, s.name)
    return (sym: string) => m.get(sym) ?? sym
  }, [uni.data])

  if (g.isLoading || eq.isLoading) return <Spin size="large" style={{ display: 'block', margin: '80px auto' }} />
  if (g.isError || eq.isError) return <Alert type="error" showIcon message="后端不可用" description="请确认 web/api 已启动 (uvicorn :8000)" />

  const metrics = g.data?.metrics ?? []
  const curve = eq.data?.curve ?? []
  const summary = eq.data?.summary
  const dailyReturns = curve
    .filter(p => p.daily_return !== null && p.daily_return !== undefined)
    .map(p => ({ date: p.date, ret: p.daily_return as number }))

  return (
    <div>
      <h2>模拟盘实盘总览</h2>
      {metrics.length ? (
        <Row gutter={[12, 12]}>
          {metrics.map((m) => (
            <Col key={m.key} xs={12} md={6}>
              <Card size="small" title={m.name}>
                <div style={{ fontSize: 20, fontWeight: 600 }}>
                  {valueText(m)}
                  <Tag color={statusMap[m.status]?.color ?? 'default'} style={{ marginLeft: 8 }}>
                    {statusMap[m.status]?.text ?? m.status}
                  </Tag>
                </div>
                <div style={{ color: '#888', fontSize: 12 }}>{m.detail}</div>
                {!m.detail.startsWith('阈值') && <div style={{ color: '#888', fontSize: 12 }}>阈值 {thresholdText(m)}</div>}
              </Card>
            </Col>
          ))}
        </Row>
      ) : (
        <Empty description="毕业指标待计算" />
      )}

      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col span={16}>
          <Card title="模拟盘净值与回撤">
            {curve.length === 0 ? <Empty description="模拟盘 8/3 开跑后产生" />
              : <EquityTriptych equity={curve.map(p => ({ date: p.date, equity: p.total_equity }))} />}
          </Card>
        </Col>
        <Col span={8}>
          <Card title="月度收益热力图">
            {dailyReturns.length === 0 ? <Empty description="暂无日收益数据" />
              : <MonthlyHeatmap dailyReturns={dailyReturns} />}
          </Card>
        </Col>
      </Row>

      <Card title="模拟盘个股盈亏贡献" style={{ marginTop: 16 }}>
        {(pnl.data?.items?.length ?? 0) === 0 ? <Empty description="暂无已实现盈亏" />
          : <PnlBarChart items={(pnl.data?.items ?? []).map(i => ({ symbol: i.symbol, total_pnl: i.total_pnl + (i.unrealized_pnl ?? 0) }))} nameOf={nameOf} />}
      </Card>
    </div>
  )
}
```

注意：删除了旧文件的 `equityOption` 手工 echarts（由 EquityTriptych 替代）与 `ReactECharts` import。若 `StatCard` 不再使用，移除其 import。

- [ ] **Step 2: 构建验证**

Run: `cd web/ui && npm run build`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add web/ui/src/pages/Overview.tsx
git commit -m "feat(web-ui): 仪表盘重构 — 三件套+月度热力图+模拟盘盈亏贡献"
```

---

### Task 8: 实验页报告卡 + 对比模式（Experiments）

**Files:**
- Modify: `web/ui/src/pages/Experiments.tsx`（整体重写）

**Interfaces:**
- Consumes: `useExperiment`（Task 4）、`ExperimentReport`（Task 5）、`fetchExperimentRegistry`, `fetchExperimentDetail`（Task 4）
- Produces: 注册表卡片列表 → 点击进报告视图（URL ?exp=）；对比模式（勾选 2-3 个 → 指标对比表 + 净值叠加图）

- [ ] **Step 1: 重写 Experiments.tsx**

```tsx
import { useMemo, useState } from 'react'
import { Card, Col, Row, Table, Tag, Checkbox, Spin, Alert, Empty, Button, Typography } from 'antd'
import { useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import { fetchExperimentRegistry, fetchExperimentDetail, fetchUniverse } from '../api'
import type { ExperimentRegistryItem } from '../api'
import { fmtNum, fmtPct } from '../lib/format'
import { useExperiment } from '../experiment-context'
import ExperimentReport from '../components/ExperimentReport'

export default function Experiments() {
  const { expId, setExpId, detail, detailLoading, registry } = useExperiment()
  const [compareIds, setCompareIds] = useState<string[]>([])
  const uni = useQuery({ queryKey: ['universe'], queryFn: fetchUniverse, staleTime: 300_000 })
  const nameOf = useMemo(() => {
    const m = new Map<string, string>()
    for (const s of uni.data?.stocks ?? []) m.set(s.symbol, s.name)
    return (sym: string) => m.get(sym) ?? sym
  }, [uni.data])

  const exps: ExperimentRegistryItem[] = registry?.experiments ?? []

  const compareQueries = compareIds.map(id => ({
    id,
    q: useQuery({ queryKey: ['exp-detail', id], queryFn: () => fetchExperimentDetail(id), staleTime: 60_000 }),
  }))

  // 指标对比: 行=指标, 列=实验
  const compareMetrics = [
    ['excess_annual', '年化超额', 'pct'], ['total_return', '总收益', 'pct'],
    ['sharpe', 'Sharpe', 'num'], ['max_drawdown', '最大回撤', 'pct'],
    ['calmar', 'Calmar', 'num'], ['avg_turnover', '平均换手', 'pct'],
  ] as const

  const compareTableData = compareMetrics.map(([key, label]) => {
    const row: any = { key, label }
    let best: number | null = null
    let bestId = ''
    for (const { id, q } of compareQueries) {
      const m = q.data?.metrics?.find(mm => mm.key === key)
      row[id] = m?.value ?? null
      if (typeof m?.value === 'number' && m.value !== null) {
        const better = m.better === 'low' ? 'min' : 'max'
        if (best === null || (better === 'max' && m.value > best) || (better === 'min' && m.value < best)) {
          best = m.value; bestId = id
        }
      }
    }
    row._best = bestId
    return row
  })

  const compareSeries: any[] = compareQueries.map(({ id, q }) => {
    const d = q.data
    const eq = d?.series?.find(s => s.name === '组合净值')
    return { id, name: id, x: eq?.x ?? [], y: eq?.y ?? [] }
  })

  return (
    <div>
      <Typography.Title level={4}>实验（可插拔注册表）</Typography.Title>
      <Typography.Paragraph type="secondary">
        新实验产出标准 JSON 后自动出现在此列表（无需改前端）。勾选最多 3 个进行对比。
      </Typography.Paragraph>

      {/* 对比勾选 */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {exps.map(e => (
            <Checkbox key={e.id} checked={compareIds.includes(e.id)}
              disabled={!compareIds.includes(e.id) && compareIds.length >= 3}
              onChange={(ev) => {
                setCompareIds(prev => ev.target.checked
                  ? [...prev, e.id] : prev.filter(x => x !== e.id))
              }}>
              {e.name}
            </Checkbox>
          ))}
        </div>
        {compareIds.length >= 2 && (
          <>
            <Card size="small" title="指标对比（绿色=最优）" style={{ marginTop: 12 }}>
              <Table size="small" rowKey="key" pagination={false}
                dataSource={compareTableData}
                columns={[
                  { title: '指标', dataIndex: 'label' },
                  ...compareIds.map(id => ({
                    title: id, dataIndex: id,
                    render: (v: number | null, r: any) => {
                      if (v === null || v === undefined) return '—'
                      const key = r.key as string
                      const isPct = compareMetrics.find(c => c[0] === key)?.[2] === 'pct'
                      const text = isPct ? fmtPct(v) : fmtNum(v, 2)
                      return <span style={{ color: r._best === id ? '#389e0d' : undefined, fontWeight: r._best === id ? 600 : 400 }}>{text}</span>
                    },
                  })),
                ]} />
            </Card>
            <Card size="small" title="净值曲线叠加" style={{ marginTop: 12 }}>
              <ReactECharts option={{
                tooltip: { trigger: 'axis' },
                legend: { data: compareSeries.map(s => s.name) },
                grid: { left: 60, right: 30, top: 40, bottom: 30 },
                xAxis: { type: 'category', data: compareSeries[0]?.x ?? [] },
                yAxis: { type: 'value', scale: true },
                series: compareSeries.map(s => ({ name: s.name, type: 'line', data: s.y, showSymbol: false })),
              }} style={{ height: 320 }} />
            </Card>
          </>
        )}
      </Card>

      {/* 实验卡片列表 */}
      <Row gutter={[12, 12]}>
        {exps.map(e => (
          <Col key={e.id} xs={24} md={12} xl={8}>
            <Card size="small" hoverable
              style={{ borderColor: expId === e.id ? '#1677ff' : undefined }}
              onClick={() => setExpId(e.id)}
              title={<span>{e.name} {e.kind === 'walkforward' ? <Tag color="blue">回测</Tag> : <Tag>实验</Tag>}</span>}
              extra={<Button type="link" size="small" onClick={(ev) => { ev.stopPropagation(); setExpId(e.id) }}>查看报告</Button>}>
              <div style={{ fontSize: 12, color: '#888' }}>{e.generated_at}</div>
              {e.summary?.excess_annual != null && (
                <div style={{ marginTop: 8 }}>
                  年化超额 <b>{fmtPct(e.summary.excess_annual)}</b> · Sharpe <b>{e.summary.sharpe != null ? fmtNum(e.summary.sharpe, 2) : '—'}</b> · 回撤 <b>{e.summary.max_drawdown != null ? fmtPct(e.summary.max_drawdown) : '—'}</b>
                </div>
              )}
            </Card>
          </Col>
        ))}
        {!exps.length && <Empty description="注册表为空" />}
      </Row>

      {/* 当前选中实验的报告视图 */}
      {expId && (
        <Card title={`实验报告: ${expId}`} style={{ marginTop: 24 }}>
          {detailLoading ? <Spin /> : detail
            ? <ExperimentReport detail={detail} nameOf={nameOf} />
            : <Alert type="error" message="加载失败" />}
        </Card>
      )}
    </div>
  )
}
```

注意：`compareQueries` 内联 useQuery 违反 hooks 规则（动态数量），改为固定 3 个查询或使用 `useQueries`：

```tsx
import { useQueries } from '@tanstack/react-query'
// 替换 compareQueries:
const compareQueries = useQueries({
  queries: compareIds.map(id => ({ queryKey: ['exp-detail', id], queryFn: () => fetchExperimentDetail(id), staleTime: 60_000 })),
})
// 用法: compareQueries[i].data; 生成 {id, q} 列表:
const compareData = compareIds.map((id, i) => ({ id, q: compareQueries[i] }))
// 下方所有 compareQueries 引用改 compareData
```

- [ ] **Step 2: 构建验证**

Run: `cd web/ui && npm run build`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add web/ui/src/pages/Experiments.tsx
git commit -m "feat(web-ui): 实验页注册表卡片+报告视图+多实验对比(指标表/净值叠加)"
```

---

### Task 9: 组合页升级（Portfolio）+ 成交页双 Tab（Trading）

**Files:**
- Modify: `web/ui/src/pages/Portfolio.tsx`
- Modify: `web/ui/src/pages/Trading.tsx`

**Interfaces:**
- Consumes: Task 3 的 `/api/portfolio`（positions 含 current_price/pnl/pnl_pct）、Task 2 的 `/api/experiments/{id}`（回测成交）、`useExperiment`（Task 4）、`PnlBarChart`（Task 5）

- [ ] **Step 1: Portfolio.tsx 补现价/盈亏率列 + 贡献图**

修改点（在现有文件上改，保留其余逻辑）：
1. `Position` 接口补 `current_price?: number | null; pnl?: number; pnl_pct?: number | null`
2. columns 中 `market_value` 后插入现价列，`avg_cost` 后保留；新增盈亏率列复用 `pnl` 函数
3. 表格上方加 `PnlBarChart`（持仓按 pnl 排序）
4. import `PnlBarChart`、`fetchUniverse` 构建 nameOf

完整代码：

```tsx
import { useMemo } from 'react'
import { Card, Row, Col, Spin, Alert, Table, Typography, Empty } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { fetchPortfolio, fetchUniverse } from '../api'
import { col } from '../lib/columns'
import { fmtNum, fmtPct } from '../lib/format'
import StatCard from '../components/StatCard'
import PnlBarChart from '../components/PnlBarChart'

interface Position { symbol: string; qty: number; avg_cost: number; market_value: number; entry_date?: string; current_price?: number | null; pnl?: number; pnl_pct?: number | null }

export default function Portfolio() {
  const navigate = useNavigate()
  const { data, isLoading, isError } = useQuery({
    queryKey: ['portfolio'], queryFn: fetchPortfolio, refetchInterval: 30_000,
  })
  const uni = useQuery({ queryKey: ['universe'], queryFn: fetchUniverse, staleTime: 300_000 })
  const nameOf = useMemo(() => {
    const m = new Map<string, string>()
    for (const s of uni.data?.stocks ?? []) m.set(s.symbol, s.name)
    return (sym: string) => m.get(sym) ?? sym
  }, [uni.data])

  const positions = useMemo(() => (data?.positions ?? []) as Position[], [data])
  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="加载失败" />

  const pnl = (p: Position) => p.pnl ?? (p.market_value ?? 0) - (p.avg_cost ?? 0) * (p.qty ?? 0)
  const pnlPct = (p: Position) => p.pnl_pct ?? ((p.avg_cost ?? 0) * (p.qty ?? 0) > 0 ? pnl(p) / ((p.avg_cost ?? 0) * (p.qty ?? 0)) : 0)
  const totalMv = positions.reduce((s, p) => s + (p.market_value ?? 0), 0)
  const totalCost = positions.reduce((s, p) => s + (p.avg_cost ?? 0) * (p.qty ?? 0), 0)
  const totalPnl = totalMv - totalCost
  const pnlColor = (v: number) => (v > 0 ? '#cf1322' : v < 0 ? '#3f8600' : undefined)

  const columns = [
    col<Position>('symbol', { title: '代码', width: 100, sorter: true, render: (v) => `${nameOf(v as string)} ${v}` }),
    col<Position>('qty', { title: '数量', width: 90, sorter: true, render: (v) => fmtNum(v as number, 0) }),
    col<Position>('avg_cost', { title: '成本价', width: 100, sorter: true, render: (v) => fmtNum(v as number, 2) }),
    col<Position>('current_price', { title: '现价', width: 100, sorter: true, render: (v) => (v == null ? '—' : fmtNum(v as number, 2)) }),
    col<Position>('market_value', { title: '市值', width: 120, sorter: true, render: (v) => fmtNum(v as number, 2) }),
    { title: '浮动盈亏', dataIndex: 'pnl', width: 170,
      sorter: (a: Position, b: Position) => pnl(a) - pnl(b),
      render: (_v: unknown, r: Position) => (
        <span style={{ color: pnlColor(pnl(r)) }}>
          {fmtNum(pnl(r), 2)}（{fmtPct(pnlPct(r))}）
        </span>
      ) },
    col<Position>('entry_date', { title: '建仓日', width: 110, sorter: true }),
  ]

  return (
    <div>
      <Typography.Title level={4}>模拟盘组合</Typography.Title>
      <Row gutter={12}>
        <Col span={6}><StatCard title="现金" value={fmtNum(data?.cash, 0)} /></Col>
        <Col span={6}><StatCard title="总市值" value={fmtNum(totalMv, 0)} /></Col>
        <Col span={6}><StatCard title="总成本" value={fmtNum(totalCost, 0)} /></Col>
        <Col span={6}><StatCard title="总盈亏" value={fmtNum(totalPnl, 0)} color={pnlColor(totalPnl)} /></Col>
      </Row>
      {positions.length ? (
        <>
          <Card size="small" title="盈亏贡献" style={{ marginTop: 16 }}>
            <PnlBarChart items={positions.map(p => ({ symbol: p.symbol, total_pnl: pnl(p) }))} nameOf={nameOf} />
          </Card>
          <Table rowKey="symbol" dataSource={positions} columns={columns} size="small"
            style={{ marginTop: 16 }}
            pagination={{ pageSize: 10, showSizeChanger: true, showTotal: (t) => `共 ${t} 条` }}
            onRow={(r) => ({ onClick: () => navigate(`/stocks?symbol=${r.symbol}`), style: { cursor: 'pointer' } })}
          />
        </>
      ) : (
        <Empty description="暂无持仓（模拟盘 8/3 开跑后产生）" style={{ marginTop: 24 }} />
      )}
    </div>
  )
}
```

- [ ] **Step 2: Trading.tsx 双 Tab**

修改点（保留现有回测视图为 Tab 1，新增模拟盘 Tab 2）：
1. import `Tabs`, `fetchPaperStockPnl`（或直接用 broker trades 表格——模拟盘 Tab 用 `fetchBroker` 的 trades + `fetchPaperStockPnl` 的个股盈亏榜）
2. 回测 Tab 用 `useExperiment()` 的 `detail`（替代硬编码 fetchBacktestTrades？**保留** fetchBacktestTrades 现有实现不变，改为从全局实验 detail 取数据需后端路由一致——为最小改动，回测 Tab 维持现有 fetchBacktestTrades，但标题显示当前实验 id）
3. 结构：`<Tabs items=[{key:'backtest', label:'回测实验成交', children: <现有内容>}, {key:'paper', label:'模拟盘实盘成交', children: <实盘表格>}] />`

具体改动（在现有文件末尾的 return 外包 Tabs，新增模拟盘 Tab 内容）：

```tsx
// 新增 import
import { Tabs } from 'antd'
import { useExperiment } from '../experiment-context'
import { fetchPaperStockPnl } from '../api'
import PnlBarChart from '../components/PnlBarChart'

// 组件内新增
const { expId } = useExperiment()
const paperPnl = useQuery({ queryKey: ['paper-stock-pnl'], queryFn: fetchPaperStockPnl, refetchInterval: 60_000 })

// 模拟盘 Tab 内容 (Tabs items)
const paperTab = (
  <div>
    <Card size="small" title="模拟盘个股盈亏（已实现+浮动）" style={{ marginBottom: 16 }}>
      <PnlBarChart items={(paperPnl.data?.items ?? []).map(i => ({ symbol: i.symbol, total_pnl: i.total_pnl + (i.unrealized_pnl ?? 0) }))} nameOf={nameOf} />
    </Card>
    <Table size="small" rowKey="symbol" dataSource={paperPnl.data?.items ?? []} pagination={{ pageSize: 15 }}
      columns={[
        { title: '代码', dataIndex: 'symbol', render: (v) => `${nameOf(v)} ${v}` },
        { title: '已实现盈亏', dataIndex: 'total_pnl', sorter: (a: any, b: any) => a.total_pnl - b.total_pnl,
          render: (v: number) => <span style={{ color: v > 0 ? '#cf1322' : v < 0 ? '#3f8600' : undefined }}>{fmtNum(v, 2)}</span> },
        { title: '浮动盈亏', dataIndex: 'unrealized_pnl', render: (v) => v == null ? '—' : <span style={{ color: v > 0 ? '#cf1322' : v < 0 ? '#3f8600' : undefined }}>{fmtNum(v, 2)}</span> },
        { title: '回合数', dataIndex: 'n_round_trips' },
        { title: '胜率', dataIndex: 'win_rate', render: (v) => v == null ? '—' : fmtPct(v) },
      ]} />
  </div>
)
```

现有回测内容整体包成 `backtestTab`，最终 return 改为：

```tsx
return (
  <div>
    <Typography.Title level={4}>成交明细{expId ? `（实验: ${expId}）` : ''}</Typography.Title>
    <Tabs defaultActiveKey="backtest" items={[
      { key: 'backtest', label: '回测实验成交', children: backtestTab },
      { key: 'paper', label: '模拟盘实盘成交', children: paperTab },
    ]} />
  </div>
)
```

注：现有 Trading.tsx 的回测视图代码（净值图/调仓分布/年份 Segmented/换仓明细表）原样移入 `backtestTab` 变量，不删功能。`fmtPct` 需确保已 import（若未 import 则补）。

- [ ] **Step 3: 构建验证**

Run: `cd web/ui && npm run build`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add web/ui/src/pages/Portfolio.tsx web/ui/src/pages/Trading.tsx
git commit -m "feat(web-ui): 组合页现价/盈亏率+贡献图; 成交页双Tab(回测实验/模拟盘实盘)"
```

---

### Task 10: 个股页买卖标记（Stocks）

**Files:**
- Modify: `web/ui/src/pages/Stocks.tsx`

**Interfaces:**
- Consumes: `useExperiment`（Task 4）的 detail.trades、现有 fetchStock K 线数据
- Produces: K 线图叠加买卖 markPoint（买入红▲ / 卖出绿▼，backtrader 风格）

- [ ] **Step 1: 先读现有 Stocks.tsx 确认 K 线 option 结构**

Run: `cat web/ui/src/pages/Stocks.tsx`

- [ ] **Step 2: 在 K 线 series 上追加 markPoint**

```tsx
// 组件内新增
const { detail } = useExperiment()
const tradesOfStock = useMemo(
  () => (detail?.trades ?? []).filter((t: any) => t.symbol === symbol),
  [detail, symbol],
)
// K 线 series[0] 追加:
const markPoint = {
  symbol: 'triangle',
  symbolSize: 10,
  data: [
    ...tradesOfStock.filter((t: any) => t.action === 'BUY').map((t: any) => ({
      name: `买入 ${t.date}`, coord: [t.date, t.price],
      itemStyle: { color: '#cf1322' },
    })),
    ...tradesOfStock.filter((t: any) => t.action === 'SELL').map((t: any) => ({
      name: `卖出 ${t.date}`, coord: [t.date, t.price],
      itemStyle: { color: '#3f8600' },
      symbolRotate: 180,
    })),
  ],
}
// series[0] = { ...原有k线配置, markPoint }
```

- [ ] **Step 3: 构建验证**

Run: `cd web/ui && npm run build`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add web/ui/src/pages/Stocks.tsx
git commit -m "feat(web-ui): 个股K线叠加实验买卖标记(backtrader风格)"
```

---

### Task 11: 全链路验收（测试 + 构建 + 截图）

**Files:** 无（验证任务）

- [ ] **Step 1: 后端测试全绿**

Run: `py -m pytest tests/ -q && py -m pytest web/api/tests/ -q`
Expected: 61 + 后端 API 测试全部 PASS

- [ ] **Step 2: 前端构建**

Run: `cd web/ui && npm run build`
Expected: PASS, 无 TS 错误

- [ ] **Step 3: 启动后端 + 前端**

```bash
# 后端 (杀残留 8000 进程后)
py -c "import subprocess,os; os.system('netstat -ano | findstr :8000')"  # 找 PID
# 或直接用: taskkill //F //PID <pid>
# 启动:
py -m uvicorn web.api.main:app --host 127.0.0.1 --port 8000  # 后台
# 前端:
cd web/ui && npm run dev  # 后台 (:5173)
```

- [ ] **Step 4: 浏览器截图验收 4 区页面**

用 Playwright（browser_navigate）依次访问并截图确认：
- `http://localhost:5173/` 仪表盘（三件套+热力图+盈亏贡献）
- `http://localhost:5173/experiments?exp=walkforward_results_v24e_pov` 实验报告（指标卡/净值/个股盈亏榜/逐笔成交）
- `http://localhost:5173/trades` 成交明细双 Tab
- `http://localhost:5173/portfolio` 组合（现价/盈亏率）
- `http://localhost:5173/stocks?symbol=600519` 个股（买卖标记）
- 验证：实验选择器切换 `?exp=walkforward_results_v25_combo` 后实验报告联动
- 检查浏览器 console 无报错（browser_console_messages）

- [ ] **Step 5: 验收问题修复**

记录截图/console 中的问题 → 修复 → 重截图确认

- [ ] **Step 6: 最终 Commit**

```bash
git add -A
git commit -m "feat(web): 前端全面重构验收 — 实验为中心四区导航+可插拔注册表+个股盈亏"
```

---

## Self-Review

**Spec 覆盖检查:**
- ✅ 注册表 API（Task 2）、统一 schema（Task 2）、个股盈亏 FIFO（Task 1）、portfolio 现价（Task 3）、基准曲线（Task 1/2）
- ✅ 全局实验选择器（Task 4/6）、四区导航（Task 6）、仪表盘（Task 7）、实验报告卡+对比（Task 8）、成交双 Tab（Task 9）、个股买卖标记（Task 10）
- ✅ 可插拔：Task 2 注册表 + Task 8 通用渲染（新 JSON 零代码）
- ✅ 错误处理：坏 JSON 跳过（Task 2）、旧格式兜底（Task 2 exp_schema）、前端 ErrorBoundary/Empty
- ✅ 验收：Task 11 截图
- ⚠️ spec 中"调仓图点击下钻"未单独成任务——Trading 页保留现有调仓分布图（不新增下钻，成本/收益不划算），在 Task 9 中标注为保留现状
- ⚠️ spec 中"研究区因子页滚动 IC 图"未纳入——Factors.tsx 保持现状（不属于本次核心 4 目标）

**Placeholder 扫描:** 无 TBD/TODO；所有代码块完整。

**类型一致性:** `StockPnlItem` 字段与后端 `aggregate_stock_pnl` 输出一致（symbol/total_pnl/realized_pnl/n_round_trips/win_rate/buy_count/sell_count/open_qty + paper 端点追加 open_qty/avg_cost/current_price/unrealized_pnl）；`MetricItem.format` 与后端 metrics 输出一致（pct/num/str）；前端 `useExperiment` 的 detail 类型与 Task 2 响应一致。
