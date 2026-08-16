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
    """data/ic_validation/walkforward_results*.json — 含无版本号的生产结果
    (walkforward_results.json, 即最新实验), 排除 _bak_ 备份。"""
    if not config.IC_DIR.exists():
        return
    for p in sorted(config.IC_DIR.glob("walkforward_results*.json")):
        if "_bak_" in p.stem:
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

    # ── 全部成交: 合并 folds + extend_val, 标注归属阶段 (2026-08-16) ──
    # 之前只取 extend_val 的 trades, 前端只能看到 2025 之后的成交;
    # 实际 fold_1~5 (2020-2024) 也有逐笔成交, 合并后按日期排序全量展示。
    legacy_fills = str(meta.get("generated_at", "")) < "2026-08-15"
    all_trades = []
    # 各阶段净值曲线 + 基准 (供前端按年份/阶段切换净值图, 2026-08-16)
    segments = []
    for seg_key in [k for k in results if k.startswith("fold_")] + ["extend_val"]:
        seg = results.get(seg_key) or {}
        seg_trades = seg.get("trades", []) or []
        # 阶段净值曲线 (主数据: series 中"组合净值"画线)
        seg_eq = seg.get("equity_curve") or []
        if seg_eq:
            # 该阶段基准曲线 (与阶段净值同基)
            seg_period = seg.get("period", "")
            sp_s, _, sp_e = (seg_period.split(" ~ ") + ["", ""])[:3]
            seg_bench = build_benchmark_curve(sp_s or "2020-01-01", sp_e or "2026-06-30")
            seg_base = seg_bench[0]["close"] if seg_bench else None
            eq0 = seg_eq[0].get("equity", 100000.0)
            segments.append({
                "key": seg_key,
                "label": "模拟考" if seg_key == "extend_val" else seg_key.replace("fold_", "Fold"),
                "val": seg.get("val", ""),
                "equity": [{"date": p["date"], "equity": p["equity"]} for p in seg_eq],
                "benchmark": ([{"date": b["date"],
                                "close": round(b["close"] / seg_base * eq0, 2)}
                               for b in seg_bench] if seg_base else []),
                "excess_annual": seg.get("excess_annual"),
                "sharpe": seg.get("sharpe"),
                "max_drawdown": seg.get("max_drawdown"),
                "n_trades": len(seg_trades),
            })
        if not seg_trades:
            continue
        # 每阶段自己的 equity_curve → equity_after 映射
        seg_eq_map = {}
        for p in seg_eq:
            seg_eq_map[p["date"]] = p.get("equity")
        for t in seg_trades:
            t["segment"] = seg_key  # fold_1..5 / extend_val
            t_date = t.get("date", "")
            if t_date in seg_eq_map and "equity_after" not in t:
                t["equity_after"] = seg_eq_map[t_date]
            if legacy_fills:
                ft = t.get("fill_times")
                if ft and ft == ["09:35"]:
                    t["fill_times"] = ["全天VWAP"]
        all_trades.extend(seg_trades)
    all_trades.sort(key=lambda t: t.get("date", ""))
    trades = all_trades
    stock_pnl = aggregate_stock_pnl(trades)

    return {"meta": meta, "metrics": metrics, "series": series, "folds": folds,
            "stock_pnl": stock_pnl, "trades": trades,
            "equity_curve": eq, "benchmark_curve": bench, "segments": segments}


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
            "equity_curve": [], "benchmark_curve": [], "segments": []}


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
    # 排序: 先按 generated_at 倒序 (最新在前), 再用稳定排序把 walkforward
    # 回测结果整体提到 exp 流水记录之前 (回测是核心资产, exp 记录是附属流水;
    # 2026-08-16 实测: exp 记录时间晚于回测时会抢占置顶, 掩盖最新实验)
    items.sort(key=lambda x: x.get("generated_at", ""), reverse=True)
    items.sort(key=lambda x: 0 if x.get("kind") == "walkforward" else 1)
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
