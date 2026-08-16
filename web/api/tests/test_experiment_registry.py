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
    # 任意版本号 walkforward 结果都应在注册表中 (不绑定具体版本, 防归档后测试失效)
    assert any(i.startswith("walkforward_results_v") for i in ids)
    # 无版本号的生产结果 (最新实验) 也必须在注册表中
    assert "walkforward_results" in ids, "生产结果 walkforward_results 应出现在注册表"
    for e in data["experiments"]:
        assert set(["id", "kind", "name", "generated_at"]) <= set(e.keys())


def test_registry_sorted_walkforward_first_then_by_time():
    """walkforward 回测结果整体在 exp 流水记录之前; 同 kind 内按时间倒序。"""
    resp = client.get("/api/experiments/registry")
    data = resp.json()
    exps = data["experiments"]
    kinds = [e["kind"] for e in exps]
    # walkforward 全部在 experiment 之前
    first_exp = kinds.index("experiment") if "experiment" in kinds else len(kinds)
    assert all(k == "walkforward" for k in kinds[:first_exp])
    # 同 kind 内 generated_at 倒序 (字符串 ISO 比较)
    wf_gen = [e["generated_at"] for e in exps if e["kind"] == "walkforward"]
    assert wf_gen == sorted(wf_gen, reverse=True)
    exp_gen = [e["generated_at"] for e in exps if e["kind"] == "experiment"]
    assert exp_gen == sorted(exp_gen, reverse=True)
    # 最新回测置顶 (v27 或更新)
    assert exps[0]["kind"] == "walkforward"


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
    # 逐笔成交增强: 成交后净值 + 时间归一化 (旧数据 09:35 → 全天VWAP 标记)
    t0 = d["trades"][0]
    assert "equity_after" in t0 and t0["equity_after"] is not None
    for t in d["trades"]:
        ft = t.get("fill_times")
        if ft:
            assert all(f != "09:35" for f in ft), "fill_times 不应再有 09:35"
            assert all(f != "早盘市价" for f in ft), "不应再有旧的早盘市价标记"


# ── fill_times 归一化判别: 按 generated_at 区分新旧数据 ──


def _mk_schema(generated_at, fills):
    from web.api.routers.experiments import _walkforward_schema
    d = {
        "meta": {"generated_at": generated_at, "description": "unit"},
        "results": {"extend_val": {
            "period": "2025-01-01 ~ 2026-06-30",
            "trades": [{"date": "2025-01-02", "symbol": "600000",
                        "action": "BUY", "price": 10.0, "qty": 100,
                        "commission": 5.0, "fill_times": fills}],
            "equity_curve": [{"date": "2025-01-02", "equity": 300000.0}],
        }},
    }
    return _walkforward_schema(Path("x.json"), d)


def test_schema_legacy_0935_normalized():
    """旧数据 (2026-08-15 前, 装饰性 09:35/VWAP) → 归一化为 全天VWAP。"""
    s = _mk_schema("2026-08-13 00:11:43", ["09:35"])
    assert s["trades"][0]["fill_times"] == ["全天VWAP"]


def test_schema_new_0935_preserved():
    """新数据 (确定性 POV, 2026-08-15 起) 的 09:35 是真实成交 → 不重写。"""
    s = _mk_schema("2026-08-16 07:00:22", ["09:35"])
    assert s["trades"][0]["fill_times"] == ["09:35"]


def test_schema_new_multibar_preserved():
    """新数据多段 POV 成交时间保持原样。"""
    s = _mk_schema("2026-08-16 07:00:22", ["09:35", "10:00", "14:30"])
    assert s["trades"][0]["fill_times"] == ["09:35", "10:00", "14:30"]


def test_schema_missing_generated_at_treated_as_legacy():
    """无 generated_at 的老文件按旧数据处理 (安全默认)。"""
    s = _mk_schema("", ["09:35"])
    assert s["trades"][0]["fill_times"] == ["全天VWAP"]


def test_experiment_detail_404():
    resp = client.get("/api/experiments/not_exist_xyz")
    assert resp.status_code == 404


def test_legacy_endpoint_kept():
    resp = client.get("/api/experiments")
    assert resp.status_code == 200
    assert "count" in resp.json()
