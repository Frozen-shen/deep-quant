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
