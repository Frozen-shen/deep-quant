import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi.testclient import TestClient
from web.api.main import app

client = TestClient(app)


def test_graduation_shape():
    """8 项指标齐全，数据不足时为 pending 而非报错。"""
    r = client.get("/api/graduation")
    assert r.status_code == 200
    metrics = r.json()["metrics"]
    keys = {m["key"] for m in metrics}
    assert keys == {"runtime_days", "excess_return", "ir", "max_drawdown",
                    "fill_rate", "ic_decay", "sharpe", "monthly_win_rate"}
    assert all(m["status"] in ("pass", "fail", "pending") for m in metrics)
