import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi.testclient import TestClient
from web.api.main import app

client = TestClient(app)


def test_broker_status():
    r = client.get("/api/broker/status")
    assert r.status_code == 200
    body = r.json()
    assert body["adapter"] == "paper"
    assert body["connected"] is True
    assert "balance" in body and "positions" in body
