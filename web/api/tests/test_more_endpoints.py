import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi.testclient import TestClient
from web.api.main import app

client = TestClient(app)


def test_signals_endpoint():
    r = client.get("/api/signals")
    assert r.status_code == 200
    assert "count" in r.json()


def test_experiments_endpoint():
    r = client.get("/api/experiments")
    assert r.status_code == 200
    assert "count" in r.json() and "by_script" in r.json()


def test_factors_ic_endpoint():
    r = client.get("/api/factors/ic")
    assert r.status_code == 200
    assert "sources" in r.json() and "results" in r.json()


def test_universe_endpoint():
    r = client.get("/api/universe")
    assert r.status_code == 200
    assert "total" in r.json()


def test_universe_search():
    r = client.get("/api/universe/search", params={"q": "茅台"})
    assert r.status_code == 200
    assert len(r.json()["stocks"]) <= 50


def test_stock_not_found():
    r = client.get("/api/stocks/999999")
    assert r.status_code == 404


def test_stock_found():
    # 用 data_cache 里确定存在的某只股票
    from web.api import config
    syms = config.get_cached_symbols() if hasattr(config, "get_cached_symbols") else []
    if syms:
        r = client.get(f"/api/stocks/{syms[0]}")
        assert r.status_code == 200
        assert "ohlc" in r.json()
