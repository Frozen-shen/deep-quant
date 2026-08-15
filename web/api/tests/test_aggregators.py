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
    # 买佣金摊到全部200股: 每股成本 10.025; 卖100股: (11-0.05-10.025)*100 = 92.5
    assert r["total_pnl"] == 92.5
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
