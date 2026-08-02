import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi.testclient import TestClient
from web.api.main import app

client = TestClient(app)


def test_equity_empty_state():
    """equity_log 为空时返回空列表 + summary 为 None，不报错。"""
    r = client.get("/api/equity")
    assert r.status_code == 200
    body = r.json()
    assert body["curve"] == []
    assert body["summary"] is None


def test_equity_with_data():
    """构造临时 equity_log 数据后能算出回撤/夏普。"""
    import storage as st
    tmp_db = os.path.join(os.path.dirname(__file__), "tmp_test.db")
    st.init_db(tmp_db)
    eqs = [1000000, 1010000, 990000, 1005000]
    for i, eq in enumerate(eqs):
        # 简报原版 daily_return 全传 0.0 时 vol=0 → sharpe=None，
        # 与 sharpe 断言矛盾；改用相邻净值推导的日收益（断言不变）。
        ret = 0.0 if i == 0 else eq / eqs[i - 1] - 1
        st.log_equity(f"2026-08-0{i+1}", 500000, eq - 500000, ret, path=tmp_db)
    # 直接验证计算函数
    from web.api.routers import equity as eqmod
    rows = st.get_equity_log(limit=10, path=tmp_db)
    summary = eqmod.compute_summary(rows)
    assert summary["max_drawdown"] < 0
    assert summary["sharpe"] is not None
    os.remove(tmp_db)


def test_portfolio_reads_json():
    """portfolio.json 存在时返回持仓列表。"""
    r = client.get("/api/portfolio")
    assert r.status_code == 200
    body = r.json()
    assert "cash" in body and "positions" in body
