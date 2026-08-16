# tests/test_offline_fetchers.py
"""netgate 接入 fetcher 测试: 离线模式下网络分支被拦截。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_minute_fetcher_offline_no_network(monkeypatch):
    """离线模式下 MinuteFetcher 不触碰网络: 本地无数据直接返回 None。"""
    from netgate import set_offline_mode
    from data.minute_fetcher import MinuteFetcher
    set_offline_mode(True)
    try:
        calls = []
        import akshare as ak

        def _boom(**kwargs):
            calls.append(kwargs)
            raise RuntimeError("network forbidden")

        monkeypatch.setattr(ak, "stock_zh_a_hist_min_em", _boom)
        # 即便调用方忘了 allow_network=False, 离线守卫仍拦截
        mf = MinuteFetcher(allow_network=True)
        # 不存在的代码: minute_5m 与滚动缓存都没有 → 原本会走网络
        out = mf.fetch("888888", days=5, end_date="2026-08-14")
        assert out is None
        assert calls == [], "离线模式下不得发起网络请求"
    finally:
        set_offline_mode(False)


def test_minute_fetcher_online_network_reachable(monkeypatch):
    """非离线 + allow_network=True 时网络路径可达 (守卫不误伤获取阶段)。"""
    from data.minute_fetcher import MinuteFetcher
    calls = []
    import akshare as ak

    def _boom(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("network forbidden in test")

    monkeypatch.setattr(ak, "stock_zh_a_hist_min_em", _boom)
    mf = MinuteFetcher(allow_network=True)
    out = mf.fetch("888888", days=5, end_date="2026-08-14")
    assert out is None  # 网络异常被吞, 返回 None
    assert calls, "非离线模式应尝试网络"


def test_fundamental_fetch_one_offline_raises():
    from netgate import set_offline_mode, OfflineViolation
    import fundamental_fetcher
    set_offline_mode(True)
    try:
        raised = False
        try:
            fundamental_fetcher.fetch_one("600519")
        except OfflineViolation:
            raised = True
        assert raised, "离线模式下 fundamental_fetcher.fetch_one 必须抛 OfflineViolation"
    finally:
        set_offline_mode(False)


def test_flow_snapshot_offline_raises():
    from netgate import set_offline_mode, OfflineViolation
    import flow_fetcher
    set_offline_mode(True)
    try:
        raised = False
        try:
            flow_fetcher.fetch_money_flow_snapshot("20日排行")
        except OfflineViolation:
            raised = True
        assert raised, "离线模式下 flow_fetcher.fetch_money_flow_snapshot 必须抛 OfflineViolation"
    finally:
        set_offline_mode(False)


def test_smart_money_offline_raises():
    from netgate import set_offline_mode, OfflineViolation
    import smart_money_fetcher
    set_offline_mode(True)
    try:
        raised = False
        try:
            smart_money_fetcher.fetch_northbound_history("600519")
        except OfflineViolation:
            raised = True
        assert raised, "离线模式下 smart_money_fetcher.fetch_northbound_history 必须抛 OfflineViolation"
    finally:
        set_offline_mode(False)


def test_smart_money_analyst_offline_raises():
    from netgate import set_offline_mode, OfflineViolation
    import smart_money_fetcher
    set_offline_mode(True)
    try:
        raised = False
        try:
            smart_money_fetcher.fetch_analyst_consensus()
        except OfflineViolation:
            raised = True
        assert raised, "离线模式下 smart_money_fetcher.fetch_analyst_consensus 必须抛 OfflineViolation"
    finally:
        set_offline_mode(False)
