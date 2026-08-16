# tests/test_pipeline_minute.py
"""daily_pipeline 分钟增量步骤测试 (T5): 标的集合 + 失败容忍。"""
import sys
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "scripts"))
sys.path.insert(0, os.path.join(BASE, "scripts", "active"))


def test_minute_symbols_union(monkeypatch):
    """标的集合 = 持仓 ∪ 近期成交。"""
    import daily_pipeline
    monkeypatch.setattr(daily_pipeline.storage, "get_all_positions",
                        lambda: [{"symbol": "600519", "qty": 100}])
    monkeypatch.setattr(daily_pipeline.storage, "get_trades",
                        lambda **k: [{"symbol": "000858"}, {"symbol": "600519"}])
    syms = daily_pipeline._minute_symbols("2026-08-14")
    assert "600519" in syms and "000858" in syms
    assert syms == sorted(syms)


def test_minute_refresh_runs_both_freqs(monkeypatch):
    """分钟增量按 5m + 15m 两档调用 fetch 脚本, 传 --since 与标的列表。"""
    import daily_pipeline
    from types import SimpleNamespace
    calls = []

    def fake_run(cmd, **k):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(daily_pipeline.subprocess, "run", fake_run)
    monkeypatch.setattr(daily_pipeline, "_minute_symbols",
                        lambda d: ["600519", "000858"])
    out = daily_pipeline._minute_refresh("2026-08-14")
    assert len(calls) == 2, f"应分别调用 5m 与 15m, got {len(calls)}"
    freqs = [c[c.index("--freq") + 1] for c in calls]
    assert freqs == ["5", "15"], f"freq 顺序应为 5,15, got {freqs}"
    for c in calls:
        assert "--since" in c and "5" in c
        joined = " ".join(c)
        assert "600519" in joined and "000858" in joined
    assert "完成" in out


def test_minute_refresh_failure_tolerated(monkeypatch):
    """fetch 抛异常 → 返回失败文案, 不抛出 (管线继续)。"""
    import daily_pipeline

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(daily_pipeline.subprocess, "run", boom)
    monkeypatch.setattr(daily_pipeline, "_minute_symbols", lambda d: ["600519"])
    out = daily_pipeline._minute_refresh("2026-08-14")
    assert "失败" in out and "boom" in out


def test_minute_refresh_no_symbols_skips(monkeypatch):
    """无交易标的时不调用 fetch 脚本。"""
    import daily_pipeline
    calls = []
    monkeypatch.setattr(daily_pipeline.subprocess, "run",
                        lambda *a, **k: calls.append(a))
    monkeypatch.setattr(daily_pipeline, "_minute_symbols", lambda d: [])
    out = daily_pipeline._minute_refresh("2026-08-14")
    assert calls == []
    assert "跳过" in out
