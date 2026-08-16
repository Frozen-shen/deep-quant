# tests/test_signal_offline.py
"""信号生成离线化测试: 入口开启离线守卫 + 无网络预拉取。"""
import sys
import os
import inspect

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "scripts", "active"))


def test_signal_sets_offline_on_entry(monkeypatch):
    """generate_signal_v3 入口必须开启离线模式 (全局开关可见)。"""
    import netgate
    netgate.set_offline_mode(False)
    import run_paper_signal
    # 配置为空 → 函数在因子加载后提前返回, 不触碰数据/执行路径
    monkeypatch.setattr(run_paper_signal, "load_factor_config", lambda: None)
    monkeypatch.setattr(run_paper_signal, "build_fallback_config",
                        lambda: {"factors": []})
    result = run_paper_signal.generate_signal_v3(date_str="2026-08-14",
                                                 dry_run=True)
    assert result is None  # 空因子配置提前返回
    assert netgate.is_offline(), "信号生成入口必须开启离线模式"
    netgate.set_offline_mode(False)


def test_signal_source_has_no_network_prefetch():
    """回归守卫: 信号路径不得包含 MinuteFetcher 网络预拉取 (2026-08-15 卡死源)。"""
    import run_paper_signal
    src = inspect.getsource(run_paper_signal.generate_signal_v3)
    assert "fetch_batch" not in src, "信号生成不得现场拉取分钟数据"
    assert "MinuteFetcher(" not in src, "信号生成不得实例化网络 MinuteFetcher"
    assert "set_offline_mode(True)" in src
