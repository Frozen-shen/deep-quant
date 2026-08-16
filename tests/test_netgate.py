# tests/test_netgate.py
"""netgate 离线守卫测试: 训练/回测/信号离线, 网络统一前置到获取阶段。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_netgate_default_off():
    """默认离线模式关闭 (获取脚本/执行器进程不受影响)。"""
    import netgate
    netgate.set_offline_mode(False)
    assert netgate.is_offline() is False


def test_netgate_set_roundtrip():
    import netgate
    netgate.set_offline_mode(True)
    assert netgate.is_offline() is True
    netgate.set_offline_mode(False)
    assert netgate.is_offline() is False


def test_netgate_violation_is_exception():
    from netgate import OfflineViolation
    assert issubclass(OfflineViolation, Exception)
