# tests/test_executor_fallback.py
"""paper_executor 分钟数据缺失时的回退行为 (离线守卫配套)。"""
import sys
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)


def test_minute_mode_falls_back_to_daily_open(monkeypatch):
    """minute_mode 下本地分钟数据缺失 → 回退日线开盘价 (而非 None 拒单)。"""
    import pandas as pd
    from execution.paper_executor import PaperExecutor
    from netgate import set_offline_mode
    set_offline_mode(True)  # 离线: 分钟路径只能读本地 (888888 无本地数据)
    try:
        ex = PaperExecutor(minute_mode=True, execution_algo="pov")
        all_data = {"888888": pd.DataFrame({
            "date": pd.to_datetime(["2026-08-14"]),
            "open": [12.34], "close": [12.50],
        })}
        px = ex._get_execution_price("888888", "2026-08-14", all_data,
                                     order_qty=1000, side="BUY")
        assert px is not None, "本地分钟缺失时必须回退日线开盘价"
        assert abs(px - 12.34) < 1e-9, f"回退价应为日线开盘 12.34, got {px}"
    finally:
        set_offline_mode(False)


def test_daily_mode_still_uses_open():
    """日线模式行为不变 (回归对照)。"""
    import pandas as pd
    from execution.paper_executor import PaperExecutor
    ex = PaperExecutor(minute_mode=False)
    all_data = {"600000": pd.DataFrame({
        "date": pd.to_datetime(["2026-08-14"]),
        "open": [9.87], "close": [10.01],
    })}
    px = ex._get_execution_price("600000", "2026-08-14", all_data)
    assert px is not None and abs(px - 9.87) < 1e-9
