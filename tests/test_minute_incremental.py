# tests/test_minute_incremental.py
"""fetch_baostock_minute 增量合并测试 (--since 模式核心逻辑)。"""
import sys
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "scripts", "active"))

import pandas as pd  # noqa: E402


def _frame(dts):
    import pandas as pd
    n = len(dts)
    dts = pd.to_datetime(dts)
    return pd.DataFrame({
        "datetime": dts,
        "day": dts.normalize(),
        "open": [10.0] * n, "high": [10.2] * n, "low": [9.9] * n,
        "close": [10.1] * n, "volume": [1000] * n, "amount": [10100] * n,
    })


def test_merge_minute_incremental_dedupe_keep_new():
    """已有缓存 + 增量帧: 按 datetime 去重保留最新值, 按时间排序。"""
    from fetch_baostock_minute import merge_minute_incremental
    old = _frame(["2026-08-01 09:35:00", "2026-08-01 09:40:00",
                  "2026-08-04 09:35:00"])
    old.loc[2, "close"] = 10.5  # 08-04 旧值
    new = _frame(["2026-08-04 09:35:00", "2026-08-05 09:35:00"])
    new.loc[0, "close"] = 10.55  # 08-04 新值 (前复权重算)
    out = merge_minute_incremental(old, new)
    assert len(out) == 4, f"去重后应为 4 行, got {len(out)}"
    assert list(out["datetime"]) == sorted(out["datetime"]), "必须按时间排序"
    row = out[out["datetime"] == pd.Timestamp("2026-08-04 09:35:00")]
    assert float(row["close"].iloc[0]) == 10.55, "重复时间戳保留最新值"


def test_merge_minute_incremental_no_existing():
    """无已有缓存时直接返回增量帧。"""
    from fetch_baostock_minute import merge_minute_incremental
    new = _frame(["2026-08-05 09:35:00"])
    out = merge_minute_incremental(None, new)
    assert len(out) == 1


def test_merge_minute_incremental_empty_new_keeps_existing():
    """增量帧为空时保留已有缓存 (写回不变)。"""
    from fetch_baostock_minute import merge_minute_incremental
    import pandas as pd
    old = _frame(["2026-08-01 09:35:00"])
    out = merge_minute_incremental(old, pd.DataFrame())
    assert len(out) == 1
