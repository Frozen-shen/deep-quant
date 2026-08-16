# tests/test_staleness_guard.py
"""T6 陈旧度守卫测试: 分钟数据最新日期 / 回测覆盖硬失败 / 信号陈旧度判定。"""
import sys
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "scripts", "active"))

import pandas as pd  # noqa: E402


def test_latest_local_minute_date(tmp_path):
    """抽样文件取最小最新日期 (保守)。"""
    from data.minute_fetcher import latest_local_minute_date
    for i, last in enumerate(["2026-08-13", "2026-08-14"]):
        pd.DataFrame({
            "day": pd.to_datetime(["2026-08-10", last]),
            "close": [10.0, 10.1],
        }).to_parquet(tmp_path / f"sym{i}.parquet", index=False)
    assert latest_local_minute_date(str(tmp_path)) == pd.Timestamp("2026-08-13").date()


def test_latest_local_minute_date_missing_dir():
    from data.minute_fetcher import latest_local_minute_date
    assert latest_local_minute_date("Z:/not/exist/dir_xyz") is None


def test_backtest_coverage_daily_insufficient():
    """日线覆盖不足 → 硬失败 (实验可复现性)。"""
    import run_walkforward_backtest as bt
    all_data = {"s1": pd.DataFrame({
        "date": pd.to_datetime(["2026-05-01", "2026-05-02"]),
        "close": [10.0, 10.1]})}
    try:
        bt._check_data_coverage(all_data, "2026-06-30")
        raised = False
    except RuntimeError as e:
        raised = True
        assert "日线数据覆盖不足" in str(e)
    assert raised, "覆盖不足必须硬失败"


def test_backtest_coverage_ok():
    import run_walkforward_backtest as bt
    all_data = {"s1": pd.DataFrame({
        "date": pd.to_datetime(["2026-06-28", "2026-06-30"]),
        "close": [10.0, 10.1]})}
    bt._check_data_coverage(all_data, "2026-06-30")  # 不抛 = 通过


def test_backtest_coverage_minute_insufficient(tmp_path):
    """分钟数据未覆盖运行区间末尾 (2022+) → 硬失败。"""
    import run_walkforward_backtest as bt
    pd.DataFrame({
        "day": pd.to_datetime(["2026-05-01", "2026-05-02"]),
        "close": [10.0, 10.1],
    }).to_parquet(tmp_path / "s1.parquet", index=False)
    all_data = {"s1": pd.DataFrame({
        "date": pd.to_datetime(["2026-06-30"]), "close": [10.0]})}
    try:
        bt._check_data_coverage(all_data, "2026-06-30", minute_dir=str(tmp_path))
        raised = False
    except RuntimeError as e:
        raised = True
        assert "分钟数据覆盖不足" in str(e)
    assert raised


def test_backtest_coverage_minute_skipped_pre_2022(tmp_path):
    """2022 前无分钟数据属正常 (POV 回退 VWAP/开盘), 不拦截。"""
    import run_walkforward_backtest as bt
    all_data = {"s1": pd.DataFrame({
        "date": pd.to_datetime(["2020-12-30", "2020-12-31"]),
        "close": [10.0, 10.1]})}
    bt._check_data_coverage(all_data, "2020-12-31", minute_dir=str(tmp_path))


def test_signal_minute_staleness_pure():
    """交易日差距判定: 分钟数据落后 >5 个交易日 → stale。"""
    from run_paper_signal import check_minute_staleness
    daily_dates = pd.to_datetime(
        ["2026-08-07", "2026-08-10", "2026-08-11", "2026-08-12",
         "2026-08-13", "2026-08-14"])
    # 落后 1 个交易日 → 不告警
    stale, gap = check_minute_staleness(pd.Timestamp("2026-08-13").date(),
                                        daily_dates)
    assert not stale and gap == 1
    # 落后 6 个交易日 → 告警
    stale, gap = check_minute_staleness(pd.Timestamp("2026-08-06").date(),
                                        daily_dates)
    assert stale and gap == 6
    # 无分钟数据 → 不判定 (由回测守卫另行处理)
    stale, gap = check_minute_staleness(None, daily_dates)
    assert stale is False and gap is None
