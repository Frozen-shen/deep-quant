# tests/test_execution_pricing.py
"""执行定价模块 (L0/L1/L2) 测试。

L0: minute_fetcher 小订单确定性 POV / 缓存截断修复 / get_pov_price 与
    get_pov_fills 统一 / start_bar 参数。
L0b: paper_executor 滑点选择 (算法单 10bp 残差, open/close 30bp)。
L1: execution/exec_quality.py 的 fill_quality 符号化偏差数值。
L2: execution/execution_overlay.py 三个因果规则 + 因果性 + 超时。
"""
import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

TIME_RE = re.compile(r"^\d{2}:\d{2}$")


# ────────────────────────── L0: minute_fetcher ──────────────────────────


def test_small_order_fill_is_plain_time_not_random_label():
    """小订单 fill_times 必须是纯 HH:MM (随机分支的 '市价@HH:MM' 已删除)。"""
    from data.minute_fetcher import MinuteFetcher
    mf = MinuteFetcher(allow_network=False)
    res = mf.get_pov_fills("600519", "2024-03-08", 100)
    assert res is not None, "600519 本地分钟数据应存在"
    assert res["n_fills"] >= 1
    for f in res["fills"]:
        assert TIME_RE.match(f["time"]), f"fill time 应为纯 HH:MM, got {f['time']}"


def test_small_order_fills_early_in_day():
    """小订单 (占日成交量 <0.1%) 应在开盘后前几根 bar 内成交完。"""
    from data.minute_fetcher import MinuteFetcher
    mf = MinuteFetcher(allow_network=False)
    res = mf.get_pov_fills("600519", "2024-03-08", 100)
    assert res is not None
    first_time = res["fills"][0]["time"]
    assert first_time <= "10:00", f"小订单应开盘后不久成交, got {first_time}"


def test_pov_fills_deterministic():
    """同一输入两次调用结果完全一致 (无随机性)。"""
    from data.minute_fetcher import MinuteFetcher
    mf = MinuteFetcher(allow_network=False)
    r1 = mf.get_pov_fills("600519", "2024-03-08", 100)
    r2 = mf.get_pov_fills("600519", "2024-03-08", 100)
    assert r1 == r2


def test_fetch_cache_not_truncated_by_first_end_date():
    """修复: 同一 symbol 先后用两个 end_date, 后者必须取到更晚的数据。"""
    from data.minute_fetcher import MinuteFetcher
    mf = MinuteFetcher(allow_network=False)
    df1 = mf.fetch("600519", days=10, end_date="2024-06-28")
    df2 = mf.fetch("600519", days=10, end_date="2024-12-31")
    assert df1 is not None and df2 is not None
    assert df2["时间"].max() > df1["时间"].max(), \
        "第二次 fetch 不应被首次 end_date 的缓存截断"


def test_get_pov_price_equals_get_pov_fills():
    """统一实现后 get_pov_price 与 get_pov_fills 的价格一致。"""
    from data.minute_fetcher import MinuteFetcher
    mf = MinuteFetcher(allow_network=False)
    for sym, date in [("600519", "2024-03-08"), ("003043", "2025-01-03"),
                      ("000001", "2024-07-15")]:
        p = mf.get_pov_price(sym, date, 100)
        f = mf.get_pov_fills(sym, date, 100)
        assert p is not None and f is not None
        assert abs(p - f["price"]) < 1e-9, f"{sym} {date}: price 不一致"


def test_pov_fills_start_bar_skips_early_bars():
    """L2 支持: start_bar=N 时首笔成交不早于第 N 根 bar。"""
    from data.minute_fetcher import MinuteFetcher
    mf = MinuteFetcher(allow_network=False)
    res = mf.get_pov_fills("600519", "2024-03-08", 100, start_bar=6)
    assert res is not None
    assert res["fills"][0]["time"] >= "10:00", \
        f"start_bar=6 应跳过前 6 根 bar, got {res['fills'][0]['time']}"


# ────────────────────────── L0b: 滑点选择 ──────────────────────────


def test_pick_slippage_algo_uses_residual():
    """pov/vwap/twap 算法单用残差滑点 (10bp), open/close 用全额滑点 (30bp)。"""
    from execution.paper_executor import pick_slippage_bps
    assert pick_slippage_bps(30, 10, minute_mode=True, algo="pov") == 10
    assert pick_slippage_bps(30, 10, minute_mode=True, algo="vwap") == 10
    assert pick_slippage_bps(30, 10, minute_mode=True, algo="twap") == 10
    assert pick_slippage_bps(30, 10, minute_mode=True, algo="open") == 30
    assert pick_slippage_bps(30, 10, minute_mode=True, algo="close") == 30
    # 非分钟模式: 日线开盘价成交, 保持全额滑点
    assert pick_slippage_bps(30, 10, minute_mode=False, algo="pov") == 30
    # 未传 residual 时向后兼容
    assert pick_slippage_bps(30, None, minute_mode=True, algo="pov") == 30


# ────────────────────────── L1: fill_quality ──────────────────────────


class _FakeFetcher:
    """返回合成单日分钟数据 (依赖注入, 供 fill_quality 测试)。"""
    def __init__(self, day_df):
        self.day_df = day_df

    def fetch(self, symbol, days=10, end_date=None):
        return self.day_df.copy()


def _make_day_df():
    """4 根 bar: close 10.0/10.1/9.9/10.2, 等量; low=9.9, high=10.2。"""
    times = pd.to_datetime(["2024-03-08 09:35", "2024-03-08 09:40",
                            "2024-03-08 09:45", "2024-03-08 09:50"])
    return pd.DataFrame({
        "时间": times,
        "开盘": [10.0, 10.1, 9.9, 10.2],
        "收盘": [10.0, 10.1, 9.9, 10.2],
        "最高": [10.0, 10.1, 9.9, 10.2],
        "最低": [9.95, 10.05, 9.9, 10.15],
        "成交量": [100, 100, 100, 100],
    })


def test_fill_quality_math():
    """等量 4 bar, VWAP=(10.0+10.1+9.9+10.2)/4=10.05。"""
    from execution.exec_quality import fill_quality
    day = _make_day_df()
    mf = _FakeFetcher(day)
    trades = [
        {"date": "2024-03-08", "symbol": "X", "action": "BUY",
         "price": 10.30, "qty": 100},
        {"date": "2024-03-08", "symbol": "X", "action": "SELL",
         "price": 9.80, "qty": 100},
    ]
    q = fill_quality(trades, mf)
    assert q["n"] == 2
    # BUY: vwap_dev = (10.30/10.05-1)*1e4 = 248.8; 到达价(首bar收盘)=10.0 → 300
    # SELL: vwap_dev = (10.05/9.80-1)*1e4 = 255.1
    b = q["by_action"]["BUY"]; s = q["by_action"]["SELL"]
    assert abs(b["vwap_dev_bps"]["mean"] - 248.8) < 0.1
    assert abs(b["arrival_dev_bps"]["mean"] - 300.0) < 0.1
    assert abs(s["vwap_dev_bps"]["mean"] - 255.1) < 0.1
    # 完美择时上限: BUY 买在最低 9.9 → (10.30/9.9-1)*1e4=404.0
    # SELL 卖在最高 10.2 → (10.2/9.8-1)*1e4=408.2
    assert abs(b["perfect_gain_bps"]["mean"] - 404.0) < 0.1
    assert abs(s["perfect_gain_bps"]["mean"] - 408.2) < 0.1
    # 汇总 perfect_gain = (404.0+408.2)/2 = 406.1
    assert abs(q["overall"]["perfect_gain_bps"]["mean"] - 406.1) < 0.1


def test_fill_quality_skips_missing_data():
    """取不到分钟数据的 trade 跳过, 不报错。"""
    from execution.exec_quality import fill_quality
    mf = _FakeFetcher(None)  # fetch 返回 None
    q = fill_quality([{"date": "2024-03-08", "symbol": "X", "action": "BUY",
                       "price": 10.0, "qty": 100}], mf)
    assert q["n"] == 0
    assert q["overall"]["vwap_dev_bps"]["mean"] is None


# ────────────────────────── L2: execution_overlay ──────────────────────────


def _bars(opens, closes, vols=None):
    """构造单日 bar DataFrame (09:35 起, 5 分钟间隔)。"""
    n = len(closes)
    times = pd.date_range("2024-03-08 09:35", periods=n, freq="5min")
    vols = vols if vols is not None else [100] * n
    return pd.DataFrame({
        "时间": times,
        "开盘": opens,
        "收盘": closes,
        "最高": [max(o, c) for o, c in zip(opens, closes)],
        "最低": [min(o, c) for o, c in zip(opens, closes)],
        "成交量": vols,
    })


def test_gap_rule_waits_for_retrace():
    """高开 4% 超阈值 → 等回撤到缺口一半以内才执行。"""
    from execution.execution_overlay import decide_start_bar
    closes = [10.40] + [10.38] * 3 + [10.14] + [10.15] * 20
    opens = [10.40] + closes[:-1]
    day = _bars(opens, closes)
    rules = {"gap_wait": {"gap_bps": 300, "timeout_bars": 12}}
    # prev_close=10.0, 缺口 400bps; bar index 4 收盘 10.14 → 距昨收 140bps < 150bps
    start = decide_start_bar(day, "BUY", rules, prev_close=10.0)
    assert start == 4


def test_gap_rule_no_gap_returns_zero():
    from execution.execution_overlay import decide_start_bar
    closes = [10.01] * 20
    day = _bars([10.01] * 20, closes)
    rules = {"gap_wait": {"gap_bps": 300, "timeout_bars": 12}}
    assert decide_start_bar(day, "BUY", rules, prev_close=10.0) == 0


def test_gap_rule_timeout_forces_execution():
    """缺口不回落 → 超时强制执行 (start_bar=timeout_bars)。"""
    from execution.execution_overlay import decide_start_bar
    closes = [10.40] * 20
    day = _bars([10.40] * 20, closes)
    rules = {"gap_wait": {"gap_bps": 300, "timeout_bars": 12}}
    assert decide_start_bar(day, "BUY", rules, prev_close=10.0) == 12


def test_momentum_rule_waits_for_cooloff():
    """前 30 分钟涨 3% 超阈值 → 等回落才执行。"""
    from execution.execution_overlay import decide_start_bar
    closes = [10.30] * 6 + [10.28] * 4 + [10.09] + [10.10] * 15
    day = _bars([10.30] + closes[:-1], closes)
    rules = {"momentum_wait": {"mom_bps": 200, "timeout_bars": 12}}
    # 前 30 分钟 +300bps; bar10 收盘 10.09 → +90bps < 100bps 阈值一半
    assert decide_start_bar(day, "BUY", rules, prev_close=10.0) == 10


def test_volume_rule_waits_until_volume_picks_up():
    """低量比 → 等放量 bar 或超时。"""
    from execution.execution_overlay import decide_start_bar
    vols = [100] + [20, 20, 20] + [200] + [100] * 20
    closes = [10.0] * len(vols)
    day = _bars([10.0] * len(vols), closes, vols=vols)
    rules = {"volume_wait": {"vol_min": 2.0, "timeout_bars": 6}}
    # bar1: 20/100=0.2<2 等; bar2: 20/20=1.0<2 等; bar3: 20/20=1.0<2 等;
    # bar4: 200/20=10≥2 → 放量, 从 bar4 起执行
    assert decide_start_bar(day, "BUY", rules) == 4


def test_volume_rule_timeout():
    from execution.execution_overlay import decide_start_bar
    vols = [100] + [20] * 20
    closes = [10.0] * len(vols)
    day = _bars([10.0] * len(vols), closes, vols=vols)
    rules = {"volume_wait": {"vol_min": 2.0, "timeout_bars": 6}}
    assert decide_start_bar(day, "BUY", rules) == 6


def test_overlay_decision_is_causal():
    """决策只依赖 start_bar 之前的 bar: 截断后续数据结果不变。"""
    from execution.execution_overlay import decide_start_bar
    closes = [10.40] + [10.38] * 3 + [10.14] + [10.15] * 20
    day = _bars([10.40] + closes[:-1], closes)
    rules = {"gap_wait": {"gap_bps": 300, "timeout_bars": 12}}
    start = decide_start_bar(day, "BUY", rules, prev_close=10.0)
    assert start > 0
    truncated = day.iloc[:start + 1].reset_index(drop=True)
    assert decide_start_bar(truncated, "BUY", rules, prev_close=10.0) == start


def test_overlay_no_rules_returns_zero():
    """rules=None 或空 → 立即执行 (行为与现在一致)。"""
    from execution.execution_overlay import decide_start_bar
    day = _bars([10.40] * 20, [10.40] * 20)
    assert decide_start_bar(day, "BUY", None, prev_close=10.0) == 0
    assert decide_start_bar(day, "BUY", {}, prev_close=10.0) == 0
