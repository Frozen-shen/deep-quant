# tests/test_backtest_repair.py
"""回测引擎修复 (2026-08-15) 测试。

- turnover_period_cap: 月频调仓期换手上限不再被 20/21 缩放压线 (选项A)
- build_calendar: 过滤周末脏日期
- safe_mark_to_market: 无有效收盘价时沿用前值, 不按 0 计价
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "active"))

import pandas as pd


# ── 换手约束 (选项 A: 月频不缩放) ──


def test_turnover_cap_monthly_not_scaled():
    """月频 (>=20 天) 期上限 = 月上限, 不再乘 20/21 自我压线。"""
    from run_walkforward_backtest import turnover_period_cap
    assert turnover_period_cap(0.5, 20) == 0.5
    assert turnover_period_cap(0.5, 21) == 0.5
    assert turnover_period_cap(0.5, 30) == 0.5


def test_turnover_cap_short_period_scaled():
    """短周期 (周频等) 仍按天数比例缩放。"""
    from run_walkforward_backtest import turnover_period_cap
    assert abs(turnover_period_cap(0.5, 5) - 0.5 * 5 / 21.0) < 1e-9
    assert abs(turnover_period_cap(0.5, 10) - 0.5 * 10 / 21.0) < 1e-9
    assert abs(turnover_period_cap(0.5, 19) - 0.5 * 19 / 21.0) < 1e-9


def test_turnover_50pct_not_skipped_under_monthly_cap():
    """修复前: 换手 50% > 期上限 47.6% → 每轮调仓被跳过。修复后不跳过。"""
    from run_walkforward_backtest import turnover_period_cap
    cap = turnover_period_cap(0.5, 20)
    turnover = (5 + 5) / (2 * 10)  # 买5卖5, 持仓10 → 50%
    assert turnover <= cap, f"换手 {turnover} 不应超过期上限 {cap}"


# ── 日历过滤周末 ──


def _fake_all_data(dates):
    """构造含指定日期的单只股票日线 DataFrame。"""
    return {"000001": pd.DataFrame({"date": pd.to_datetime(dates),
                                    "close": [10.0] * len(dates)})}


def test_build_calendar_filters_weekends():
    """脏数据中的周末日期不进入交易日历。"""
    from run_walkforward_backtest import build_calendar
    all_data = _fake_all_data(["2021-05-28", "2021-05-30", "2021-05-31"])
    cal = build_calendar(all_data, min_coverage=1)
    dates = {d.date() for d in cal}
    assert pd.Timestamp("2021-05-28").date() in dates
    assert pd.Timestamp("2021-05-31").date() in dates
    assert pd.Timestamp("2021-05-30").date() not in dates  # 周日


def test_build_calendar_keeps_weekdays_and_sorts():
    """正常工作日保留且排序。"""
    from run_walkforward_backtest import build_calendar
    all_data = _fake_all_data(["2021-06-02", "2021-06-01", "2021-06-03"])
    cal = build_calendar(all_data, min_coverage=1)
    assert [d.date() for d in cal] == [
        pd.Timestamp("2021-06-01").date(),
        pd.Timestamp("2021-06-02").date(),
        pd.Timestamp("2021-06-03").date(),
    ]


# ── mark-to-market 防御 ──


def test_safe_mtm_carries_forward_when_no_prices():
    """持仓存在但当日无任何有效收盘价 → 沿用前值 (不按 0 计价)。"""
    from run_walkforward_backtest import safe_mark_to_market
    equity_fn = lambda cp: 1000.0 + sum(cp.get(s, 0) for s in ("a", "b"))
    positions = {"a": {"qty": 100}, "b": {"qty": 200}}
    eq = safe_mark_to_market(equity_fn, positions, {}, prev_equity=15000.0)
    assert eq == 15000.0


def test_safe_mtm_marks_normally_with_prices():
    """有收盘价时正常计价。"""
    from run_walkforward_backtest import safe_mark_to_market
    equity_fn = lambda cp: 1000.0 + sum(cp.get(s, 0) for s in ("a", "b"))
    positions = {"a": {"qty": 100}, "b": {"qty": 200}}
    eq = safe_mark_to_market(equity_fn, positions, {"a": 10.0, "b": 20.0},
                             prev_equity=15000.0)
    assert eq == 1000.0 + 10.0 + 20.0


def test_safe_mtm_empty_positions_uses_fn():
    """无持仓时 (空仓) 直接计价 (cash 场景, 不误用前值)。"""
    from run_walkforward_backtest import safe_mark_to_market
    equity_fn = lambda cp: 1000.0
    eq = safe_mark_to_market(equity_fn, {}, {}, prev_equity=15000.0)
    assert eq == 1000.0


# ── 等权基准收益 winsorize ──


def test_benchmark_ret_winsorize_clips_extreme():
    """单只股票单日 +100 万% 的脏价格不炸掉基准。"""
    from run_walkforward_backtest import _benchmark_daily_ret
    import pandas as pd
    all_data = {
        # 正常股票: 每天 +1%
        "a": pd.DataFrame({"date": pd.to_datetime(
            ["2023-01-03", "2023-01-04", "2023-01-05"]),
            "close": [100.0, 101.0, 102.01]}),
        # 脏股票: 1 月 4 日价格从 1 跳到 10000 (数据源错误)
        "b": pd.DataFrame({"date": pd.to_datetime(
            ["2023-01-03", "2023-01-04", "2023-01-05"]),
            "close": [1.0, 10000.0, 10001.0]}),
    }
    eqw = _benchmark_daily_ret(all_data, min_coverage=1)
    # 1月4日的等权收益 = mean(clip(1%), clip(999900%)) ≈ 0.505/2 之内
    r_0104 = eqw.loc[pd.Timestamp("2023-01-04")]
    assert 0.0 < r_0104 <= 0.5 + 1e-9, f"脏收益应被截断, got {r_0104}"
    # 1月5日: 两只都正常 (a: +1%, b: +0.01%)
    r_0105 = eqw.loc[pd.Timestamp("2023-01-05")]
    assert abs(r_0105 - (0.01 + 0.0001) / 2) < 1e-6


def test_benchmark_ret_winsorize_normal_days_untouched():
    """无脏数据时基准收益不变。"""
    from run_walkforward_backtest import _benchmark_daily_ret
    import pandas as pd
    all_data = {
        "a": pd.DataFrame({"date": pd.to_datetime(
            ["2023-01-03", "2023-01-04"]),
            "close": [100.0, 101.0]}),
        "b": pd.DataFrame({"date": pd.to_datetime(
            ["2023-01-03", "2023-01-04"]),
            "close": [50.0, 49.0]}),
    }
    eqw = _benchmark_daily_ret(all_data, min_coverage=1)
    assert abs(eqw.loc[pd.Timestamp("2023-01-04")] - (0.01 - 0.02) / 2) < 1e-9


# ── 覆盖度过滤 (春节假期只有 2 只退市股有数据 → 假交易日) ──


def test_benchmark_ret_drops_low_coverage_days():
    """覆盖度 < min_coverage 的日期不计入基准 (防假交易日污染)。"""
    from run_walkforward_backtest import _benchmark_daily_ret
    import pandas as pd
    all_data = {}
    for i in range(120):
        all_data[f"s{i}"] = pd.DataFrame({
            "date": pd.to_datetime(["2026-02-13", "2026-02-24"]),
            "close": [100.0, 101.0]})
    # 2 只退市股在假期 (02-19) 有脏行, 单日 +30%
    all_data["dirty1"] = pd.DataFrame({
        "date": pd.to_datetime(["2026-02-13", "2026-02-19", "2026-02-24"]),
        "close": [1.0, 1.3, 1.3]})
    all_data["dirty2"] = pd.DataFrame({
        "date": pd.to_datetime(["2026-02-13", "2026-02-19", "2026-02-24"]),
        "close": [2.0, 2.0, 2.0]})
    eqw = _benchmark_daily_ret(all_data, min_coverage=100)
    # 02-19 只有 2 只股票有数据 → 剔除
    assert pd.Timestamp("2026-02-19") not in eqw.index
    # 正常日保留
    assert pd.Timestamp("2026-02-24") in eqw.index


def test_benchmark_ret_low_coverage_default_100():
    """默认 min_coverage=100。"""
    from run_walkforward_backtest import _benchmark_daily_ret
    import pandas as pd
    all_data = {}
    for i in range(99):
        all_data[f"s{i}"] = pd.DataFrame({
            "date": pd.to_datetime(["2026-02-13", "2026-02-19"]),
            "close": [100.0, 101.0]})
    eqw = _benchmark_daily_ret(all_data)
    assert pd.Timestamp("2026-02-19") not in eqw.index


def test_build_calendar_drops_low_coverage_days():
    """日历剔除覆盖 <100 只的假交易日 (春节假期只有退市股有数据)。"""
    from run_walkforward_backtest import build_calendar
    import pandas as pd
    all_data = {}
    for i in range(120):
        all_data[f"s{i}"] = pd.DataFrame({
            "date": pd.to_datetime(["2026-02-13", "2026-02-24"]),
            "close": [100.0, 101.0]})
    all_data["dirty"] = pd.DataFrame({
        "date": pd.to_datetime(["2026-02-13", "2026-02-19", "2026-02-24"]),
        "close": [1.0, 1.3, 1.3]})
    cal = build_calendar(all_data, min_coverage=100)
    dates = {d.date() for d in cal}
    assert pd.Timestamp("2026-02-13").date() in dates
    assert pd.Timestamp("2026-02-24").date() in dates
    assert pd.Timestamp("2026-02-19").date() not in dates


# ── 单票上限 (v27: 约束从日志改为实际执行) ──


def test_decision_weights_capped_by_max_single():
    """买入 5 只、上限 20%: 等权 20% 不超限则等权; 3 只等权 33% 超限则每只 20%。"""
    from run_walkforward_backtest import apply_portfolio_constraints
    c = {"max_single_pct": 0.20}
    w5 = apply_portfolio_constraints({s: 1.0 for s in "abcde"}, c)
    assert abs(w5["a"] - 0.20) < 1e-9  # 5 只等权=20% 恰好不超限
    w3 = apply_portfolio_constraints({s: 1.0 for s in "abc"}, c)
    assert abs(w3["a"] - 0.20) < 1e-9  # 3 只等权 33% > 20% → 缩到 20%
    assert abs(sum(w3.values()) - 0.60) < 1e-9  # 剩余 40% 留现金


def test_cap_single_weights_preserves_relative_scale():
    """已有权重 (inv_vol 等) 只做个股封顶, 不改变未超限股票的相对比例。"""
    from run_walkforward_backtest import _cap_single_weights
    w = {"a": 0.30, "b": 0.10, "c": 0.10, "d": 0.10}  # a 超限
    out = _cap_single_weights(w, max_single_pct=0.20)
    assert abs(out["a"] - 0.20) < 1e-9  # 封顶
    assert abs(out["b"] - 0.10) < 1e-9  # 其余不变
    assert abs(out["c"] - 0.10) < 1e-9
    assert abs(out["d"] - 0.10) < 1e-9


# ── 行业中性 (v28) ──


def test_industry_map_code_alignment():
    """行业映射键带 sh/sz 前缀, 对齐后按 6 位代码可查。"""
    from run_walkforward_backtest import _load_industry_map
    m = _load_industry_map()
    assert m, "行业映射文件应存在"
    assert "600519" in m, "对齐后应可用 6 位代码查询"


def test_industry_cap_weights():
    """同一行业权重之和超上限时, 该行业所有股票按比例缩减。"""
    from run_walkforward_backtest import _industry_cap_weights
    ind = {"a": "银行", "b": "银行", "c": "医药", "d": "医药", "e": "机械"}
    w = {s: 0.20 for s in "abcde"}  # 等权 20%
    out = _industry_cap_weights(w, ind, max_industry_pct=0.30)
    # 银行 40% > 30% → 缩到 30% (每只 15%)
    assert abs(out["a"] + out["b"] - 0.30) < 1e-9
    assert abs(out["a"] - 0.15) < 1e-9
    # 医药 40% > 30% → 同样缩减
    assert abs(out["c"] + out["d"] - 0.30) < 1e-9
    # 机械 20% 不超限 → 不变
    assert abs(out["e"] - 0.20) < 1e-9
    # 无行业映射的股票不参与约束
    w2 = {"a": 0.20, "b": 0.20, "zzz": 0.20}
    out2 = _industry_cap_weights(w2, {"a": "银行", "b": "银行"},
                                 max_industry_pct=0.30)
    assert abs(out2["zzz"] - 0.20) < 1e-9


# ── 风格预算 (v29) ──


def test_style_budget_caps_family_weight():
    """同因子家族权重之和超上限时按比例缩减 (风格预算)。"""
    from run_walkforward_backtest import _style_budget_weights
    weights = {"amihud_5d": 0.3, "amihud_20d": 0.3, "volatility_20d": 0.2,
               "return_30d": 0.2}
    out = _style_budget_weights(weights, style_cap=0.4)
    # amihud 家族 0.6 > 0.4 → 缩到 0.4 (每只 0.2)
    assert abs(out["amihud_5d"] + out["amihud_20d"] - 0.4) < 1e-9
    assert abs(out["amihud_5d"] - 0.2) < 1e-9
    # 其他家族不超限 → 不变
    assert abs(out["volatility_20d"] - 0.2) < 1e-9
    assert abs(out["return_30d"] - 0.2) < 1e-9


def test_style_budget_weighted_by_abs_icir():
    """家族缩放保持组内相对权重 (按 |权重| 比例缩, 符号保留)。"""
    from run_walkforward_backtest import _style_budget_weights
    weights = {"amihud_5d": 0.6, "amihud_20d": -0.3, "return_30d": 0.1}
    out = _style_budget_weights(weights, style_cap=0.4)
    assert abs(abs(out["amihud_5d"]) + abs(out["amihud_20d"]) - 0.4) < 1e-9
    assert out["amihud_5d"] > 0 and out["amihud_20d"] < 0  # 符号保留
    assert abs(out["return_30d"] - 0.1) < 1e-9
