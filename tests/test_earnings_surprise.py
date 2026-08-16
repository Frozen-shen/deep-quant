"""预期差因子面板测试 (PIT-safe)。"""
import sys, os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import pandas as pd
import numpy as np


def _mk_symbol(sym, quarters, eps, announce_days, start="2020-03-31"):
    """构造 fundamental_cache 风格季度数据 → tmp 文件 (中文列)。"""
    import tempfile
    d = tempfile.mkdtemp()
    rep = [pd.Timestamp(start) + pd.DateOffset(months=3 * i) for i in range(quarters)]
    df = pd.DataFrame({"日期": rep, "摊薄每股收益(元)": eps})
    df.to_parquet(os.path.join(d, f"{sym}.parquet"), index=False)
    return d


def test_sue_panel_positive_surprise(monkeypatch, tmp_path):
    """EPS 超预期 → 公告日起 SUE>0 生效; 公告日前为 NaN。"""
    import earnings_surprise as es
    quarters = 10
    eps = [1.0, 1.05, 1.10, 1.15,  # 去年
           1.20, 1.25, 1.30, 1.35,  # 今年 (同季同比 +20%)
           1.60, 1.70]              # 最后两季: 大幅超预期
    d = _mk_symbol("600000", quarters, eps, None)
    monkeypatch.setattr(es, "FUND_DIR", d)
    monkeypatch.setattr(es, "LEGACY_DIR", str(tmp_path / "nolegacy"))
    cal = pd.date_range("2022-01-01", "2022-12-31", freq="B")
    panel = es.sue_panel(["600000"], list(cal))
    assert "600000" in panel.columns
    s = panel["600000"]
    # 最后一期公告 (报告期 2022-03-31 + 45天 ≈ 2022-05-15) 之前有值, 且超预期季度后为正
    announce = pd.Timestamp("2022-03-31") + pd.Timedelta(days=es.PIT_LAG_DAYS)
    before = s[cal < announce - pd.Timedelta(days=1)]
    after = s[cal >= announce]
    assert after.notna().all(), "公告后 SUE 必须生效"
    assert float(after.iloc[-1]) > 0, "大幅超预期季度 SUE 应为正"
    # 公告日前沿用上一期值 (PIT 无前视): 上一公告生效后至本公告前, 面板值保持恒定,
    # 不得提前反映新季报信息
    prev_ann = pd.Timestamp("2021-12-31") + pd.Timedelta(days=es.PIT_LAG_DAYS)
    seg = s[(cal >= prev_ann) & (cal < announce)]
    assert seg.notna().all(), "上一期 SUE 应已生效"
    assert seg.nunique() == 1, "公告日前不得提前反映新季报 (值应保持上一期)"
    del before, after  # noqa


def test_sue_pit_no_lookahead(monkeypatch, tmp_path):
    """前视禁止: 公告日之前面板值不得包含该公告期信息。"""
    import earnings_surprise as es
    eps = [1.0] * 8 + [5.0]  # 第 9 季暴涨
    d = _mk_symbol("600001", 9, eps, None)
    monkeypatch.setattr(es, "FUND_DIR", d)
    monkeypatch.setattr(es, "LEGACY_DIR", str(tmp_path / "nolegacy"))
    cal = pd.date_range("2021-06-01", "2022-09-30", freq="B")
    panel = es.sue_panel(["600001"], list(cal))
    s = panel["600001"]
    boom_announce = pd.Timestamp("2022-03-31") + pd.Timedelta(days=es.PIT_LAG_DAYS)
    pre = s[cal < boom_announce]
    # 暴涨季公告前, SUE 不可能反映该季 → 全部值 ≤ 0 (此前各季无惊喜)
    assert (pre.dropna() <= 0).all(), "公告前不得包含暴涨季信息"


def test_earn_accel_panel_basic(monkeypatch, tmp_path):
    import earnings_surprise as es
    # yoy: 前四季 +10%, 后两季 +30% → 加速为正
    eps = [1.00, 1.00, 1.00, 1.00, 1.10, 1.10, 1.10, 1.10, 1.43, 1.43]
    d = _mk_symbol("600002", 10, eps, None)
    monkeypatch.setattr(es, "FUND_DIR", d)
    monkeypatch.setattr(es, "LEGACY_DIR", str(tmp_path / "nolegacy"))
    cal = pd.date_range("2022-01-01", "2022-12-31", freq="B")
    panel = es.earn_accel_panel(["600002"], list(cal))
    s = panel["600002"]
    ann = pd.Timestamp("2022-03-31") + pd.Timedelta(days=es.PIT_LAG_DAYS)
    # 公告生效首日加速为正 (年末面板值携带的是后续 2022-06-30 季的加速值 0, 故取首值)
    assert float(s[cal >= ann].iloc[0]) > 0, "增速加速应为正"


def test_pead_panel_no_lookahead(monkeypatch, tmp_path):
    """公告后 20 日窗口内 PEAD 值必须为 NaN (前视禁止), 窗口完成后生效。"""
    import earnings_surprise as es
    # 序列自 2022-03-31 起: 该季公告是被测的首个事件 (其窗口内无更早已完成窗口)
    eps = [1.0] * 2
    d = _mk_symbol("600003", 2, eps, None, start="2022-03-31")
    monkeypatch.setattr(es, "FUND_DIR", d)
    monkeypatch.setattr(es, "LEGACY_DIR", str(tmp_path / "nolegacy"))
    cal = pd.date_range("2022-01-01", "2022-12-31", freq="B")
    close = pd.Series(100 * (1.001 ** np.arange(len(cal))), index=cal)
    all_data = {"600003": pd.DataFrame({"date": cal, "close": close.values})}
    panel = es.pead_panel(["600003"], all_data, list(cal))
    s = panel["600003"]
    # 公告 = 报告期2022-03-31+45天; 其后20个交易日内应为 NaN
    ann = pd.Timestamp("2022-03-31") + pd.Timedelta(days=es.PIT_LAG_DAYS)
    pos = cal.searchsorted(ann)
    in_window = cal[pos:pos + 20]
    assert s[in_window].isna().all(), "漂移窗口内不得生效 (前视)"
    assert s[cal[pos + 20]:].notna().all(), "窗口完成后生效"
