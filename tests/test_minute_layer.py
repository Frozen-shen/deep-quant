# tests/test_minute_layer.py
"""分钟因子独立叠加层 (方案B) 测试。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "active"))


def test_minute_layer_config():
    """config.yaml 的 minute_layer 段存在且含默认值。"""
    import yaml
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ml = cfg.get("minute_layer", {})
    assert ml.get("enabled") is True
    assert ml.get("lambda") == 0.3
    assert ml.get("min_icir") == 0.3
    assert ml.get("validate_folds") == [4, 5]


# ── T2: validate_minute_factors (fold 4-5 独立验证) ──


def test_validate_minute_factors_empty():
    """无 min_* 因子时返回空 dict。"""
    from run_walkforward_backtest import validate_minute_factors
    import pandas as pd
    result = validate_minute_factors({}, pd.DataFrame(), [], {}, [], [])
    assert result == {}


def test_validate_minute_factors_filters(monkeypatch):
    """ICIR 达标的分钟因子保留, 噪声因子被过滤 (确定性信号)。"""
    from run_walkforward_backtest import validate_minute_factors
    import numpy as np
    import pandas as pd
    import minute_factors as mf

    # 真实 MINUTE_FACTOR_NAMES 是 min_realized_vol 等 10 个名字, 不含测试用的
    # min_a/min_b → 不 patch 则 minute_names 过滤为空, 验证主路径根本不会执行。
    monkeypatch.setattr(mf, "MINUTE_FACTOR_NAMES", ["min_a", "min_b"])

    rng = np.random.default_rng(42)
    cal = pd.date_range("2022-01-03", periods=600, freq="B")
    cal_idx = {d: i for i, d in enumerate(cal)}
    n_stocks = 100
    syms = [f"s{i}" for i in range(n_stocks)]

    # 价格: 随机游走
    rets = rng.normal(0, 0.01, (len(cal), n_stocks))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)),
                         index=pd.DatetimeIndex(cal), columns=syms)

    # 21日前瞻收益 (训练期内标签已实现, 无前视)
    fwd = close.shift(-21) / close - 1

    panels = {}
    # min_a: 与前瞻收益负相关 → 高 |ICIR| (信号幅值远大于噪声, 且噪声足以
    # 避免 IC 恒为 ±1 导致 sd=0 → icir 被归 0)
    panels["min_a"] = pd.DataFrame(
        (-fwd + rng.normal(0, 0.001, (len(cal), n_stocks))).to_numpy(dtype=np.float32),
        index=pd.DatetimeIndex(cal), columns=syms)
    # min_b: 纯噪声 → ICIR≈0 (compute_icir_weights 用 .loc 取日截面, 必须是 DataFrame)
    panels["min_b"] = pd.DataFrame(
        rng.normal(0, 1.0, (len(cal), n_stocks)).astype(np.float32),
        index=pd.DatetimeIndex(cal), columns=syms)

    result = validate_minute_factors(
        panels, close, list(cal), cal_idx, ["min_a", "min_b"],
        [("2022-01-03", "2023-12-29")], min_icir=0.3)
    assert "min_a" in result, f"min_a 应通过验证, got {result}"
    assert "min_b" not in result, f"min_b 噪声应被过滤, got {result}"
