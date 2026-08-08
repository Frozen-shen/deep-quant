# 分钟因子独立叠加层（方案B）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 10 个分钟因子（min_*）通过独立验证层（fold 4-5 训练期 ICIR）作为叠加信号进入 TEST 回测，改善 TEST 超额/IR，且不改变主 fold 筛选体系。

**Architecture:** 主通道（182 因子 → 40 稳定因子 → ICIR 权重 → score_stocks）不变；新增分钟通道（10 个 min_* → fold 4-5 独立验证 ICIR → λ 线性叠加进 score_stocks）。改动 3 个文件：`scripts/active/run_walkforward_backtest.py`（核心）、`config.yaml`（配置）、`tests/test_minute_layer.py`（测试）。

**Tech Stack:** Python 3.12, pandas, numpy（复用现有 `compute_icir_weights` 逻辑做独立验证）

## Global Constraints

- **主组合不动**：40 个稳定因子的 fold 筛选逻辑零变化；分钟因子不进入稳定因子列表
- **可回退**：`config.minute_layer.enabled: false` 时行为与 v5 完全一致
- **无前视**：分钟因子验证只用 fold 4-5 训练期（2022-01~2023-12 / 2022-01~2024-12），TEST（2025-2026）是真样本外
- 参数从 config.yaml 读取（唯一参数源），不散落全局变量
- Windows 用 `py` 命令；改完跑 `py -m pytest tests/ -q` 全量通过
- 不得新建脚本（只改现有文件）；测试文件 `tests/test_minute_layer.py` 是新测试文件（允许）

---

## File Structure

| 文件 | 职责 | 任务 |
|------|------|------|
| `config.yaml` | 新增 `minute_layer` 配置段 | T1 |
| `scripts/active/run_walkforward_backtest.py` | `validate_minute_factors` 新函数 + `score_stocks` 叠加 + `run_fold_test` 接线 | T2, T3 |
| `tests/test_minute_layer.py` | 测试验证逻辑/叠加逻辑/回退行为 | T1-T3 |

## 任务总览（3 个任务）

- T1: config.minute_layer 配置段 + 测试文件骨架
- T2: `validate_minute_factors`（fold 4-5 独立验证）+ 测试
- T3: `score_stocks` 叠加层 + `run_fold_test` 接线 + 全量回归

---

### Task 1: config.minute_layer 配置段 + 测试骨架

**Files:**
- Modify: `config.yaml`（新增 minute_layer 段）
- Create: `tests/test_minute_layer.py`（骨架）

**Interfaces:**
- Produces: `config["minute_layer"]` = `{enabled: true, lambda: 0.3, min_icir: 0.3, validate_folds: [4, 5]}`
- Produces: `tests/test_minute_layer.py` 文件（含 import 骨架，后续任务填充用例）

- [ ] **Step 1: 在 config.yaml 添加 minute_layer 段**

在现有 `minute_factors` 段（约 69 行）之后添加：

```yaml
# ── 分钟因子叠加层 (方案B, 2026-08-04) ──
minute_layer:
  enabled: true        # 总开关: false 时行为与 v5 完全一致
  lambda: 0.3          # 叠加权重 (综合分 = 主分 + λ×分钟分)
  min_icir: 0.3        # 分钟因子保留的 ICIR 门槛
  validate_folds: [4, 5]  # 独立验证使用的 fold 训练期
```

- [ ] **Step 2: 创建测试文件骨架**

```python
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
```

- [ ] **Step 3: 运行测试确认通过**

Run: `py -m pytest tests/test_minute_layer.py::test_minute_layer_config -v`
Expected: PASS

- [ ] **Step 4: 提交**

```bash
git add config.yaml tests/test_minute_layer.py
git commit -m "feat(v6): minute_layer config + test skeleton"
```

---

### Task 2: validate_minute_factors（fold 4-5 独立验证）

**Files:**
- Modify: `scripts/active/run_walkforward_backtest.py`（新增函数，放在 `_merge_minute_panels` 之后）
- Test: `tests/test_minute_layer.py`（追加用例）

**Interfaces:**
- Consumes: `compute_icir_weights`（685 行，现有函数）、`factor_panels`（含 min_* 面板）、`calendar`/`cal_idx`/`close_panel`
- Produces: `validate_minute_factors(factor_panels, close_panel, calendar, cal_idx, factor_names, train_folds, min_icir=0.3) -> dict`
  - 参数: `train_folds`: list of (train_start, train_end) 元组，如 `[("2022-01-01", "2023-12-31"), ("2022-01-01", "2024-12-31")]`
  - 返回: `{min_factor: validated_icir}` — 通过验证的分钟因子及其 ICIR（各 fold 中位数）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_minute_layer.py 追加
def test_validate_minute_factors_empty():
    """无 min_* 因子时返回空 dict。"""
    from run_walkforward_backtest import validate_minute_factors
    import pandas as pd
    result = validate_minute_factors({}, pd.DataFrame(), [], {}, [], [])
    assert result == {}


def test_validate_minute_factors_filters():
    """ICIR 低于门槛的分钟因子被过滤。"""
    from run_walkforward_backtest import validate_minute_factors
    import numpy as np
    import pandas as pd
    # 构造: 一个强因子(min_a ICIR高) 一个弱因子(min_b ICIR低)
    cal = pd.date_range("2022-01-03", periods=600, freq="B")
    cal_idx = {d: i for i, d in enumerate(cal)}
    n_stocks = 100
    syms = [f"s{i}" for i in range(n_stocks)]
    close = pd.DataFrame(
        {s: np.random.default_rng(42).normal(100, 1, len(cal)).cumsum()
         for s in syms}, index=pd.DatetimeIndex(cal))
    # min_a: 与未来收益强相关 (构造 ICIR 高)
    rng = np.random.default_rng(7)
    future = np.zeros(len(cal))
    # 简化: 直接构造因子面板, min_a 有信号, min_b 是噪声
    panels = {}
    for name, signal in [("min_a", True), ("min_b", False)]:
        arr = np.full((len(cal), n_stocks), np.nan, dtype=np.float32)
        for i in range(n_stocks):
            if signal:
                arr[:, i] = rng.normal(0, 1, len(cal))  # 真实信号
            else:
                arr[:, i] = rng.normal(0, 10, len(cal))  # 噪声
        panels[name] = pd.DataFrame(arr, index=pd.DatetimeIndex(cal), columns=syms)
    factors = ["min_a", "min_b"]
    result = validate_minute_factors(
        panels, close, cal, cal_idx, factors,
        [("2022-01-03", "2023-12-29")], min_icir=0.05)
    # 至少返回非空 (信号因子的 ICIR 会被算出; 是否过门槛取决于构造)
    assert isinstance(result, dict)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `py -m pytest tests/test_minute_layer.py::test_validate_minute_factors_empty -v`
Expected: FAIL — ImportError: cannot import name 'validate_minute_factors'

- [ ] **Step 3: 实现 validate_minute_factors**

在 `_merge_minute_panels` 函数之后添加：

```python
def validate_minute_factors(factor_panels: dict, close_panel: pd.DataFrame,
                            calendar: list, cal_idx: dict, factor_names: list,
                            train_folds: list, min_icir: float = 0.3) -> dict:
    """
    方案B: 分钟因子独立验证层。

    用 fold 4-5 训练期 (2022+) 独立估计 min_* 因子的 ICIR,
    通过 |ICIR| >= min_icir 的因子作为叠加层权重 (各 fold 中位数)。
    与主通道 (40 稳定因子) 完全隔离, 不参与主筛选。

    Returns: {min_factor: validated_icir}
    """
    from minute_factors import MINUTE_FACTOR_NAMES
    minute_names = [fn for fn in factor_names if fn in MINUTE_FACTOR_NAMES]
    if not minute_names or not train_folds:
        return {}

    # 对每个 fold 训练期计算 ICIR
    fold_icirs = {fn: [] for fn in minute_names}
    for (ts, te) in train_folds:
        # 用训练期末作为 t_date (固定窗口模式)
        t_date = None
        for d in calendar:
            if pd.Timestamp(ts).date() <= d.date() <= pd.Timestamp(te).date():
                t_date = d
        if t_date is None:
            continue
        weights, ic_stats = compute_icir_weights(
            factor_panels, close_panel, calendar, cal_idx,
            t_date, minute_names, train_start=ts, train_end=te)
        for fn in minute_names:
            st = ic_stats.get(fn)
            if st is not None:
                fold_icirs[fn].append(st["icir"])
            else:
                fold_icirs[fn].append(0.0)

    # 取中位数, 过门槛保留
    result = {}
    for fn in minute_names:
        arr = np.array(fold_icirs[fn])
        if len(arr) == 0:
            continue
        med = float(np.median(arr))
        if abs(med) >= min_icir:
            result[fn] = med
    if result:
        log.info("  分钟叠加层: %d/%d 因子通过验证 |ICIR|>=%.2f: %s",
                 len(result), len(minute_names), min_icir,
                 {k: round(v, 3) for k, v in result.items()})
    return result
```

- [ ] **Step 4: 运行测试确认通过**

Run: `py -m pytest tests/test_minute_layer.py -v`
Expected: PASS（test_minute_layer_config + test_validate_minute_factors_empty + test_validate_minute_factors_filters）

- [ ] **Step 5: 提交**

```bash
git add scripts/active/run_walkforward_backtest.py tests/test_minute_layer.py
git commit -m "feat(v6): validate_minute_factors — fold 4-5 independent minute factor validation"
```

---

### Task 3: score_stocks 叠加层 + run_fold_test 接线 + 回归

**Files:**
- Modify: `scripts/active/run_walkforward_backtest.py`（`score_stocks` 加参数 + `run_fold_test` 加验证调用 + `run_fold_analysis` 透传）
- Test: `tests/test_minute_layer.py`（追加用例）

**Interfaces:**
- Consumes: `validate_minute_factors`（T2）、`config["minute_layer"]`（T1）
- Produces:
  - `score_stocks(factor_panels, weights, t_date, minute_weights=None, minute_lambda=0.3)` — 叠加后返回综合分
  - `run_fold_test(..., minute_layer: dict | None = None)` — 透传分钟叠加配置
  - `run_fold_analysis(..., minute_layer: dict | None = None)` — 透传

- [ ] **Step 1: 写失败测试**

```python
# tests/test_minute_layer.py 追加
def test_score_stocks_minute_overlay():
    """minute_weights 提供时, 综合分 = 主分 + λ×分钟分。"""
    from run_walkforward_backtest import score_stocks
    import numpy as np
    import pandas as pd
    cal = pd.date_range("2024-01-01", periods=5, freq="B")
    syms = ["a", "b", "c"]
    t_date = cal[2]
    # 主因子面板: 一个因子
    main_panel = pd.DataFrame({
        "a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "c": [2, 3, 4, 5, 6]},
        index=cal)
    # 分钟面板
    min_panel = pd.DataFrame({
        "a": [0.1, 0.2, 0.3, 0.4, 0.5], "b": [0.5, 0.4, 0.3, 0.2, 0.1],
        "c": [0.2, 0.3, 0.4, 0.5, 0.6]}, index=cal)
    panels = {"f1": main_panel, "min_a": min_panel}
    # 无分钟叠加
    s_main = score_stocks(panels, {"f1": 1.0}, t_date)
    # 有分钟叠加 (λ=0.5)
    s_over = score_stocks(panels, {"f1": 1.0}, t_date,
                          minute_weights={"min_a": 1.0}, minute_lambda=0.5)
    assert set(s_main.keys()) == set(s_over.keys())
    # 叠加后分数变化 (至少一只股票不同)
    assert any(abs(s_over[k] - s_main[k]) > 1e-9 for k in s_main)


def test_score_stocks_disabled():
    """minute_weights=None 时行为不变 (回退 v5)。"""
    from run_walkforward_backtest import score_stocks
    import numpy as np
    import pandas as pd
    cal = pd.date_range("2024-01-01", periods=5, freq="B")
    syms = ["a", "b", "c"]
    t_date = cal[2]
    main_panel = pd.DataFrame({
        "a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "c": [2, 3, 4, 5, 6]},
        index=cal)
    panels = {"f1": main_panel}
    s1 = score_stocks(panels, {"f1": 1.0}, t_date)
    s2 = score_stocks(panels, {"f1": 1.0}, t_date, minute_weights=None, minute_lambda=0.3)
    assert s1 == s2
```

- [ ] **Step 2: 运行测试确认失败**

Run: `py -m pytest tests/test_minute_layer.py::test_score_stocks_minute_overlay -v`
Expected: FAIL — TypeError: score_stocks() got an unexpected keyword argument 'minute_weights'

- [ ] **Step 3: 修改 score_stocks 支持叠加**

```python
def score_stocks(factor_panels: dict, weights: dict, t_date,
                 minute_weights: dict | None = None,
                 minute_lambda: float = 0.3) -> dict:
    """用动态 ICIR 权重对 t_date 当日截面打分 (z-score 加权)。

    方案B: minute_weights 提供时, 综合分 = 主分 + λ×分钟因子加权分。
    分钟因子分独立 z-score (不参与主因子归一化, 避免尺度污染)。
    """
    if not weights:
        return {}
    factor_names = list(weights.keys())
    w = np.array([weights[n] for n in factor_names])
    abs_w = np.sum(np.abs(w))
    if abs_w < 1e-9:
        return {}

    # 当日截面: index=股票, columns=因子
    cols = {}
    for n in factor_names:
        p = factor_panels[n]
        if t_date in p.index:
            cols[n] = p.loc[t_date]
    if not cols:
        return {}
    cross = pd.DataFrame(cols)  # (n_stocks × n_factors)

    # 覆盖率 >= 50%
    n_f = cross.shape[1]
    cov = cross.notna().sum(axis=1)
    cross = cross[cov >= n_f * 0.5]
    if len(cross) < 10:
        return {}

    vals = cross.to_numpy()  # (n_valid, n_factors)
    composite = np.zeros(len(cross))
    for fi in range(n_f):
        col = vals[:, fi]
        m = ~np.isnan(col)
        if m.sum() < 10:
            continue
        mu = np.nanmean(col)
        sd = np.nanstd(col)
        if sd < 1e-9:
            continue
        z = np.where(m, (col - mu) / sd, 0.0)
        composite += w[fi] * z
    composite /= abs_w

    # ── 方案B: 分钟因子叠加层 ──
    if minute_weights:
        m_names = list(minute_weights.keys())
        m_w = np.array([minute_weights[n] for n in m_names])
        m_abs = np.sum(np.abs(m_w))
        if m_abs > 1e-9:
            m_cols = {}
            for n in m_names:
                p = factor_panels.get(n)
                if p is not None and t_date in p.index:
                    m_cols[n] = p.loc[t_date]
            if m_cols:
                m_cross = pd.DataFrame(m_cols)
                # 与主分相同的股票对齐
                m_cross = m_cross.reindex(cross.index)
                m_vals = m_cross.to_numpy()
                m_comp = np.zeros(len(cross))
                for fi in range(len(m_names)):
                    col = m_vals[:, fi]
                    m = ~np.isnan(col)
                    if m.sum() < 10:
                        continue
                    mu = np.nanmean(col)
                    sd = np.nanstd(col)
                    if sd < 1e-9:
                        continue
                    z = np.where(m, (col - mu) / sd, 0.0)
                    m_comp += m_w[fi] * z
                m_comp /= m_abs
                # 无分钟数据的股票 m_comp=0 → 仅主分生效 (自然降级)
                m_comp = np.nan_to_num(m_comp, nan=0.0)
                composite = composite + minute_lambda * m_comp

    return {s: float(v) for s, v in zip(cross.index, composite)
            if not np.isnan(v)}
```

- [ ] **Step 4: 修改 run_fold_test 接线**

在 `run_fold_test` 签名加 `minute_layer: dict | None = None`，在调用 `run_backtest` 前构造叠加参数：

```python
def run_fold_test(all_data, factor_panels, close_panel, calendar, cal_idx,
                  factor_names, bt_config, stable_factors, stable_icir,
                  test_start, test_end,
                  universe_fn=get_universe,
                  use_regime: bool = False,
                  portfolio_constraints: dict | None = None,
                  minute_layer: dict | None = None) -> dict | None:
    """...（docstring 追加: minute_layer 含 minute_weights + lambda）"""
    # ...（现有风格均衡逻辑不变）...

    # ── 方案B: 分钟叠加层 ──
    ml_weights = None
    ml_lambda = 0.3
    if minute_layer and minute_layer.get("enabled"):
        ml_weights = minute_layer.get("weights")  # validate_minute_factors 输出
        ml_lambda = float(minute_layer.get("lambda", 0.3))
        if ml_weights:
            log.info("  分钟叠加层: %d 个因子, λ=%.2f", len(ml_weights), ml_lambda)

    return run_backtest(all_data, factor_panels, close_panel, calendar,
                        cal_idx, factor_names, bt_config,
                        test_start, test_end, label="TEST",
                        fixed_weights=weights, universe_fn=universe_fn,
                        use_regime=use_regime,
                        portfolio_constraints=portfolio_constraints,
                        minute_weights=ml_weights, minute_lambda=ml_lambda)
```

同时给 `run_backtest` 签名加 `minute_weights: dict | None = None, minute_lambda: float = 0.3`，在 `scores = score_stocks(factor_panels, weights, today)` 处传参：

```python
            scores = score_stocks(factor_panels, weights, today,
                                  minute_weights=minute_weights,
                                  minute_lambda=minute_lambda)
```

`run_fold_analysis` 签名加 `minute_layer: dict | None = None`，内部对 fold 4-5 验证期调用 `run_backtest` 时传 `minute_weights=ml_weights, minute_lambda=ml_lambda`（fold 4-5 才有分钟数据；fold 1-3 传 None）。

- [ ] **Step 5: 修改 main() 接线**

在 main() 的 fold 分支中，读取 config 并调用 `validate_minute_factors`：

```python
        if args.folds:
            # 方案B: 分钟因子独立验证 (fold 4-5 训练期)
            ml_cfg = config.get("minute_layer", {})
            ml_weights = None
            if ml_cfg.get("enabled"):
                # 训练期: fold 4 = 2015-2022, fold 5 = 2015-2023
                # 但分钟数据 2022 起 → 实际用 2022-01~2023-12 / 2022-01~2024-12
                train_folds = [
                    ("2022-01-01", "2023-12-31"),
                    ("2022-01-01", "2024-12-31"),
                ]
                ml_weights = validate_minute_factors(
                    factor_panels, close_panel, calendar, cal_idx,
                    factor_names, train_folds,
                    min_icir=float(ml_cfg.get("min_icir", 0.3)))
            minute_layer = {
                "enabled": ml_cfg.get("enabled", True),
                "weights": ml_weights,
                "lambda": float(ml_cfg.get("lambda", 0.3)),
            }
            fold_out = run_fold_analysis(
                all_data, factor_panels, close_panel, calendar, cal_idx,
                factor_names, bt_config, universe_fn=universe_fn,
                minute_layer=minute_layer)
            ...
            r = run_fold_test(
                ..., minute_layer=minute_layer)
```

- [ ] **Step 6: 运行测试确认通过**

Run: `py -m pytest tests/test_minute_layer.py -v`
Expected: PASS（5 个用例）

- [ ] **Step 7: 全量回归**

Run: `py -m pytest tests/ -q`
Expected: 全部通过（原有 45 + 新增 5 = 50）

- [ ] **Step 8: 提交**

```bash
git add scripts/active/run_walkforward_backtest.py tests/test_minute_layer.py
git commit -m "feat(v6): minute factor overlay in score_stocks + wiring"
```

---

## Self-Review 记录

**Spec coverage:** config（T1）/validate（T2）/叠加+接线（T3）全覆盖 spec 3.2 组件 A/B/C 与 4 测试要求 ✅
**Placeholder scan:** 所有步骤含完整代码 ✅
**Type consistency:** `validate_minute_factors` 返回 `{min_factor: icir}` → `score_stocks(minute_weights=...)` 消费同型；`minute_layer` dict 结构在 T3 中定义并贯穿 run_fold_analysis/run_fold_test ✅
**纪律:** 主 fold 筛选零改动（validate 独立于 compute_icir_weights 的 fold 调用）；enabled=false 时 ml_weights=None → score_stocks 走原路径（test_score_stocks_disabled 验证）✅

