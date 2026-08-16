# 多风格 Sleeve 架构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 v27（50 稳定因子单通道）基础上加入"核心 + 动量 + 成长"三风格 sleeve 的预算混合，验证是否能获得跨风格覆盖（2024-2026 AI 成长行情）而不损失既有成绩，并产出最新优化版训练结果。

**Architecture:** 评分层混合——每个 sleeve 内部仍走"fold ICIR 估计 → 截面 z-score 加权（内部归一）"，组合分 = (1-Σ预算)×核心分 + Σ 预算×sleeve 分。sleeve 预算为 config 固定参数（不择时）。引擎/调仓/约束层完全不动。

**Tech Stack:** Python 3.12 / pandas / numpy / pytest；回测脚本 `scripts/active/run_walkforward_backtest.py`；分支 `feature/multi-style-sleeve`。

## Global Constraints

- 只修改现有 `scripts/active/` 脚本，不得新建脚本；新模块必须被生产链路引用或有 tests 覆盖
- `config.yaml` 是唯一参数源；改前备份、实验后复位
- 回测全程离线（`netgate` 守卫已在位）；只用 `--folds-only` + `--extend-val`（不消耗 TEST 锁）
- `styles.enabled: false` 时行为必须与 v27 **逐位一致**（A/B 可复现）
- TDD：每个代码任务先写失败测试再实现；每任务结束跑 `py -m pytest tests/ -q` 全绿
- Windows Git Bash；用 `py` 启动 Python
- 工作分支 `feature/multi-style-sleeve`；每任务一次 commit

---

### Task 0: 前置校验（数据 schema + 分支基线）

**Files:** 无代码改动（只读校验）

- [ ] **Step 1: 校验 fundamental_cache schema 一致性**

Run:
```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter && py -c "
import os, pandas as pd
d = 'data/fundamental_cache'
files = [f for f in os.listdir(d) if f.endswith('.parquet')] if os.path.isdir(d) else []
print('files:', len(files))
if files:
    cols = set()
    for f in files[:200]:
        cols |= set(pd.read_parquet(os.path.join(d, f)).columns)
    print('union cols:', sorted(cols))
"
```
Expected: 输出列集合中同时存在英文 `report_date`（v2 脚本）与中文 `日期`（生产）→ 记录结论；若只有中文列 → schema 无冲突，Task 直接继续。若混合存在，成长 sleeve 实验用 `fundamental.py` 的 `fetch_financials` 路径（生产读取器），不修数据（记录为已知限制）。

- [ ] **Step 2: 分支基线测试**

Run: `py -m pytest tests/ -q && py -m pytest web/api/tests/ -q`
Expected: `124 passed` + `30 passed`（已在分支上验证，重复确认一次）

- [ ] **Step 3: Commit（若有变更则提交，否则跳过）**

```bash
git add -A && git commit -m "chore: multi-style sleeve 前置校验" || echo "nothing to commit"
```

---

### Task 1: config `styles:` 段 + 解析辅助函数

**Files:**
- Modify: `config.yaml`（在 `portfolio_optimizer` 段之后追加 styles 段）
- Modify: `scripts/active/run_walkforward_backtest.py`（加 3 个模块级函数，放在 `_style_budget_weights` 之后）
- Test: `tests/test_sleeve_blend.py`（新建）

**Interfaces:**
- Produces:
  - `load_styles_config(config: dict) -> dict | None`（enabled=false/缺失 → None）
  - `split_sleeve_factors(factor_names: list, styles_cfg: dict) -> tuple[list, dict]`
  - `parse_budget_combos(s: str) -> list`（Task 5 备用；本轮实验用 config 编辑跑网格，此函数仍实现并测试）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_sleeve_blend.py
"""多风格 sleeve 配置解析测试。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "active"))


def _base_styles():
    return {
        "styles": {
            "enabled": True,
            "budgets": {"momentum": 0.25, "growth": 0.15},
            "sleeves": {
                "momentum": {"min_hits": 3,
                             "factors": ["mom_60d", "return_30d"]},
                "growth": {"min_hits": 0, "fallback_weight": 0.1,
                           "factors": ["fund_profit_growth"]},
            },
        }
    }


def test_load_styles_config_disabled_returns_none():
    from run_walkforward_backtest import load_styles_config
    assert load_styles_config({}) is None
    assert load_styles_config({"styles": {"enabled": False}}) is None


def test_load_styles_config_valid():
    from run_walkforward_backtest import load_styles_config
    assert load_styles_config(_base_styles()) is not None


def test_load_styles_config_budget_over_1_raises():
    from run_walkforward_backtest import load_styles_config
    import pytest
    cfg = _base_styles()
    cfg["styles"]["budgets"] = {"momentum": 0.8, "growth": 0.5}
    with pytest.raises(ValueError):
        load_styles_config(cfg)


def test_split_sleeve_factors():
    from run_walkforward_backtest import split_sleeve_factors
    core, sleeves = split_sleeve_factors(
        ["mom_60d", "return_30d", "fund_bp", "amihud_5d", "fund_profit_growth"],
        _base_styles())
    assert core == ["fund_bp", "amihud_5d"]
    assert set(sleeves["momentum"]) == {"mom_60d", "return_30d"}
    assert sleeves["growth"] == ["fund_profit_growth"]


def test_split_sleeve_factors_unknown_skipped():
    from run_walkforward_backtest import split_sleeve_factors
    cfg = _base_styles()
    cfg["styles"]["sleeves"]["momentum"]["factors"] = ["mom_60d", "not_a_factor"]
    core, sleeves = split_sleeve_factors(["mom_60d", "fund_bp"], cfg)
    assert sleeves["momentum"] == ["mom_60d"]
    assert core == ["fund_bp"]


def test_parse_budget_combos_valid():
    from run_walkforward_backtest import parse_budget_combos
    combos = parse_budget_combos("0.25/0.15,0.2/0.2")
    assert combos == [[("momentum", 0.25), ("growth", 0.15)],
                      [("momentum", 0.2), ("growth", 0.2)]]


def test_parse_budget_combos_errors():
    from run_walkforward_backtest import parse_budget_combos
    import pytest
    with pytest.raises(ValueError):
        parse_budget_combos("0.25")            # 元素数不足
    with pytest.raises(ValueError):
        parse_budget_combos("0.8/0.5")         # 合计 > 1
    with pytest.raises(ValueError):
        parse_budget_combos("")                # 空
```

- [ ] **Step 2: 运行确认失败**

Run: `py -m pytest tests/test_sleeve_blend.py -q`
Expected: FAIL（ImportError: cannot import name 'load_styles_config'）

- [ ] **Step 3: config.yaml 加 styles 段**（在 `portfolio_optimizer: "equal"` 行后插入）

```yaml
# ── 多风格 sleeve (2026-08-16 实验; enabled=false 时与 v27 完全一致) ──
styles:
  enabled: false
  budgets:                 # 组合分 = (1-Σbudget)×核心 + Σ budget×sleeve分
    momentum: 0.25
    growth: 0.15
  sleeves:
    momentum:              # 5/5 fold 命中但被 top50 截断的动量因子
      min_hits: 3
      factors:
        - return_30d
        - return_60d
        - return_90d
        - vol_adj_mom_60d
        - momentum_7d
        - momentum_20d
        - mom_60d
        - mom_120d
        - mom_12_1
    growth:                # 成长 sleeve: 风格预算实验, 不强求 fold 命中
      min_hits: 0
      fallback_weight: 0.1  # fold 中位数 |ICIR|<0.02 时的默认权重 (符号取正, p6 ICIR 均为正)
      factors:
        - fund_profit_growth
        - fund_profit_growth_ded
        - fund_revenue_growth
        - aux_yjkb_profit_growth
        - fund_ocf_ps
        - fund_ocf_yield
        - fund_net_margin
```

- [ ] **Step 4: 实现解析函数**（`run_walkforward_backtest.py`，插入到 `_style_budget_weights` 函数之后）

```python
# ── 多风格 sleeve 配置 (2026-08-16, config styles 段) ──
SLEEVE_BUDGET_ORDER = ("momentum", "growth")


def load_styles_config(config: dict) -> dict | None:
    """styles 段: enabled=false/缺失 → None (行为与 v27 完全一致)。"""
    cfg = config.get("styles") or {}
    if not cfg.get("enabled"):
        return None
    budgets = cfg.get("budgets") or {}
    for name in SLEEVE_BUDGET_ORDER:
        b = float(budgets.get(name, 0.0))
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"styles.budgets.{name} 必须在 [0,1], got {b}")
    if sum(float(budgets.get(n, 0.0)) for n in SLEEVE_BUDGET_ORDER) > 1.0:
        raise ValueError("styles.budgets 合计不得超过 1.0 (core 隐含 1-Σ)")
    for name, scfg in (cfg.get("sleeves") or {}).items():
        if not scfg.get("factors"):
            raise ValueError(f"styles.sleeves.{name}.factors 为空")
    return cfg


def split_sleeve_factors(factor_names: list, styles_cfg: dict):
    """核心池 = 全部因子 - sleeve 因子; 返回 (core_names, {name: [factors]})。"""
    sleeves = {}
    sleeve_set = set()
    for name, scfg in (styles_cfg.get("sleeves") or {}).items():
        fs = [f for f in scfg.get("factors", []) if f in factor_names]
        sleeves[name] = fs
        sleeve_set.update(fs)
    core = [f for f in factor_names if f not in sleeve_set]
    return core, sleeves


def parse_budget_combos(s: str) -> list:
    """'0.25/0.15,0.2/0.2' → [[('momentum',0.25),('growth',0.15)], ...]"""
    combos = []
    for part in s.split(","):
        vals = [p.strip() for p in part.split("/")]
        if len(vals) != len(SLEEVE_BUDGET_ORDER):
            raise ValueError(
                f"预算组合需 {len(SLEEVE_BUDGET_ORDER)} 个值 (mom/growth): '{part}'")
        combo = [(SLEEVE_BUDGET_ORDER[i], float(vals[i]))
                 for i in range(len(SLEEVE_BUDGET_ORDER))]
        if any(not 0.0 <= v <= 1.0 for _, v in combo):
            raise ValueError(f"预算值须在 [0,1]: '{part}'")
        if sum(v for _, v in combo) > 1.0:
            raise ValueError(f"预算合计不得超过 1.0: '{part}'")
        combos.append(combo)
    if not combos:
        raise ValueError("预算组合列表不能为空")
    return combos
```

- [ ] **Step 5: 运行确认通过**

Run: `py -m pytest tests/test_sleeve_blend.py -q`
Expected: 7 passed

- [ ] **Step 6: 全量回归**

Run: `py -m pytest tests/ -q`
Expected: 131 passed

- [ ] **Step 7: Commit**

```bash
git add config.yaml scripts/active/run_walkforward_backtest.py tests/test_sleeve_blend.py
git commit -m "feat(sleeve): styles 配置段 + 解析辅助函数 (enabled=false 行为不变)"
```

---

### Task 2: score_stocks 多通道重构

**Files:**
- Modify: `scripts/active/run_walkforward_backtest.py`（`score_stocks` 1213-1294 区段）
- Test: `tests/test_sleeve_blend.py`（追加）

**Interfaces:**
- Consumes: 无（独立于 Task 1）
- Produces:
  - `_weighted_z_composite(cross: pd.DataFrame, weights: dict) -> np.ndarray`（内部 z-score 加权和，按 Σ|w| 归一）
  - `score_stocks(factor_panels, weights, t_date, sleeve_weights: list | None = None, minute_weights=None, minute_lambda=0.3) -> dict`
  - `sleeve_weights` 元素格式: `{"name": str, "weights": dict, "budget": float}`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_sleeve_blend.py 追加
def _panels():
    import pandas as pd
    import numpy as np
    cal = pd.date_range("2024-01-01", periods=3, freq="B")
    syms = [f"s{i}" for i in range(10)]
    rng = np.random.default_rng(7)
    rets = rng.normal(0, 0.02, (3, 10))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)),
                         index=cal, columns=syms)
    panels = {}
    for name in ["f1", "f2", "m1", "g1"]:
        vals = close / close.shift(1).fillna(1.0) - 1.0
        panels[name] = pd.DataFrame(
            vals.to_numpy(), index=cal, columns=syms, dtype=np.float32)
    return panels, cal


def test_score_stocks_no_sleeve_unchanged():
    """无 sleeve 时与单通道旧实现数值一致。"""
    from run_walkforward_backtest import score_stocks
    panels, cal = _panels()
    t = cal[1]
    s1 = score_stocks(panels, {"f1": 1.0, "f2": -0.5}, t)
    s2 = score_stocks(panels, {"f1": 1.0, "f2": -0.5}, t, sleeve_weights=None)
    assert s1 == s2
    # 手工复算: composite = (1*z1 - 0.5*z2)/1.5
    import numpy as np
    cross = panels["f1"].loc[t]
    z1 = (cross - cross.mean()) / cross.std()
    cross2 = panels["f2"].loc[t]
    z2 = (cross2 - cross2.mean()) / cross2.std()
    expect = (1.0 * z1 - 0.5 * z2) / 1.5
    for k in s1:
        assert abs(s1[k] - expect[k]) < 1e-9


def test_score_stocks_sleeve_blend_math():
    """composite = (1-Σbudget)×core + Σ budget×sleeve (各通道内部归一)。"""
    from run_walkforward_backtest import score_stocks
    import numpy as np
    panels, cal = _panels()
    t = cal[1]
    sw = [
        {"name": "momentum", "weights": {"m1": 1.0}, "budget": 0.25},
        {"name": "growth", "weights": {"g1": 1.0}, "budget": 0.15},
    ]
    s = score_stocks(panels, {"f1": 1.0}, t, sleeve_weights=sw)
    c = panels["f1"].loc[t]
    z1 = (c - c.mean()) / c.std()
    m = panels["m1"].loc[t]
    zm = (m - m.mean()) / m.std()
    g = panels["g1"].loc[t]
    zg = (g - g.mean()) / g.std()
    expect = 0.6 * z1 + 0.25 * zm + 0.15 * zg
    for k in s:
        assert abs(s[k] - expect[k]) < 1e-9, f"{k}: {s[k]} vs {expect[k]}"
```

- [ ] **Step 2: 运行确认失败**

Run: `py -m pytest tests/test_sleeve_blend.py -q -k "sleeve_blend"`
Expected: FAIL（sleeve_weights 参数不存在 → TypeError）

- [ ] **Step 3: 重构实现**（整体替换现有 `score_stocks` 函数体，保留分钟叠加层语义）

```python
def _weighted_z_composite(cross: pd.DataFrame, weights: dict) -> np.ndarray:
    """cross (n × 因子) → 因子内 z-score 后按权重加权和, 除以 Σ|w| (长度 n)。"""
    names = [n for n in weights if n in cross.columns]
    vals = cross[names].to_numpy(dtype=np.float64)
    n, m = vals.shape
    w = np.array([weights[n] for n in names], dtype=np.float64)
    comp = np.zeros(n)
    for fi in range(m):
        col = vals[:, fi]
        mask = ~np.isnan(col)
        if mask.sum() < 10:
            continue
        mu = np.nanmean(col)
        sd = np.nanstd(col)
        if sd < 1e-9:
            continue
        z = np.where(mask, (col - mu) / sd, 0.0)
        comp += w[fi] * z
    denom = np.sum(np.abs(w))
    if denom < 1e-9:
        return comp
    return comp / denom


def score_stocks(factor_panels: dict, weights: dict, t_date,
                 sleeve_weights: list | None = None,
                 minute_weights: dict | None = None,
                 minute_lambda: float = 0.3) -> dict:
    """ICIR 加权 z-score 打分, 支持多 sleeve 预算混合 (2026-08-16)。

    sleeve_weights: [{"name","weights","budget"}, ...]
      composite = (1-Σbudget)×主分 + Σ budget×sleeve分 (各通道内部分别归一)。
    sleeve_weights=None 时行为与 v27 逐位一致 (回归测试钉住)。
    """
    if not weights:
        return {}
    factor_names = list(weights.keys())

    cols = {}
    for n in factor_names:
        p = factor_panels[n]
        if t_date in p.index:
            cols[n] = p.loc[t_date]
    if not cols:
        return {}
    cross = pd.DataFrame(cols)

    n_f = cross.shape[1]
    cov = cross.notna().sum(axis=1)
    cross = cross[cov >= n_f * 0.5]
    if len(cross) < 10:
        return {}

    base = _weighted_z_composite(cross, weights)
    core_budget = 1.0
    if sleeve_weights:
        for sw in sleeve_weights:
            s_w = sw.get("weights") or {}
            s_budget = float(sw.get("budget", 0.0))
            if not s_w or s_budget <= 0:
                continue
            s_cols = {}
            for n in s_w:
                p = factor_panels.get(n)
                if p is not None and t_date in p.index:
                    s_cols[n] = p.loc[t_date]
            if not s_cols:
                continue
            s_cross = pd.DataFrame(s_cols).reindex(cross.index)
            s_comp = _weighted_z_composite(s_cross, s_w)
            # 无该 sleeve 数据的股票 s_comp=0 → 仅主分生效 (自然降级)
            s_comp = np.nan_to_num(s_comp, nan=0.0)
            base = base + s_budget * s_comp
            core_budget -= s_budget
    composite = core_budget * base

    # ── 方案B: 分钟因子叠加层 (语义不变) ──
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
                m_cross = pd.DataFrame(m_cols).reindex(cross.index)
                m_comp = _weighted_z_composite(m_cross, minute_weights)
                m_comp = np.nan_to_num(m_comp, nan=0.0)
                composite = composite + minute_lambda * m_comp

    return {s: float(v) for s, v in zip(cross.index, composite)
            if not np.isnan(v)}
```

- [ ] **Step 4: 运行确认通过**

Run: `py -m pytest tests/test_sleeve_blend.py -q`
Expected: 全部通过（含 Task 1 的 7 个）

- [ ] **Step 5: 全量回归**（含既有 test_minute_layer 的叠加层测试）

Run: `py -m pytest tests/ -q`
Expected: 133 passed

- [ ] **Step 6: Commit**

```bash
git add scripts/active/run_walkforward_backtest.py tests/test_sleeve_blend.py
git commit -m "feat(sleeve): score_stocks 多通道预算混合 (无 sleeve 时逐位一致)"
```

---

### Task 3: run_backtest 传参 + run_fold_analysis 分 sleeve 权重

**Files:**
- Modify: `scripts/active/run_walkforward_backtest.py`
- Test: `tests/test_sleeve_blend.py`（追加）

**Interfaces:**
- Consumes: `score_stocks(..., sleeve_weights=...)`（Task 2）、`load_styles_config`/`split_sleeve_factors`（Task 1）
- Produces:
  - `run_backtest(..., sleeve_weights: list | None = None)`（新参数透传给 score_stocks）
  - `run_fold_analysis(..., styles_cfg: dict | None = None)`（分池选因子）
  - `sleeve_median_weights(sleeve_icirs: dict, sleeves_cfg: dict) -> dict`（模块级纯函数）
  - `run_fold_analysis` 返回 dict 新增键 `"sleeve_median_weights"`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_sleeve_blend.py 追加
def test_sleeve_median_weights_fallback():
    """|median|<0.02 用 fallback_weight 正号; 否则用 median ICIR。"""
    from run_walkforward_backtest import sleeve_median_weights
    icirs = {
        "momentum": {"m1": [0.05, 0.06, 0.04, 0.05, 0.05],
                     "m2": [0.01, -0.01, 0.0, 0.0, 0.0]},
        "growth": {"g1": [0.0, 0.0, 0.0, 0.0, 0.0]},
    }
    cfg = {"momentum": {"fallback_weight": 0.1},
           "growth": {"fallback_weight": 0.1}}
    out = sleeve_median_weights(icirs, cfg)
    assert abs(out["momentum"]["m1"] - 0.05) < 1e-9
    assert out["momentum"]["m2"] == 0.1   # median≈0 → fallback
    assert out["growth"]["g1"] == 0.1     # 全 0 → fallback
```

- [ ] **Step 2: 运行确认失败**

Run: `py -m pytest tests/test_sleeve_blend.py -q -k median`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现 sleeve_median_weights**（插入到 `run_fold_analysis` 之前）

```python
def sleeve_median_weights(sleeve_icirs: dict, sleeves_cfg: dict) -> dict:
    """每 sleeve 因子 5 折 ICIR 中位数 → 权重; |median|<0.02 用 fallback_weight。

    fallback 符号取正 (成长因子 p6 全样本 ICIR 均为正; 2026-08-16)。
    """
    out = {}
    for name, icirs in sleeve_icirs.items():
        scfg = (sleeves_cfg or {}).get(name, {}) or {}
        fb = float(scfg.get("fallback_weight", 0.1))
        w = {}
        for fn, arr in icirs.items():
            med = float(np.median(arr)) if arr else 0.0
            w[fn] = med if abs(med) >= 0.02 else fb
        out[name] = w
    return out
```

- [ ] **Step 4: run_backtest 加参数并透传**

签名处（现 `def run_backtest(all_data, factor_panels, close_panel, calendar, cal_idx, factor_names, bt_config, start, end, label="", fixed_weights..., minute_lambda..., weight_mode...)`）追加 `sleeve_weights: list | None = None,`（放在 `minute_lambda` 之前），并在评分调用处：

```python
            # 评分
            scores = score_stocks(factor_panels, weights, today,
                                  sleeve_weights=sleeve_weights,
                                  minute_weights=minute_weights,
                                  minute_lambda=minute_lambda)
```

同时在 run_backtest 的 docstring 追加：
```python
    sleeve_weights: 多风格 sleeve 列表 [{"name","weights","budget"}, ...],
      与 fixed_weights 配套 (fold 模式), None=单通道 (v27 行为)。
```

- [ ] **Step 5: run_fold_analysis 分池改造**

签名追加 `styles_cfg: dict | None = None,`。函数体改动三处：

① 函数开头（`fold_results = {}` 之前）：
```python
    core_names = factor_names
    sleeves = {}
    if styles_cfg:
        core_names, sleeves = split_sleeve_factors(factor_names, styles_cfg)
        log.info("  sleeve 模式: 核心 %d 因子 + %s",
                 len(core_names),
                 ", ".join(f"{n}({len(fs)} 因子)" for n, fs in sleeves.items()))
```

② 因子命中统计与 fold 权重估计（现有 `factor_hits`/`factor_icirs` 初始化与 `compute_icir_weights` 调用、`weights` 显著性过滤、稳定因子筛选 `cand` 循环）中所有 `factor_names` 改为 `core_names`；新增 sleeve 侧统计（在 `weights, ic_stats = compute_icir_weights(...)` 之后插入）：
```python
        # ── sleeve 权重估计 (独立通道, 2026-08-16) ──
        fold_sleeve_weights = {}
        if sleeves:
            for sname, sfs in sleeves.items():
                s_weights, s_stats = compute_icir_weights(
                    factor_panels, close_panel, calendar, cal_idx,
                    val_first, sfs, train_start=ts, train_end=te,
                    universe_fn=universe_fn)
                min_hits = int(styles_cfg["sleeves"][sname].get("min_hits", 3))
                if min_hits > 0:
                    s_weights = {fn: w for fn, w in s_weights.items()
                                 if fn in _sleeve_sig(s_stats)}
                else:
                    # min_hits=0 (成长): 保留 |icir|>=MIN_ICIR 的因子, 符号随 ICIR
                    s_weights = {fn: w for fn, w in s_weights.items()
                                 if abs(w) >= MIN_ICIR}
                fold_sleeve_weights[sname] = s_weights
                for fn in sfs:
                    st = s_stats.get(fn)
                    sleeve_icirs.setdefault(sname, {}).setdefault(fn, []).append(
                        float(st["icir"]) if st is not None else 0.0)
                log.info("  sleeve[%s] 本折权重: %d 因子", sname, len(s_weights))
```
其中 `sleeve_icirs = {}` 在 fold 循环前初始化；辅助函数：
```python
def _sleeve_sig(ic_stats: dict) -> set:
    """与核心同口径的显著因子: |ICIR|>=FOLD_ICIR_MIN 且 t 统计 >= FOLD_T_STAT_MIN。"""
    sig = set()
    for fn, st in ic_stats.items():
        n_obs = st.get("n_obs", 0)
        t_stat = abs(st["icir"]) * np.sqrt(n_obs) if n_obs > 0 else 0.0
        if t_stat >= FOLD_T_STAT_MIN and abs(st["icir"]) >= FOLD_ICIR_MIN:
            sig.add(fn)
    return sig
```

③ 验证期 run_backtest 调用追加 sleeve 参数：
```python
        r = run_backtest(all_data, factor_panels, close_panel, calendar,
                         cal_idx, factor_names, bt_config, vs, ve,
                         label=f"VAL{fi+1}", fixed_weights=weights,
                         sleeve_weights=([{
                             "name": sname,
                             "weights": fold_sleeve_weights.get(sname) or {},
                             "budget": float(styles_cfg["budgets"][sname]),
                         } for sname in sleeves] if sleeves else None),
                         universe_fn=universe_fn, use_regime=use_regime,
                         portfolio_constraints=portfolio_constraints,
                         minute_weights=ml_weights if fi >= 3 else None,
                         minute_lambda=ml_lambda,
                         weight_mode=weight_mode,
                         pool_filter_cfg=pool_filter_cfg,
                         vol_target_cfg=vol_target_cfg,
                         trend_timing_cfg=trend_timing_cfg)
```
（`factor_names` 传给 run_backtest 保持全量——面板引用用；核心权重已由 fixed_weights 传入。）

④ 返回值新增键：
```python
    return {
        "folds": fold_results,
        "factor_hits": factor_hits,
        "stable_factors": stable,
        "stable_factor_icir_median": {
            k: round(v, 4) for k, v in stable_icir.items()},
        "sleeve_median_weights": (
            sleeve_median_weights(sleeve_icirs, styles_cfg.get("sleeves", {}))
            if styles_cfg else {}),
    }
```

- [ ] **Step 6: 运行确认通过**

Run: `py -m pytest tests/test_sleeve_blend.py -q`
Expected: 全部通过

- [ ] **Step 7: 全量回归**

Run: `py -m pytest tests/ -q`
Expected: 134 passed

- [ ] **Step 8: Commit**

```bash
git add scripts/active/run_walkforward_backtest.py tests/test_sleeve_blend.py
git commit -m "feat(sleeve): fold 分析分池选因子 + run_backtest sleeve 透传"
```

---

### Task 4: main() extend 接线 + 实验登记

**Files:**
- Modify: `scripts/active/run_walkforward_backtest.py`
- Test: `tests/test_sleeve_blend.py`（追加，用 monkeypatch 冒烟）

**Interfaces:**
- Consumes: `run_fold_analysis` 返回值新键 `sleeve_median_weights`（Task 3）
- Produces: extend 模拟考在多 sleeve 配置下运行；每次运行自动写 `experiments/exp_*.json`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_sleeve_blend.py 追加
def test_build_extend_sleeve_weights():
    """fold_out + styles_cfg → extend 用 sleeve 列表 (空预算跳过)。"""
    from run_walkforward_backtest import build_extend_sleeve_weights
    fold_out = {"sleeve_median_weights": {
        "momentum": {"m1": 0.05, "m2": 0.1},
        "growth": {"g1": 0.1}}}
    cfg = {"budgets": {"momentum": 0.25, "growth": 0.0},
           "sleeves": {"momentum": {}, "growth": {}}}
    out = build_extend_sleeve_weights(fold_out, cfg)
    assert len(out) == 1
    assert out[0]["name"] == "momentum" and out[0]["budget"] == 0.25
    assert out[0]["weights"] == {"m1": 0.05, "m2": 0.1}


def test_build_extend_sleeve_weights_no_styles():
    from run_walkforward_backtest import build_extend_sleeve_weights
    assert build_extend_sleeve_weights({}, None) is None
```

- [ ] **Step 2: 运行确认失败**

Run: `py -m pytest tests/test_sleeve_blend.py -q -k extend`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现 build_extend_sleeve_weights**（放在 `sleeve_median_weights` 之后）

```python
def build_extend_sleeve_weights(fold_out: dict, styles_cfg: dict | None) -> list | None:
    """extend 模拟考用 sleeve 权重列表 (中位数 ICIR + config 预算)。"""
    if not styles_cfg:
        return None
    med = fold_out.get("sleeve_median_weights") or {}
    budgets = styles_cfg.get("budgets") or {}
    out = []
    for name, w in med.items():
        budget = float(budgets.get(name, 0.0))
        if w and budget > 0:
            out.append({"name": name, "weights": w, "budget": budget})
    return out or None
```

- [ ] **Step 4: main() 接线**

① 在 main() 的 fold 执行前（`fold_out = run_fold_analysis(...)` 调用处上方）解析 styles：
```python
    styles_cfg = load_styles_config(config)
    if styles_cfg:
        log.info("  多风格 sleeve: %s (budgets=%s)",
                 ", ".join(styles_cfg["sleeves"].keys()),
                 styles_cfg.get("budgets"))
```
② `run_fold_analysis(...)` 调用追加 `styles_cfg=styles_cfg,`。
③ extend 的 `run_backtest(...)` 调用追加：
```python
                    sleeve_weights=build_extend_sleeve_weights(
                        fold_out, styles_cfg),
```
（放在 `minute_weights=...` 参数之前。）

- [ ] **Step 5: 实验登记**（main() 保存结果后、`log.info("  结果: %s", OUTPUT_PATH)` 之前）

```python
    # ── 实验登记 (2026-08-16): 每次回测自动写入 experiments/ ──
    try:
        from experiment_tracker import log_experiment
        _styles = config.get("styles") or {}
        log_experiment(
            script_name="run_walkforward_backtest",
            partition="development+extend_val",
            config={"top_k": bt_config.get("top_k"),
                    "styles_enabled": bool(_styles.get("enabled")),
                    "styles_budgets": _styles.get("budgets"),
                    "styles_sleeves": {k: v.get("factors") for k, v in
                                       (_styles.get("sleeves") or {}).items()}},
            results={k: {"excess_annual": v.get("excess_annual"),
                         "total_return": v.get("total_return"),
                         "sharpe": v.get("sharpe"),
                         "max_drawdown": v.get("max_drawdown")}
                     for k, v in results.items() if isinstance(v, dict)},
            notes=(f"styles={bool(_styles.get('enabled'))} "
                   f"budgets={_styles.get('budgets')}"),
            experiments_dir=os.path.join(BASE_DIR, "experiments"),
        )
    except Exception as e:
        log.warning("experiment_tracker 登记失败 (不影响主结果): %s", e)
```

- [ ] **Step 6: 运行确认通过 + 全量回归**

Run: `py -m pytest tests/test_sleeve_blend.py -q && py -m pytest tests/ -q`
Expected: 136 passed

- [ ] **Step 7: Commit**

```bash
git add scripts/active/run_walkforward_backtest.py tests/test_sleeve_blend.py
git commit -m "feat(sleeve): extend 接线 + 回测自动实验登记"
```

---

### Task 5: 实验执行（预算网格 + top_k 验证 + 训练结果报告）

**Files:** 无代码改动（config 编辑 + 运行 + 存档）；每档实验前备份 config

**运行方式**（每档 ~2h，顺序执行）：
```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
py scripts/active/run_walkforward_backtest.py --folds-only --liquid --extend-val 2025-01-01 2026-06-30
```

- [ ] **Step 1: 实验 A —— 基线复现（enabled: false）**

config 保持 `styles.enabled: false` → 跑一次全量。期望：与 v27 结果一致（extend +93.6%/Sharpe 2.17）。完成后：
```bash
cp data/ic_validation/walkforward_results.json data/ic_validation/sleeve_A_baseline_off.json
```
对照 v27 存档（`walkforward_results_v27_singlecap20_lowturnover.json`）：extend 总收益/Sharpe/DD 应一致（±1pp）。不一致 → 停，查回归（Task 2-4 的改动泄漏）。

- [ ] **Step 2: 实验 B —— 预算 60/25/15**

config：`styles.enabled: true`，budgets momentum 0.25 / growth 0.15。跑全量。存档：
```bash
cp data/ic_validation/walkforward_results.json data/ic_validation/sleeve_B_60_25_15.json
```

- [ ] **Step 3: 实验 C —— 预算 60/20/20**

budgets 0.20 / 0.20。跑全量。存档 `sleeve_C_60_20_20.json`。

- [ ] **Step 4: 实验 D —— 预算 50/30/20**

budgets 0.30 / 0.20。跑全量。存档 `sleeve_D_50_30_20.json`。

- [ ] **Step 5: 汇总对比 + 选优**

对比表（各档）：extend 总收益/年化/超额/Sharpe/DD/Calmar/换手、folds 均值超额、单票最大权重、持仓中是否出现成长/动量风格标的（对照 v27 的 30 只医药资源名单，检查新名单变化）。
选优规则（预先声明，防事后挑选）：**在"folds 均值超额不劣于 v27 的 +5.8% 且 extend DD 不劣于 -15%"的档中，取 extend Sharpe 最高者**。若无档满足 → 结论为"不通过"，回到 v27 定稿并记录。

- [ ] **Step 6: top_k 验证（选优档上）**

对选优档的 config，分别设 `execution.top_k: 10` 与 `30` 各跑一次全量（备份恢复 config 其余不变）。对比 20：换手/集中度/Sharpe/收益。若 10 或 30 的 extend Sharpe + 收益组合显著更优（Sharpe 提升 >0.15 且 DD 不恶化）→ 采用；否则维持 20。
存档 `sleeve_E_topk10.json` / `sleeve_F_topk30.json`。

- [ ] **Step 7: 训练结果报告**

生成 `docs/superpowers/plans/2026-08-16-multi-style-sleeve-results.md`：各档成绩表、选优结论、最终 config 快照、与 v27 的逐项对比、局限与下一步。更新 AGENTS.md 生产状态段。

- [ ] **Step 8: Commit 结果文档与存档清单**

```bash
git add docs/ config.yaml
git commit -m "docs(sleeve): 预算网格实验结果与最终定稿"
```

---

### Task 6: 部署链适配（仅当 Task 5 通过）+ 定稿

**Files:**
- Modify: `scripts/active/deploy_v24b_paper.py`、`scripts/active/run_paper_signal.py`（仅当选优档通过）
- Modify: `AGENTS.md`（生产状态定稿）

- [ ] **Step 1: deploy_v24b_paper.py**：p5 报告写入 sleeve 结构（`sleeves` 键：各 sleeve 因子权重 + budgets，来源 walkforward 结果 `sleeve_median_weights` + config budgets）
- [ ] **Step 2: run_paper_signal.py**：`compute_composite_scores_live` 增加与回测同语义的 sleeve 混合（复用 `_weighted_z_composite` 逻辑：因子 z 分 × ICIR 权重 / Σ|w|，按 budgets 混合）
- [ ] **Step 3: 信号冒烟**：`py scripts/active/run_paper_signal.py --dry-run <最近交易日>` 跑通 + 结果 data_as_of 字段正常
- [ ] **Step 4: AGENTS.md 定稿**（生产状态 = 新版本号 + 成绩 + sleeve 配置说明）
- [ ] **Step 5: Commit + 合并回 main**

```bash
git add -A && git commit -m "feat(sleeve): 部署链适配 + 生产定稿"
git checkout main && git merge feature/multi-style-sleeve --no-ff
```
