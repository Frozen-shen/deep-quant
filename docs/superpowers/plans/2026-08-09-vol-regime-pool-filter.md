# 波动率分层 × Regime 乘数（股票池分域）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现选股层波动率分域——候选股按自身波动率三分位分层，按市场波动率状态（regime vol_pct）施加分数乘数，使高波动市场偏好低波池。

**Architecture:** 新增 `pool_filter.py` 模块（vol_bucket 分档 + apply_pool_filter 乘数），在 run_walkforward_backtest.py 的调仓日流程中（score_stocks 之后、ranker.rank 之前）插入调用；config.yaml 新增 pool_filter 段控制开关与乘数。乘数作用于选股分（分数层），与 v9b inv_vol（权重层）正交。

**Tech Stack:** Python 3.12, pandas, numpy, pytest；复用现有 regime_detector.detect_v2 的 vol_pct。

## Global Constraints

- 数据分区纪律由 gate.py 强制，回测仅限 development 分区（folds-only，不消耗 TEST 锁）
- 所有参数走 config.yaml（唯一参数源）
- PIT 安全：股票波动率只用 ≤ today 的日线数据
- config `pool_filter.enabled: false` 时行为必须与现有完全一致（回退安全）
- 不新建平行代码库：pool_filter.py 必须被 run_walkforward_backtest.py 引用且有测试覆盖
- Python 命令用 `py`（Windows 环境）

---

### Task 1: pool_filter.py 模块（分档 + 乘数）

**Files:**
- Create: `C:\Users\Frozen\ZCodeProject\quant-starter\pool_filter.py`
- Test: `C:\Users\Frozen\ZCodeProject\quant-starter\tests\test_pool_filter.py`

**Interfaces:**
- Consumes: 无（独立模块）
- Produces:
  - `vol_bucket(scores: dict, all_data: dict, today, lookback: int = 60) -> dict[str, str]` — 返回 {sym: 'low'|'mid'|'high'}，波动率数据不足的股票返回 'mid'
  - `apply_pool_filter(scores: dict, buckets: dict, vol_pct: float, mults: dict) -> dict` — 按档位乘数调整分数，返回新 dict（不原地修改）

- [ ] **Step 1: 写失败测试**

```python
"""tests/test_pool_filter.py — 波动率分层 × regime 乘数"""
import numpy as np
import pandas as pd
import pytest

from pool_filter import vol_bucket, apply_pool_filter


def _mk_data(n_dates=80, n_stocks=9):
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    all_data = {}
    vols = [0.005, 0.01, 0.02] * 3  # 低/中/高各3只
    for i, v in enumerate(vols):
        rng = np.random.default_rng(i)
        rets = rng.normal(0, v, n_dates)
        px = 10 * np.exp(np.cumsum(rets))
        all_data[f"S{i}"] = pd.DataFrame(
            {"date": dates, "open": px, "close": px, "high": px * 1.01,
             "low": px * 0.99, "volume": np.full(n_dates, 1e6),
             "amount": np.full(n_dates, 1e7)})
    return all_data, dates[-1]


def test_vol_bucket_three_tiers():
    all_data, today = _mk_data()
    scores = {f"S{i}": 1.0 for i in range(9)}
    buckets = vol_bucket(scores, all_data, today)
    assert buckets["S0"] == "low" and buckets["S1"] == "low" and buckets["S2"] == "low"
    assert buckets["S3"] == "mid" and buckets["S4"] == "mid" and buckets["S5"] == "mid"
    assert buckets["S6"] == "high" and buckets["S7"] == "high" and buckets["S8"] == "high"


def test_vol_bucket_insufficient_data_defaults_mid():
    all_data, today = _mk_data(n_dates=5)  # 波动率数据不足
    scores = {f"S{i}": 1.0 for i in range(9)}
    buckets = vol_bucket(scores, all_data, today)
    assert all(b == "mid" for b in buckets.values())


def test_apply_pool_filter_high_vol_regime():
    scores = {"a": 1.0, "b": 2.0, "c": 3.0}
    buckets = {"a": "low", "b": "mid", "c": "high"}
    mults = {"low": 1.5, "mid": 1.0, "high": 0.5}
    out = apply_pool_filter(scores, buckets, vol_pct=0.8, mults=mults)
    assert out["a"] == pytest.approx(1.5)
    assert out["b"] == pytest.approx(2.0)
    assert out["c"] == pytest.approx(1.5)
    assert out is not scores  # 不原地修改


def test_apply_pool_filter_does_not_mutate_input():
    scores = {"a": 1.0}
    buckets = {"a": "low"}
    apply_pool_filter(scores, buckets, 0.8, {"low": 1.5, "mid": 1.0, "high": 0.5})
    assert scores["a"] == 1.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `py -m pytest tests/test_pool_filter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pool_filter'`

- [ ] **Step 3: 实现 pool_filter.py**

```python
"""
pool_filter.py — 波动率分层 × regime 乘数 (股票池分域, v10, 2026-08-09)

设计文档: docs/superpowers/specs/2026-08-09-vol-regime-pool-filter-design.md

选股层软偏好:
  - vol_bucket: 候选股按自身 lookback 日波动率跨截面三分位 (low/mid/high)
  - apply_pool_filter: 按市场波动率状态 (vol_pct) 选择乘数表, 对分数加权
  - 波动率数据不足的股票按 mid (×1.0) 处理, 静默跳过
"""

import numpy as np
import pandas as pd


def vol_bucket(scores: dict, all_data: dict, today,
               lookback: int = 60) -> dict:
    """候选股波动率三分位分档。

    Args:
        scores: {sym: score} 候选股票分数 (仅键有意义)
        all_data: {sym: DataFrame(date, open, close, ...)} 日线
        today: 调仓日 (pd.Timestamp), 只用 <= today 数据 (PIT)
        lookback: 波动率回看天数 (日收益 std)

    Returns:
        {sym: 'low'|'mid'|'high'}, 数据不足的股票返回 'mid'
    """
    vols = {}
    for sym in scores:
        if sym not in all_data:
            continue
        df = all_data[sym][all_data[sym]["date"] <= today]
        if len(df) < 20:
            continue
        rets = df["close"].pct_change().dropna().tail(lookback)
        if len(rets) < 10:
            continue
        v = float(rets.std())
        if v > 0 and not np.isnan(v):
            vols[sym] = v
    if not vols:
        return {s: "mid" for s in scores}

    # 跨截面分位 (波动率排序, 无聚类)
    arr = np.array(sorted(vols.values()))
    q30 = np.percentile(arr, 30)
    q70 = np.percentile(arr, 70)

    buckets = {}
    for sym in scores:
        v = vols.get(sym)
        if v is None:
            buckets[sym] = "mid"
        elif v < q30:
            buckets[sym] = "low"
        elif v > q70:
            buckets[sym] = "high"
        else:
            buckets[sym] = "mid"
    return buckets


def apply_pool_filter(scores: dict, buckets: dict, vol_pct: float,
                      mults: dict) -> dict:
    """按市场波动率状态对选股分施加档位乘数 (软偏好)。

    Args:
        scores: {sym: score} 原始选股分
        buckets: {sym: 'low'|'mid'|'high'} (vol_bucket 输出)
        vol_pct: 市场波动率百分位 (0-1, regime_detector.detect_v2 输出)
        mults: 乘数表 {"low": x, "mid": x, "high": x} (当前市场状态下)

    Returns:
        新分数 dict (不原地修改)
    """
    # 选择乘数表: 高波动市场用 mults 原表; 低波动市场由调用方传入对应表
    out = dict(scores)
    for sym in out:
        tier = buckets.get(sym, "mid")
        out[sym] = out[sym] * float(mults.get(tier, 1.0))
    return out
```

- [ ] **Step 4: 运行测试确认通过**

Run: `py -m pytest tests/test_pool_filter.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add pool_filter.py tests/test_pool_filter.py
git commit -m "feat: add vol-regime pool filter (bucket + multiplier)"
```

---

### Task 2: config.yaml 新增 pool_filter 段

**Files:**
- Modify: `C:\Users\Frozen\ZCodeProject\quant-starter\config.yaml`（在 portfolio_optimizer 段后）

**Interfaces:**
- Consumes: 无
- Produces: `config["pool_filter"]` dict：
  ```python
  {"enabled": bool, "low_vol_mult": float, "high_vol_mult": float,
   "low_vol_up": float, "high_vol_up": float}
  ```

- [ ] **Step 1: 在 config.yaml 的 portfolio_optimizer 段后添加**

```yaml
# ── 股票池分域 (v10, 波动率分层×regime乘数, 设计文档 2026-08-09) ──
pool_filter:
  enabled: false          # v10 实验时置 true; false 时行为与 v9 完全一致
  low_vol_mult: 1.5       # 高波动市场: 低波档乘数
  high_vol_mult: 0.5      # 高波动市场: 高波档乘数
  low_vol_up: 0.8         # 低波动市场: 低波档乘数 (牛市弹性)
  high_vol_up: 1.2        # 低波动市场: 高波档乘数
```

- [ ] **Step 2: 验证 config 可解析**

Run: `py -c "import yaml; cfg=yaml.safe_load(open('config.yaml',encoding='utf-8')); print(cfg['pool_filter'])"`
Expected: `{'enabled': False, 'low_vol_mult': 1.5, 'high_vol_mult': 0.5, 'low_vol_up': 0.8, 'high_vol_up': 1.2}`

- [ ] **Step 3: 提交**

```bash
git add config.yaml
git commit -m "feat: add pool_filter config section (disabled by default)"
```

---

### Task 3: run_backtest 接入 pool_filter

**Files:**
- Modify: `C:\Users\Frozen\ZCodeProject\quant-starter\scripts\active\run_walkforward_backtest.py`

**Interfaces:**
- Consumes: `pool_filter.vol_bucket`, `pool_filter.apply_pool_filter`（Task 1）；`config["pool_filter"]`（Task 2）；`regime_det.detect_v2(date_str)` 返回 `(regime, vol_pct)`（现有）
- Produces: `run_backtest(..., pool_filter_cfg: dict | None = None)`；`run_fold_analysis(..., pool_filter_cfg=None)`；main 从 config 读取并传入

- [ ] **Step 1: run_backtest 签名加 pool_filter_cfg 参数**

在 `def run_backtest(...)` 签名末尾（weight_mode 之后）添加：

```python
                 weight_mode: str = "equal",
                 pool_filter_cfg: dict | None = None):
```

- [ ] **Step 2: 调仓日流程插入 pool_filter 调用**

在 score_stocks 调用后、PIT 过滤前（`scores = {s: v for s, v in scores.items() if s in pit_stocks}` 之前）插入：

```python
            # ★ 股票池分域 (v10): 波动率分层 × regime 乘数 (选股层软偏好)
            if pool_filter_cfg and pool_filter_cfg.get("enabled"):
                from pool_filter import vol_bucket, apply_pool_filter
                if regime_det is not None:
                    _reg, _vol_pct = regime_det.detect_v2(str(today.date()))
                else:
                    _vol_pct = 0.5
                _buckets = vol_bucket(scores, all_data, today)
                if _vol_pct > 0.70:  # 高波动市场: 避险偏好
                    _mults = {"low": pool_filter_cfg.get("low_vol_mult", 1.5),
                              "mid": 1.0,
                              "high": pool_filter_cfg.get("high_vol_mult", 0.5)}
                elif _vol_pct < 0.30:  # 低波动市场: 弹性偏好
                    _mults = {"low": pool_filter_cfg.get("low_vol_up", 0.8),
                              "mid": 1.0,
                              "high": pool_filter_cfg.get("high_vol_up", 1.2)}
                else:
                    _mults = {"low": 1.0, "mid": 1.0, "high": 1.0}
                scores = apply_pool_filter(scores, _buckets, _vol_pct, _mults)
                log.info("  [%s] 调仓日 %s: pool_filter vol_pct=%.2f (%s)",
                         label, today.date(), _vol_pct,
                         "高波避险" if _vol_pct > 0.70 else
                         ("低波弹性" if _vol_pct < 0.30 else "中性"))
```

- [ ] **Step 3: run_fold_analysis 签名加 pool_filter_cfg 并透传**

签名改为 `..., weight_mode: str = "equal", pool_filter_cfg: dict | None = None) -> dict:`；
内部 run_backtest 调用（1548 行附近）加参数：

```python
                         weight_mode=weight_mode,
                         pool_filter_cfg=pool_filter_cfg)
```

run_fold_test 同样处理（签名 + run_backtest 调用透传）。

- [ ] **Step 4: main() 从 config 读取并传入**

在 main 的 fold 分支调用 run_fold_analysis 处添加：

```python
                pool_filter_cfg=config.get("pool_filter"))
```

- [ ] **Step 5: 语法检查 + 行为验证**

Run: `py -m py_compile scripts/active/run_walkforward_backtest.py`
Expected: 无输出（编译通过）

Run: `py -c "
import sys; sys.path.insert(0, '.')
from scripts.active.run_walkforward_backtest import run_backtest
import inspect
sig = inspect.signature(run_backtest)
assert 'pool_filter_cfg' in sig.parameters
print('pool_filter_cfg 参数已接入')
"`
Expected: `pool_filter_cfg 参数已接入`

- [ ] **Step 6: 回归验证（enabled=false 行为不变）**

Run: `py -m pytest tests/test_pool_filter.py tests/test_minute_layer.py tests/test_gate.py -q`
Expected: 全部通过（pool_filter disabled 不影响现有路径）

- [ ] **Step 7: 提交**

```bash
git add scripts/active/run_walkforward_backtest.py
git commit -m "feat: wire pool_filter into walkforward backtest (score-layer)"
```

---

### Task 4: v10 回测验证

**Files:**
- 无代码改动（纯实验运行）

**Interfaces:**
- Consumes: Task 1-3 全部

- [ ] **Step 1: 确认 TEST 锁存在（不应被触碰）**

Run: `ls data/ic_validation/.test_lock_v4`
Expected: 文件存在

- [ ] **Step 2: 开启 pool_filter 并跑 v10**

修改 config.yaml：`pool_filter.enabled: true`

Run: `py scripts/active/run_walkforward_backtest.py --folds --folds-only --liquid > logs_v10_walkforward.log 2>&1`（后台）

- [ ] **Step 3: 结果对比**

Run: `grep -E "FOLD_[1-5]" logs_v10_walkforward.log`
Expected: 记录 5 个 fold 超额收益，与 v9（-11.7/+8.0/+6.3/+1.9/+8.2，均值 +2.54%）对比

通过标准（设计文档 §6）：
1. fold 均值 > +2.54%
2. 换手率无剧变（avg_turnover < 基线的 1.5 倍）
3. bootstrap 95% CI 不显著变差（用日度超额序列 circular block bootstrap，沿用 v6-v7 对比方法）

- [ ] **Step 4: 结果记录（experiment_tracker）**

```python
from experiment_tracker import log_experiment
log_experiment(script_name='run_walkforward_backtest --folds --folds-only (v10: pool_filter)',
               partition='folds', config={'pool_filter': 'enabled', 'low_vol_mult': 1.5, 'high_vol_mult': 0.5},
               results={'fold_excess': {...}, 'fold_mean': ...},
               notes='...')
```

- [ ] **Step 5: 汇总汇报（向用户）**

汇报：v10 vs v9 对比、bootstrap 结论、换手变化、是否进入生产（enabled 置 true 或回退 false）
