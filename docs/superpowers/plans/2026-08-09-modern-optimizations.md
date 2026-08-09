# P0/P1/P2 现代优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 不引入 LLM 的三个现代优化：P0 组合层（风险平价权重 + 波动率目标仓位）、P1 LightGBM fold 对比、P2 因子正交化。

**Architecture:** P0 在 run_backtest 权重层扩展（weight_mode + 仓位缩放，与 pool_filter 正交）；P1 复用 model/baselines.py 的 fit/predict 接口做 fold 对比（不改变生产路径，纯研究对比输出）；P2 在因子面板层做正交化（fold 权重计算前）。

**Tech Stack:** Python 3.12, pandas, numpy, sklearn (Ridge), lightgbm（若可用）, pytest

## Global Constraints

- 数据分区纪律由 gate.py 强制；回测仅 folds-only（不消耗 TEST② 锁）
- 所有参数走 config.yaml（唯一参数源）
- P0 开关默认关闭（equal），开启后行为变化可控、回退零成本
- P1 是研究对比输出（results["lgbm_compare"]），不改变生产信号路径
- P2 正交化不改变因子语义（保留方向，去冗余）
- 复用 `_inv_vol_weights` 的模式（PIT：只用 ≤ today 数据）
- 测试：pytest；`py` 命令（Windows）

---

### Task 1 (P0-1): 风险平价权重

**Files:**
- Modify: `scripts/active/run_walkforward_backtest.py`（_inv_vol_weights 旁新增 _risk_parity_weights）
- Modify: `config.yaml`（portfolio_optimizer 注释更新）
- Test: `tests/test_risk_parity.py`

**Interfaces:**
- Consumes: 无
- Produces: `_risk_parity_weights(all_data: dict, buy_list: list, today, lookback: int = 60, shrink: float = 0.5) -> dict | None` — 风险平价权重（用样本协方差 + Ledoit-Wolf 收缩近似），返回 {sym: weight} 或 None；run_backtest 的 weight_mode 支持 "risk_parity"

- [ ] **Step 1: 写失败测试**

```python
"""tests/test_risk_parity.py"""
import numpy as np
import pandas as pd
import pytest
import scripts.active.run_walkforward_backtest as rw


def _mk_data(n_dates=120, n_stocks=4):
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    all_data = {}
    rng = np.random.default_rng(5)
    # 低波动股票 (S0) 与高波动 (S3), 部分相关
    base = rng.normal(0, 0.01, n_dates)
    vols = [0.008, 0.015, 0.022, 0.030]
    for i, v in enumerate(vols):
        rets = base * (0.3 if i < 2 else 0.6) + rng.normal(0, v, n_dates)
        px = 10 * np.exp(np.cumsum(rets))
        all_data[f"S{i}"] = pd.DataFrame(
            {"date": dates, "open": px, "close": px, "high": px * 1.01,
             "low": px * 0.99, "volume": np.full(n_dates, 1e6),
             "amount": np.full(n_dates, 1e7)})
    return all_data, dates[-1]


def test_risk_parity_weights_normalized():
    all_data, today = _mk_data()
    w = rw._risk_parity_weights(all_data, ["S0", "S1", "S2", "S3"], today)
    assert w is not None
    assert abs(sum(w.values()) - 1.0) < 1e-6
    # 高波动股票权重低于低波动
    assert w["S0"] > w["S3"]


def test_risk_parity_insufficient_data_returns_none():
    all_data, today = _mk_data(n_dates=10)
    w = rw._risk_parity_weights(all_data, ["S0", "S1", "S2", "S3"], today)
    assert w is None or all(v == 0 for v in w.values())
```

- [ ] **Step 2: 运行确认失败**

Run: `py -m pytest tests/test_risk_parity.py -v`
Expected: FAIL — AttributeError: no attribute '_risk_parity_weights'

- [ ] **Step 3: 实现 _risk_parity_weights（在 _inv_vol_weights 后）**

```python
def _risk_parity_weights(all_data: dict, buy_list: list, today,
                         lookback: int = 60, shrink: float = 0.5) -> dict | None:
    """风险平价权重 (P0, 2026-08-09): 按协方差风险贡献均等分配。

    简化实现 (小样本友好):
      - 样本协方差 + Ledoit-Wolf 收缩 (shrink 比例)
      - 风险贡献均等: w_i ∝ 1/(ΣΣ w_j σ_ij 的边际贡献) — 用迭代近似:
        权重 ∝ 对角元素倒数 → 迭代 3 次风险平价
    PIT: 只用 <= today 数据。
    """
    rets = {}
    for s in buy_list:
        if s not in all_data:
            continue
        df = all_data[s][all_data[s]["date"] <= today]
        if len(df) < 20:
            continue
        r = df["close"].pct_change().dropna().tail(lookback)
        if len(r) < 10:
            continue
        rets[s] = r.to_numpy(dtype=np.float64)
    if len(rets) < 2:
        return None
    # 对齐长度
    n = min(len(v) for v in rets.values())
    X = np.column_stack([v[-n:] for v in rets.values()])
    syms = list(rets.keys())
    # 样本协方差 + 收缩
    S = np.cov(X, rowvar=False)
    diag = np.diag(S)
    target = np.eye(len(syms)) * np.mean(diag)
    S_shrunk = (1 - shrink) * S + shrink * target
    # 迭代风险平价 (3 次)
    w = 1.0 / np.sqrt(np.diag(S_shrunk))
    w = w / w.sum()
    for _ in range(3):
        port_var = w @ S_shrunk @ w
        mrc = S_shrunk @ w / port_var  # 边际风险贡献
        w = w * (1.0 / np.maximum(mrc, 1e-9))
        w = w / w.sum()
    return {s: float(wi) for s, wi in zip(syms, w) if wi > 0}
```

- [ ] **Step 4: 运行确认通过**

Run: `py -m pytest tests/test_risk_parity.py -v`
Expected: 2 passed

- [ ] **Step 5: 接入 weight_mode**

在 run_backtest 的 `if weight_mode == "inv_vol":` 处扩展：

```python
                    if weight_mode == "inv_vol":
                        w = _inv_vol_weights(all_data, decision.get("buy", []), today)
                        if w:
                            decision["weights"] = w
                    elif weight_mode == "risk_parity":
                        w = _risk_parity_weights(all_data, decision.get("buy", []), today)
                        if w:
                            decision["weights"] = w
```

- [ ] **Step 6: 回归 + 提交**

Run: `py -m pytest tests/ -q`（全量通过）
```bash
git add scripts/active/run_walkforward_backtest.py tests/test_risk_parity.py
git commit -m "feat(P0): risk parity portfolio weights (weight_mode=risk_parity)"
```

---

### Task 2 (P0-2): 波动率目标仓位

**Files:**
- Modify: `scripts/active/run_walkforward_backtest.py`
- Modify: `config.yaml`（vol_target 段）
- Test: `tests/test_vol_target.py`

**Interfaces:**
- Consumes: regime_det.detect_v2 的 vol_pct（已有）
- Produces: `_vol_target_scale(vol_pct: float, cfg: dict) -> float` — 仓位缩放系数；run_backtest 的 `vol_target_cfg: dict | None = None` 参数（None=不启用）

- [ ] **Step 1: 写失败测试**

```python
"""tests/test_vol_target.py"""
import pytest
import scripts.active.run_walkforward_backtest as rw


def test_vol_target_scale_low_vol_full():
    cfg = {"target_pct": 0.70, "max_scale": 1.0, "min_scale": 0.4}
    s = rw._vol_target_scale(0.20, cfg)  # 市场低波动 → 满仓
    assert s == pytest.approx(1.0)


def test_vol_target_scale_high_vol_reduced():
    cfg = {"target_pct": 0.70, "max_scale": 1.0, "min_scale": 0.4}
    s = rw._vol_target_scale(0.95, cfg)  # 市场高波动 → 降仓
    assert s < 0.6
    assert s >= 0.4


def test_vol_target_disabled_when_none():
    assert rw._vol_target_scale(0.95, None) == 1.0
```

- [ ] **Step 2: 运行确认失败**

Run: `py -m pytest tests/test_vol_target.py -v`
Expected: FAIL — AttributeError

- [ ] **Step 3: 实现 _vol_target_scale**

```python
def _vol_target_scale(vol_pct: float, cfg: dict | None) -> float:
    """波动率目标仓位缩放 (P0, Moreira & Muir 2017 简化版)。

    市场已实现波动率百分位 vol_pct (0-1) 高于目标 → 降仓:
      scale = clip(target_pct / (vol_pct + eps), min_scale, max_scale)
    cfg: {"target_pct": 0.7, "max_scale": 1.0, "min_scale": 0.4}
    """
    if not cfg:
        return 1.0
    target = float(cfg.get("target_pct", 0.7))
    mx = float(cfg.get("max_scale", 1.0))
    mn = float(cfg.get("min_scale", 0.4))
    if vol_pct <= 0 or target <= 0:
        return 1.0
    scale = target / max(vol_pct, 1e-6)
    return float(np.clip(scale, mn, mx))
```

- [ ] **Step 4: 运行确认通过**

Run: `py -m pytest tests/test_vol_target.py -v`
Expected: 3 passed

- [ ] **Step 5: 接入 run_backtest（调仓日 pending 生成后缩放）**

run_backtest 签名加 `vol_target_cfg: dict | None = None`；在 decision 生成后：

```python
                    # ★ 波动率目标仓位 (P0): 按市场波动率缩放总仓位
                    if vol_target_cfg and regime_det is not None:
                        _r2, _vp2 = regime_det.detect_v2(str(today.date()))
                        _scale = _vol_target_scale(_vp2, vol_target_cfg)
                        if _scale < 1.0 and decision.get("buy"):
                            # 买入金额按 scale 缩放的实现: 记录 scale 供 execute 用
                            decision["cash_scale"] = _scale
                            log.info("  [%s] 调仓日 %s: vol_target scale=%.2f (vol_pct=%.2f)",
                                     label, today.date(), _scale, _vp2)
```

engine.execute 支持 `decision.get("cash_scale", 1.0)`（现金池 × scale）。run_fold_analysis/run_fold_test/main 透传 `vol_target_cfg=config.get("vol_target")`。

- [ ] **Step 6: config.yaml 加 vol_target 段**

```yaml
# ── 波动率目标仓位 (P0, 2026-08-09) ──
vol_target:
  enabled: false          # 实验时置 true
  target_pct: 0.70        # 目标波动率百分位
  max_scale: 1.0          # 满仓上限
  min_scale: 0.40         # 最低仓位
```

- [ ] **Step 7: 回归 + 提交**

Run: `py -m pytest tests/ -q`
```bash
git add scripts/active/run_walkforward_backtest.py config.yaml tests/test_vol_target.py
git commit -m "feat(P0): vol-target position scaling (Moreira-Muir style)"
```

---

### Task 3 (P1): LightGBM fold 对比

**Files:**
- Create: `scripts/active/run_lgbm_fold_compare.py`（独立研究脚本，folds-only 对比）
- Modify: 无（不改变生产路径）

**Interfaces:**
- Consumes: model/baselines.py（L2 Ridge / L3 LightGBM，fit/predict 接口已统一）；run_walkforward_backtest 的因子面板（复用其构建逻辑或独立构建简化版）
- Produces: `data/ic_validation/lgbm_fold_compare.json` — {fold_i: {"lgbm_ic": x, "ridge_ic": y, "lgbm_excess": x, "ridge_excess": y}}

**设计要点**（严格纪律）：
- fold 结构同主回测：fold i 训练期（2015-训练年）训练 LGBM，验证期预测 → 截面 IC + 组合超额
- 特征 = 稳定因子面板值（fold 筛选的因子）；标签 = 21 日前瞻收益（与主回测一致）
- LGBM 参数受 gate 约束（n_estimators≤100、max_depth≤3）——防过拟合
- 输出对比不进入生产信号（纯研究结论）

- [ ] **Step 1: 写脚本骨架（复用 factor_panels 构建）**

```python
"""scripts/active/run_lgbm_fold_compare.py — P1: LGBM vs 线性 fold 对比 (2026-08-09)

严格 walk-forward: fold i 训练期训练, 验证期预测, 无泄漏。
不改变生产路径, 输出研究对比结果。
"""
import os, sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gate import load_config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config = load_config(os.path.join(BASE_DIR, "config.yaml"))

def main():
    # 1. 构建因子面板 (复用 run_walkforward_backtest 的 precompute)
    from scripts.active.run_walkforward_backtest import (
        precompute_factor_panels, _load_partitions, load_bt_config,
        _nearest_idx, FOLDS)
    from scripts.active.run_walkforward_backtest import _load_all_data  # 需确认存在
    # ... (实现见下方步骤)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 检查 run_walkforward_backtest 是否暴露数据加载函数（load_all_data/calendar）**

Run: `grep -n "def .*all_data\|def .*calendar\|def load_bt_config" scripts/active/run_walkforward_backtest.py`
若数据加载逻辑在 main() 内部，则本脚本内联复刻（读 data_store 日线 + 交易日历）。

- [ ] **Step 3: 实现 fold 对比主循环**

```python
def run_fold_compare():
    # 复用主回测的数据/面板构建 (同 needed_dates 逻辑: fold 训练+验证期)
    # 每个 fold:
    #   train X = 因子面板[fold训练期] 展平截面 (行=股票×日期, 列=因子)
    #   y = 21日前瞻收益 (close[t+21]/close[t]-1)
    #   LGBM fit (早停 on val 20%) → predict 验证期 → 截面 IC (Spearman)
    #   Ridge 同流程 → 对比 IC
    #   组合超额: top-30 等权 验证期 (可选, 简化版只报 IC)
    pass
```

- [ ] **Step 4: 运行并输出对比 JSON**

Run: `py scripts/active/run_lgbm_fold_compare.py`
Expected: `data/ic_validation/lgbm_fold_compare.json` 含 5 个 fold 的 lgbm/ridge 验证期 IC

- [ ] **Step 5: 提交**

```bash
git add scripts/active/run_lgbm_fold_compare.py
git commit -m "feat(P1): LGBM vs ridge fold compare (research only)"
```

---

### Task 4 (P2): 因子正交化

**Files:**
- Create: `orthogonalize.py`（模块，被 run_walkforward_backtest 引用）
- Modify: `scripts/active/run_walkforward_backtest.py`（fold 权重计算前可选正交化）
- Modify: `config.yaml`（factor_orthogonalize 开关）
- Test: `tests/test_orthogonalize.py`

**Interfaces:**
- Consumes: 无
- Produces: `orthogonalize_panels(panels: dict, factor_names: list, method: str = "gs") -> dict` — Gram-Schmidt 正交化（按 |ICIR| 降序逐因子去相关）；`config["factor_orthogonalize"]` 开关

- [ ] **Step 1: 写失败测试**

```python
"""tests/test_orthogonalize.py"""
import numpy as np
import pandas as pd
import pytest
from orthogonalize import orthogonalize_panels


def test_orthogonalize_removes_correlation():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-02", periods=60)
    syms = [f"S{i}" for i in range(20)]
    base = rng.normal(0, 1, (len(dates), 1))
    f1 = pd.DataFrame(np.tile(base, (1, 20)) + rng.normal(0, 0.1, (len(dates), 20)),
                      index=dates, columns=syms).astype(np.float32)
    f2 = pd.DataFrame(np.tile(base * 2, (1, 20)) + rng.normal(0, 0.1, (len(dates), 20)),
                      index=dates, columns=syms).astype(np.float32)
    panels = {"f1": f1, "f2": f2}
    out = orthogonalize_panels(panels, ["f1", "f2"], method="gs")
    # f2 正交化后与 f1 相关性 ≈ 0
    r = np.corrcoef(out["f1"].to_numpy().ravel(), out["f2"].to_numpy().ravel())[0, 1]
    assert abs(r) < 0.05


def test_orthogonalize_keeps_first_factor():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-02", periods=60)
    syms = [f"S{i}" for i in range(20)]
    f1 = pd.DataFrame(rng.normal(0, 1, (len(dates), 20)), index=dates, columns=syms).astype(np.float32)
    panels = {"f1": f1}
    out = orthogonalize_panels(panels, ["f1"], method="gs")
    assert np.allclose(out["f1"].to_numpy(), f1.to_numpy(), atol=1e-6)
```

- [ ] **Step 2: 运行确认失败**

Run: `py -m pytest tests/test_orthogonalize.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: 实现 orthogonalize.py**

```python
"""
orthogonalize.py — 因子正交化 (P2, 2026-08-09)

Gram-Schmidt 正交化 (按给定顺序): 每个因子减去其在已正交因子上的投影。
用于去除稳定因子间的冗余 (corr 0.6 剪枝之外的系统性去相关)。
不改变第一个因子的值 (方向保留)。
"""
import numpy as np
import pandas as pd


def orthogonalize_panels(panels: dict, factor_names: list,
                         method: str = "gs") -> dict:
    """按 factor_names 顺序 Gram-Schmidt 正交化面板。

    Args:
        panels: {factor: DataFrame(date × symbol)} (float32)
        factor_names: 正交化顺序 (先正交的优先保留)
        method: 仅支持 "gs" (Gram-Schmidt)

    Returns:
        新面板 dict (原 panels 不修改)
    """
    names = [fn for fn in factor_names if fn in panels]
    if not names:
        return panels
    out = {}
    basis = []  # 已正交因子 (展平向量)
    for fn in names:
        v = panels[fn].to_numpy(dtype=np.float64).ravel()
        m = ~np.isnan(v)
        vc = v.copy()
        if m.sum() > 10:
            for b in basis:
                bm = m & ~np.isnan(b)
                if bm.sum() < 10:
                    continue
                # 回归投影: v = alpha + beta*b + eps → 残差
                b_ = b[bm]
                v_ = vc[bm]
                beta = np.cov(b_, v_)[0, 1] / (np.var(b_) + 1e-12)
                alpha = v_.mean() - beta * b_.mean()
                vc[bm] = v_ - (alpha + beta * b_)
        out[fn] = pd.DataFrame(
            vc.reshape(panels[fn].shape), index=panels[fn].index,
            columns=panels[fn].columns).astype(np.float32)
        basis.append(vc)
    return out
```

- [ ] **Step 4: 运行确认通过**

Run: `py -m pytest tests/test_orthogonalize.py -v`
Expected: 2 passed

- [ ] **Step 5: 接入 fold 流程（可选开关）**

run_walkforward_backtest.py：fold 分析前，若 config `factor_orthogonalize.enabled` 则对 panels 正交化（在 precompute 后、compute_icir_weights 前）。main 透传。

config.yaml：
```yaml
# ── 因子正交化 (P2, 2026-08-09) ──
factor_orthogonalize:
  enabled: false          # 实验时置 true (fold 前 Gram-Schmidt 去冗余)
```

- [ ] **Step 6: 回归 + 提交**

Run: `py -m pytest tests/ -q`
```bash
git add orthogonalize.py tests/test_orthogonalize.py scripts/active/run_walkforward_backtest.py config.yaml
git commit -m "feat(P2): factor orthogonalization (Gram-Schmidt, optional)"
```

---

### Task 5: 汇总验证

- [ ] **Step 1: 全量测试**

Run: `py -m pytest tests/ -q`
Expected: 全量通过（原 54 + 新增）

- [ ] **Step 2: 冒烟验证（runbook 门禁）**

Run: `py scripts/active/run_walkforward_backtest.py --folds --folds-only --liquid --sample 50`
检查 4 关键行（universe/面板日期/实验状态/调仓买入）

- [ ] **Step 3: 提交 + 汇报**

```bash
git add -A
git commit -m "feat(P0-P2): risk parity + vol target + lgbm fold compare + orthogonalization"
```
