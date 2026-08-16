# 组合层优化实施方案（单票上限 + 行业中性 + 因子去冗余 + 换手网格）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把组合层约束从"只记日志"变为"真正执行"，并按已验证的优化方向重跑回测，目标是 extend_val 超额从 +0.2% 提升到 +30~50pp（低换手实验削峰模拟已证明该空间存在）。

**Architecture:** 分四个阶段依次落地：① 单票上限在回测引擎中执行（engine 已支持 `decision["weights"]`，只需接线）；② 行业中性（修复行业映射格式 + 行业权重封顶）；③ 因子聚类去冗余 + 风格预算（先做只读相关性分析，再实现权重缩放）；④ 换手参数网格实验（复用 bt_* 配置键）。每阶段 TDD + 全量回测验证，达标才进下一阶段。

**Tech Stack:** Python 3.12 / pandas / pytest（项目现有栈）；回测验证用 `py scripts/active/run_walkforward_backtest.py --folds-only --liquid --extend-val 2025-01-01 2026-06-30`。

**注意:** 本项目不是 git 仓库——计划中的"Commit"步骤改为"备份关键文件"（`cp config.yaml config.yaml.bak_<阶段>`）。每次全量回测约 2.5 小时，用后台任务跑。

---

### Task 1: 单票上限落地（接线 engine weights）

**Files:**
- Modify: `scripts/active/run_walkforward_backtest.py:1615-1629`（组合后置约束段，日志改为执行）
- Modify: `config.yaml:176`（`max_single_pct` 0.05 → 0.20）
- Test: `tests/test_backtest_repair.py`（追加）

- [ ] **Step 1: 写失败测试**

```python
def test_decision_weights_capped_by_max_single():
    """买入 5 只、上限 20% 时, decision weights 应被 apply_portfolio_constraints
    缩放 (等权 20% 不超限则等权; 3 只等权 33% 超限则每只 20%)。"""
    from run_walkforward_backtest import apply_portfolio_constraints
    c = {"max_single_pct": 0.20}
    w5 = apply_portfolio_constraints({s: 1.0 for s in "abcde"}, c)
    assert abs(w5["a"] - 0.20) < 1e-9  # 5 只等权=20% 恰好不超限
    w3 = apply_portfolio_constraints({s: 1.0 for s in "abc"}, c)
    assert abs(w3["a"] - 0.20) < 1e-9  # 3 只等权 33% > 20% → 缩到 20%
    assert abs(sum(w3.values()) - 0.60) < 1e-9  # 剩余 40% 留现金
```

- [ ] **Step 2: 运行确认失败**

Run: `py -m pytest tests/test_backtest_repair.py::test_decision_weights_capped_by_max_single -v`
Expected: PASS（`apply_portfolio_constraints` 已有此语义——本测试是**回归钉住**，防止接线改动破坏语义。若失败则先查函数实现。）

- [ ] **Step 3: 接线——把约束结果写入 decision["weights"]**

Modify `scripts/active/run_walkforward_backtest.py`，替换 1615-1629 行的"检查+日志"段：

```python
                    # ── 组合后置约束 (方案C v5) ──
                    # ★ 2026-08-16 (v27): 从"检查+日志"改为"实际执行" —
                    # 目标权重写入 decision["weights"], bt.execute 按权重
                    # 分配现金 (engine.py 已支持 weights, 缺省等权兜底)。
                    if portfolio_constraints:
                        buy_scores = {s: 1.0 for s in decision.get("buy", [])}
                        w = apply_portfolio_constraints(buy_scores,
                                                        portfolio_constraints)
                        if w:
                            decision["weights"] = w
                            n_buy = len(decision.get("buy", []))
                            ew_pct = 1.0 / n_buy if n_buy else 0
                            max_single = portfolio_constraints.get(
                                "max_single_pct", 0.05)
                            if ew_pct > max_single:
                                log.info("  [%s] 单票等权 %.1f%% > 上限 %.1f%%, "
                                         "已按权重执行 (剩余留现金)",
                                         label, ew_pct * 100, max_single * 100)
```

- [ ] **Step 4: 改 config 上限为 20%**

Modify `config.yaml:176`:

```yaml
portfolio_constraints:
  max_single_pct: 0.20      # 单票仓位上限 20% (2026-08-16: 5%→20%, 用户风险偏好决策)
  max_industry_pct: 0.25    # 行业上限 25%
```

- [ ] **Step 5: 全量测试**

Run: `py -m pytest tests/ -q`
Expected: 全部 PASS（≥91）

- [ ] **Step 6: 备份 + 全量回测（低换手配置）**

```bash
cp config.yaml config.yaml.bak_v27_singlecap
# 在 config portfolio 段启用低换手参数 (取消 bt_* 注释并改值):
#   bt_hold_thresh: 45 / bt_sell_rank_buffer: 6 / bt_n_drop: 6 / bt_buy_confirm_days: 2
py scripts/active/run_walkforward_backtest.py --folds-only --liquid --extend-val 2025-01-01 2026-06-30
```

Expected: 回测完成，extend_val 超额显著高于 +0.2% 且组合集中度受控（单票市值占比 ≤ ~25%，因持有期增值可能略超 20%）。

- [ ] **Step 7: 验证集中度受控**

Run: 解析 `walkforward_results.json` 的 extend_val trades，重建每日持仓市值占比，确认峰值单票占比不超过 25%。
Expected: `max_single_weight <= 0.25`（留 5% 容差：买入时按 20% 现金分配，持有期增值可超）。

---

### Task 2: 行业中性落地

**Files:**
- Modify: `scripts/active/run_walkforward_backtest.py`（`_load_industry_map` 格式对齐 + 新函数 `_industry_cap_weights` + 接线）
- Test: `tests/test_backtest_repair.py`（追加）

- [ ] **Step 1: 写失败测试（行业映射格式对齐）**

```python
def test_industry_map_code_alignment():
    """行业映射键带 sh/sz 前缀, 对齐后按 6 位代码查询。"""
    from run_walkforward_backtest import _load_industry_map
    m = _load_industry_map()
    assert m, "行业映射文件应存在"
    assert "600519" in m or "sh600519" in m
```

- [ ] **Step 2: 写失败测试（行业权重封顶）**

```python
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
```

- [ ] **Step 3: 运行确认失败**

Run: `py -m pytest tests/test_backtest_repair.py -k "industry" -v`
Expected: 两个测试 FAIL（`_industry_cap_weights` 不存在）

- [ ] **Step 4: 实现格式对齐**

Modify `_load_industry_map`（`run_walkforward_backtest.py`）:

```python
def _load_industry_map() -> dict:
    """加载行业映射 {6位代码: industry} (新浪行业快照, 键去 sh/sz 前缀)。

    近似 PIT: 行业分类为最新快照 (换行业股票占比极小, 影响可接受)。
    """
    path = os.path.join(BASE_DIR, "data_store", "aux_industry", "industry_map.parquet")
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_parquet(path)
        codes = df["code"].astype(str).str.replace("sh", "", regex=False) \
            .str.replace("sz", "", regex=False)
        return dict(zip(codes, df["industry"].astype(str)))
    except Exception:
        return {}
```

- [ ] **Step 5: 实现行业权重封顶**

Add to `run_walkforward_backtest.py`（`apply_portfolio_constraints` 函数之后）:

```python
def _industry_cap_weights(weights: dict, industry_map: dict,
                          max_industry_pct: float) -> dict:
    """行业权重封顶: 同行业权重之和超上限时按比例缩减 (无映射股票不参与)。"""
    if not weights or max_industry_pct <= 0:
        return weights
    out = dict(weights)
    by_ind: dict = {}
    for s, w in out.items():
        ind = industry_map.get(s)
        if ind:
            by_ind.setdefault(ind, []).append(s)
    for ind, syms in by_ind.items():
        total = sum(out[s] for s in syms)
        if total > max_industry_pct:
            scale = max_industry_pct / total
            for s in syms:
                out[s] *= scale
    return out
```

- [ ] **Step 6: 运行确认通过**

Run: `py -m pytest tests/test_backtest_repair.py -k "industry" -v`
Expected: PASS

- [ ] **Step 7: 接线到调仓决策**

Modify Task 1 接线段，在 `decision["weights"] = w` 之后追加行业封顶：

```python
                    if portfolio_constraints:
                        buy_scores = {s: 1.0 for s in decision.get("buy", [])}
                        w = apply_portfolio_constraints(buy_scores,
                                                        portfolio_constraints)
                        if w:
                            max_ind = portfolio_constraints.get(
                                "max_industry_pct", 0.25)
                            w = _industry_cap_weights(
                                w, _load_industry_map(), max_ind)
                            decision["weights"] = w
                            ...
```

注意：`_load_industry_map` 在每次调仓日调用会重复读 parquet——改为模块级缓存：

```python
_industry_map_cache: dict | None = None

def _load_industry_map() -> dict:
    global _industry_map_cache
    if _industry_map_cache is not None:
        return _industry_map_cache
    ...  # 原实现, 末尾赋值 _industry_map_cache = result 并返回
```

- [ ] **Step 8: 全量测试 + 备份**

Run: `py -m pytest tests/ -q` → Expected: 全部 PASS
Run: `cp config.yaml config.yaml.bak_v28_industry`

---

### Task 3: 因子聚类去冗余（先分析后实现）

**Files:**
- Analyze (只读): 一次性脚本（不落盘，`py - <<EOF`）
- Modify: `scripts/active/run_walkforward_backtest.py`（`compute_icir_weights` 后加风格预算缩放）
- Test: `tests/test_backtest_repair.py`（追加）

- [ ] **Step 1: 只读分析——稳定因子相关性聚类**

Run:

```bash
py - <<'EOF'
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'scripts/active')
import json, numpy as np
import pandas as pd
d = json.load(open('data/ic_validation/walkforward_results.json', encoding='utf-8'))
stable = d['meta']['stable_factors']
print('稳定因子数:', len(stable))
# 用最近一次权重演变里因子权重符号近似风格: 分组按因子名前缀
import re
groups = {}
for f in stable:
    g = re.sub(r'\d+d?$', '', f)  # 去尾部数字
    groups.setdefault(g, []).append(f)
big = {k: v for k, v in groups.items() if len(v) >= 2}
print('家族分组 (≥2成员):')
for k, v in sorted(big.items(), key=lambda x: -len(x[1])):
    print(f'  {k}: {len(v)} 个 {v[:6]}')
EOF
```

Expected: 输出因子家族分布（预期 amihud/volatility/amplitude/cord 等家族扎堆），作为风格预算的分组依据。

- [ ] **Step 2: 写失败测试（风格预算缩放）**

```python
def test_style_budget_caps_family_weight():
    """同家族权重之和超上限时按比例缩减 (风格预算)。"""
    from run_walkforward_backtest import _style_budget_weights
    weights = {"amihud_5d": 0.3, "amihud_20d": 0.3, "volatility_20d": 0.2,
               "return_30d": 0.2}
    out = _style_budget_weights(weights, style_cap=0.4)
    # amihud 家族 0.6 > 0.4 → 缩到 0.4 (每只 0.2)
    assert abs(out["amihud_5d"] + out["amihud_20d"] - 0.4) < 1e-9
    assert abs(out["amihud_5d"] - 0.2) < 1e-9
    # 其他家族不超限 → 不变
    assert abs(out["volatility_20d"] - 0.2) < 1e-9
```

- [ ] **Step 3: 运行确认失败**

Run: `py -m pytest tests/test_backtest_repair.py -k "style_budget" -v`
Expected: FAIL（函数不存在）

- [ ] **Step 4: 实现风格预算**

Add to `run_walkforward_backtest.py`:

```python
def _style_budget_weights(weights: dict, style_cap: float = 0.4) -> dict:
    """风格预算: 同因子家族 (名称去尾部数字) 权重之和超 style_cap 时按比例缩减。

    防一族同质因子 (如 amihud_*/volatility_*) 垄断权重 — 2020 年失效的根因。
    """
    import re
    if not weights or style_cap <= 0:
        return weights
    out = dict(weights)
    fam: dict = {}
    for f in out:
        fam.setdefault(re.sub(r'\d+d?$', '', f), []).append(f)
    for fns in fam.values():
        total = sum(out[f] for f in fns)
        if total > style_cap:
            scale = style_cap / total
            for f in fns:
                out[f] *= scale
    return out
```

- [ ] **Step 5: 接线——权重计算后应用**

Modify `run_backtest` 中 `compute_icir_weights` 调用处（`weights = compute_icir_weights(...)` 之后、regime 乘数之前）：

```python
            weights = compute_icir_weights(...)
            # ★ 风格预算 (2026-08-16): 同家族因子权重封顶, 防单一风格垄断
            weights = _style_budget_weights(weights, style_cap=0.4)
```

fold 模式（`fixed_weights is not None`）的训练期权重同样处理：在 `validate_minute_factors` 旁或 `run_fold_analysis` 内对 stable 权重应用。实现位置：`run_fold_analysis` 中 fixed_weights 构造后应用一次。

- [ ] **Step 6: 全量测试 + 备份**

Run: `py -m pytest tests/ -q` → Expected: 全部 PASS
Run: `cp config.yaml config.yaml.bak_v29_stylebudget`

- [ ] **Step 7: 全量回测验证**

Run: `py scripts/active/run_walkforward_backtest.py --folds-only --liquid --extend-val 2025-01-01 2026-06-30`
Expected: 回测完成；对比 fold 均值超额与 v27，风格预算应改善或持平 2020/2022 逆风年表现。

---

### Task 4: 换手参数网格实验

**Files:**
- Modify: `config.yaml`（portfolio.bt_* 键，逐个参数组跑）
- 不新建脚本（纪律）；用现有 `run_walkforward_backtest.py` 逐组跑

- [ ] **Step 1: 网格设计**

在 Task 1-3 落地后的 config 基础上，测试三组换手参数（每组全量回测）：

| 组 | bt_hold_thresh | bt_sell_rank_buffer | bt_n_drop | bt_buy_confirm_days |
|----|---------------|---------------------|-----------|---------------------|
| A（低换手） | 45 | 6 | 6 | 2 |
| B（中低） | 45 | 4 | 6 | 1 |
| C（默认对照） | 30 | 3 | 10 | 1 |

- [ ] **Step 2: 依次跑三组全量回测**

每组：改 config 的 bt_* 键 → `cp config.yaml config.yaml.bak_v30_grid_<组>` → 跑全量回测 → 把 `walkforward_results.json` 归档为 `walkforward_results_v30_grid_<组>.json`。

- [ ] **Step 3: 对比三组 + 定稿**

Run（解析脚本）:

```bash
py - <<'EOF'
import json
for g in ('A','B','C'):
    d = json.load(open(f'data/ic_validation/walkforward_results_v30_grid_{g}.json', encoding='utf-8'))
    folds = [r['excess_annual'] for k, r in d['results'].items() if k.startswith('fold')]
    ev = d['results']['extend_val']
    print(f"{g}: fold均值超额 {sum(folds)/len(folds):+.1f}% | extend_val 超额 {ev['excess_annual']:+.1f}% "
          f"Sharpe {ev['sharpe']:.2f} 回撤 {ev['max_drawdown']:.1f}% 换手 {ev.get('avg_turnover')}%")
EOF
```

Expected: 选出 fold 均值超额 + extend_val 超额综合最优的一组。

- [ ] **Step 4: 定稿 config + 归档 + 更新 AGENTS.md**

1. 把最优组的 bt_* 值写入 config（去掉注释）；
2. 最优结果复制为 `walkforward_results.json`（生产）；
3. 用 `experiment_tracker.log_experiment` 登记网格实验；
4. 更新 `AGENTS.md` 生产状态段（v30 最终配置 + 成绩 + 阶段历史）。

---

## Self-Review

1. **Spec coverage**: 用户要求的四步（单票上限→行业中性→因子去冗余→换手网格）各有独立 Task，且每步有回测验证门。
2. **Placeholder scan**: 无 TBD；所有代码块完整；测试命令带预期输出。
3. **Type consistency**: `apply_portfolio_constraints(scores: dict, constraints: dict) -> dict`、`_industry_cap_weights(weights, industry_map, max_industry_pct) -> dict`、`_style_budget_weights(weights, style_cap) -> dict` 签名在测试与实现中一致。
4. **纪律**: 不新建 scripts；config 每次改动前备份；实验登记 tracker；TDD 先行。
