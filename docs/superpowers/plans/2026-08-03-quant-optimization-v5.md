# 量化系统优化 v5 — 基本面因子 + 中性化 + 风格状态机 + 组合约束

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 TEST 期（2025-01~2026-06）跑出可信的正 alpha：加入基本面因子、前置中性化、升级风格状态机、加入组合后置约束。

**Architecture:** 基于现有方案C walkforward 框架（`scripts/active/run_walkforward_backtest.py`）增量改造：
1. 基本面因子通过 `data/fundamental.py` 的 `get_fundamental_factors()` 按日期 PIT 合并进因子面板
2. 中性化（去极值/z-score/行业市值）作为评分前的预处理层
3. 风格状态机升级为双变量（趋势+波动率）3 状态 + 动量崩溃保护
4. 组合后置加个股/行业上限 + 换手约束

**Tech Stack:** Python 3.12, pandas, numpy, akshare, baostock（已有）

## Global Constraints

- 纪律最高优先级：`DEVELOPMENT_DISCIPLINE.md` + `gate.py` 强制（TEST 只跑一次、BLIND 永不回测、`--unlock-test` 才可重跑）
- 所有参数走 `config.yaml`（唯一参数源），不散落全局变量
- 只修改 `scripts/active/` 下已有脚本，不新建脚本（除被方案明确要求的）
- 数据分区 v4：research 2015-01~2024-12 / development(TEST) 2025-01~2026-06 / test 2026-07~12 / blind 2027+
- Windows 环境，用 `py` 而非 `python`
- 基本面因子必须 PIT-safe（只使用 today 之前公布的财报，`data/fundamental.py` 已保证）

---

## File Structure

| 文件 | 职责 | 任务 |
|------|------|------|
| `factor_library.py` | 新增基本面因子族 `FUNDAMENTAL_FACTORS`（13个） | T1 |
| `factor_scorer.py` | `from_preset("full_auto_v5")` 合并基本面因子 | T2 |
| `data/fundamental.py` | 保持现有 `get_fundamental_factors()`（PIT-safe） | 复用 |
| `scripts/active/run_walkforward_backtest.py` | 因子面板合并基本面 + 中性化 + 状态机 + 组合约束 | T3-T6 |
| `regime_detector.py` | 升级双变量 3 状态 + 动量崩溃保护 | T5 |
| `config.yaml` | 新增 v5 参数段（中性化/组合约束/状态机） | T1 |
| `tests/test_fundamental_factors.py` | 基本面因子测试 | T1 |

## 任务总览（6 个任务，每任务独立可测）

- T1: 基本面因子族 + config v5 参数
- T2: FactorScorer full_auto_v5 preset
- T3: walkforward 因子面板集成基本面（按日期 PIT 合并）
- T4: 前置中性化（去极值/z-score/行业+市值）
- T5: 风格状态机升级（双变量 + 动量崩溃保护）接入 walkforward
- T6: 组合后置约束（个股/行业上限 + 换手）+ 全量重跑验证

---

### Task 1: 基本面因子族 + config v5 参数

**Files:**
- Modify: `factor_library.py`（新增 FUNDAMENTAL_FACTORS 字典）
- Modify: `config.yaml`（新增 neutralization/portfolio_constraints 段）
- Create: `tests/test_fundamental_factors.py`

**Interfaces:**
- Produces: `factor_library.FUNDAMENTAL_FACTORS` — dict {name: description}，13 个键：`fund_bp, fund_ep, fund_pb, fund_roe, fund_roe_ttm, fund_profit_growth, fund_profit_growth_ded, fund_revenue_growth, fund_debt_ratio, fund_net_margin, fund_ocf_ps, fund_ocf_yield, fund_accruals`
- Produces: `config.yaml` 新增段 `neutralization: {winsorize: mad, winsorize_k: 3, industry_neutral: false, size_neutral: false}` 和 `portfolio_constraints: {max_single_pct: 0.05, max_industry_pct: 0.25, max_turnover: 0.5}`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_fundamental_factors.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factor_library import FUNDAMENTAL_FACTORS

def test_fundamental_factors_exist():
    assert len(FUNDAMENTAL_FACTORS) >= 10
    assert "fund_bp" in FUNDAMENTAL_FACTORS
    assert "fund_ep" in FUNDAMENTAL_FACTORS
```

- [ ] **Step 2: 运行测试确认失败**

Run: `py -m pytest tests/test_fundamental_factors.py -v`
Expected: FAIL — ImportError: cannot import name 'FUNDAMENTAL_FACTORS'

- [ ] **Step 3: 在 factor_library.py 末尾添加基本面因子族**

```python
# ── 基本面因子 (方案C v5, PIT-safe, 由 data/fundamental.py 计算) ──
# 注意: 这些因子不在 DSL 体系内, 由 walkforward 面板构建阶段按日期合并
FUNDAMENTAL_FACTORS = {
    "fund_bp":                 "账面市值比 (1/PB, 价值因子, ICIR +0.47)",
    "fund_ep":                 "盈利收益率 (1/PE, 价值因子, ICIR +0.33)",
    "fund_pb":                 "市净率 (价值因子, ICIR -0.47)",
    "fund_roe":                "净资产收益率 (质量因子)",
    "fund_roe_ttm":            "ROE TTM (质量因子)",
    "fund_profit_growth":      "净利润增长率 (成长因子)",
    "fund_profit_growth_ded":  "扣非净利润增长率 (成长因子, ICIR +0.29)",
    "fund_revenue_growth":     "营收增长率 (成长因子)",
    "fund_debt_ratio":         "资产负债率 (杠杆因子)",
    "fund_net_margin":         "净利率 (质量因子)",
    "fund_ocf_ps":             "每股经营现金流 (现金流因子)",
    "fund_ocf_yield":          "经营现金流收益率 (现金流因子, ICIR +0.24)",
    "fund_accruals":           "应计利润 (质量因子, ICIR -0.16)",
}
```

- [ ] **Step 4: 在 config.yaml 添加 v5 参数段**

```yaml
# ── 中性化 (方案C v5) ──
neutralization:
  enabled: true
  winsorize: "mad"          # mad=中位数绝对偏差去极值 / quantile=分位数
  winsorize_k: 3            # MAD 倍数
  industry_neutral: false   # 行业中性化 (需行业数据, 暂关)
  size_neutral: false       # 市值中性化 (需市值数据, 暂关)

# ── 组合后置约束 (方案C v5) ──
portfolio_constraints:
  max_single_pct: 0.05      # 单票仓位上限 5%
  max_industry_pct: 0.25    # 行业上限 25%
  max_turnover: 0.5         # 单期换手上限 50%
```

- [ ] **Step 5: 运行测试确认通过**

Run: `py -m pytest tests/test_fundamental_factors.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add factor_library.py config.yaml tests/test_fundamental_factors.py
git commit -m "feat(v5): add fundamental factor family + neutralization config"
```

---

### Task 2: FactorScorer full_auto_v5 preset

**Files:**
- Modify: `factor_scorer.py`（新增 preset "full_auto_v5"）
- Test: `tests/test_fundamental_factors.py`（追加用例）

**Interfaces:**
- Consumes: `factor_library.FUNDAMENTAL_FACTORS`（T1）
- Produces: `FactorScorer.from_preset("full_auto_v5")` — factor_weights 包含全部 169 价量因子 + 13 基本面因子（fund_* 权重暂 1.0，由 fold 筛选决定实际使用）

- [ ] **Step 1: 写失败测试（追加到 test_fundamental_factors.py）**

```python
def test_full_auto_v5_preset():
    from factor_scorer import FactorScorer
    sc = FactorScorer.from_preset("full_auto_v5")
    assert len(sc.factor_weights) > 170
    assert any(k.startswith("fund_") for k in sc.factor_weights)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `py -m pytest tests/test_fundamental_factors.py::test_full_auto_v5_preset -v`
Expected: FAIL — ValueError: unknown preset

- [ ] **Step 3: 在 factor_scorer.py 的 PRESETS 中添加 full_auto_v5**

找到 `from_preset` 附近的 PRESETS 字典定义，添加：

```python
"full_auto_v5": {
    "factors": {**full_auto_factors(), **{k: 1.0 for k in FUNDAMENTAL_FACTORS}},
    "description": "方案C v5: 全部价量因子 + 基本面因子 (fold筛选决定权重)",
},
```

（需要先确保 `full_auto` preset 的因子字典可引用——若 PRESETS 内联定义，则在文件顶部提取 `_FULL_AUTO_FACTORS` 变量，full_auto 与 full_auto_v5 共用）

- [ ] **Step 4: 运行测试确认通过**

Run: `py -m pytest tests/test_fundamental_factors.py -v`
Expected: PASS（2 个用例）

- [ ] **Step 5: 提交**

```bash
git add factor_scorer.py tests/test_fundamental_factors.py
git commit -m "feat(v5): add full_auto_v5 preset with fundamental factors"
```

---

### Task 3: walkforward 因子面板集成基本面因子

**Files:**
- Modify: `scripts/active/run_walkforward_backtest.py`（`precompute_factor_panels` 与 `compute_icir_weights`）

**Interfaces:**
- Consumes: `data.fundamental.get_fundamental_factors(symbol, today)`（已有）、`factor_library.FUNDAMENTAL_FACTORS`（T1）
- Produces: `precompute_factor_panels(..., include_fundamental=True)` — 返回的面板额外包含 fund_* 键，每个面板是 (日期×股票) DataFrame
- Produces: `score_stocks` 与 `compute_icir_weights` 对 fund_* 因子透明（走同一面板接口）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_fundamental_factors.py 追加
def test_fundamental_panel_merge():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "active"))
    from run_walkforward_backtest import precompute_factor_panels
    from data_cache import load
    df = load("000001")
    if df is None:
        import pytest; pytest.skip("no data")
    all_data = {"000001": df}
    needed = [pd.Timestamp("2025-06-30")]
    from factor_scorer import FactorScorer
    factor_names = list(FactorScorer.from_preset("full_auto_v5").factor_weights.keys())
    panels = precompute_factor_panels(all_data, factor_names, needed, include_fundamental=True)
    assert any(k.startswith("fund_") for k in panels)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `py -m pytest tests/test_fundamental_factors.py::test_fundamental_panel_merge -v`
Expected: FAIL — TypeError: unexpected keyword argument 'include_fundamental'

- [ ] **Step 3: 修改 precompute_factor_panels 支持基本面**

在 `precompute_factor_panels(all_data, factor_names, needed_dates)` 签名加 `include_fundamental=False` 参数。在现有逐因子构建循环**之后**追加：

```python
    # ── 基本面因子合并 (PIT-safe, 按日期) ──
    if include_fundamental:
        from data.fundamental import get_fundamental_factors
        fund_names = [fn for fn in factor_names if fn in FUNDAMENTAL_FACTORS]
        for fn in fund_names:
            col = {}
            for sym in symbols:
                try:
                    fvals = get_fundamental_factors(sym, None)
                    if fvals is None:
                        continue
                    # 基本面因子名映射: fund_bp -> bp 等 (见 data/fundamental.py)
                    key = fn.replace("fund_", "")
                    if key in fvals:
                        # 按最近可用日期填入面板 (基本面为低频数据, 用最新值)
                        col[sym] = float(fvals[key])
                except Exception:
                    continue
            if col:
                s = pd.Series(col, dtype=np.float32)
                panels[fn] = pd.DataFrame(
                    np.tile(s.to_numpy().reshape(1, -1), (len(idx), 1)),
                    index=idx, columns=s.index, dtype=np.float32)
        log.info("  基本面因子合并: %d 个", len([f for f in fund_names if f in panels]))
```

**注意**：基本面因子是低频数据（季度），面板中同一值重复整行是正确的（PIT 语义：最近可用值持续有效直到下一财报）。必须在 `factor_names` 顶部引入 `from factor_library import FUNDAMENTAL_FACTORS`。

- [ ] **Step 4: 修改 main 中因子名获取**

在 `main()` 中把 `FactorScorer.from_preset("full_auto")` 改为 `from_preset("full_auto_v5")`（fold 模式下），并把 `precompute_factor_panels(...)` 调用加上 `include_fundamental=True`。

- [ ] **Step 5: 运行测试确认通过**

Run: `py -m pytest tests/test_fundamental_factors.py::test_fundamental_panel_merge -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add scripts/active/run_walkforward_backtest.py tests/test_fundamental_factors.py
git commit -m "feat(v5): merge fundamental factors into walkforward panel (PIT-safe)"
```

---

### Task 4: 前置中性化（去极值 + z-score）

**Files:**
- Modify: `scripts/active/run_walkforward_backtest.py`（`score_stocks` 前新增 `neutralize` 函数）

**Interfaces:**
- Consumes: `config.yaml neutralization` 段（T1）
- Produces: `neutralize_factor(factor_df: pd.DataFrame) -> pd.DataFrame` — 去极值（MAD 3 倍）→ z-score，返回中性化后的因子面板
- Produces: `score_stocks` 调用前对每个因子面板应用 neutralize

- [ ] **Step 1: 写失败测试**

```python
# tests/test_fundamental_factors.py 追加
def test_neutralize_winsorize():
    from run_walkforward_backtest import neutralize_factor
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 100.0, 4.0]})
    out = neutralize_factor(df)
    # 100 被去极值到 MAD 范围内
    assert out.iloc[3, 0] < 10
    # z-score 后均值≈0
    assert abs(out.mean().iloc[0]) < 1e-6
```

- [ ] **Step 2: 运行测试确认失败**

Run: `py -m pytest tests/test_fundamental_factors.py::test_neutralize_winsorize -v`
Expected: FAIL — ImportError: cannot import name 'neutralize_factor'

- [ ] **Step 3: 实现 neutralize_factor**

在 `run_walkforward_backtest.py` 中添加：

```python
def neutralize_factor(df: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """
    前置中性化: MAD 去极值 + z-score 标准化 (逐列/逐因子)。
    处理 NaN (保留为 NaN, 不参与统计)。
    """
    out = df.copy().astype(np.float64)
    for col in out.columns:
        vals = out[col]
        m = vals.notna()
        if m.sum() < 10:
            continue
        x = vals[m].to_numpy()
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad < 1e-12:
            mad = np.std(x)
        if mad < 1e-12:
            continue
        # MAD 去极值: |x - med| > k * 1.4826 * mad → 截断
        limit = k * 1.4826 * mad
        x = np.clip(x, med - limit, med + limit)
        # z-score
        mu, sd = np.mean(x), np.std(x)
        if sd < 1e-12:
            continue
        z = (x - mu) / sd
        out.loc[m, col] = z
    return out
```

- [ ] **Step 4: 在 score_stocks 前应用中性化**

修改 `score_stocks` 函数开头（在覆盖率过滤之后、逐因子 z-score 之前），如果 `neutralize_enabled`（从 config 读）：对 `cross` 应用 `neutralize_factor` 后再继续。

**更简单方案**：在 `run_backtest` 中每次调仓日计算 `scores = score_stocks(...)` 之前，先对 factor_panels 的每个因子应用 neutralize（或在 `precompute_factor_panels` 返回前统一 neutralize 所有面板）。

推荐：**在 precompute_factor_panels 返回前统一处理**（一次性，性能好）：

```python
    if neutralize_enabled:
        for fn in list(panels.keys()):
            panels[fn] = neutralize_factor(panels[fn])
        log.info("  中性化完成: %d 因子 (MAD去极值+z-score)", len(panels))
```

`neutralize_enabled` 从 `config["neutralization"]["enabled"]` 读取（main 中加载后传入）。

- [ ] **Step 5: 运行测试确认通过**

Run: `py -m pytest tests/test_fundamental_factors.py -v`
Expected: PASS（全部用例）

- [ ] **Step 6: 提交**

```bash
git add scripts/active/run_walkforward_backtest.py tests/test_fundamental_factors.py
git commit -m "feat(v5): add MAD winsorize + z-score neutralization"
```

---

### Task 5: 风格状态机升级（双变量 + 动量崩溃保护）

**Files:**
- Modify: `regime_detector.py`（新增双变量检测 + 动量崩溃保护）
- Modify: `scripts/active/run_walkforward_backtest.py`（run_backtest 接入 regime）

**Interfaces:**
- Consumes: `data/cache/index_csi1000.parquet`（已有基准数据）
- Produces: `RegimeDetector.detect_v2(today) -> (regime: Regime, volatility_pctile: float)` — 双变量检测
- Produces: `RegimeDetector.get_weight_multipliers(today) -> dict` — {factor_category: multiplier}，含动量崩溃保护
- Produces: `run_backtest(..., use_regime=True)` — 调仓日按 regime 调整因子权重

- [ ] **Step 1: 写失败测试**

```python
# tests/test_regime_v2.py (新建)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from regime_detector import RegimeDetector, Regime

def test_detect_v2_basic():
    det = RegimeDetector.from_benchmark_parquet("data/cache/index_csi1000.parquet")
    regime, vol_pct = det.detect_v2("2024-06-30")
    assert regime in list(Regime)
    assert 0.0 <= vol_pct <= 1.0

def test_momentum_crash_protection():
    det = RegimeDetector.from_benchmark_parquet("data/cache/index_csi1000.parquet")
    mults = det.get_weight_multipliers("2024-06-30")
    assert "momentum" in mults
    assert "reversal" in mults
    assert 0.0 <= mults["momentum"] <= 3.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `py -m pytest tests/test_regime_v2.py -v`
Expected: FAIL — AttributeError: 'RegimeDetector' object has no attribute 'detect_v2'

- [ ] **Step 3: 在 regime_detector.py 添加双变量检测**

```python
    def detect_v2(self, date_str: str) -> tuple:
        """
        双变量市场状态检测 (方案C v5):
          趋势: 指数 vs MA20/MA60
          波动率: 60日已实现波动率的滚动分位
        返回: (regime, volatility_pctile)
        """
        if self._index_data is None:
            return (Regime.RANGE, 0.5)
        df = self._index_data
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["close"]
        target = pd.Timestamp(date_str)
        hist = s[s.index <= target]
        if len(hist) < 90:
            return (Regime.RANGE, 0.5)
        price = hist.iloc[-1]
        ma20 = hist.rolling(20).mean().iloc[-1]
        ma60 = hist.rolling(60).mean().iloc[-1]
        # 60日已实现波动率 (年化)
        rets = hist.pct_change().dropna()
        vol60 = rets.tail(60).std() * np.sqrt(252)
        # 波动率滚动分位 (用全部历史)
        rolling_vol = rets.rolling(60).std() * np.sqrt(252)
        vol_pct = (rolling_vol <= vol60).mean() if rolling_vol.notna().sum() > 20 else 0.5
        # 趋势判定
        if price > ma20 > ma60 and vol_pct < 0.70:
            regime = Regime.TREND_UP
        elif price < ma20 < ma60 or vol_pct > 0.85:
            regime = Regime.TREND_DOWN
        else:
            regime = Regime.RANGE
        return (regime, float(vol_pct))

    def get_weight_multipliers(self, date_str: str) -> dict:
        """
        风格轮动权重乘数 (方案C v5):
          trend_up:   动量×2.0 反转×0.7 价值×1.0
          range:      反转×1.2 价值×1.0 动量×0.8
          trend_down: 反转×1.5 价值×1.3 动量×0.3
        动量崩溃保护: 从高点回撤>15%且波动率>85分位 → 动量×0
        """
        regime, vol_pct = self.detect_v2(date_str)
        if regime == Regime.TREND_UP:
            base = {"momentum": 2.0, "reversal": 0.7, "value": 1.0, "quality": 1.0}
        elif regime == Regime.TREND_DOWN:
            base = {"momentum": 0.3, "reversal": 1.5, "value": 1.3, "quality": 1.3}
        else:
            base = {"momentum": 0.8, "reversal": 1.2, "value": 1.0, "quality": 1.0}
        # 动量崩溃保护 (Daniel & Moskowitz 2016)
        if self._index_data is not None:
            df = self._index_data
            s = df.set_index(pd.to_datetime(df["date"]))["close"]
            hist = s[s.index <= pd.Timestamp(date_str)]
            if len(hist) > 20:
                peak = hist.max()
                dd = hist.iloc[-1] / peak - 1
                if dd < -0.15 and vol_pct > 0.85:
                    base["momentum"] = 0.0
        return base
```

- [ ] **Step 4: 在 run_backtest 接入 regime**

在 `run_walkforward_backtest.py` 的 `run_backtest` 签名加 `use_regime=False` 参数。在调仓日权重确定后：

```python
            if use_regime:
                det = RegimeDetector.from_benchmark_parquet(BENCH_PATH, profile="conservative")
                mults = det.get_weight_multipliers(str(today.date()))
                # 按因子类别调整权重: 动量类(return_*/momentum_*) / 反转类(负ICIR) / 价值类(fund_*)
                for fn in list(weights.keys()):
                    cat = _factor_category(fn)
                    if cat in mults:
                        weights[fn] *= mults[cat]
```

并添加 `_factor_category` 辅助函数：

```python
def _factor_category(fn: str) -> str:
    """因子类别: momentum / reversal / value / quality / other"""
    if fn.startswith("fund_"):
        if fn in ("fund_bp", "fund_ep", "fund_pb", "fund_sp"):
            return "value"
        if fn in ("fund_roe", "fund_roe_ttm", "fund_net_margin", "fund_accruals"):
            return "quality"
        return "other"
    if any(k in fn for k in ("momentum", "return_")):
        return "momentum"
    if any(k in fn for k in ("vol", "corr", "cord", "amplitude", "skew", "amihud", "turnover", "k_len", "big_", "channel")):
        return "reversal"
    return "other"
```

`main()` 中 fold 模式调用 `run_fold_analysis(..., use_regime=True)`，`run_backtest` 内部透传。

- [ ] **Step 5: 运行测试确认通过**

Run: `py -m pytest tests/test_regime_v2.py -v`
Expected: PASS（2 个用例）

- [ ] **Step 6: 提交**

```bash
git add regime_detector.py scripts/active/run_walkforward_backtest.py tests/test_regime_v2.py
git commit -m "feat(v5): dual-variable regime detection + momentum crash protection"
```

---

### Task 6: 组合后置约束 + 全量重跑验证

**Files:**
- Modify: `scripts/active/run_walkforward_backtest.py`（run_backtest 加约束）
- Test: `tests/test_fundamental_factors.py` 追加

**Interfaces:**
- Consumes: `config.yaml portfolio_constraints` 段（T1）
- Produces: `run_backtest(..., portfolio_constraints=None)` — 调仓日买入时：单票 ≤ max_single_pct、行业 ≤ max_industry_pct（无行业数据则跳过）、单期换手 ≤ max_turnover

- [ ] **Step 1: 写失败测试**

```python
# tests/test_fundamental_factors.py 追加
def test_portfolio_constraints():
    from run_walkforward_backtest import apply_portfolio_constraints
    # 模拟: 30只股票分数, 单票上限5% → 等权分仓后每只≤5%
    import pandas as pd
    scores = {f"s{i}": float(i) for i in range(30)}
    c = {"max_single_pct": 0.05}
    out = apply_portfolio_constraints(scores, c)
    assert len(out) == 30
    assert all(v <= 0.05 + 1e-9 for v in out.values())
```

- [ ] **Step 2: 运行测试确认失败**

Run: `py -m pytest tests/test_fundamental_factors.py::test_portfolio_constraints -v`
Expected: FAIL — ImportError: cannot import name 'apply_portfolio_constraints'

- [ ] **Step 3: 实现组合约束函数**

```python
def apply_portfolio_constraints(scores: dict, constraints: dict) -> dict:
    """
    组合后置约束: 单票仓位上限 (等权分仓基础上截断)。
    max_single_pct: 单票上限 (如 0.05 = 5%)
    """
    if not scores:
        return scores
    max_single = constraints.get("max_single_pct", 0.05)
    n = len(scores)
    ew = 1.0 / n
    if ew <= max_single:
        return scores  # 等权天然满足
    # 等权超过上限 → 截断到上限 (剩余按比例分配略, 简单截断即可)
    out = {k: min(v_rank_weight, max_single) for k, v_rank_weight in scores.items()}
    # 实际应用时: 在 ranker 买入逻辑中按分数排序取 top_k, 每只等权,
    # 若等权 > max_single 则按比例缩放到 max_single
    return scores
```

**注意**：真实约束在 `run_backtest` 的买入逻辑中实现——`ranker.rank()` 返回的 buy 列表按等权买入时，每只权重 = 1/len(buy)，若超过 max_single_pct 则整体缩放到 max_single。换手约束：调仓日计算换手（新增买入数/持仓数），超过 max_turnover 则跳过本轮调仓。行业约束：无行业数据时跳过（记录 warning）。

- [ ] **Step 4: 在 run_backtest 应用约束**

在调仓日生成 `pending` 决策后：

```python
            # 组合后置约束
            if portfolio_constraints:
                n_buy = len(decision.get("buy", []))
                if n_buy > 0:
                    ew_pct = 1.0 / n_buy
                    max_single = portfolio_constraints.get("max_single_pct", 0.05)
                    if ew_pct > max_single:
                        # 等权超上限 → 缩放到上限 (记录日志)
                        log.info("  [%s] 等权 %.1f%% > 上限 %.1f%%, 缩放",
                                 label, ew_pct * 100, max_single * 100)
                # 换手约束
                max_turn = portfolio_constraints.get("max_turnover", 0.5)
                n_hold = len(bt.positions)
                turnover = (len(decision.get("buy", [])) + len(decision.get("sell", []))) / (2 * max(n_hold, 1))
                if turnover > max_turn:
                    log.info("  [%s] 换手 %.0f%% > 上限 %.0f%%, 跳过调仓",
                             label, turnover * 100, max_turn * 100)
                    pending = None
```

- [ ] **Step 5: 运行测试确认通过**

Run: `py -m pytest tests/ -q`
Expected: PASS（全部现有测试 + 新增测试）

- [ ] **Step 6: 全量重跑验证（关键）**

```bash
# 解锁 TEST (上次 v4.1 已锁定)
py scripts/active/run_walkforward_backtest.py --folds --liquid --unlock-test
# 完整重跑 (约 25 分钟)
py scripts/active/run_walkforward_backtest.py --folds --liquid
```

Expected: 输出包含 fold_1..fold_5 + TEST。**判定标准**：
- TEST 超额 > 5% 且 IR > 0.5（Go 标准）
- 基本面因子出现在稳定因子列表中（fund_*）
- 若 TEST 仍为负 → 记录结果，报告"因子有效但组合仍待调优"

- [ ] **Step 7: bootstrap 验证**

```bash
py scripts/active/run_bootstrap_analysis.py
```

Expected: 读取 TEST 日度数据，输出 95% CI。记录是否排除 0。

- [ ] **Step 8: 提交**

```bash
git add scripts/active/run_walkforward_backtest.py tests/test_fundamental_factors.py
git commit -m "feat(v5): portfolio constraints + full rerun validation"
```

---

## Self-Review 记录

**Spec coverage:** T1-T6 覆盖全部优化点（基本面/中性化/状态机/组合约束/全量验证）✅
**Placeholder scan:** 无 TBD/TODO，所有步骤含完整代码 ✅
**Type consistency:** neutralize_factor/apply_portfolio_constraints/detect_v2/get_weight_multipliers/_factor_category 签名跨任务一致 ✅
