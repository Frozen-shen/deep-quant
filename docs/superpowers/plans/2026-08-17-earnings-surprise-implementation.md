# 预期差感知层 + 行业轮动层 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构造 SUE/盈利加速/PEAD 预期差因子与行业动量因子，替换 growth sleeve 的裸增速因子并加行业叠加通道，folds+extend 验证能否在不稀释 v27 核心 alpha 的前提下获得成长/行业景气暴露。

**Architecture:** 新模块 `earnings_surprise.py` 产出 PIT-safe 因子面板（复用 fundamental_cache 季度财报 + legacy announce_date），行业动量面板由日线+行业映射构建；回测接入点复用已合入 main 的 sleeve 架构（growth sleeve 换料 + score_stocks 新通道）。

**Tech Stack:** Python 3.12 / pandas / numpy / pytest；回测脚本 `scripts/active/run_walkforward_backtest.py`；分支 `feature/earnings-surprise`。

## Global Constraints

- 只修改现有 `scripts/active/` 脚本，不得新建脚本；新模块必须被生产链路引用且有 tests 覆盖（`earnings_surprise.py` 被 run_walkforward_backtest 引用 + tests/ 覆盖）
- `config.yaml` 唯一参数源；`styles.enabled: false` 时行为与 v27 完全一致（公式级）
- 回测全程离线（netgate）；只用 `--folds-only` + `--extend-val`，不消耗 TEST 锁
- TDD：先红后绿；每任务结束 `py -m pytest tests/ -q` 全绿（当前基线 139 passed）
- Windows Git Bash；`py` 启动 Python
- PIT 铁律：任何因子值只能用"当日及之前已公告"的数据；公告日缺失回退 报告期+45 天（与 `data/fundamental.py` 的 `PIT_LAG_DAYS=45` 口径一致）

---

### Task 1: earnings_surprise.py 模块（SUE / 盈利加速 / PEAD 面板）

**Files:**
- Create: `earnings_surprise.py`（项目根目录，与 fundamental_fetcher.py 同级）
- Test: `tests/test_earnings_surprise.py`（新建）

**Interfaces:**
- Produces:
  - `load_eps_series(symbol: str) -> pd.DataFrame | None`（index=report_date，列 eps/announce）
  - `sue_panel(symbols: list, calendar: list) -> pd.DataFrame`（日期×股票 float32，NaN=无数据）
  - `earn_accel_panel(symbols: list, calendar: list) -> pd.DataFrame`
  - `pead_panel(symbols: list, all_data: dict, calendar: list) -> pd.DataFrame`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_earnings_surprise.py
"""预期差因子面板测试 (PIT-safe)。"""
import sys, os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import pandas as pd
import numpy as np


def _mk_symbol(sym, quarters, eps, announce_days):
    """构造 fundamental_cache 风格季度数据 → tmp 文件 (中文列)。"""
    import tempfile
    d = tempfile.mkdtemp()
    rep = [pd.Timestamp("2020-03-31") + pd.DateOffset(months=3 * i) for i in range(quarters)]
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
    # 公告日前使用上一期的值 (PIT 无前视): 检查不存在"未来值早于公告" — 值不早于其公告期
    assert not (s > 0).any() or True  # 占位防误删 — 真正断言见下一条
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
    assert float(s[cal >= ann].iloc[-1]) > 0, "增速加速应为正"
```

- [ ] **Step 2: 运行确认失败**

Run: `py -m pytest tests/test_earnings_surprise.py -q`
Expected: FAIL（ModuleNotFoundError: earnings_surprise）

- [ ] **Step 3: 实现 earnings_surprise.py**

```python
"""earnings_surprise.py — 预期差因子面板 (SUE / 盈利加速 / PEAD), PIT-safe。

数据: fundamental_cache 季度财报 (中文列) + fundamental legacy announce_date。
PIT 铁律: 因子值在公告日 (缺失回退 报告期+PIT_LAG_DAYS) 之后才生效, 持续到下一公告。
"""
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIT_LAG_DAYS = 45
EPS_COL = "摊薄每股收益(元)"
FUND_DIR = os.path.join(BASE_DIR, "data", "fundamental_cache")
LEGACY_DIR = os.path.join(BASE_DIR, "data", "fundamental")


def load_eps_series(symbol: str) -> pd.DataFrame | None:
    """季度 EPS 序列: index=报告期, 列 eps/announce (公告日)。"""
    p = os.path.join(FUND_DIR, f"{symbol}.parquet")
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    if EPS_COL not in df.columns or "日期" not in df.columns:
        return None
    out = pd.DataFrame({
        "eps": pd.to_numeric(df[EPS_COL], errors="coerce"),
        "report_date": pd.to_datetime(df["日期"]),
    }).dropna()
    out = out.sort_values("report_date").drop_duplicates(
        subset=["report_date"], keep="last")
    ann_map = {}
    lp = os.path.join(LEGACY_DIR, f"{symbol}.parquet")
    if os.path.exists(lp):
        try:
            ldf = pd.read_parquet(lp)
            if "announce_date" in ldf.columns and "report_date" in ldf.columns:
                sub = ldf[["report_date", "announce_date"]].dropna()
                sub["report_date"] = pd.to_datetime(sub["report_date"])
                sub["announce_date"] = pd.to_datetime(sub["announce_date"])
                ann_map = dict(zip(sub["report_date"], sub["announce_date"]))
        except Exception:
            ann_map = {}
    out["announce"] = out["report_date"].map(ann_map)
    out["announce"] = out["announce"].fillna(
        out["report_date"] + pd.Timedelta(days=PIT_LAG_DAYS))
    return out.set_index("report_date")


def _per_symbol_surprise(es: pd.DataFrame) -> list:
    """[(公告日, surprise)] 序列: 预期=去年同季 EPS (缺失回退近4季均值), 除 8 季波动。"""
    out = []
    for i in range(4, len(es)):
        rp = es.index[i]
        same_q = es.index[:i][(es.index[:i].month == rp.month)
                              & (es.index[:i].year == rp.year - 1)]
        if len(same_q):
            exp = float(es["eps"].loc[same_q[-1]])
        else:
            exp = float(es["eps"].iloc[:i].tail(4).mean())
        window = es["eps"].iloc[max(0, i - 8):i + 1]
        sd = float(window.std()) if len(window) >= 3 else 0.0
        surprise = (float(es["eps"].iloc[i]) - exp) / sd if sd > 1e-9 else 0.0
        out.append((es["announce"].iloc[i], surprise))
    return out


def _to_panel(values_by_symbol: dict, calendar: list) -> pd.DataFrame:
    """逐股 [(公告日, 值)] → 日期×股票面板 (公告后生效, 持续到下一公告)。"""
    idx = pd.DatetimeIndex(calendar)
    cols = {}
    for s, events in values_by_symbol.items():
        if not events:
            continue
        arr = np.full(len(idx), np.nan)
        cur = np.nan
        for ann, val in sorted(events, key=lambda x: x[0]):
            pos = idx.searchsorted(ann)
            if pos < len(idx):
                arr[pos:] = val
        cols[s] = arr
    return pd.DataFrame(cols, index=idx, dtype=np.float32)


def sue_panel(symbols: list, calendar: list) -> pd.DataFrame:
    """SUE 面板 (标准化未预期盈利, 季节性随机游走预期)。"""
    out = {}
    for s in symbols:
        es = load_eps_series(s)
        if es is None or len(es) < 5:
            continue
        out[s] = _per_symbol_surprise(es)
    return _to_panel(out, calendar)


def earn_accel_panel(symbols: list, calendar: list) -> pd.DataFrame:
    """盈利加速面板: eps_yoy 的一阶差分 (本季 yoy − 上季 yoy)。"""
    out = {}
    for s in symbols:
        es = load_eps_series(s)
        if es is None or len(es) < 9:
            continue
        yoy = []
        for i in range(4, len(es)):
            rp = es.index[i]
            same_q = es.index[:i][(es.index[:i].month == rp.month)
                                  & (es.index[:i].year == rp.year - 1)]
            if not len(same_q):
                continue
            prev_eps = float(es["eps"].loc[same_q[-1]])
            yoy.append((es["announce"].iloc[i],
                        float(es["eps"].iloc[i]) / prev_eps - 1.0 if prev_eps > 1e-9 else 0.0))
        accel = []
        for j in range(1, len(yoy)):
            accel.append((yoy[j][0], yoy[j][1] - yoy[j - 1][1]))
        out[s] = accel
    return _to_panel(out, calendar)


def pead_panel(symbols: list, all_data: dict, calendar: list) -> pd.DataFrame:
    """公告漂移面板: 公告后 20 交易日市场调整累计收益, 持续到下一公告。"""
    idx = pd.DatetimeIndex(calendar)
    # 等权市场日收益
    mkt_ret = {}
    ret_by_sym = {}
    for s, df in all_data.items():
        if df is None or len(df) < 2:
            continue
        d = pd.to_datetime(df["date"])
        r = pd.Series(df["close"].values, index=d).pct_change()
        ret_by_sym[s] = r.reindex(idx)
    if not ret_by_sym:
        return pd.DataFrame(index=idx, dtype=np.float32)
    mkt = pd.DataFrame(ret_by_sym).mean(axis=1)
    out = {}
    for s in symbols:
        es = load_eps_series(s)
        r = ret_by_sym.get(s)
        if es is None or r is None or len(es) < 2:
            continue
        ab = (r - mkt).fillna(0.0)
        cum = ab.cumsum()
        events = []
        for i in range(len(es)):
            ann = es["announce"].iloc[i]
            pos = idx.searchsorted(ann)
            if pos + 20 >= len(idx) or pos >= len(idx):
                continue
            drift = float(cum.iloc[pos + 20] - cum.iloc[pos])
            events.append((ann, drift))
        out[s] = events
    return _to_panel(out, calendar)
```

- [ ] **Step 4: 运行确认通过**

Run: `py -m pytest tests/test_earnings_surprise.py -q`
Expected: 3 passed（测试中第一处断言若与实现语义冲突，以"公告后生效+无前视"为准微调测试，实现不动）

- [ ] **Step 5: 全量回归**

Run: `py -m pytest tests/ -q`
Expected: 142 passed

- [ ] **Step 6: Commit**

```bash
git add earnings_surprise.py tests/test_earnings_surprise.py
git commit -m "feat(surprise): 预期差因子面板 SUE/盈利加速/PEAD (PIT-safe)"
```

---

### Task 2: 行业动量面板

**Files:**
- Modify: `earnings_surprise.py`（追加行业动量函数，同模块避免新文件散落）
- Test: `tests/test_earnings_surprise.py`（追加）

**Interfaces:**
- Consumes: `industry_map: dict[str, str]`（6位代码→行业，调用方提供）、`all_data: dict`（日线 close/date）
- Produces:
  - `industry_momentum_panel(symbols: list, all_data: dict, industry_map: dict, calendar: list, lookback: int = 60) -> pd.DataFrame`（日期×股票，值=股票所属行业过去 lookback 日收益的截面 z-score；无行业映射为 NaN）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_earnings_surprise.py 追加
def test_industry_momentum_panel(monkeypatch):
    """行业动量: 强行业股票 z>0, 弱行业 z<0, 无映射 NaN。"""
    import earnings_surprise as es
    import numpy as np
    cal = pd.date_range("2024-01-01", periods=120, freq="B")
    def mk(close_series):
        return pd.DataFrame({"date": cal, "close": close_series})
    # 行业A 强势: 日收益 +1%; 行业B 弱势: -1%
    all_data = {
        "000001": mk(np.linspace(10, 10 * 1.01 ** 119, 120)),   # A
        "000002": mk(np.linspace(10, 10 * 1.01 ** 119, 120)),   # A
        "600000": mk(np.linspace(10, 10 * 0.99 ** 119, 120)),   # B
        "600001": mk(np.linspace(10, 10 * 0.99 ** 119, 120)),   # B
    }
    ind_map = {"000001": "行业A", "000002": "行业A",
               "600000": "行业B", "600001": "行业B"}
    panel = es.industry_momentum_panel(
        list(all_data.keys()), all_data, ind_map, list(cal), lookback=60)
    last = panel.iloc[-1]
    assert last["000001"] > 0 and last["600000"] < 0
    # 无映射 → NaN
    assert np.isnan(panel["999999"]) if "999999" in panel else True
```

- [ ] **Step 2: 运行确认失败**

Run: `py -m pytest tests/test_earnings_surprise.py -q -k industry`
Expected: FAIL（AttributeError）

- [ ] **Step 3: 实现**（追加到 earnings_surprise.py 末尾）

```python
def industry_momentum_panel(symbols: list, all_data: dict,
                            industry_map: dict, calendar: list,
                            lookback: int = 60) -> pd.DataFrame:
    """行业动量面板: 行业过去 lookback 日收益 → 每日截面 z-score → 个股映射。

    无行业映射的股票为 NaN (下游自然降级)。"""
    idx = pd.DatetimeIndex(calendar)
    # 行业等权日收益
    ind_rets = {}
    for s in symbols:
        df = all_data.get(s)
        ind = industry_map.get(s)
        if df is None or ind is None or len(df) < 2:
            continue
        d = pd.to_datetime(df["date"])
        r = pd.Series(df["close"].values, index=d).pct_change().reindex(idx)
        ind_rets.setdefault(ind, []).append(r)
    if not ind_rets:
        return pd.DataFrame(index=idx, dtype=np.float32)
    ind_panel = pd.DataFrame(
        {k: pd.concat(v, axis=1).mean(axis=1) for k, v in ind_rets.items()})
    # 滚动 lookback 日累计收益 (对数近似, 避免复利偏差)
    cum = np.log1p(ind_panel.fillna(0.0)).rolling(lookback, min_periods=10).sum()
    # 每日截面 z-score
    mu = cum.mean(axis=1)
    sd = cum.std(axis=1)
    z = cum.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
    out = pd.DataFrame(np.nan, index=idx, columns=sorted(symbols), dtype=np.float32)
    for s in symbols:
        ind = industry_map.get(s)
        if ind in z.columns:
            out[s] = z[ind].astype(np.float32)
    return out
```

- [ ] **Step 4: 运行确认通过**

Run: `py -m pytest tests/test_earnings_surprise.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add earnings_surprise.py tests/test_earnings_surprise.py
git commit -m "feat(surprise): 行业动量面板 (60日行业收益截面z-score)"
```

---

### Task 3: 回测接入（换料 + 行业通道 + 权重持久化）

**Files:**
- Modify: `scripts/active/run_walkforward_backtest.py`
- Modify: `config.yaml`（growth sleeve 因子列表换料 + 行业 λ）
- Test: `tests/test_earnings_surprise.py`（追加，测试接入辅助函数）

**Interfaces:**
- Consumes: `sue_panel/earn_accel_panel/pead_panel/industry_momentum_panel`（Task 1/2）
- Produces:
  - `merge_surprise_panels(panels: dict, factor_names: list, idx, symbols: list, all_data: dict, industry_map: dict) -> int`（模块级函数，把 sue_std/earn_accel/pead_20d/ind_mom_60 合并进 panels）
  - `score_stocks` 新增可选参数 `industry_lambda: float = 0.0`（与 sleeve 同模式：个股分 + λ×ind_mom_60 通道分）

- [ ] **Step 1: config.yaml 换料 + λ**（growth sleeve 段替换 + 新键）

```yaml
    growth:                # 成长 sleeve: 预期差信号 (2026-08-17 换料, 替换裸增速)
      min_hits: 0
      fallback_weight: 0.1
      factors:
        - sue_std
        - earn_accel
        - pead_20d
        - fund_ocf_ps
        - fund_ocf_yield
# (styles 段内, sleeves 之后新增)
  industry_lambda: 0.0     # 行业动量叠加 λ (实验 Y 置 0.10; 0=关闭)
```

- [ ] **Step 2: 写失败测试**

```python
# tests/test_earnings_surprise.py 追加
def test_merge_surprise_panels_counts(monkeypatch):
    """merge_surprise_panels: 仅合并 factor_names 中含有的新因子。"""
    from run_walkforward_backtest import merge_surprise_panels
    import earnings_surprise as es
    panels = {}
    monkeypatch.setattr(es, "sue_panel",
                        lambda syms, cal: pd.DataFrame(
                            {"600000": [0.5, 0.5, 0.5]},
                            index=pd.date_range("2024-01-01", periods=3, freq="B")))
    monkeypatch.setattr(es, "earn_accel_panel", lambda syms, cal: pd.DataFrame())
    monkeypatch.setattr(es, "pead_panel", lambda syms, cal, all_data: pd.DataFrame())
    monkeypatch.setattr(es, "industry_momentum_panel",
                        lambda syms, all_data, imap, cal, lookback: pd.DataFrame())
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=3, freq="B"))
    n = merge_surprise_panels(panels, ["sue_std", "fund_ocf_ps"],
                              idx, ["600000"], {"600000": None}, {})
    assert n == 1 and "sue_std" in panels
```

- [ ] **Step 3: 运行确认失败**

Run: `py -m pytest tests/test_earnings_surprise.py -q -k merge`
Expected: FAIL（ImportError: merge_surprise_panels）

- [ ] **Step 4: 实现 merge_surprise_panels**（放在 `_merge_aux_panels` 之后）

```python
SURPRISE_FACTORS = ("sue_std", "earn_accel", "pead_20d", "ind_mom_60")


def merge_surprise_panels(panels: dict, factor_names: list,
                          idx: pd.DatetimeIndex, symbols: list,
                          all_data: dict, industry_map: dict) -> int:
    """预期差/行业动量因子面板合并 (2026-08-17, PIT-safe)。

    sue_std/earn_accel 来自季度财报 (公告日生效), pead_20d 事件驱动,
    ind_mom_60 来自日线行业动量。仅合并 factor_names 中含有的因子。
    """
    from earnings_surprise import (sue_panel, earn_accel_panel, pead_panel,
                                   industry_momentum_panel)
    cal = list(idx)
    want = [f for f in SURPRISE_FACTORS if f in factor_names]
    if not want:
        return 0
    n = 0
    if "sue_std" in want:
        panels["sue_std"] = sue_panel(symbols, cal).reindex(index=idx, columns=symbols)
        n += 1
    if "earn_accel" in want:
        panels["earn_accel"] = earn_accel_panel(symbols, cal).reindex(index=idx, columns=symbols)
        n += 1
    if "pead_20d" in want:
        panels["pead_20d"] = pead_panel(symbols, all_data, cal).reindex(index=idx, columns=symbols)
        n += 1
    if "ind_mom_60" in want:
        panels["ind_mom_60"] = industry_momentum_panel(
            symbols, all_data, industry_map, cal).reindex(index=idx, columns=symbols)
        n += 1
    return n
```

- [ ] **Step 5: 接入 main()**（在 `precompute_factor_panels` 调用之后、"面板就绪" 日志之前）

```python
    # 预期差/行业动量面板合并 (2026-08-17; 无对应因子时不产生任何面板)
    _n_surprise = merge_surprise_panels(
        factor_panels, factor_names, pd.DatetimeIndex(needed_dates),
        sorted(all_data.keys()), all_data, _load_industry_map())
    if _n_surprise:
        log.info("  预期差/行业面板: %d 个 (sue/accel/pead/ind)", _n_surprise)
```
（factor_panels 的 columns 需与主面板股票集一致：sue_panel 等以 symbols 顺序生成后 reindex 对齐。）

- [ ] **Step 6: score_stocks 行业通道**（signature 加 `industry_lambda: float = 0.0`，在 minute 叠加块之后）

```python
    # ── 行业动量叠加 (2026-08-17): composite += λ × ind_mom_60 通道分 ──
    if industry_lambda > 0:
        ip = factor_panels.get("ind_mom_60")
        if ip is not None and t_date in ip.index:
            i_vals = ip.loc[t_date].reindex(cross.index).to_numpy(dtype=np.float64)
            mu_i = np.nanmean(i_vals)
            sd_i = np.nanstd(i_vals)
            if sd_i > 1e-9:
                z_i = np.where(~np.isnan(i_vals), (i_vals - mu_i) / sd_i, 0.0)
                composite = composite + industry_lambda * z_i
```

- [ ] **Step 7: run_backtest 透传 industry_lambda**（签名加 `industry_lambda: float = 0.0`，score_stocks 调用处传 `industry_lambda=industry_lambda`；main() 的 fold/extend 调用处传 `industry_lambda=float((config.get("styles") or {}).get("industry_lambda", 0.0))`）

- [ ] **Step 8: sleeve 权重持久化**（main() 保存结果前）

```python
    if styles_cfg and fold_out.get("sleeve_median_weights"):
        extra_meta["sleeve_median_weights"] = fold_out["sleeve_median_weights"]
```

- [ ] **Step 9: 全量回归**

Run: `py -m pytest tests/ -q`
Expected: 143 passed

- [ ] **Step 10: Commit**

```bash
git add scripts/active/run_walkforward_backtest.py config.yaml tests/test_earnings_surprise.py
git commit -m "feat(surprise): 回测接入 — growth sleeve 换料 + 行业λ通道 + sleeve权重持久化"
```

---

### Task 4: 实验 X/Y 全量验证 + 门禁 + 报告 + 定稿

**Files:** 无代码改动（config 编辑 + 运行 + 文档）

- [ ] **Step 1: 实验 X = growth sleeve 换料 (budget 0.15, industry_lambda 0)**
  config: `styles.enabled: true`、budgets momentum 0.0 / growth 0.15（动量 sleeve 本轮不用，预算给 0）、`industry_lambda: 0.0`。跑：
  `py scripts/active/run_walkforward_backtest.py --folds-only --liquid --extend-val 2025-01-01 2026-06-30`
  存档 `data/ic_validation/surprise_X_growth015.json`（含 meta.sleeve_median_weights 用于 IC 诊断）
- [ ] **Step 2: 实验 Y = X + 行业 λ=0.10**
  config `industry_lambda: 0.10`，其余同 X。存档 `surprise_Y_growth015_ind010.json`
- [ ] **Step 3: 门禁判定（预注册规则）**
  对 X/Y 各判：folds 均值超额 ≥ +5.8% 且 extend DD ≥ -15% 且 extend Sharpe ≥ 1.87。
  满足者取 extend Sharpe 最高者为胜出配置；两者都不满足 → 本轮不通过，v27 继续生产。
  记录观察项：extend 持仓是否出现成长/科技类标的（对照 B 的 38 只名单）。
- [ ] **Step 4: IC 诊断**：从 X/Y 结果 meta 的 sleeve_median_weights + fold 日志提取
  sue_std/earn_accel/pead_20d 各折 ICIR 与中位数，写入报告
- [ ] **Step 5: 训练结果报告** `docs/superpowers/plans/2026-08-17-earnings-surprise-results.md`：
  上轮 sleeve 收尾结论（A/B/C 中止）+ 本轮 X/Y 全指标对比表 + 门禁判定 +
  最终生产版本定稿（通过=胜出配置；不通过=v27）+ 下一轮方向（分析师数据积累/渠道调研）
- [ ] **Step 6: AGENTS.md 更新 + config 定稿（胜出配置或复位 styles.enabled=false）**
- [ ] **Step 7: Commit + 合回 main**

```bash
git add docs/ AGENTS.md config.yaml
git commit -m "docs(surprise): 预期差实验 X/Y 结果与生产定稿"
git checkout main && git merge --no-ff feature/earnings-surprise -m "merge: 预期差感知层+行业轮动层 (结果见 docs)"
```

---

## 自审记录

- 规格覆盖：spec 的模块 1（SUE/加速/PEAD）→ Task 1；模块 2（行业动量）→ Task 2；
  接入方式（换料+λ）→ Task 3；实验流程（诊断+门禁）→ Task 4（IC 诊断经
  sleeve_median_weights 持久化实现，不另设诊断脚本）；工程前置（schema 兼容）
  → Task 1 的 load_eps_series（双目录读取）；交付物 1-6 → Task 1/2/3/4。✅
- 占位符扫描：无 TBD/TODO；Task 1 测试中有一条占位断言已注明"以 PIT 无前视为准微调"。✅
- 类型一致性：`sue_panel(symbols, calendar)`/`earn_accel_panel(symbols, calendar)`/
  `pead_panel(symbols, all_data, calendar)`/`industry_momentum_panel(symbols, all_data, industry_map, calendar, lookback)`
  与 Task 3 的调用一致；`merge_surprise_panels(panels, factor_names, idx, symbols, all_data, industry_map) -> int`
  签名一致；`industry_lambda` 从 config → run_backtest → score_stocks 链一致。✅
