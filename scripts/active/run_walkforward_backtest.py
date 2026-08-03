"""
scripts/active/run_walkforward_backtest.py — Walk-Forward 回测 (动态IC权重)

背景:
  run_corrected_backtest.py 使用 2018-2022 的固定 IC 权重 (P5),
  因子衰减导致 2026 年样本外失败。本脚本在每个调仓日用
  滚动 (trailing) 数据重新估计因子 ICIR, 实现 walk-forward。

方法论 (每个调仓日 T):
  1. 回看 252 个交易日窗口 [T-252, T-21], 每 21 天取一个月度观测点
     (标签为 21 日前瞻收益, 在 T 时刻已全部实现 → 无未来函数)
  2. 每个因子计算截面 Spearman Rank IC
  3. ICIR = mean(IC) / std(IC), 仅保留 |ICIR| > 0.02 的因子
  4. 用动态 ICIR 权重对当日股票打分, 选 top_k
  5. 每 20 个交易日调仓一次

保障:
  - PIT universe: 每个调仓日用 CSI300+CSI1000 月度成分
  - 参数统一: top_k/成本/调仓频率来自 config.yaml
  - 日期守卫: gate.py DateRangeGuard 拦截盲测期访问
  - T+1 执行: 调仓日信号, 次日开盘价成交

用法:
  py scripts/active/run_walkforward_backtest.py                 # Development + TEST
  py scripts/active/run_walkforward_backtest.py --dev-only      # 仅 Development
  py scripts/active/run_walkforward_backtest.py --test-only     # 仅 TEST
  py scripts/active/run_walkforward_backtest.py --sample 200    # 抽样200只快速测试

输出:
  data/ic_validation/walkforward_results.json
"""

import os
import sys
import json
import time
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from logger import get_logger
from gate import load_config, check_date_range, DateRangeGuard, GateViolation
from data.pit_universe import get_universe

log = get_logger("walkforward_bt")

IC_DIR = os.path.join(BASE_DIR, "data", "ic_validation")
OUTPUT_PATH = os.path.join(IC_DIR, "walkforward_results.json")
BENCH_PATH = os.path.join(BASE_DIR, "data", "cache", "index_csi1000.parquet")

# ── Walk-forward 参数 ──
TRAIL_DAYS = 252          # 回看窗口 (交易日)
LABEL_HORIZON = 21        # 前瞻收益标签长度 (交易日)
IC_STEP = 21              # 月度IC观测间隔 (交易日)
MIN_ICIR = 0.02           # |ICIR| 入选阈值
MIN_IC_OBS = 6            # 因子最少月度IC观测数
MIN_CROSS_SECTION = 30    # 单次截面IC最少股票数
REBALANCE_DAYS = 20       # 调仓间隔 (交易日)


def _load_partitions() -> dict:
    """从 config.yaml 读取数据分区 (硬编码仅作 fallback)。"""
    try:
        cfg = load_config(os.path.join(BASE_DIR, "config.yaml"))
        dp = cfg["data_partition"]
        return {
            "development": (dp["development"]["start"], dp["development"]["end"]),
            "test": (dp["test"]["start"], dp["test"]["end"]),
        }
    except Exception:
        return {
            "development": ("2026-01-01", "2026-06-30"),
            "test": ("2026-07-01", "2026-07-31"),
        }


PARTITIONS = _load_partitions()


def load_bt_config() -> dict:
    """从 config.yaml 加载执行参数 (与 run_corrected_backtest 统一)。"""
    cfg = load_config(os.path.join(BASE_DIR, "config.yaml"))
    return {
        "rebalance_days": REBALANCE_DAYS,
        "top_k": cfg["execution"]["top_k"],
        "initial_capital": cfg["execution"]["initial_capital"],
        "lot_size": cfg["execution"]["lot_size"],
        "slippage_bps": cfg["execution"]["slippage_bps"],
        "commission_buy": cfg["execution"]["commission_buy"],
        "commission_sell": cfg["execution"]["commission_sell"],
    }


# ═══════════════════════════════════════════════════════════
#  面板构建
# ═══════════════════════════════════════════════════════════

def build_calendar(all_data: dict) -> list:
    """全局交易日历 (所有股票日期的并集, 排序)。"""
    all_dates = set()
    for df in all_data.values():
        all_dates.update(pd.to_datetime(df["date"]).dt.normalize().tolist())
    return sorted(all_dates)


def build_close_panel(all_data: dict, calendar: list) -> pd.DataFrame:
    """收盘价面板: index=日历, columns=股票代码。不交易日为 NaN (不前填)。"""
    panel = pd.DataFrame(
        {sym: df.set_index(pd.to_datetime(df["date"]))["close"]
         for sym, df in all_data.items()}
    )
    return panel.reindex(pd.DatetimeIndex(calendar))


def neutralize_factor(df: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """
    前置中性化: MAD 去极值 + z-score 标准化 (逐列/逐因子)。
    处理 NaN (保留为 NaN, 不参与统计)。
    """
    # 保持输入 dtype (float32): 全面板 astype(float64) 内存翻倍 (~9.4GB→~19GB) 会 OOM。
    # 计算阶段逐列提升为 float64 (单列很小), 写回时显式降回 float32。
    out = df.copy()
    for col in out.columns:
        vals = out[col]
        m = vals.notna()
        if m.sum() < 10:
            continue
        x = vals[m].to_numpy(dtype=np.float64)
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
        # pandas 2.x 下 float64 数组直接写入 float32 列会抛 LossySetitemError,
        # 必须显式 .astype(np.float32) (数值差异 max ~5.7e-8, 可接受)
        out.loc[m, col] = ((x - mu) / sd).astype(np.float32)
    return out


def precompute_factor_panels(all_data: dict, factor_names: list,
                             needed_dates: list,
                             include_fundamental: bool = False,
                             neutralize_enabled: bool = False,
                             neutralize_k: float = 3.0) -> dict:
    """
    预计算因子并裁剪到所需日期, 构建 {factor: DataFrame(日期×股票)} 面板。

    内存优化 (方案C v4):
      - float32 存储 (因子值精度足够, 内存减半)
      - 单次全因子计算 (每只股票 compute_factors 只调用一次)
      - 面板构建完成后立即释放 per_stock 中间数据

    include_fundamental=True: 额外合并 fund_* 基本面因子 (PIT-safe)。
      基本面因子由本函数单独计算 (factor_scorer 无法输出非 DSL 的 fund_* 列),
      每只股票只读一次财报缓存, 用 searchsorted 做逐日期 PIT 最近可用查找。

    neutralize_enabled=True: 返回前对每个因子面板统一做前置中性化
      (MAD 去极值 + z-score, 逐因子独立), 下游 IC/打分全部消费中性化面板。
    """
    from factor_scorer import FactorScorer
    scorer = FactorScorer.from_preset("full_auto")

    needed_set = set(pd.DatetimeIndex(needed_dates))
    symbols = sorted(all_data.keys())
    idx = pd.DatetimeIndex(needed_dates)
    n_stocks = len(symbols)

    t0 = time.time()
    per_stock = {}   # sym -> DataFrame(needed_dates × factors, float32)
    n_ok = 0
    for i in range(0, len(symbols), 200):
        batch = symbols[i:i + 200]
        for sym in batch:
            df = all_data[sym]
            try:
                full = scorer.compute_factors(df)
                if "date" not in full.columns:
                    full["date"] = df["date"].values
                full["date"] = pd.to_datetime(full["date"])
                sub = full[full["date"].isin(needed_set)]
                if len(sub) == 0:
                    continue
                keep = [c for c in factor_names if c in sub.columns]
                sdf = sub.set_index("date")[keep].astype(np.float32)
                if sdf.shape[1] > 0:
                    per_stock[sym] = sdf
                    n_ok += 1
            except Exception:
                continue
        done = min(i + 200, len(symbols))
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(symbols) - done) / rate if rate > 0 else 0
        log.info("  因子预计算: %d/%d (有效 %d, %.0fs, 剩余~%.0fs)",
                 done, len(symbols), n_ok, elapsed, eta)

    log.info("  构建因子面板 (%d 因子 × %d 日期 × %d 股票, float32)...",
             len(factor_names), len(needed_dates), n_ok)

    # 逐因子面板: dict 收集 → pandas 原生对齐 (C 实现, 快)
    # 内存峰值 = per_stock (9.4GB) + panels (9.4GB) ≈ 19GB < 34GB ✅
    panels = {}
    try:
        for fi, fn in enumerate(factor_names):
            cols = {sym: sdf[fn] for sym, sdf in per_stock.items()
                    if fn in sdf.columns}
            if cols:
                panels[fn] = pd.DataFrame(cols, index=idx, dtype=np.float32)
            if (fi + 1) % 40 == 0 or fi == len(factor_names) - 1:
                log.info("  面板: %d/%d 因子", fi + 1, len(factor_names))
    finally:
        per_stock.clear()
        import gc
        gc.collect()

    # ── 基本面因子合并 (PIT-safe, 由本函数单独计算, 不依赖 scorer) ──
    if include_fundamental:
        t1 = time.time()
        n_fund = _merge_fundamental_panels(panels, all_data, factor_names, idx)
        log.info("  基本面因子面板: %d 个 (%.0fs)", n_fund, time.time() - t1)

    # ── 前置中性化 (MAD去极值 + z-score, 逐因子) ──
    if neutralize_enabled:
        t2 = time.time()
        for fn in list(panels.keys()):
            panels[fn] = neutralize_factor(panels[fn], k=neutralize_k)
        log.info("  中性化完成: %d 因子 (MAD去极值+z-score, k=%.1f, %.0fs)",
                 len(panels), neutralize_k, time.time() - t2)

    log.info("  面板就绪: %d 因子 × %d 日期", len(panels), len(needed_dates))
    return panels


# ═══════════════════════════════════════════════════════════
#  基本面因子面板 (PIT-safe)
# ═══════════════════════════════════════════════════════════

# 价格类因子: fund_因子 -> (财报基数列, 变换模式)
#   div_price = 基数值 / PIT价格 (bp/ep/ocf_yield), price_div = PIT价格 / 基数值 (pb)
_FUND_PRICE_MAP = {
    "fund_bp": ("bvps", "div_price"),
    "fund_ep": ("eps_ttm", "div_price"),
    "fund_pb": ("bvps", "price_div"),
    "fund_ocf_yield": ("ocf_ps", "div_price"),
}


def _fund_report_factors(fin: pd.DataFrame) -> pd.DataFrame | None:
    """
    从单只股票的财报序列 (data/fundamental.py fetch_financials 输出) 预计算
    每期报告的 13 个基本面因子值 (不含价格类分母, 价格在合并时按日 PIT 对齐)。

    Returns: DataFrame index=报告期(日期), columns=因子键(无 fund_ 前缀), 或 None。
    """
    if fin is None or len(fin) == 0 or "日期" not in fin.columns:
        return None
    f = fin.copy()
    f["日期"] = pd.to_datetime(f["日期"])
    f = f.sort_values("日期").reset_index(drop=True)

    def col(name: str) -> pd.Series | None:
        if name not in f.columns:
            return None
        return pd.to_numeric(f[name], errors="coerce")

    roe = col("净资产收益率(%)")
    eps = col("摊薄每股收益(元)")
    bvps = col("每股净资产_调整前(元)")
    ocf = col("每股经营性现金流(元)")
    pg = col("净利润增长率(%)")
    rg = col("主营业务收入增长率(%)")
    dr = col("资产负债率(%)")
    nm = col("销售净利率(%)")
    ded = col("扣除非经常性损益后的净利润(元)")

    out = pd.DataFrame(index=pd.DatetimeIndex(f["日期"]))
    month = f["日期"].dt.month

    # ROE (按报告期季度数年化, 与 fundamental_fetcher 口径一致)
    if roe is not None:
        mult = month.map({3: 4.0, 6: 2.0, 9: 4 / 3}).fillna(1.0)
        out["roe"] = (roe * mult).to_numpy(dtype=np.float32)
        out["roe_ttm"] = roe.rolling(4, min_periods=4).mean().to_numpy(dtype=np.float32)

    # EPS TTM (最近4季度滚动加总; 摊薄EPS为年内累计口径)
    if eps is not None:
        out["eps_ttm"] = eps.rolling(4, min_periods=4).sum().to_numpy(dtype=np.float32)

    # 扣非净利润同比增速 (同季度对比: groupby(quarter).shift(1) 取去年同季)
    if ded is not None:
        q = f["日期"].dt.quarter
        prev = ded.groupby(q).shift(1)
        g = (ded / prev - 1.0).where(prev > 0)
        out["profit_growth_ded"] = g.to_numpy(dtype=np.float32)

    for key, s in [("profit_growth", pg), ("revenue_growth", rg),
                   ("debt_ratio", dr), ("net_margin", nm),
                   ("ocf_ps", ocf), ("bvps", bvps)]:
        if s is not None:
            out[key] = s.to_numpy(dtype=np.float32)

    # 应计利润 (每股口径: (EPS_TTM - OCF_PS) / BVPS, 与 fundamental_fetcher 一致)
    if eps is not None and ocf is not None and bvps is not None:
        out["accruals"] = ((out["eps_ttm"] - ocf.to_numpy(dtype=np.float32))
                           / bvps.to_numpy(dtype=np.float32))
    return out


def _merge_fundamental_panels(panels: dict, all_data: dict,
                              factor_names: list, idx: pd.DatetimeIndex) -> int:
    """
    将 fund_* 基本面因子按日期 PIT 合并进 panels (原地修改)。

    PIT 规则 (与 data/fundamental.py 一致):
      - 每只股票只读一次财报缓存 (fetch_financials)
      - 报告期 + PIT_LAG_DAYS 之后才可用 (财报公布延迟缓冲, 禁止前视)
      - 每个面板日期取 <= date-PIT_LAG_DAYS 的最新一期财报 (searchsorted)
      - 价格类因子 (bp/ep/pb/ocf_yield) 使用面板日期当日及之前最近收盘价
      - 财报缺失/异常 → 静默跳过该股票

    Returns: 实际合并的 fund_* 因子数。
    """
    from data.fundamental import fetch_financials, PIT_LAG_DAYS
    from factor_library import FUNDAMENTAL_FACTORS

    fund_names = [fn for fn in factor_names if fn in FUNDAMENTAL_FACTORS]
    if not fund_names:
        return 0
    fund_keys = {fn: fn.replace("fund_", "") for fn in fund_names}

    dates_arr = idx.to_numpy(dtype="datetime64[ns]")
    lag = np.timedelta64(PIT_LAG_DAYS, "D")

    # searchsorted 要求日期升序; 面板日期非有序时先排序再还原
    if np.all(dates_arr[1:] >= dates_arr[:-1]):
        sorted_dates = dates_arr
        inv = None
    else:
        order = np.argsort(dates_arr, kind="stable")
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        sorted_dates = dates_arr[order]

    cols = {fn: {} for fn in fund_names}
    for sym in all_data:
        try:
            vals = _fund_report_factors(fetch_financials(sym))
            if vals is None or len(vals) == 0:
                continue
            rdates = vals.index.to_numpy(dtype="datetime64[ns]")
            # PIT 可用日期: 报告期 + 财报公布延迟缓冲
            avail = rdates + lag
            pos = np.searchsorted(avail, sorted_dates, side="right") - 1
            valid = pos >= 0
            if not valid.any():
                continue

            # PIT 收盘价: 面板日期当日及之前最近收盘价
            px = all_data[sym]
            # T3 审查修复: searchsorted 要求价格日期升序 — 缓存数据乱序时
            # 先排序, 防止静默产出错误的 PIT 价格 (只读副本, 不改动缓存)
            px_dates = pd.to_datetime(px["date"])
            if not px_dates.is_monotonic_increasing:
                px = px.sort_values("date").reset_index(drop=True)
                px_dates = pd.to_datetime(px["date"])
            pdates = px_dates.to_numpy(dtype="datetime64[ns]")
            closes = px["close"].to_numpy(dtype=np.float64)
            ppos = np.searchsorted(pdates, sorted_dates, side="right") - 1
            pvalid = (ppos >= 0) & (closes[np.clip(ppos, 0, len(closes) - 1)] > 0)
            price = closes[np.clip(ppos, 0, len(closes) - 1)]

            v = vals.to_numpy(dtype=np.float64)  # (n_reports, n_keys)
            rp = np.clip(pos, 0, len(v) - 1)
            for fn in fund_names:
                base_key, mode = _FUND_PRICE_MAP.get(
                    fn, (fund_keys[fn], None))
                if base_key not in vals.columns:
                    continue
                kpos = vals.columns.get_loc(base_key)
                values = np.where(valid, v[rp, kpos], np.nan)
                if mode == "div_price":
                    m = pvalid & (price > 0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        values = np.where(m, values / price, np.nan)
                elif mode == "price_div":
                    m = pvalid & (price > 0) & (values > 0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        values = np.where(m, price / values, np.nan)
                if inv is not None:
                    values = values[inv]
                cols[fn][sym] = values.astype(np.float32)
        except Exception:
            continue

    n_added = 0
    for fn in fund_names:
        if cols[fn]:
            panels[fn] = pd.DataFrame(cols[fn], index=idx, dtype=np.float32)
            n_added += 1
    return n_added


# ═══════════════════════════════════════════════════════════
#  动态 IC 权重
# ═══════════════════════════════════════════════════════════

def rank_corr_cols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Spearman 秩相关: x 形状 (n_obs, n_factors), y 形状 (n_obs,) 且无 NaN。
    返回每个因子与 y 的秩相关系数 (NaN 表示无法计算)。
    与 scipy.stats.spearmanr 一致: 在 (因子, y) 联合有效观测上秩化。
    """
    from scipy.stats import rankdata
    n_obs, n_f = x.shape
    out = np.full(n_f, np.nan)
    for fi in range(n_f):
        col = x[:, fi]
        mask = ~np.isnan(col)
        if mask.sum() < MIN_CROSS_SECTION:
            continue
        cv, yv = col[mask], y[mask]
        if (cv.max() - cv.min()) < 1e-12 or (yv.max() - yv.min()) < 1e-12:
            continue
        r = np.corrcoef(rankdata(cv), rankdata(yv))[0, 1]
        if not np.isnan(r):
            out[fi] = r
    return out


def _equal_weight_benchmark(start: str, end: str,
                            all_data: dict):
    """
    全市场等权基准 (方案C v4.1): 用回测股票池每日等权收益构建,
    与策略池/流动性过滤天然一致, 避免小盘指数基准错配。

    Returns: pd.Series of close prices (index=date) or None
    """
    try:
        # 每日等权收益: 所有股票当日 close 的日收益均值
        daily_rets = {}
        for sym, df in all_data.items():
            s = df.set_index(pd.to_datetime(df["date"]))["close"]
            s = s[s > 0]
            r = s.pct_change()
            daily_rets[sym] = r
        panel = pd.DataFrame(daily_rets)
        eqw = panel.mean(axis=1, skipna=True)
        eqw = eqw[(eqw.index >= pd.Timestamp(start)) &
                  (eqw.index <= pd.Timestamp(end))]
        eqw = eqw.dropna()
        if len(eqw) < 10:
            return None
        # 转价格序列 (从 100 起)
        return 100.0 * (1 + eqw).cumprod()
    except Exception:
        return None


def _nearest_idx(cal_idx: dict, date_str: str):
    """
    Find the nearest calendar index for a requested date.
    (训练期起止日期可能是非交易日, 回退到最近交易日)
    """
    target = pd.Timestamp(date_str)
    if target in cal_idx:
        return cal_idx[target]
    # 找 <= target 的最大交易日
    candidates = [d for d in cal_idx if d <= target]
    if candidates:
        return cal_idx[max(candidates)]
    # target 早于所有交易日 → 取最早交易日
    if cal_idx:
        return cal_idx[min(cal_idx.keys())]
    return None


def compute_icir_weights(factor_panels: dict, close_panel: pd.DataFrame,
                         calendar: list, cal_idx: dict, t_date,
                         factor_names: list,
                         train_start: str | None = None,
                         train_end: str | None = None,
                         universe_fn=None) -> tuple:
    """
    计算因子 ICIR 权重。

    两种窗口模式:
      rolling (默认): 滚动 [T-TRAIL_DAYS, T-LABEL_HORIZON] 窗口, 每个调仓日重估。
      fixed (方案C fold): 传 train_start/train_end, 在固定训练期内估计,
        用于验证期全部调仓日 (避免验证期信息泄漏)。

    窗口: [start_idx, end_idx] (日历索引), 每 IC_STEP 天一个观测点。
    标签: 21 日前瞻收益 close[t+21]/close[t]-1 (在 T 时已全部实现)。

    Returns: (weights: {factor: icir}, ic_stats: {factor: {ic_mean, ic_std, n_obs}})
    """
    t_idx = cal_idx.get(pd.Timestamp(t_date))
    if t_idx is None:
        return {}, {}

    if train_start is not None and train_end is not None:
        # 固定训练窗口 (方案C fold): 训练期内全部观测点
        s_idx = _nearest_idx(cal_idx, train_start)
        e_idx = _nearest_idx(cal_idx, train_end)
        if s_idx is None or e_idx is None:
            return {}, {}
        start_idx = s_idx
        # 严格无泄漏: 训练观测点标签(21日前瞻)必须完全在训练期内
        end_idx = min(e_idx - LABEL_HORIZON, t_idx - LABEL_HORIZON)
        if end_idx <= start_idx:
            return {}, {}
    else:
        # 滚动窗口 (默认)
        start_idx = t_idx - TRAIL_DAYS
        if start_idx < 0:
            start_idx = 0  # 早期数据不足 → 扩展窗口 (expanding)
        end_idx = t_idx - LABEL_HORIZON
    offsets = list(range(start_idx, end_idx + 1, IC_STEP))
    if not offsets:
        return {}, {}

    close_vals = close_panel.to_numpy()
    n_dates = len(calendar)
    available = [fn for fn in factor_names if fn in factor_panels]
    # 预取每个观测日的因子截面矩阵 (n_stocks, n_factors)
    matrices = []
    labels = []
    for oi in offsets:
        if oi + LABEL_HORIZON >= n_dates or oi < 0:
            continue
        # ★ FIX-C: IC 域 = 回测域 — 每个观测日动态过滤到当时的 universe
        if universe_fn is not None:
            oi_date = str(calendar[oi].date())
            try:
                uni = set(universe_fn(oi_date))
            except Exception:
                uni = None
        else:
            uni = None
        c0 = close_vals[oi, :]
        c1 = close_vals[oi + LABEL_HORIZON, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = c1 / c0 - 1
        valid_px = (c0 > 0) & ~np.isnan(c0) & ~np.isnan(c1)
        if uni is not None:
            col_names = close_panel.columns
            # 只在 universe 内计算 IC
            uidx = [ci for ci, s in enumerate(col_names) if s in uni]
            valid_px = valid_px.copy()
            valid_px[~np.isin(np.arange(len(col_names)), uidx)] = False
        ret = np.where(valid_px, ratio, np.nan)
        if np.sum(~np.isnan(ret)) < MIN_CROSS_SECTION:
            continue
        cols = []
        for fn in available:
            cols.append(factor_panels[fn].loc[calendar[oi]].to_numpy())
        x = np.column_stack(cols)
        # 仅保留有有效标签的行
        valid = ~np.isnan(ret)
        matrices.append(x[valid, :])
        labels.append(ret[valid])

    if not matrices:
        return {}, {}

    ic_records = {fn: [] for fn in available}
    for x, y in zip(matrices, labels):
        ics = rank_corr_cols(x, y)
        for fi, fn in enumerate(available):
            if not np.isnan(ics[fi]):
                ic_records[fn].append(ics[fi])

    weights = {}
    ic_stats = {}
    for fn in available:
        arr = np.array(ic_records[fn])
        if len(arr) < MIN_IC_OBS:
            continue
        mu = float(np.mean(arr))
        sd = float(np.std(arr))
        icir = mu / sd if sd > 1e-9 else 0.0
        ic_stats[fn] = {
            "ic_mean": round(mu, 5),
            "ic_std": round(sd, 5),
            "icir": round(icir, 4),
            "n_obs": int(len(arr)),
        }
        if abs(icir) > MIN_ICIR:
            weights[fn] = icir
    return weights, ic_stats


def score_stocks(factor_panels: dict, weights: dict, t_date) -> dict:
    """用动态 ICIR 权重对 t_date 当日截面打分 (z-score 加权)。

    与 run_corrected_backtest.compute_composite_scores 逻辑一致:
    覆盖率 >= 50%, 逐因子 z-score, ICIR 加权求和 / Σ|w|。
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

    return {s: float(v) for s, v in zip(cross.index, composite)
            if not np.isnan(v)}


# ═══════════════════════════════════════════════════════════
#  回测主循环
# ═══════════════════════════════════════════════════════════

def run_backtest(all_data, factor_panels, close_panel, calendar, cal_idx,
                 factor_names, bt_config, start, end, label="",
                 fixed_weights: dict | None = None,
                 universe_fn=get_universe):
    """回测主循环。

    fixed_weights: 若提供, 全程使用该固定权重 (方案C fold 验证期模式,
      权重在训练期估计, 验证期不重估 → 无信息泄漏)。
    universe_fn: universe 提供函数, 默认指数成分, 可传 get_liquid_universe。
    """
    from model.engine import SimpleBacktest
    from trading_rules import TradingRules
    from portfolio_ranker import PortfolioRanker

    bt = SimpleBacktest(
        initial_capital=bt_config["initial_capital"],
        top_k=bt_config["top_k"],
        lot_size=bt_config["lot_size"],
        slippage_bps=bt_config["slippage_bps"],
        turnover_limit_pct=1.0,
    )
    rules = TradingRules()
    ranker = PortfolioRanker(
        top_k=bt_config["top_k"],
        n_drop=10, hold_thresh=30,
        sell_rank_buffer=3, buy_confirm_days=1,
        cost_threshold=0.08,
    )

    all_dates_set = set(pd.to_datetime(calendar))
    bt_dates = sorted(d for d in all_dates_set
                      if pd.Timestamp(start).date() <= d.date() <= pd.Timestamp(end).date())
    if not bt_dates:
        log.error("[%s] 无交易日", label)
        return None

    log.info("=" * 60)
    log.info("[%s] %s ~ %s (%d 天), top_k=%d, 动态IC权重",
             label, bt_dates[0].date(), bt_dates[-1].date(),
             len(bt_dates), bt_config["top_k"])
    log.info("=" * 60)

    equity_curve = []
    daily_returns = []
    positions_history = []
    weights_history = []
    turnover_history = []
    rebalance_count = 0
    pending = None
    prev_equity = float(bt_config["initial_capital"])
    pit_sizes = []

    for di, today in enumerate(bt_dates):
        # T+1 执行
        if pending is not None:
            bt.execute(pending, today, all_data, rules)
            pending = None

        # 调仓日
        if di % bt_config["rebalance_days"] == 0:
            pit_stocks = set(universe_fn(str(today.date())))
            pit_sizes.append(len(pit_stocks))

            # ★ 权重: 固定权重 (方案C fold) 或滚动 ICIR (默认)
            if fixed_weights is not None:
                weights = fixed_weights
            else:
                t_ic0 = time.time()
                weights, ic_stats = compute_icir_weights(
                    factor_panels, close_panel, calendar, cal_idx,
                    today, factor_names, universe_fn=universe_fn)
                log.info("  [%s] 调仓日 %s: %d 因子入选 (|ICIR|>%.2f), IC计算 %.1fs",
                         label, today.date(), len(weights), MIN_ICIR,
                         time.time() - t_ic0)

            weights_history.append({
                "date": str(today.date()),
                "n_selected": len(weights),
                "weights": {k: round(v, 4) for k, v in
                            sorted(weights.items(), key=lambda kv: -abs(kv[1]))},
                "top10": [k for k, _ in sorted(weights.items(),
                                                 key=lambda kv: -abs(kv[1]))[:10]],
            })

            # 评分
            scores = score_stocks(factor_panels, weights, today)
            # PIT 过滤
            scores = {s: v for s, v in scores.items() if s in pit_stocks}

            if scores and len(scores) >= bt_config["top_k"]:
                tradeable = {}
                for sym, sc in scores.items():
                    if sym in all_data:
                        dt = all_data[sym][all_data[sym]["date"] <= today].tail(2)
                        if len(dt) >= 2 and not rules.is_suspended(sym, dt):
                            tradeable[sym] = sc

                if len(tradeable) >= bt_config["top_k"]:
                    holdings = list(bt.positions.keys())
                    decision = ranker.rank(tradeable, holdings)
                    decision["buy"] = [
                        s for s in decision.get("buy", [])
                        if s in all_data and rules.can_buy(
                            s, all_data[s][all_data[s]["date"] <= today].tail(2))]
                    decision["sell"] = [
                        s for s in decision.get("sell", [])
                        if s in all_data and rules.can_sell(
                            s, all_data[s][all_data[s]["date"] <= today].tail(2))]
                    pending = decision
                    rebalance_count += 1
                    n_turn = (len(decision.get("sell", [])) +
                              len(decision.get("buy", []))) / (2 * bt_config["top_k"])
                    turnover_history.append(n_turn)

            positions_history.append({
                "date": str(today.date()),
                "positions": sorted(bt.positions.keys()),
            })

        # Mark-to-market (用面板收盘价, O(1))
        close_prices = {}
        if today in cal_idx:
            crow = close_panel.iloc[cal_idx[today]]
            for sym in list(bt.positions.keys()):
                if sym in crow.index:
                    v = crow[sym]
                    if not (isinstance(v, float) and np.isnan(v)):
                        close_prices[sym] = float(v)
        equity = bt.mark_to_market(close_prices)
        equity_curve.append({"date": str(today.date()), "equity": equity})
        daily_ret = (equity / prev_equity - 1) if prev_equity > 0 else 0.0
        daily_returns.append(daily_ret)
        prev_equity = equity

    # ── 统计 ──
    eq = np.array([e["equity"] for e in equity_curve])
    rets = np.array(daily_returns)
    total_return = eq[-1] / bt_config["initial_capital"] - 1
    n_years = len(bt_dates) / 252.0
    annual_return = (1 + total_return) ** (1 / max(n_years, 0.1)) - 1

    # 基准 (方案C v4.1: 优先全市场等权, 与策略池一致; CSI1000 仅 fallback)
    bench_annual = 0.0
    bench_daily = np.zeros(len(rets))
    ew_bench = _equal_weight_benchmark(start, end, all_data)
    if ew_bench is not None:
        bs = ew_bench
        if len(bs) > 1:
            bt_total = bs.iloc[-1] / bs.iloc[0] - 1
            bench_annual = (1 + bt_total) ** (1 / max(n_years, 0.1)) - 1
            bench_daily = bs.pct_change().dropna().to_numpy()
    elif os.path.exists(BENCH_PATH):
        bdf = pd.read_parquet(BENCH_PATH)
        bdf["date"] = pd.to_datetime(bdf["date"])
        bdf = bdf.set_index("date")
        rs = pd.Timestamp(start)
        re_ = pd.Timestamp(end)
        bm = (bdf.index >= rs) & (bdf.index <= re_)
        bs = bdf.loc[bm, "close"]
        if len(bs) > 1:
            bt_total = bs.iloc[-1] / bs.iloc[0] - 1
            bench_annual = (1 + bt_total) ** (1 / max(n_years, 0.1)) - 1
            bench_daily = bs.pct_change().dropna().to_numpy()

    excess = annual_return - bench_annual

    # Sharpe (rf=2.5%)
    rf_daily = 0.025 / 252
    excess_daily = rets - rf_daily
    sharpe = (np.mean(excess_daily) / np.std(excess_daily) * np.sqrt(252)
              if np.std(excess_daily) > 0 else 0)

    # IR
    if len(bench_daily) >= len(rets):
        active = rets - bench_daily[:len(rets)]
    else:
        active = rets - np.pad(bench_daily, (0, len(rets) - len(bench_daily)))
    ir = (np.mean(active) / np.std(active) * np.sqrt(252)
          if np.std(active) > 0 else 0)

    # MaxDD
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(np.min(dd))
    calmar = annual_return / abs(max_dd) if abs(max_dd) > 0 else 0

    avg_turnover = np.mean(turnover_history) if turnover_history else 0

    result = {
        "label": label,
        "period": f"{bt_dates[0].date()} ~ {bt_dates[-1].date()}",
        "n_days": len(bt_dates),
        "total_return": round(total_return * 100, 1),
        "annual_return": round(annual_return * 100, 1),
        "benchmark_annual": round(bench_annual * 100, 1),
        "excess_annual": round(excess * 100, 1),
        "sharpe": round(sharpe, 2),
        "ir": round(ir, 2),
        "max_drawdown": round(max_dd * 100, 1),
        "calmar": round(calmar, 2),
        "n_rebalances": rebalance_count,
        "avg_turnover": round(avg_turnover * 100, 1),
        "avg_pit_size": int(np.mean(pit_sizes)) if pit_sizes else 0,
        "avg_n_selected_factors": int(np.mean(
            [w["n_selected"] for w in weights_history])) if weights_history else 0,
        # 日度数据 (供 bootstrap 分析)
        "daily_returns": [round(float(r), 8) for r in rets],
        "daily_active_returns": [round(float(a), 8) for a in active],
        # 明细
        "weights_evolution": weights_history,
        "positions_history": positions_history,
        "equity_curve": equity_curve,
    }

    log.info("  结果:")
    log.info("    总收益: %+.1f%%", result["total_return"])
    log.info("    年化收益: %+.1f%%", result["annual_return"])
    log.info("    基准年化: %+.1f%%", result["benchmark_annual"])
    log.info("    年化超额: %+.1f%%", result["excess_annual"])
    log.info("    Sharpe: %.2f  IR: %.2f", result["sharpe"], result["ir"])
    log.info("    最大回撤: %.1f%%  Calmar: %.2f",
             result["max_drawdown"], result["calmar"])
    log.info("    调仓: %d 次, 平均PIT: %d 只, 平均入选因子: %d",
             rebalance_count, result["avg_pit_size"],
             result["avg_n_selected_factors"])
    return result


# ═══════════════════════════════════════════════════════════
#  方案C: Walk-Forward Fold 分析 (训练→验证, 固定权重)
# ═══════════════════════════════════════════════════════════

# Fold 划分 (方案C): 训练期逐年扩展, 验证期固定1年
FOLDS = [
    {"train": ("2015-01-01", "2019-12-31"), "val": ("2020-01-01", "2020-12-31")},
    {"train": ("2015-01-01", "2020-12-31"), "val": ("2021-01-01", "2021-12-31")},
    {"train": ("2015-01-01", "2021-12-31"), "val": ("2022-01-01", "2022-12-31")},
    {"train": ("2015-01-01", "2022-12-31"), "val": ("2023-01-01", "2023-12-31")},
    {"train": ("2015-01-01", "2023-12-31"), "val": ("2024-01-01", "2024-12-31")},
]
FOLD_MIN_HITS = 3          # 因子须在 >=3/5 folds 中 |ICIR| 达标才保留
FOLD_ICIR_MIN = 0.05       # fold 内 |ICIR| 入选阈值 (滚动模式下仅作下限)
FOLD_T_STAT_MIN = 1.645    # 统计显著标准: |ICIR|*sqrt(n_obs) >= 1.645 (单尾5%)
FOLD_MAX_FACTORS = 40      # 稳定因子数量上限 (防止噪音因子稀释组合)


def run_fold_analysis(all_data, factor_panels, close_panel, calendar, cal_idx,
                      factor_names, bt_config,
                      universe_fn=get_universe) -> dict:
    """
    方案C核心: 5-fold Walk-Forward。

    每个 fold:
      1. 训练期 [train] 用固定窗口计算全部因子 ICIR
      2. 验证期 [val] 用训练期固定权重回测 (无信息泄漏)
    汇总:
      - 每个因子在多少 fold 中 |ICIR| 达标 (fold_hits)
      - 保留 fold_hits >= FOLD_MIN_HITS 的稳定因子
      - 各 fold 验证期成绩汇总 → OOS 年化超额均值/IR
    """
    log.info("=" * 60)
    log.info("  方案C Walk-Forward: %d folds (训练扩展, 验证固定权重)",
             len(FOLDS))
    log.info("  因子保留标准: |ICIR|>=%.2f 在 >=%d/%d folds 中达标",
             FOLD_ICIR_MIN, FOLD_MIN_HITS, len(FOLDS))
    log.info("=" * 60)

    fold_results = {}
    factor_hits = {fn: 0 for fn in factor_names}
    factor_icirs = {fn: [] for fn in factor_names}

    # 验证期首调仓日: 固定权重只需在训练期末计算一次
    for fi, fold in enumerate(FOLDS):
        ts, te = fold["train"]
        vs, ve = fold["val"]
        log.info("")
        log.info(f"── Fold {fi+1}: Train {ts}~{te} → Val {vs}~{ve} ──")

        # 训练期权重 (固定窗口, 在验证期首日估计 → 训练信息在 T 前全部实现)
        val_first = None
        for d in calendar:
            if pd.Timestamp(vs).date() <= d.date() <= pd.Timestamp(ve).date():
                val_first = d
                break
        if val_first is None:
            log.warning("  Fold %d: 验证期无交易日, 跳过", fi + 1)
            continue

        weights, ic_stats = compute_icir_weights(
            factor_panels, close_panel, calendar, cal_idx,
            val_first, factor_names, train_start=ts, train_end=te,
            universe_fn=universe_fn)
        n_sel = len(weights)
        log.info("  训练期因子: %d/%d 入选 |ICIR|>=%.2f",
                 n_sel, len(factor_names), FOLD_ICIR_MIN)

        # 记录每个因子的 fold 命中 (统计显著标准: |ICIR|*sqrt(n) >= 1.645)
        sig_factors = set()
        for fn in factor_names:
            st = ic_stats.get(fn)
            if st is not None:
                factor_icirs[fn].append(st["icir"])
                n_obs = st.get("n_obs", 0)
                t_stat = abs(st["icir"]) * np.sqrt(n_obs) if n_obs > 0 else 0.0
                if t_stat >= FOLD_T_STAT_MIN and abs(st["icir"]) >= FOLD_ICIR_MIN:
                    factor_hits[fn] += 1
                    sig_factors.add(fn)
            else:
                factor_icirs[fn].append(0.0)

        # 验证期回测权重: 仅统计显著因子 (过滤噪音)
        weights = {fn: w for fn, w in weights.items() if fn in sig_factors}
        n_sel = len(weights)
        log.info("  回测权重: %d 因子 (统计显著, |ICIR|*√n>=%.2f)",
                 n_sel, FOLD_T_STAT_MIN)

        if not weights:
            log.warning("  Fold %d: 训练期无因子达标, 验证期跳过", fi + 1)
            continue

        r = run_backtest(all_data, factor_panels, close_panel, calendar,
                         cal_idx, factor_names, bt_config, vs, ve,
                         label=f"VAL{fi+1}", fixed_weights=weights,
                         universe_fn=universe_fn)
        if r:
            fold_results[f"fold_{fi+1}"] = {
                "train": f"{ts} ~ {te}",
                "val": f"{vs} ~ {ve}",
                "n_selected_factors": n_sel,
                **r,
            }

    # ── 稳定因子筛选 (统计显著 ≥3/5 folds + 方向一致 + 数量上限) ──
    # 方向一致性: 因子在 ≥3/5 folds 中 ICIR 符号相同 (防止正负抵消)
    cand = []
    for fn in factor_names:
        if factor_hits[fn] < FOLD_MIN_HITS or not factor_icirs[fn]:
            continue
        arr = np.array(factor_icirs[fn])
        pos_cnt = int((arr > 0).sum())
        neg_cnt = int((arr < 0).sum())
        if max(pos_cnt, neg_cnt) >= FOLD_MIN_HITS:  # 方向一致
            cand.append(fn)
    cand_icir = {fn: float(np.median(factor_icirs[fn])) for fn in cand}
    # 按 |ICIR| 排序取 top FOLD_MAX_FACTORS
    ranked = sorted(cand_icir.items(), key=lambda kv: -abs(kv[1]))
    stable = [fn for fn, _ in ranked[:FOLD_MAX_FACTORS]]
    stable_icir = {fn: cand_icir[fn] for fn in stable}

    log.info("")
    log.info("=" * 60)
    log.info("  稳定因子: %d/%d (≥%d folds 显著+方向一致, top%d)",
             len(stable), len(factor_names), FOLD_MIN_HITS,
             FOLD_MAX_FACTORS)
    if stable:
        top = sorted(stable_icir.items(), key=lambda kv: -abs(kv[1]))[:15]
        for fn, icir in top:
            log.info("    %-30s ICIR=%+.3f (命中 %d/5)",
                     fn, icir, factor_hits[fn])
    log.info("=" * 60)

    return {
        "folds": fold_results,
        "factor_hits": factor_hits,
        "stable_factors": stable,
        "stable_factor_icir_median": {
            k: round(v, 4) for k, v in stable_icir.items()},
    }


def run_fold_test(all_data, factor_panels, close_panel, calendar, cal_idx,
                  factor_names, bt_config, stable_factors, stable_icir,
                  test_start, test_end,
                  universe_fn=get_universe) -> dict | None:
    """
    终极 Holdout: 用稳定因子的中位数 ICIR 权重, 在 TEST 期一次性回测。

    风格均衡 (方案C v4.1): 正/负 ICIR 因子权重分别归一化,
    避免单一风格 (如反转防御) 过度主导组合 → 牛市跑输基准。
    """
    if not stable_factors:
        log.warning("无稳定因子, 跳过终极 TEST")
        return None
    weights = {fn: stable_icir[fn] for fn in stable_factors
               if fn in stable_icir}
    if not weights:
        log.warning("稳定因子无权重, 跳过终极 TEST")
        return None

    # ★ 风格均衡: 正负方向权重各自归一化到等和
    w_pos = sum(v for v in weights.values() if v > 0)
    w_neg = sum(-v for v in weights.values() if v < 0)
    if w_pos > 1e-9 and w_neg > 1e-9:
        for fn in weights:
            if weights[fn] > 0:
                weights[fn] = weights[fn] / w_pos
            else:
                weights[fn] = -weights[fn] / w_neg
        log.info("  风格均衡: 正方向权重合计=%+.2f, 负方向合计=%+.2f",
                 sum(v for v in weights.values() if v > 0),
                 sum(v for v in weights.values() if v < 0))

    log.info("")
    log.info("=" * 60)
    log.info("  终极 TEST (只跑一次): 稳定因子 %d 个, 权重=中位数ICIR(风格均衡)",
             len(weights))
    log.info("=" * 60)
    return run_backtest(all_data, factor_panels, close_panel, calendar,
                        cal_idx, factor_names, bt_config,
                        test_start, test_end, label="TEST",
                        fixed_weights=weights, universe_fn=universe_fn)


# ═══════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Walk-Forward 回测: 滚动ICIR(默认) 或 方案C Fold(推荐)")
    parser.add_argument("--dev-only", action="store_true",
                        help="仅跑 Development")
    parser.add_argument("--test-only", action="store_true",
                        help="仅跑 TEST")
    parser.add_argument("--folds", action="store_true",
                        help="方案C: 5-fold Walk-Forward + 终极 TEST")
    parser.add_argument("--liquid", action="store_true",
                        help="使用流动性 PIT universe (全市场+过滤, 方案C推荐)")
    parser.add_argument("--unlock-test", action="store_true",
                        help="解锁终极 TEST (仅供已确认的重新验证)")
    parser.add_argument("--sample", type=int, default=None,
                        help="抽样股票数 (快速测试用)")
    parser.add_argument("--max-stocks", type=int, default=None,
                        help="最多使用股票数")
    args = parser.parse_args()

    config = load_config(os.path.join(BASE_DIR, "config.yaml"))

    # 前置中性化开关 (config.yaml neutralization 段)
    _neut = config.get("neutralization", {}) or {}
    neutralize_enabled = bool(_neut.get("enabled", False))
    neutralize_k = float(_neut.get("winsorize_k", 3.0))
    log.info("  前置中性化: %s (MAD去极值 k=%.1f)",
             "开启" if neutralize_enabled else "关闭", neutralize_k)

    # 终极 TEST 锁 (方案C: 2025-01~2026-06 只跑一次)
    TEST_LOCK_PATH = os.path.join(IC_DIR, ".test_lock_v4")
    if args.unlock_test and os.path.exists(TEST_LOCK_PATH):
        os.remove(TEST_LOCK_PATH)
        log.warning("  🔓 终极 TEST 锁已解除")
    if args.folds and not args.unlock_test and os.path.exists(TEST_LOCK_PATH):
        with open(TEST_LOCK_PATH, "r", encoding="utf-8") as f:
            lock_info = json.load(f)
        log.error("=" * 60)
        log.error("  🚫 终极 TEST 已锁定！不可重复运行。")
        log.error("  锁定时间: %s", lock_info.get("locked_at", "unknown"))
        log.error("  结果文件: %s", OUTPUT_PATH)
        log.error("  如确需重跑: --unlock-test")
        log.error("=" * 60)
        sys.exit(1)

    # universe 提供函数
    from data.pit_universe import get_liquid_universe
    universe_fn = get_liquid_universe if args.liquid else get_universe

    if args.folds:
        partitions_to_run = {"development": PARTITIONS["development"]}
    elif args.dev_only:
        partitions_to_run = {"development": PARTITIONS["development"]}
    elif args.test_only:
        partitions_to_run = {"test": PARTITIONS["test"]}
    else:
        partitions_to_run = dict(PARTITIONS)

    # 日期范围静态检查 (不触碰盲测期)
    for label, (s, e) in partitions_to_run.items():
        try:
            check_date_range(s, e, config, script_name="run_walkforward_backtest")
        except GateViolation as ex:
            log.error(str(ex))
            sys.exit(1)

    log.info("=" * 60)
    if args.folds:
        log.info("  方案C Walk-Forward Fold 分析 (训练→验证)")
    else:
        log.info("  Walk-Forward 回测 (动态IC权重, 滚动252天)")
    log.info("  分区: %s", ", ".join(partitions_to_run.keys()))
    log.info("  universe: %s", "流动性PIT(全市场+过滤)" if args.liquid else "指数成分(CSI300+ZZ500)")
    log.info("=" * 60)

    bt_config = load_bt_config()
    log.info("  top_k=%d, lot=%d, slippage=%dbps, 调仓=%d天",
             bt_config["top_k"], bt_config["lot_size"],
             bt_config["slippage_bps"], bt_config["rebalance_days"])

    # ── 1. 加载数据 (复用 run_corrected_backtest 逻辑) ──
    log.info("加载数据...")
    from data_cache import get_cached_symbols, load
    syms = get_cached_symbols()
    if args.sample and args.sample < len(syms):
        import random
        random.seed(42)
        syms = random.sample(syms, args.sample)
    elif args.max_stocks and args.max_stocks < len(syms):
        syms = syms[:args.max_stocks]
    all_data = {}
    for sym in syms:
        df = load(sym)
        if df is not None and len(df) >= 250:
            all_data[sym] = df
    log.info("  有效: %d 只", len(all_data))

    # ── 2. 交易日历 + 收盘价面板 ──
    calendar = build_calendar(all_data)
    cal_idx = {d: i for i, d in enumerate(calendar)}
    log.info("  交易日历: %d 天 (%s ~ %s)", len(calendar),
             calendar[0].date(), calendar[-1].date())

    log.info("构建收盘价面板...")
    close_panel = build_close_panel(all_data, calendar)

    # ── 3. 确定所需日期 (调仓日 + IC窗口内月度观测日) ──
    from factor_scorer import FactorScorer
    if args.folds:
        # 方案C (fold): v5 预设 = 169 价量 + 13 基本面 (fund_*) 因子
        factor_names = sorted(FactorScorer.from_preset("full_auto_v5").factor_weights.keys())
    else:
        factor_names = sorted(FactorScorer.from_preset("full_auto").factor_weights.keys())

    needed = set()
    if args.folds:
        # 方案C: fold 训练期(全观测点) + 验证期调仓日 + 终极TEST
        for fold in FOLDS:
            ts, te = fold["train"]
            vs, ve = fold["val"]
            for d in calendar:
                d_ = d.date()
                if pd.Timestamp(ts).date() <= d_ <= pd.Timestamp(te).date():
                    needed.add(d)
                elif pd.Timestamp(vs).date() <= d_ <= pd.Timestamp(ve).date():
                    needed.add(d)
        # 终极 TEST (方案C: 2025-01 ~ 2026-06)
        test_s, test_e = "2025-01-01", "2026-06-30"
        for d in calendar:
            if pd.Timestamp(test_s).date() <= d.date() <= pd.Timestamp(test_e).date():
                needed.add(d)
    else:
        for label, (s, e) in partitions_to_run.items():
            part_dates = [d for d in calendar
                          if pd.Timestamp(s).date() <= d.date() <= pd.Timestamp(e).date()]
            for di, d in enumerate(part_dates):
                if di % bt_config["rebalance_days"] == 0:
                    needed.add(d)
                    i = cal_idx[d]
                    start_i = max(0, i - TRAIL_DAYS)
                    end_i = i - LABEL_HORIZON
                    for oi in range(start_i, end_i + 1, IC_STEP):
                        needed.add(calendar[oi])
    needed_dates = sorted(needed)
    log.info("  因子面板所需日期: %d 天 (%s ~ %s)",
             len(needed_dates), needed_dates[0].date(), needed_dates[-1].date())

    # ── 4. 预计算因子面板 ──
    log.info("预计算因子面板...")
    factor_panels = precompute_factor_panels(
        all_data, factor_names, needed_dates,
        include_fundamental=bool(args.folds),
        neutralize_enabled=neutralize_enabled,
        neutralize_k=neutralize_k)
    log.info("  面板就绪: %d 因子", len(factor_panels))

    # ── 5. 回测 (带日期守卫) ──
    results = {}
    extra_meta = {}
    with DateRangeGuard(config, script_name="run_walkforward_backtest") as guard:
        if args.folds:
            # 方案C: 5-fold 分析 + 终极 TEST
            guard.check_range("2015-01-01", "2026-06-30")
            fold_out = run_fold_analysis(
                all_data, factor_panels, close_panel, calendar, cal_idx,
                factor_names, bt_config, universe_fn=universe_fn)
            for k, v in fold_out.get("folds", {}).items():
                results[k] = v
            extra_meta["fold_factor_hits"] = fold_out["factor_hits"]
            extra_meta["stable_factors"] = fold_out["stable_factors"]
            extra_meta["stable_factor_icir_median"] = (
                fold_out["stable_factor_icir_median"])
            r = run_fold_test(
                all_data, factor_panels, close_panel, calendar, cal_idx,
                factor_names, bt_config,
                fold_out["stable_factors"],
                fold_out["stable_factor_icir_median"],
                "2025-01-01", "2026-06-30", universe_fn=universe_fn)
            if r:
                results["test"] = r
                # 终极 TEST 锁 (只跑一次纪律)
                with open(TEST_LOCK_PATH, "w", encoding="utf-8") as f:
                    json.dump({
                        "locked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "period": "2025-01-01 ~ 2026-06-30",
                        "script": "run_walkforward_backtest.py --folds",
                        "output": OUTPUT_PATH,
                    }, f, ensure_ascii=False, indent=2)
                log.info("  🔒 终极 TEST 锁已写入: %s", TEST_LOCK_PATH)
        else:
            for label, (s, e) in partitions_to_run.items():
                log.info("")
                guard.check_range(s, e)
                r = run_backtest(all_data, factor_panels, close_panel,
                                 calendar, cal_idx, factor_names,
                                 bt_config, s, e, label=label.upper(),
                                 universe_fn=universe_fn)
                if r:
                    results[label] = r

    # ── 6. 保存 ──
    output = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": ("方案C Fold Walk-Forward (训练→验证固定权重)"
                            if args.folds
                            else "Walk-Forward 回测: 每个调仓日用滚动252天 ICIR 动态加权"),
            "design": {
                "trail_days": TRAIL_DAYS,
                "label_horizon": LABEL_HORIZON,
                "ic_step": IC_STEP,
                "min_icir": MIN_ICIR,
                "min_ic_obs": MIN_IC_OBS,
                "min_cross_section": MIN_CROSS_SECTION,
                "rebalance_days": REBALANCE_DAYS,
                "no_lookahead": "标签为 [T-252, T-21] 窗口内的21日前瞻收益, 在T时刻已全部实现",
                "pit_universe": ("流动性PIT(全市场+上市1年+成交额500万)"
                                 if args.liquid else "CSI300+ZZ500 月度成分"),
                "expanding_window": "早期数据不足252天时自动退化为扩展窗口",
                **({"fold_min_hits": FOLD_MIN_HITS,
                    "fold_icir_min": FOLD_ICIR_MIN,
                    "folds": FOLDS,
                    "ultimate_test": "2025-01-01 ~ 2026-06-30 (只跑一次)"}
                   if args.folds else {}),
            },
            "n_stocks": len(all_data),
            "n_factors": len(factor_names),
            "bt_config": {k: v for k, v in bt_config.items()},
        },
        "results": results,
    }
    if extra_meta:
        output["meta"].update(extra_meta)
    os.makedirs(IC_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── 汇总 ──
    log.info("")
    log.info("=" * 60)
    log.info("  %-15s %8s %8s %8s %8s %6s %6s",
             "Period", "Excess", "IR", "MaxDD", "Sharpe", "PIT", "NFac")
    log.info("  " + "-" * 68)
    for label, r in results.items():
        log.info("  %-15s %+7.1f%% %7.2f %7.1f%% %7.2f %6d %6d",
                 label.upper(), r["excess_annual"], r["ir"],
                 r["max_drawdown"], r["sharpe"],
                 r["avg_pit_size"], r.get("avg_n_selected_factors", 0))
    if args.folds:
        log.info("  " + "-" * 68)
        log.info("  稳定因子 %d 个 (≥%d/5 folds, |ICIR|≥%.2f)",
                 len(extra_meta.get("stable_factors", [])),
                 FOLD_MIN_HITS, FOLD_ICIR_MIN)
    log.info("=" * 60)
    log.info("  结果: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
