"""
scripts/active/run_lgbm_fold_compare.py — P1: LGBM vs Ridge fold 对比 (2026-08-09)

严格 walk-forward: fold i 训练期训练模型, 验证期预测 → 截面 Spearman IC 对比。
纯研究脚本, 不改变生产路径 (run_walkforward_backtest.py 不动), 结果不进生产信号。

方法论 (与主回测对齐):
  - fold 结构: 复用 run_walkforward_backtest.FOLDS (5 折, 训练期逐年扩展, 验证期固定 1 年)
  - 特征: full_auto_v5 因子面板 (precompute_factor_panels 直接复用, 同主回测 flags:
    fundamental + aux + minute(按config) + 前置中性化(按config))
  - 训练期因子筛选: compute_icir_weights 固定窗口模式 (|ICIR|>=0.02, n_obs>=6,
    与主回测 MIN_ICIR/MIN_IC_OBS 一致) — 只用训练期数据筛选, 无验证期泄漏
  - 标签: 21 日前瞻收益 close[t+21]/close[t]-1 (LABEL_HORIZON=21, 与主回测一致)
  - 严格无泄漏:
      * 训练样本特征日期 ≤ train_end - 21 (标签完全在验证期开始前实现,
        与 compute_icir_weights 固定窗口的 end_idx 规则一致)
      * 特征标准化 (z-score) 的均值/方差只从训练行估计, 验证期用同一统计量
      * 验证期 IC 逐月观测 (IC_STEP=21), 标签在验证期内实现 (≤ val_end - 21)
      * 早停验证集按时间顺序取训练期最后 20% 日期 (不随机打乱, 防时序泄漏)
  - 模型 (gate 约束: n_estimators<=100, max_depth<=3):
      * LGBM: lightgbm 回归, max_depth=3 / num_leaves=8 / num_boost_round=100 / 早停
      * Ridge: model.baselines.L2LinearRanker(alpha=1.0) (同一 X, 同一行过滤)
  - IC: 截面 Spearman 秩相关 (与 rank_corr_cols 同法), universe 默认指数成分 PIT
    (get_universe, 同主回测默认), --liquid 切换流动性 PIT universe

输出:
  data/ic_validation/lgbm_fold_compare.json
    {fold_i: {"lgbm_ic_mean": x, "ridge_ic_mean": y,
              "lgbm_icir": x, "ridge_icir": y, "n_days": n, ...}}
    另含 _meta: 参数/因子数/样本保留率/耗时

可靠性 (长任务防中断, 后台任务约 60min 即被终止):
  - 因子面板分两阶段落盘缓存 (data/ic_validation/panels_cache_lgbm/):
      阶段1 原始面板 (precompute, ~40min) → 阶段2 中性化面板 (~30min)
    缓存键 = 日期集+因子集+minute开关+股票集+中性化标记, 不匹配自动重建;
    每个阶段独立缓存, 中断后重跑只补未完成的阶段
  - 逐 fold 落盘: 每个 fold 完成后立即写 JSON, 中断后重跑自动合并已完成的 fold
    (仅当 _meta 运行键一致时, 防止不同宇宙/参数的结果混入)
  - 训练矩阵按 fold 一次性向量化切片 (np.stack 三维块), 减少逐日 column_stack 开销

用法:
  py scripts/active/run_lgbm_fold_compare.py            # 全量 (5005 只, 5 folds)
  py scripts/active/run_lgbm_fold_compare.py --sample 300   # 抽样调试
  py scripts/active/run_lgbm_fold_compare.py --liquid       # 流动性 PIT universe
  py scripts/active/run_lgbm_fold_compare.py --no-minute    # 排除分钟因子
  py scripts/active/run_lgbm_fold_compare.py --folds 2      # 只跑前 2 个 fold
"""

import argparse
import gc
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from logger import get_logger
from gate import load_config, DateRangeGuard

# ── 复用主回测构建逻辑 (只读导入, 不修改生产路径) ──
from scripts.active.run_walkforward_backtest import (
    FOLDS,
    LABEL_HORIZON,
    IC_STEP,
    MIN_CROSS_SECTION,
    MIN_ICIR,
    MIN_IC_OBS,
    build_calendar,
    build_close_panel,
    precompute_factor_panels,
    neutralize_factor,
    compute_icir_weights,
    _nearest_idx,
    _load_industry_map,
)

log = get_logger("lgbm_fold_compare")

IC_DIR = os.path.join(BASE_DIR, "data", "ic_validation")
OUTPUT_PATH = os.path.join(IC_DIR, "lgbm_fold_compare.json")
PANEL_CACHE_DIR = os.path.join(IC_DIR, "panels_cache_lgbm")

# ── 研究期 (fold 1-5 训练+验证全覆盖, 2015~2024; 不触碰 TEST②/blind) ──
RESEARCH_START = "2015-01-01"
RESEARCH_END = "2024-12-31"

# gate 模型复杂度约束 (代码强制, 防过拟合)
MAX_N_ESTIMATORS = 100
MAX_DEPTH = 3

# LGBM 超参 (gate 内; 学习率/正则参考 config model.research_lgb 与生产 MLRanker)
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "max_depth": MAX_DEPTH,
    "num_leaves": 2 ** MAX_DEPTH,
    "learning_rate": 0.05,
    "min_data_in_leaf": 50,
    "lambda_l1": 0.5,
    "lambda_l2": 1.0,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 42,
    "deterministic": True,
}
EARLY_STOPPING_ROUNDS = 20
VAL_SPLIT_RATIO = 0.20        # 训练期按日期数分割的最后 20% 作为早停验证
RIDGE_ALPHA = 1.0


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """单列 Spearman 秩相关 (与 rank_corr_cols 同法, 无 NaN 输入)。"""
    from scipy.stats import rankdata
    if len(x) < MIN_CROSS_SECTION:
        return np.nan
    if (x.max() - x.min()) < 1e-12 or (y.max() - y.min()) < 1e-12:
        return np.nan
    r = np.corrcoef(rankdata(x), rankdata(y))[0, 1]
    return float(r) if not np.isnan(r) else np.nan


def build_needed_dates(calendar: list) -> list:
    """fold 训练期 + 验证期全部日期 (2015~2024 并集)。"""
    needed = set()
    for fold in FOLDS:
        ts, te = fold["train"]
        vs, ve = fold["val"]
        for d in calendar:
            d_ = d.date()
            if pd.Timestamp(ts).date() <= d_ <= pd.Timestamp(te).date():
                needed.add(d)
            elif pd.Timestamp(vs).date() <= d_ <= pd.Timestamp(ve).date():
                needed.add(d)
    return sorted(needed)


def _date_universe_mask(universe_fn, date_ts, close_panel: pd.DataFrame) -> np.ndarray | None:
    """某日期 PIT universe 掩码 (全股票 bool), 失败返回 None (不过滤)。"""
    if universe_fn is None:
        return None
    try:
        uni = set(universe_fn(str(date_ts.date())))
    except Exception:
        return None
    if not uni:
        return None
    return np.array([s in uni for s in close_panel.columns])


# ── 面板缓存 (构建 ~1h, 防中断重跑白费) ──

def panels_cache_key(needed_dates: list, factor_names: list,
                     minute_enabled: bool, stock_codes: list,
                     neutralized: bool = False) -> dict:
    """缓存键: 日期集 + 请求因子集 + minute 开关 + 股票集 + 中性化标记
    (任一变化即失效重建; 原始/中性化两阶段独立缓存)。"""
    return {
        "dates": [str(d.date()) for d in needed_dates],
        "factors": sorted(factor_names),
        "minute": bool(minute_enabled),
        "stocks": sorted(stock_codes),
        "neutralized": bool(neutralized),
    }


def _key_hash(cache_key: dict) -> str:
    """缓存键哈希: 不同键 (股票集/因子集/阶段) 的文件互不覆盖。"""
    import hashlib
    raw = json.dumps(cache_key, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:12]


def _cache_paths(cache_key: dict) -> tuple:
    h = _key_hash(cache_key)
    return (os.path.join(PANEL_CACHE_DIR, f"meta_{h}.json"),
            os.path.join(PANEL_CACHE_DIR, f"panel_{h}"))


def try_load_panels_cache(cache_key: dict) -> dict | None:
    """从磁盘加载面板缓存; 键不匹配/文件缺失返回 None。"""
    meta_path, panel_dir = _cache_paths(cache_key)
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("cache_key") != cache_key:
            log.info("  面板缓存键不匹配, 重建")
            return None
        panels = {}
        for fn in meta["factors_actual"]:
            p = pd.read_parquet(os.path.join(panel_dir, f"{fn}.parquet"))
            panels[fn] = p
        log.info("  面板缓存加载: %d 因子 × %d 日期 × %d 股票",
                 len(panels), len(meta["cache_key"]["dates"]),
                 len(panels[list(panels)[0]].columns) if panels else 0)
        return panels
    except Exception as e:
        log.warning("  面板缓存加载失败 (%s), 重建", e)
        return None


def save_panels_cache(factor_panels: dict, cache_key: dict) -> None:
    """保存面板到磁盘 (每因子一个 parquet + meta, 按键哈希隔离)。"""
    os.makedirs(PANEL_CACHE_DIR, exist_ok=True)
    factors_actual = sorted(factor_panels.keys())
    meta_path, panel_dir = _cache_paths(cache_key)
    os.makedirs(panel_dir, exist_ok=True)
    meta = {
        "cache_key": cache_key,
        "factors_actual": factors_actual,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    t0 = time.time()
    for fn in factors_actual:
        factor_panels[fn].to_parquet(os.path.join(panel_dir, f"{fn}.parquet"))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    log.info("  面板缓存已写入: %d 因子 (%.0fs)", len(factors_actual), time.time() - t0)


# ── 中性化并行化 (同一 neutralize_factor 函数, 逐因子进程级并行;
#    worker 直接从 raw 缓存 parquet 读/写, 避免大对象 pickle) ──
_NEUT_TMP_DIR = os.path.join(IC_DIR, "panels_cache_lgbm_tmp")


def _neutralize_worker(args) -> str:
    """单因子 worker: 读 raw 面板 parquet → neutralize_factor → 写 tmp。"""
    fn, k, industry_map, raw_dir = args
    from scripts.active.run_walkforward_backtest import neutralize_factor
    p = pd.read_parquet(os.path.join(raw_dir, f"{fn}.parquet"))
    out = neutralize_factor(p, k=k, industry_map=industry_map)
    out.to_parquet(os.path.join(_NEUT_TMP_DIR, f"{fn}.parquet"))
    return fn


def neutralize_panels_parallel(factor_names: list, k: float,
                               industry_map: dict | None,
                               workers: int = 8,
                               raw_dir: str | None = None) -> dict:
    """逐因子并行中性化 (与主回测 neutralize_factor 完全同函数/同参数)。"""
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(_NEUT_TMP_DIR, exist_ok=True)
    args = [(fn, k, industry_map, raw_dir) for fn in factor_names]
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(_neutralize_worker, args, chunksize=4):
            done += 1
            if done % 40 == 0:
                log.info("  中性化并行: %d/%d 因子 (%.0fs)",
                         done, len(factor_names), time.time() - t0)
    panels = {fn: pd.read_parquet(os.path.join(_NEUT_TMP_DIR, f"{fn}.parquet"))
              for fn in factor_names}
    import shutil
    shutil.rmtree(_NEUT_TMP_DIR, ignore_errors=True)
    return panels


def run_fold(fi: int, fold: dict, factor_panels: dict, close_panel: pd.DataFrame,
             calendar: list, cal_idx: dict, needed_dates: list,
             universe_fn, max_factors: int, log) -> dict | None:
    """单个 fold: 训练期选因子 + 训练 LGBM/Ridge → 验证期逐月截面 IC。"""
    ts, te = fold["train"]
    vs, ve = fold["val"]
    t0 = time.time()

    # ── 验证期首个交易日 (与 run_fold_analysis 同逻辑) ──
    val_first = None
    for d in calendar:
        if pd.Timestamp(vs).date() <= d.date() <= pd.Timestamp(ve).date():
            val_first = d
            break
    if val_first is None:
        log.warning("  Fold %d: 验证期无交易日, 跳过", fi + 1)
        return None

    # ── 训练期因子筛选 (固定窗口 ICIR, 严格无泄漏) ──
    weights, ic_stats = compute_icir_weights(
        factor_panels, close_panel, calendar, cal_idx,
        val_first, list(factor_panels.keys()),
        train_start=ts, train_end=te, universe_fn=universe_fn)
    if not weights:
        log.warning("  Fold %d: 训练期无因子达标 (|ICIR|>%.2f), 跳过", fi + 1, MIN_ICIR)
        return None
    # top max_factors (同主回测 fold.max_factors=40 上限): 控维度 + 提升特征覆盖率
    selected = sorted(weights.keys(), key=lambda fn: -abs(weights[fn]))[:max_factors]
    log.info("  Fold %d: 训练期筛选因子 %d/%d 个 (|ICIR|>%.2f, top%d)",
             fi + 1, len(selected), len(weights), MIN_ICIR, max_factors)

    # ── 训练/验证日期索引 ──
    s_idx = _nearest_idx(cal_idx, ts)
    e_idx = _nearest_idx(cal_idx, te)
    t_idx = cal_idx[val_first]
    # 严格无泄漏: 训练观测点标签(21日前瞻)完全在验证期开始前实现
    end_idx = min(e_idx - LABEL_HORIZON, t_idx - LABEL_HORIZON)
    v_s_idx = _nearest_idx(cal_idx, vs)
    v_e_idx = _nearest_idx(cal_idx, ve)
    if end_idx <= s_idx or v_e_idx - LABEL_HORIZON < v_s_idx:
        log.warning("  Fold %d: 训练/验证窗口过短, 跳过", fi + 1)
        return None

    # 面板行定位: 所有面板共享同一 DatetimeIndex (needed_dates)
    row_of = {d: i for i, d in enumerate(needed_dates)}
    sel_arrs = [factor_panels[fn].to_numpy(dtype=np.float32) for fn in selected]
    close_vals = close_panel.to_numpy()

    # ── 训练矩阵 (逐日展平截面, 保留标签有效 + 至少一个特征有效 + PIT universe 行) ──
    # 缺失特征 (部分因子 NaN) 在标准化后按训练均值(0)填补 — 与生产 IC 按因子
    # 部分有效计算/打分部分缺失可比的建模侧对应; 全特征缺失的行 (无信息) 剔除。
    train_dates = list(range(s_idx, end_idx + 1))          # 日历位置
    group_of = {cal_pos: gi for gi, cal_pos in enumerate(train_dates)}
    # 向量化: 训练期日期在 needed_dates 中连续 → 一次切片全部选中因子面板为三维块
    row_start = row_of[calendar[s_idx]]
    row_end = row_of[calendar[end_idx]]
    if row_end - row_start + 1 == len(train_dates):
        blocks = [factor_panels[fn].iloc[row_start:row_end + 1]
                  .to_numpy(dtype=np.float32) for fn in selected]
        train_block = np.stack(blocks, axis=2)             # (n_dates, n_stocks, n_sel)
    else:
        train_block = None
    x_chunks, y_chunks, g_chunks = [], [], []
    n_rows_possible = 0
    n_rows_kept = 0
    for i in train_dates:
        d = calendar[i]
        if train_block is not None:
            x = train_block[i - s_idx]
        else:
            x = np.column_stack([p[row_of[d]] for p in sel_arrs])  # (n_stocks, n_sel)
        c0 = close_vals[i]
        c1 = close_vals[i + LABEL_HORIZON]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = c1 / c0 - 1
        valid_px = (c0 > 0) & ~np.isnan(c0) & ~np.isnan(c1)
        feat_any = ~np.isnan(x).all(axis=1)
        valid = valid_px & feat_any
        um = _date_universe_mask(universe_fn, d, close_panel)
        if um is not None:
            valid = valid & um
        n_rows_possible += valid_px.sum()
        n_rows_kept += int(valid.sum())
        if int(valid.sum()) < MIN_CROSS_SECTION:
            continue
        x_chunks.append(x[valid])
        y_chunks.append(ratio[valid])
        g_chunks.append(np.full(int(valid.sum()), group_of[i], dtype=np.int32))

    if train_block is not None:
        del train_block

    if not x_chunks:
        log.warning("  Fold %d: 训练期无有效样本, 跳过", fi + 1)
        return None
    X = np.concatenate(x_chunks).astype(np.float32)
    y = np.concatenate(y_chunks).astype(np.float32)
    groups = np.concatenate(g_chunks)
    del x_chunks, y_chunks, g_chunks
    retention = n_rows_kept / n_rows_possible if n_rows_possible > 0 else 0.0
    nan_frac = float(np.isnan(X).mean())
    log.info("  Fold %d: 训练样本 %d 行 × %d 因子 (保留率 %.1f%%, 特征缺失率 %.1f%%)",
             fi + 1, len(y), X.shape[1], retention * 100, nan_frac * 100)

    # ── 特征标准化 (统计量只从训练行估计) + NaN→0 (训练均值填补) ──
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-9)] = 1.0
    X = (X - mu) / sd
    X = np.where(np.isnan(X), 0.0, X).astype(np.float32)

    # ── 时间顺序早停切分 (最后 20% 日期组) ──
    uni_groups = np.unique(groups)
    split_g = int(len(uni_groups) * (1 - VAL_SPLIT_RATIO))
    tr_mask = groups < uni_groups[split_g]
    va_mask = ~tr_mask
    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_va, y_va = X[va_mask], y[va_mask]
    log.info("  Fold %d: 训练 %d 行 / 早停验证 %d 行 (按日期 %d/%d 天分割)",
             fi + 1, len(y_tr), len(y_va), split_g, len(uni_groups))

    # ── LGBM (gate: n_estimators<=100, max_depth<=3) ──
    assert LGBM_PARAMS["max_depth"] <= MAX_DEPTH
    import lightgbm as lgb
    tr_ds = lgb.Dataset(X_tr, label=y_tr)
    va_ds = lgb.Dataset(X_va, label=y_va, reference=tr_ds)
    lgb_model = lgb.train(
        LGBM_PARAMS, tr_ds, num_boost_round=MAX_N_ESTIMATORS,
        valid_sets=[va_ds],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    best_iter = getattr(lgb_model, "best_iteration", MAX_N_ESTIMATORS)
    log.info("  Fold %d: LGBM 完成 (best_iter=%d/%d)",
             fi + 1, best_iter, MAX_N_ESTIMATORS)

    # ── Ridge (同一 X, 同一训练行) ──
    from model.baselines import L2LinearRanker
    ridge = L2LinearRanker(alpha=RIDGE_ALPHA)
    ridge.fit(X_tr, y_tr)
    log.info("  Fold %d: Ridge 完成 (alpha=%.1f)", fi + 1, RIDGE_ALPHA)

    # ── 验证期逐月截面 IC (IC_STEP=21, 标签在验证期内实现) ──
    offsets = list(range(v_s_idx, v_e_idx - LABEL_HORIZON + 1, IC_STEP))
    lgbm_ics, ridge_ics, n_obs_list = [], [], []
    for oi in offsets:
        d = calendar[oi]
        x = np.column_stack([p[row_of[d]] for p in sel_arrs])
        c0 = close_vals[oi]
        c1 = close_vals[oi + LABEL_HORIZON]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = c1 / c0 - 1
        valid_px = (c0 > 0) & ~np.isnan(c0) & ~np.isnan(c1)
        feat_any = ~np.isnan(x).all(axis=1)
        valid = valid_px & feat_any
        um = _date_universe_mask(universe_fn, d, close_panel)
        if um is not None:
            valid = valid & um
        if int(valid.sum()) < MIN_CROSS_SECTION:
            continue
        xs = (x - mu) / sd
        xs = np.where(np.isnan(xs), 0.0, xs).astype(np.float32)
        xs = xs[valid]
        ret = ratio[valid]
        s_lgbm = lgb_model.predict(xs)
        s_ridge = ridge.predict(xs)
        ic_l = _spearman(s_lgbm, ret)
        ic_r = _spearman(s_ridge, ret)
        if not np.isnan(ic_l):
            lgbm_ics.append(ic_l)
        if not np.isnan(ic_r):
            ridge_ics.append(ic_r)
        n_obs_list.append(int(valid.sum()))

    if not lgbm_ics or not ridge_ics:
        log.warning("  Fold %d: 验证期无有效 IC 观测, 跳过", fi + 1)
        return None

    def _ic_stats(ics: list) -> tuple:
        arr = np.array(ics)
        sd = arr.std()
        return float(arr.mean()), float(arr.mean() / sd) if sd > 1e-9 else 0.0

    lgbm_mean, lgbm_icir = _ic_stats(lgbm_ics)
    ridge_mean, ridge_icir = _ic_stats(ridge_ics)
    log.info("  Fold %d: LGBM IC=%.4f ICIR=%.3f | Ridge IC=%.4f ICIR=%.3f | %d obs",
             fi + 1, lgbm_mean, lgbm_icir, ridge_mean, ridge_icir, len(lgbm_ics))

    return {
        "train": f"{ts} ~ {te}",
        "val": f"{vs} ~ {ve}",
        "n_selected_factors": len(selected),
        "n_train_rows": int(len(y)),
        "train_retention": round(retention, 4),
        "feature_nan_frac": round(nan_frac, 4),
        "lgbm_best_iter": int(best_iter),
        "n_days": int(len(lgbm_ics)),
        "n_obs_avg": int(np.mean(n_obs_list)) if n_obs_list else 0,
        "lgbm_ic_mean": round(lgbm_mean, 5),
        "ridge_ic_mean": round(ridge_mean, 5),
        "lgbm_icir": round(lgbm_icir, 4),
        "ridge_icir": round(ridge_icir, 4),
        "lgbm_ic_median": round(float(np.median(lgbm_ics)), 5),
        "ridge_ic_median": round(float(np.median(ridge_ics)), 5),
        "runtime_s": round(time.time() - t0, 1),
    }


def write_output(results: dict, run_key: dict, factor_names: list,
                 needed_dates: list, args, minute_enabled: bool,
                 neutralize_enabled: bool, max_factors: int) -> None:
    """写结果 JSON (逐 fold 落盘; _meta 含 run_key 供恢复校验)。"""
    os.makedirs(IC_DIR, exist_ok=True)
    out = {
        **results,
        "_meta": {
            "script": "scripts/active/run_lgbm_fold_compare.py",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **run_key,
            "label_horizon": LABEL_HORIZON,
            "ic_step": IC_STEP,
            "lgbm_params": {k: LGBM_PARAMS[k] for k in
                            ("max_depth", "num_leaves", "learning_rate",
                             "min_data_in_leaf", "lambda_l1", "lambda_l2")},
            "lgbm_num_boost_round": MAX_N_ESTIMATORS,
            "lgbm_early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "ridge_alpha": RIDGE_ALPHA,
            "min_icir": MIN_ICIR,
            "min_ic_obs": MIN_IC_OBS,
        },
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="P1: LGBM vs Ridge fold 对比 (研究, 严格 walk-forward)")
    parser.add_argument("--sample", type=int, default=None,
                        help="抽样股票数 (快速调试)")
    parser.add_argument("--max-stocks", type=int, default=None,
                        help="最多使用股票数 (前 N 只)")
    parser.add_argument("--liquid", action="store_true",
                        help="使用流动性 PIT universe (默认: 指数成分)")
    parser.add_argument("--no-minute", action="store_true",
                        help="排除分钟因子 (默认按 config minute_factors.enabled)")
    parser.add_argument("--folds", type=int, default=None,
                        help="只跑前 N 个 fold (调试用)")
    parser.add_argument("--neutralize-workers", type=int, default=8,
                        help="中性化并行 worker 数 (默认 8; 1=单进程串行)")
    args = parser.parse_args()

    neutralize_workers = max(1, args.neutralize_workers)

    config = load_config(os.path.join(BASE_DIR, "config.yaml"))

    # ── 日期守卫: 研究期 2015~2024 在 research 分区内, 不触碰 TEST②/blind ──
    with DateRangeGuard(config, script_name="run_lgbm_fold_compare") as guard:
        guard.check_range(RESEARCH_START, RESEARCH_END)

    # ── 数据加载 (内联复刻主回测: data_cache 读 parquet) ──
    from data_cache import get_cached_symbols, load
    syms = get_cached_symbols()
    if args.sample and args.sample < len(syms):
        import random
        random.seed(42)
        syms = random.sample(syms, args.sample)
    elif args.max_stocks and args.max_stocks < len(syms):
        syms = syms[:args.max_stocks]
    log.info("加载日线数据: %d 只候选", len(syms))
    t0 = time.time()
    # 日线加载顺序与因子面板保持一致: 面板按 sorted(all_data.keys()) 构建,
    # close_panel 按 all_data 插入序 → 必须 sorted, 否则行掩码与矩阵错位。
    syms = sorted(syms)
    all_data = {}
    for sym in syms:
        try:
            df = load(sym)   # data_store 根目录含少量 aux_* 非日线 parquet, 防御性跳过
        except Exception:
            continue
        if df is not None and "date" in df.columns and len(df) >= 250:
            all_data[sym] = df
    log.info("  有效: %d 只 (%.0fs)", len(all_data), time.time() - t0)

    calendar = build_calendar(all_data)
    cal_idx = {d: i for i, d in enumerate(calendar)}
    close_panel = build_close_panel(all_data, calendar)
    log.info("  交易日历: %d 天 (%s ~ %s)", len(calendar),
             calendar[0].date(), calendar[-1].date())

    # ── 因子集 (同主回测 fold 路径: full_auto_v5 + 分钟因子按 config) ──
    from factor_scorer import FactorScorer
    factor_names = sorted(FactorScorer.from_preset("full_auto_v5").factor_weights.keys())
    minute_enabled = bool(config.get("minute_factors", {}).get("enabled", True)) and not args.no_minute
    if minute_enabled:
        from minute_factors import get_minute_factor_names
        factor_names = sorted(set(factor_names) | set(get_minute_factor_names()))
    log.info("  因子集: %d 个 (minute=%s)", len(factor_names), minute_enabled)

    # ── 面板所需日期 = fold 训练+验证期 (2015~2024) ──
    needed_dates = build_needed_dates(calendar)
    log.info("  因子面板所需日期: %d 天", len(needed_dates))

    # ── 前置中性化 (按 config, 同主回测) ──
    _neut = config.get("neutralization", {}) or {}
    neutralize_enabled = bool(_neut.get("enabled", False))
    neutralize_k = float(_neut.get("winsorize_k", 3.0))
    industry_neutral = bool(_neut.get("industry_neutral", False))

    # ── 因子面板 (复用主回测构建 + 两阶段磁盘缓存, 防 ~60min 后台终止白费) ──
    # 阶段1: 原始面板 (precompute, 不含中性化) → 阶段2: 中性化面板 (同主回测参数)。
    # neutralize_enabled=False 时两阶段合并为原始面板。
    if neutralize_enabled:
        factor_panels = try_load_panels_cache(
            panels_cache_key(needed_dates, factor_names, minute_enabled,
                             sorted(all_data.keys()), neutralized=True))
        if factor_panels is None:
            raw_panels = try_load_panels_cache(
                panels_cache_key(needed_dates, factor_names, minute_enabled,
                                 sorted(all_data.keys()), neutralized=False))
            if raw_panels is None:
                t1 = time.time()
                raw_panels = precompute_factor_panels(
                    all_data, factor_names, needed_dates,
                    include_fundamental=True,
                    include_aux=True,
                    include_minute=minute_enabled,
                    minute_lookback=int(config.get("minute_factors", {}).get("lookback", 20)),
                    neutralize_enabled=False,
                    neutralize_k=neutralize_k,
                    industry_map=None)
                log.info("  原始面板就绪: %d 因子 (%.0fs)", len(raw_panels), time.time() - t1)
                save_panels_cache(raw_panels, panels_cache_key(
                    needed_dates, factor_names, minute_enabled,
                    sorted(all_data.keys()), neutralized=False))
            factor_names_actual = sorted(raw_panels.keys())
            t2 = time.time()
            industry_map = _load_industry_map() if industry_neutral else None
            raw_dir = _cache_paths(panels_cache_key(
                needed_dates, factor_names, minute_enabled,
                sorted(all_data.keys()), neutralized=False))[1]
            del raw_panels          # worker 直接从磁盘读, 释放 ~10GB
            gc.collect()
            if neutralize_workers > 1:
                factor_panels = neutralize_panels_parallel(
                    list(factor_names_actual), neutralize_k, industry_map,
                    workers=neutralize_workers, raw_dir=raw_dir)
            else:
                from scripts.active.run_walkforward_backtest import neutralize_factor
                factor_panels = {}
                for fn in factor_names_actual:
                    p = pd.read_parquet(os.path.join(raw_dir, f"{fn}.parquet"))
                    factor_panels[fn] = neutralize_factor(p, k=neutralize_k,
                                                          industry_map=industry_map)
            log.info("  中性化完成: %d 因子 (MAD k=%.1f, 行业=%s, %.0fs)",
                     len(factor_panels), neutralize_k, industry_neutral, time.time() - t2)
            save_panels_cache(factor_panels, panels_cache_key(
                needed_dates, factor_names, minute_enabled,
                sorted(all_data.keys()), neutralized=True))
    else:
        factor_panels = try_load_panels_cache(
            panels_cache_key(needed_dates, factor_names, minute_enabled,
                             sorted(all_data.keys()), neutralized=False))
        if factor_panels is None:
            t1 = time.time()
            factor_panels = precompute_factor_panels(
                all_data, factor_names, needed_dates,
                include_fundamental=True,
                include_aux=True,
                include_minute=minute_enabled,
                minute_lookback=int(config.get("minute_factors", {}).get("lookback", 20)),
                neutralize_enabled=False,
                neutralize_k=neutralize_k,
                industry_map=None)
            log.info("  面板就绪: %d 因子 (%.0fs)", len(factor_panels), time.time() - t1)
            save_panels_cache(factor_panels, panels_cache_key(
                needed_dates, factor_names, minute_enabled,
                sorted(all_data.keys()), neutralized=False))
    log.info("  面板就绪: %d 因子", len(factor_panels))

    # ── 面板列对齐 (必要安全网): 部分因子面板列集可能不同
    # (如某股票缺某因子列), compute_icir_weights/column_stack 要求所有面板
    # 与 close_panel 列集一致 → 统一到并集 (缺列 NaN 填充), 列序均为排序序。
    panel_cols = sorted(set().union(*[set(p.columns) for p in factor_panels.values()]))
    n_reindex = 0
    for fn, p in factor_panels.items():
        if len(p.columns) != len(panel_cols):
            factor_panels[fn] = p.reindex(columns=panel_cols)
            n_reindex += 1
    close_panel = close_panel.reindex(columns=panel_cols)
    if n_reindex:
        log.info("  面板列对齐: %d 因子补列到 %d 只股票并集", n_reindex, len(panel_cols))

    # 释放原始数据 (面板已就绪, 后续不再需要)
    del all_data
    gc.collect()

    # ── universe 提供函数 (同主回测默认: 指数成分 PIT) ──
    if args.liquid:
        from data.pit_universe import get_liquid_universe
        universe_fn = get_liquid_universe
        log.info("  universe: 流动性 PIT (全市场+过滤)")
    else:
        from data.pit_universe import get_universe
        universe_fn = get_universe
        log.info("  universe: 指数成分 PIT (CSI300+ZZ500)")

    # ── fold 主循环 (逐 fold 落盘, 中断后可恢复) ──
    max_factors = int(config.get("fold", {}).get("max_factors", 40))
    run_key = {
        "n_stocks": int(len(close_panel.columns)),
        "n_dates": int(len(needed_dates)),
        "n_factors_total": len(factor_names),
        "universe": "liquid_pit" if args.liquid else "index_pit",
        "minute_factors": minute_enabled,
        "neutralize": neutralize_enabled,
        "max_factors_per_fold": max_factors,
    }
    results = {}
    # 恢复: 仅当已有 JSON 的 _meta 运行键一致时合并已完成 fold
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                prev = json.load(f)
            prev_meta = prev.get("_meta", {}) or {}
            if all(prev_meta.get(k) == v for k, v in run_key.items()):
                for k, v in prev.items():
                    if k.startswith("fold_"):
                        results[k] = v
                if results:
                    log.info("  恢复已完成 fold: %s", ", ".join(sorted(results)))
        except Exception:
            pass

    for fi, fold in enumerate(FOLDS):
        if args.folds and fi >= args.folds:
            break
        key = f"fold_{fi + 1}"
        if key in results:
            log.info("  跳过已完成: %s", key)
            continue
        log.info("─" * 60)
        log.info("  Fold %d: Train %s~%s → Val %s~%s",
                 fi + 1, fold["train"][0], fold["train"][1],
                 fold["val"][0], fold["val"][1])
        r = run_fold(fi, fold, factor_panels, close_panel, calendar, cal_idx,
                     needed_dates, universe_fn, max_factors, log)
        if r:
            results[key] = r
            write_output(results, run_key, factor_names, needed_dates, args,
                         minute_enabled, neutralize_enabled, max_factors)
        gc.collect()

    # ── 最终输出 ──
    os.makedirs(IC_DIR, exist_ok=True)
    write_output(results, run_key, factor_names, needed_dates, args,
                 minute_enabled, neutralize_enabled, max_factors)
    log.info("=" * 60)
    log.info("  结果已写入: %s", OUTPUT_PATH)
    log.info("  fold 数: %d / %d", len(results), len(FOLDS))
    for k, v in results.items():
        log.info("  %s: LGBM IC=%.4f/ICIR=%.3f | Ridge IC=%.4f/ICIR=%.3f | %d obs",
                 k, v["lgbm_ic_mean"], v["lgbm_icir"],
                 v["ridge_ic_mean"], v["ridge_icir"], v["n_days"])


if __name__ == "__main__":
    main()
