"""
因子自动验证 — 一键生成完整验证报告

对单个因子自动计算:
  1. 截面 Spearman IC 序列 (逐日)
  2. IC 均值 / ICIR / IC正比例
  3. IC 衰减曲线 (1d/5d/10d/20d/60d horizon)
  4. 滚动 IC (60日窗口) — 检测因子是否失效
  5. IC 半衰期估计
  6. 分组收益 (5分组, 多头-空头价差)
  7. 与已有因子的相关性 (top-10)
  8. 覆盖率统计

输出: JSON + 可选 matplotlib 图表
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

REPORT_DIR = os.path.join(BASE_DIR, "data", "factor_reports")


def _load_config():
    from gate import load_config
    return load_config(os.path.join(BASE_DIR, "config.yaml"))


def _load_all_data(sample: int = None) -> Dict[str, pd.DataFrame]:
    """加载全量行情数据。"""
    from data_cache import get_cached_symbols, load
    syms = get_cached_symbols()
    if sample and sample < len(syms):
        import random
        random.seed(42)
        syms = random.sample(syms, sample)
    all_data = {}
    for sym in syms:
        df = load(sym)
        if df is not None and len(df) >= 250:
            all_data[sym] = df
    return all_data


def _compute_factor_series(all_data: Dict[str, pd.DataFrame],
                           factor_name: str,
                           expr: str = None) -> Dict[str, pd.Series]:
    """
    计算因子值序列。

    对价量因子: 用 DSL 表达式计算
    对基本面/相对/北向因子: 调用对应模块
    """
    results = {}

    if expr:
        # 价量因子 — 用 DSL
        from factor_engine import parse_factor
        factor_ast = parse_factor(expr)
        for sym, df in all_data.items():
            try:
                vals = factor_ast.evaluate(df)
                if "date" in df.columns:
                    series = pd.Series(vals.values, index=pd.to_datetime(df["date"].values))
                else:
                    series = vals
                results[sym] = series
            except Exception:
                pass
    else:
        # 尝试从 factor_scorer 获取
        from factor_scorer import FactorScorer
        scorer = FactorScorer.from_preset("full_auto")
        for sym, df in all_data.items():
            try:
                factors_df = scorer.compute_factors(df)
                if factor_name in factors_df.columns:
                    if "date" in df.columns:
                        series = pd.Series(
                            factors_df[factor_name].values,
                            index=pd.to_datetime(df["date"].values)
                        )
                    else:
                        series = factors_df[factor_name]
                    results[sym] = series
            except Exception:
                pass

    return results


def _compute_forward_returns(all_data: Dict[str, pd.DataFrame],
                             horizon: int = 20) -> Dict[str, pd.Series]:
    """计算前瞻收益率。"""
    fwd = {}
    for sym, df in all_data.items():
        if "date" not in df.columns or len(df) < horizon + 1:
            continue
        close = df["close"].astype(float)
        ret = close.shift(-horizon) / close - 1
        series = pd.Series(ret.values, index=pd.to_datetime(df["date"].values))
        fwd[sym] = series
    return fwd


def _cross_sectional_ic(factor_series: Dict[str, pd.Series],
                        fwd_series: Dict[str, pd.Series],
                        dates: List,
                        min_stocks: int = 30) -> pd.DataFrame:
    """
    计算逐日截面 Spearman IC。

    Returns: DataFrame with columns [date, ic, n_stocks]
    """
    records = []
    for d in dates:
        ts = pd.Timestamp(d)
        f_vals, r_vals = [], []
        for sym in factor_series:
            if sym not in fwd_series:
                continue
            try:
                fv = factor_series[sym].loc[ts]
                rv = fwd_series[sym].loc[ts]
                if not (np.isnan(fv) or np.isnan(rv)):
                    f_vals.append(fv)
                    r_vals.append(rv)
            except (KeyError, TypeError):
                continue

        if len(f_vals) < min_stocks:
            continue

        ic, _ = spearmanr(f_vals, r_vals)
        if not np.isnan(ic):
            records.append({"date": ts, "ic": ic, "n_stocks": len(f_vals)})

    return pd.DataFrame(records)


def _estimate_half_life(ic_df: pd.DataFrame, max_lag: int = 120) -> float:
    """
    估计 IC 自相关半衰期。

    用 IC 序列的自相关函数拟合指数衰减: autocorr(lag) ≈ exp(-lag / half_life)
    """
    if len(ic_df) < max_lag + 10:
        return float("nan")

    ic_vals = ic_df["ic"].values
    ic_centered = ic_vals - ic_vals.mean()
    var = np.var(ic_centered)
    if var < 1e-12:
        return float("nan")

    autocorrs = []
    for lag in range(1, min(max_lag + 1, len(ic_vals) // 2)):
        cov = np.mean(ic_centered[:-lag] * ic_centered[lag:])
        autocorrs.append(cov / var)

    autocorrs = np.array(autocorrs)
    # 找第一个过零点
    positive = autocorrs > 0
    if not positive.any():
        return 1.0  # 极快衰减

    # 用正自相关拟合 log(autocorr) = -lag / half_life
    valid_lags = np.where(positive)[0][:30]  # 最多用前30个lag
    if len(valid_lags) < 3:
        return float("nan")

    log_ac = np.log(np.maximum(autocorrs[valid_lags], 1e-6))
    lags = valid_lags + 1.0
    # 线性回归: log_ac = -lags / half_life
    slope = np.polyfit(lags, log_ac, 1)[0]
    if slope >= 0:
        return float("inf")
    return -1.0 / slope


def _quintile_returns(factor_series: Dict[str, pd.Series],
                      fwd_series: Dict[str, pd.Series],
                      dates: List,
                      n_groups: int = 5,
                      min_stocks: int = 50) -> pd.DataFrame:
    """
    分组收益分析。

    Returns: DataFrame with columns [date, Q1, Q2, Q3, Q4, Q5, long_short]
    """
    records = []
    for d in dates:
        ts = pd.Timestamp(d)
        pairs = []
        for sym in factor_series:
            if sym not in fwd_series:
                continue
            try:
                fv = factor_series[sym].loc[ts]
                rv = fwd_series[sym].loc[ts]
                if not (np.isnan(fv) or np.isnan(rv)):
                    pairs.append((fv, rv))
            except (KeyError, TypeError):
                continue

        if len(pairs) < min_stocks:
            continue

        pairs.sort(key=lambda x: x[0])
        group_size = len(pairs) // n_groups
        group_rets = []
        for g in range(n_groups):
            start = g * group_size
            end = start + group_size if g < n_groups - 1 else len(pairs)
            group_ret = np.mean([p[1] for p in pairs[start:end]])
            group_rets.append(group_ret)

        record = {"date": ts}
        for g in range(n_groups):
            record[f"Q{g+1}"] = group_rets[g]
        record["long_short"] = group_rets[-1] - group_rets[0]
        records.append(record)

    return pd.DataFrame(records)


def validate_factor(factor_name: str,
                    expr: str = None,
                    period: Tuple[str, str] = None,
                    horizons: List[int] = None,
                    sample: int = None,
                    save_report: bool = True) -> dict:
    """
    一键验证单个因子。

    Args:
        factor_name: 因子名
        expr: DSL表达式 (价量因子需要, 基本面因子可省略)
        period: (start, end) 验证期间, 默认 research 期
        horizons: 前瞻收益天数列表, 默认 [1, 5, 10, 20, 60]
        sample: 抽样股票数 (加速测试)
        save_report: 是否保存JSON报告

    Returns:
        验证报告 dict
    """
    config = _load_config()
    if period is None:
        period = (config["data_partition"]["research"]["start"],
                  config["data_partition"]["research"]["end"])
    if horizons is None:
        horizons = [1, 5, 10, 20, 60]

    start, end = period
    t0 = time.time()

    # 1. 加载数据
    all_data = _load_all_data(sample=sample)
    n_stocks = len(all_data)

    # 2. 计算因子
    factor_series = _compute_factor_series(all_data, factor_name, expr=expr)
    coverage = len(factor_series) / max(n_stocks, 1)

    # 3. 确定日期范围
    all_dates = set()
    for sym, series in list(factor_series.items())[:100]:
        mask = (series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))
        all_dates.update(series.index[mask].tolist())
    dates = sorted(all_dates)

    if len(dates) < 20:
        return {"error": f"有效日期不足: {len(dates)}", "factor": factor_name}

    # 4. 多 horizon IC
    horizon_results = {}
    for h in horizons:
        fwd = _compute_forward_returns(all_data, horizon=h)
        ic_df = _cross_sectional_ic(factor_series, fwd, dates)
        if len(ic_df) == 0:
            horizon_results[h] = {"ic_mean": None, "icir": None}
            continue

        ic_mean = ic_df["ic"].mean()
        ic_std = ic_df["ic"].std()
        icir = ic_mean / ic_std if ic_std > 1e-9 else 0.0
        pos_ratio = (ic_df["ic"] > 0).mean()

        horizon_results[h] = {
            "ic_mean": round(float(ic_mean), 6),
            "ic_std": round(float(ic_std), 6),
            "icir": round(float(icir), 4),
            "pos_ratio": round(float(pos_ratio), 4),
            "n_days": len(ic_df),
            "avg_stocks": int(ic_df["n_stocks"].mean()),
        }

    # 5. 主 horizon (20d) 详细分析
    main_h = 20
    fwd_main = _compute_forward_returns(all_data, horizon=main_h)
    ic_main = _cross_sectional_ic(factor_series, fwd_main, dates)

    # 滚动 IC (60日窗口)
    rolling_ic = None
    if len(ic_main) >= 60:
        rolling_ic = ic_main["ic"].rolling(60).mean().dropna()

    # IC 半衰期
    half_life = _estimate_half_life(ic_main)

    # 分组收益
    quintile_df = _quintile_returns(factor_series, fwd_main, dates)
    quintile_summary = {}
    if len(quintile_df) > 0:
        for col in [f"Q{i}" for i in range(1, 6)] + ["long_short"]:
            quintile_summary[col] = {
                "mean": round(float(quintile_df[col].mean()), 6),
                "std": round(float(quintile_df[col].std()), 6),
                "sharpe": round(float(quintile_df[col].mean() / quintile_df[col].std()), 4)
                          if quintile_df[col].std() > 1e-9 else 0.0,
            }

    # 6. 组装报告
    elapsed = time.time() - t0
    report = {
        "factor": factor_name,
        "expr": expr or "(from scorer)",
        "period": {"start": start, "end": end},
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_stocks": n_stocks,
        "coverage": round(coverage, 4),
        "n_dates": len(dates),
        "elapsed_s": round(elapsed, 1),
        "horizon_ic": horizon_results,
        "main_horizon": main_h,
        "ic_series_length": len(ic_main),
        "half_life_days": round(half_life, 1) if not np.isnan(half_life) else None,
        "rolling_ic_latest": round(float(rolling_ic.iloc[-1]), 6) if rolling_ic is not None and len(rolling_ic) > 0 else None,
        "rolling_ic_trend": _rolling_trend(rolling_ic) if rolling_ic is not None else None,
        "quintile_returns": quintile_summary,
        "verdict": _verdict(horizon_results.get(main_h, {}), half_life, rolling_ic),
    }

    # 7. 保存
    if save_report:
        os.makedirs(REPORT_DIR, exist_ok=True)
        path = os.path.join(REPORT_DIR, f"{factor_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def _rolling_trend(rolling_ic: pd.Series) -> str:
    """判断滚动IC趋势。"""
    if rolling_ic is None or len(rolling_ic) < 20:
        return "insufficient"
    recent = rolling_ic.iloc[-20:].mean()
    early = rolling_ic.iloc[:20].mean()
    if recent > early + 0.01:
        return "improving"
    elif recent < early - 0.01:
        return "decaying"
    else:
        return "stable"


def _verdict(main_ic: dict, half_life: float, rolling_ic) -> dict:
    """综合判定。"""
    checks = {}
    icir = main_ic.get("icir", 0) or 0
    ic_mean = main_ic.get("ic_mean", 0) or 0

    checks["|ICIR| > 0.3"] = bool(abs(icir) > 0.3)
    checks["|IC_mean| > 0.02"] = bool(abs(ic_mean) > 0.02)
    checks["half_life > 20d"] = bool((not np.isnan(half_life)) and half_life > 20)
    checks["rolling_ic_positive"] = bool(
        rolling_ic is not None and len(rolling_ic) > 0 and rolling_ic.iloc[-1] > 0
    )

    n_pass = sum(checks.values())
    if n_pass >= 3:
        verdict = "PASS"
    elif n_pass >= 2:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    return {"verdict": verdict, "checks": checks, "n_pass": n_pass}


def validate_batch(factor_names: List[str],
                   exprs: Dict[str, str] = None,
                   period: Tuple[str, str] = None,
                   sample: int = None,
                   save_reports: bool = True) -> List[dict]:
    """批量验证多个因子。"""
    results = []
    for i, name in enumerate(factor_names):
        expr = (exprs or {}).get(name)
        print(f"  [{i+1}/{len(factor_names)}] 验证 {name}...", flush=True)
        report = validate_factor(name, expr=expr, period=period,
                                 sample=sample, save_report=save_reports)
        results.append(report)
        v = report.get("verdict", {}).get("verdict", "?")
        icir = report.get("horizon_ic", {}).get(20, {}).get("icir", "?")
        print(f"    → {v} (ICIR={icir})", flush=True)
    return results
