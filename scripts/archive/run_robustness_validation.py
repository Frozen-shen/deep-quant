"""
Statistical Robustness Validation Framework
============================================

Determines whether the IC-Linear strategy's backtest results (IR=1.69, +20.4% excess)
are statistically significant or just noise/overfitting.

Tests:
  1. Bootstrap Confidence Interval on excess return
  2. Monte Carlo Permutation Test (sign-shuffling)
  3. Sub-sample Consistency (4 equal periods)
  4. Parameter Sensitivity (grid over key params)
  5. Year-by-Year Analysis
  6. Out-of-Sample Validation (2026 H1)

Usage:
  py scripts/run_robustness_validation.py
"""

import os
import sys
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, BASE_DIR)

from quant.factors.engine import FactorEngine
from quant.model.linear import ICWeightedLinear, RankICCalculator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 244
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")

# Active factors from the IC-Linear strategy
ACTIVE_FACTORS = {
    "illiquidity_trend": (
        "Mean(Abs($close/Ref($close,1)-1)/($amount+1), 5) / "
        "(Mean(Abs($close/Ref($close,1)-1)/($amount+1), 20) + 0.0001)"
    ),
    "amihud": "Mean(Abs($close/Ref($close,1)-1) / ($amount+1), 20)",
    "ret_20d": "Ref($close, 20) / $close - 1",
    "ret_10d": "Ref($close, 10) / $close - 1",
    "overnight_return": "Mean($open/Ref($close,1)-1, 20)",
}

# Default strategy parameters
DEFAULT_PARAMS = {
    "top_k": 30,
    "ic_lookback": 252,
    "decay_halflife": 126,
    "rebalance_freq": 21,  # ~monthly
    "cost_bps": 30,
}

# In-sample period
IS_START = "2019-01-01"
IS_END = "2024-06-30"

# Out-of-sample period
OOS_START = "2026-01-01"
OOS_END = "2026-06-30"


# ===========================================================================
#  Data Loading
# ===========================================================================

def load_all_stocks(max_stocks=None):
    """Load all parquet files from data_cache. Returns dict {symbol: DataFrame}."""
    files = sorted(f for f in os.listdir(DATA_CACHE_DIR) if f.endswith(".parquet"))
    if max_stocks:
        files = files[:max_stocks]

    all_data = {}
    for f in files:
        sym = f.replace(".parquet", "")
        try:
            df = pd.read_parquet(os.path.join(DATA_CACHE_DIR, f))
            df["date"] = pd.to_datetime(df["date"])
            if len(df) >= 250:  # need at least 1 year of data
                all_data[sym] = df
        except Exception:
            continue
    return all_data


def build_price_panels(all_data, start_date, end_date):
    """
    Build close/open/high/low/amount price panels (date x symbol).
    Returns dict of DataFrames.
    """
    # Collect all dates
    all_dates = set()
    for df in all_data.values():
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        all_dates.update(df.loc[mask, "date"].tolist())
    all_dates = sorted(all_dates)

    if not all_dates:
        return {}

    date_index = pd.DatetimeIndex(all_dates)
    symbols = sorted(all_data.keys())

    panels = {"close": pd.DataFrame(index=date_index, columns=symbols, dtype=float),
              "open": pd.DataFrame(index=date_index, columns=symbols, dtype=float),
              "amount": pd.DataFrame(index=date_index, columns=symbols, dtype=float)}

    for sym, df in all_data.items():
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        sub = df.loc[mask].set_index("date")
        if sub.empty:
            continue
        panels["close"][sym] = sub["close"]
        panels["open"][sym] = sub["open"] if "open" in sub.columns else sub["close"]
        panels["amount"][sym] = sub.get("amount", sub.get("close", pd.Series(dtype=float)))

    return panels


def compute_factor_panels(all_data, factor_exprs, start_date, end_date):
    """
    Compute factor panels using FactorEngine.
    Returns dict {factor_name: DataFrame(date x symbol)}.
    """
    engine = FactorEngine()
    factor_panels = {name: {} for name in factor_exprs}

    symbols = sorted(all_data.keys())
    n_sym = len(symbols)

    for i, sym in enumerate(symbols):
        if (i + 1) % 200 == 0:
            print(f"    Computing factors: {i+1}/{n_sym}...")
        df = all_data[sym]
        # Need warmup before start_date
        warmup_start = pd.Timestamp(start_date) - pd.Timedelta(days=120)
        mask = df["date"] >= warmup_start
        work = df.loc[mask].copy()
        if len(work) < 60:
            continue

        for name, expr in factor_exprs.items():
            try:
                vals = engine.compute(expr, work)
                # Filter to target period
                date_mask = work["date"] >= pd.Timestamp(start_date)
                date_mask &= work["date"] <= pd.Timestamp(end_date)
                dates = work.loc[date_mask, "date"].values
                v = vals.values[date_mask]
                for d, val in zip(dates, v):
                    if not np.isnan(val):
                        factor_panels[name].setdefault(d, {})[sym] = val
            except Exception:
                continue

    # Convert to DataFrames
    result = {}
    for name, date_dict in factor_panels.items():
        if not date_dict:
            continue
        df = pd.DataFrame(date_dict).T
        df.index = pd.DatetimeIndex(df.index)
        df = df.sort_index()
        result[name] = df

    return result


def compute_forward_returns(close_panel, horizon=20):
    """Compute forward returns panel: ret(t) = close(t+horizon)/close(t) - 1."""
    fwd = close_panel.shift(-horizon) / close_panel - 1.0
    return fwd


# ===========================================================================
#  Simplified Backtest (Top-K Equal Weight)
# ===========================================================================

def run_simple_backtest(scores_panel, close_panel, top_k=30, rebalance_freq=21, cost_bps=30):
    """
    Run a simplified top-k equal-weight backtest.
    Returns equity_curve (Series) and daily_returns (Series).
    """
    dates = scores_panel.index
    returns_panel = close_panel.pct_change().fillna(0)

    n_days = len(dates)
    nav = 1.0
    equity = np.ones(n_days)
    holdings = []
    cost_rate = cost_bps / 10000.0

    for i in range(n_days):
        if i % rebalance_freq == 0:
            day_scores = scores_panel.iloc[i].copy()
            valid = day_scores.notna()
            if valid.sum() >= top_k:
                day_scores_clean = day_scores[valid]
                top_symbols = day_scores_clean.nlargest(top_k).index.tolist()

                if holdings:
                    old_set = set(holdings)
                    new_set = set(top_symbols)
                    n_changed = len(old_set.symmetric_difference(new_set))
                    turnover = n_changed / (2 * max(len(holdings), 1))
                    nav *= (1 - turnover * 2 * cost_rate)
                else:
                    nav *= (1 - cost_rate)

                holdings = top_symbols

        if holdings and i > 0:
            day_rets = returns_panel.iloc[i].reindex(holdings).fillna(0)
            port_ret = day_rets.mean()
            nav *= (1 + port_ret)

        equity[i] = nav

    equity_curve = pd.Series(equity, index=dates, name="equity")
    daily_returns = equity_curve.pct_change().fillna(0)
    return equity_curve, daily_returns


def compute_benchmark_returns(close_panel):
    """Equal-weight benchmark (all stocks in universe)."""
    returns_panel = close_panel.pct_change().fillna(0)
    bench_ret = returns_panel.mean(axis=1)
    bench_curve = (1 + bench_ret).cumprod()
    return bench_curve, bench_ret


def generate_ic_scores(factor_panels, close_panel, ic_lookback=252,
                       decay_halflife=126, forward_horizon=20):
    """
    Generate IC-weighted scores panel using the ICWeightedLinear model.
    Walk-forward: at each rebalance date, use trailing IC to weight factors.
    """
    dates = close_panel.index
    symbols = close_panel.columns.tolist()
    fwd_returns = compute_forward_returns(close_panel, forward_horizon)

    model = ICWeightedLinear(ic_lookback=ic_lookback, decay_halflife=decay_halflife, min_ic=0.01)
    calc = RankICCalculator(forward_period=forward_horizon, min_samples=30)

    scores_data = {}

    for i, date in enumerate(dates):
        # Update IC with realized data from forward_horizon days ago
        if i >= forward_horizon:
            past_date = dates[i - forward_horizon]
            # Compute IC for each factor at past_date
            ic_dict = {}
            for fname, fpanel in factor_panels.items():
                if past_date not in fpanel.index or past_date not in fwd_returns.index:
                    continue
                fvals = fpanel.loc[past_date].dropna()
                rvals = fwd_returns.loc[past_date].dropna()
                common = fvals.index.intersection(rvals.index)
                if len(common) < 30:
                    continue
                ic = calc.compute(fvals[common], rvals[common])
                ic_dict[fname] = ic
            if ic_dict:
                model.update_ic_batch(ic_dict, date=past_date)

        # Generate scores for today
        factor_names = list(factor_panels.keys())
        n_stocks = len(symbols)
        factor_matrix = np.full((n_stocks, len(factor_names)), np.nan)

        for j, fname in enumerate(factor_names):
            fpanel = factor_panels[fname]
            if date in fpanel.index:
                row = fpanel.loc[date]
                for k, sym in enumerate(symbols):
                    if sym in row.index and pd.notna(row[sym]):
                        factor_matrix[k, j] = row[sym]

        # Cross-sectional standardization
        col_means = np.nanmean(factor_matrix, axis=0)
        col_stds = np.nanstd(factor_matrix, axis=0)
        col_stds[col_stds == 0] = 1.0
        factor_matrix_norm = (factor_matrix - col_means) / col_stds

        scores = model.predict(factor_matrix_norm, factor_names)
        scores_data[date] = pd.Series(scores, index=symbols)

    scores_panel = pd.DataFrame(scores_data).T
    scores_panel.index = pd.DatetimeIndex(scores_panel.index)
    return scores_panel.sort_index()


# ===========================================================================
#  Test 1: Bootstrap Confidence Interval
# ===========================================================================

def bootstrap_excess_ci(strategy_returns, benchmark_returns, n_bootstrap=2000, confidence=0.95):
    """
    Resample (strategy_return, benchmark_return) pairs with replacement.
    Compute excess return for each bootstrap sample.
    Return 95% CI for annualized excess return.
    """
    rng = np.random.default_rng(42)

    # Align
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    strat = strategy_returns.reindex(common_idx).dropna().values
    bench = benchmark_returns.reindex(common_idx).dropna().values
    n = min(len(strat), len(bench))
    strat = strat[:n]
    bench = bench[:n]

    excess = strat - bench
    observed_annual_excess = excess.mean() * TRADING_DAYS_PER_YEAR

    bootstrap_excesses = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample_excess = excess[idx]
        bootstrap_excesses[i] = sample_excess.mean() * TRADING_DAYS_PER_YEAR

    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(bootstrap_excesses, alpha * 100)
    ci_upper = np.percentile(bootstrap_excesses, (1 - alpha) * 100)

    return {
        "observed_annual_excess": observed_annual_excess,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "n_days": n,
        "pass": ci_lower > 0,
    }


# ===========================================================================
#  Test 2: Monte Carlo Permutation Test
# ===========================================================================

def monte_carlo_pvalue(strategy_returns, n_simulations=2000):
    """
    H0: strategy returns are random (no skill).
    Shuffle the sign of daily returns randomly (maintaining distribution).
    Compute Sharpe for each simulation.
    p-value = fraction of simulations with Sharpe >= observed Sharpe.
    """
    rng = np.random.default_rng(42)
    returns_arr = strategy_returns.dropna().values
    n = len(returns_arr)

    if n < 30:
        return {"p_value": 1.0, "observed_sharpe": 0.0, "pass": False}

    # Observed Sharpe
    observed_sharpe = returns_arr.mean() / returns_arr.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Sign-shuffle: multiply each return by +1 or -1 randomly
    # This preserves the distribution but destroys any signal
    n_better = 0
    signs = rng.choice([-1, 1], size=(n_simulations, n))
    simulated_returns = returns_arr[np.newaxis, :] * signs  # vectorized
    sim_means = simulated_returns.mean(axis=1)
    sim_stds = simulated_returns.std(axis=1, ddof=1)
    sim_stds[sim_stds == 0] = 1e-10
    sim_sharpes = sim_means / sim_stds * np.sqrt(TRADING_DAYS_PER_YEAR)

    n_better = np.sum(sim_sharpes >= observed_sharpe)
    p_value = (n_better + 1) / (n_simulations + 1)

    return {
        "p_value": float(p_value),
        "observed_sharpe": float(observed_sharpe),
        "n_simulations": n_simulations,
        "n_days": n,
        "pass": p_value < 0.05,
    }


# ===========================================================================
#  Test 3: Sub-sample Consistency
# ===========================================================================

def sub_sample_analysis(equity_curve, benchmark_curve, n_splits=4):
    """
    Split the period into 4 equal sub-periods.
    Compute excess return and IR for each.
    Requirement: at least 3/4 sub-periods must have positive excess.
    """
    common_idx = equity_curve.index.intersection(benchmark_curve.index)
    eq = equity_curve.reindex(common_idx)
    bm = benchmark_curve.reindex(common_idx)

    strat_ret = eq.pct_change().dropna()
    bench_ret = bm.pct_change().dropna()
    common = strat_ret.index.intersection(bench_ret.index)
    strat_ret = strat_ret.reindex(common)
    bench_ret = bench_ret.reindex(common)
    excess = strat_ret - bench_ret

    n = len(excess)
    split_size = n // n_splits
    records = []

    for i in range(n_splits):
        start = i * split_size
        end = (i + 1) * split_size if i < n_splits - 1 else n
        sub_excess = excess.iloc[start:end]

        ann_excess = sub_excess.mean() * TRADING_DAYS_PER_YEAR
        tracking_error = sub_excess.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        ir = ann_excess / tracking_error if tracking_error > 0 else 0.0

        records.append({
            "period": i + 1,
            "start_date": str(sub_excess.index[0].date()),
            "end_date": str(sub_excess.index[-1].date()),
            "annual_excess": ann_excess,
            "ir": ir,
            "n_days": len(sub_excess),
        })

    df = pd.DataFrame(records)
    n_positive = (df["annual_excess"] > 0).sum()
    std_excess = df["annual_excess"].std()

    return {
        "sub_periods": records,
        "n_positive": int(n_positive),
        "n_splits": n_splits,
        "std_excess": float(std_excess),
        "pass": n_positive >= 3,
    }


# ===========================================================================
#  Test 4: Parameter Sensitivity
# ===========================================================================

def parameter_sensitivity(all_data, factor_exprs, start_date, end_date):
    """
    Vary key parameters and check if strategy still works.
    Uses rank-IC correlation with forward returns as a fast proxy
    (no full portfolio simulation needed).
    """
    print("    Computing factor panels for sensitivity...")
    factor_panels = compute_factor_panels(all_data, factor_exprs, start_date, end_date)
    if not factor_panels:
        return {"pass": False, "error": "No factor panels computed"}

    # Build close panel from factor panels' index
    all_dates = sorted(set().union(*[fp.index.tolist() for fp in factor_panels.values()]))
    symbols = sorted(set().union(*[fp.columns.tolist() for fp in factor_panels.values()]))
    close_panel = pd.DataFrame(index=pd.DatetimeIndex(all_dates), columns=symbols, dtype=float)
    for sym, df in all_data.items():
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        sub = df.loc[mask].set_index("date")
        if not sub.empty and sym in close_panel.columns:
            close_panel[sym] = sub["close"]

    fwd_returns = compute_forward_returns(close_panel, horizon=20)

    # Parameter grid
    param_grid = {
        "top_k": [20, 30, 40],
        "ic_lookback": [126, 252, 378],
        "decay_halflife": [63, 126, 252],
    }

    results = []
    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)

    combo_idx = 0
    for lookback, halflife in product(param_grid["ic_lookback"], param_grid["decay_halflife"]):
        combo_idx += 1
        if combo_idx % 3 == 0:
            print(f"    Sensitivity combo {combo_idx}/{total_combos // 3}...")

        # Generate IC-weighted scores with these params
        model = ICWeightedLinear(ic_lookback=lookback, decay_halflife=halflife, min_ic=0.01)
        calc = RankICCalculator(forward_period=20, min_samples=30)

        dates = close_panel.index
        factor_names = list(factor_panels.keys())

        # Walk-forward IC update and score generation
        date_ics = []
        for i, date in enumerate(dates):
            if i >= 20:
                past_date = dates[i - 20]
                ic_dict = {}
                for fname, fpanel in factor_panels.items():
                    if past_date not in fpanel.index or past_date not in fwd_returns.index:
                        continue
                    fvals = fpanel.loc[past_date].dropna()
                    rvals = fwd_returns.loc[past_date].dropna()
                    common = fvals.index.intersection(rvals.index)
                    if len(common) < 30:
                        continue
                    ic = calc.compute(fvals[common], rvals[common])
                    ic_dict[fname] = ic
                if ic_dict:
                    model.update_ic_batch(ic_dict, date=past_date)

            # Compute composite score IC for this date
            weights = model.get_weights()
            active = {k: v for k, v in weights.items() if v > 0}
            if not active:
                continue

            # Weighted score
            w_sum = sum(active.values())
            composite = pd.Series(0.0, index=symbols)
            for fname, w in active.items():
                if fname in factor_panels and date in factor_panels[fname].index:
                    fvals = factor_panels[fname].loc[date].fillna(0)
                    composite += fvals * (w / w_sum)

            # Rank-IC of composite vs forward returns
            if date in fwd_returns.index:
                fwd = fwd_returns.loc[date].dropna()
                comp_valid = composite[fwd.index].dropna()
                common = comp_valid.index.intersection(fwd.index)
                if len(common) >= 30:
                    rank_ic = comp_valid[common].rank().corr(fwd[common].rank())
                    if rank_ic is not None and not np.isnan(rank_ic):
                        date_ics.append(rank_ic)

        if date_ics:
            mean_ic = np.mean(date_ics)
            ic_ir = mean_ic / (np.std(date_ics) + 1e-10)
        else:
            mean_ic = 0.0
            ic_ir = 0.0

        results.append({
            "ic_lookback": lookback,
            "decay_halflife": halflife,
            "mean_rank_ic": mean_ic,
            "ic_ir": ic_ir,
            "n_dates": len(date_ics),
        })

    df = pd.DataFrame(results)

    # Also vary top_k (doesn't affect IC, but affects portfolio return proxy)
    # Use the base IC-IR as the metric; top_k sensitivity is secondary
    # For top_k, we approximate: higher top_k -> lower IC-IR (dilution)
    # We'll just report the IC-based results

    mean_ic = df["mean_rank_ic"].mean()
    std_ic = df["mean_rank_ic"].std()
    min_ic = df["mean_rank_ic"].min()
    max_ic = df["mean_rank_ic"].max()

    mean_ir = df["ic_ir"].mean()
    min_ir = df["ic_ir"].min()

    return {
        "n_combos": len(results),
        "mean_rank_ic": float(mean_ic),
        "std_rank_ic": float(std_ic),
        "min_rank_ic": float(min_ic),
        "max_rank_ic": float(max_ic),
        "mean_ic_ir": float(mean_ir),
        "min_ic_ir": float(min_ir),
        "all_positive_ic": bool((df["mean_rank_ic"] > 0).all()),
        "all_positive_ir": bool((df["ic_ir"] > 0).all()),
        "pass": bool(min_ic > 0),
        "details": results,
    }


# ===========================================================================
#  Test 5: Year-by-Year Analysis
# ===========================================================================

def yearly_analysis(equity_curve, benchmark_curve):
    """
    Compute excess return for each calendar year.
    Report: win rate, worst year, best year.
    Requirement: win rate >= 4/6 years (67%).
    """
    common_idx = equity_curve.index.intersection(benchmark_curve.index)
    eq = equity_curve.reindex(common_idx)
    bm = benchmark_curve.reindex(common_idx)

    strat_ret = eq.pct_change().dropna()
    bench_ret = bm.pct_change().dropna()
    common = strat_ret.index.intersection(bench_ret.index)
    excess = (strat_ret.reindex(common) - bench_ret.reindex(common))

    years = sorted(excess.index.year.unique())
    records = []

    for year in years:
        yr_excess = excess[excess.index.year == year]
        if len(yr_excess) < 20:
            continue
        ann_excess = yr_excess.mean() * TRADING_DAYS_PER_YEAR
        records.append({
            "year": int(year),
            "annual_excess": ann_excess,
            "n_days": len(yr_excess),
        })

    df = pd.DataFrame(records)
    if df.empty:
        return {"pass": False, "error": "No yearly data"}

    n_years = len(df)
    n_positive = (df["annual_excess"] > 0).sum()
    win_rate = n_positive / n_years

    return {
        "yearly_returns": records,
        "n_years": n_years,
        "n_positive": int(n_positive),
        "win_rate": float(win_rate),
        "worst_year": df.loc[df["annual_excess"].idxmin()].to_dict(),
        "best_year": df.loc[df["annual_excess"].idxmax()].to_dict(),
        "pass": win_rate >= 0.67,
    }


# ===========================================================================
#  Test 6: Out-of-Sample Validation
# ===========================================================================

def oos_validation(strategy_returns_oos, benchmark_returns_oos,
                   strategy_returns_is, benchmark_returns_is):
    """
    Compare OOS performance to IS performance.
    If OOS IR < 0.3 x IS IR, likely overfit.
    """
    # IS metrics
    common_is = strategy_returns_is.index.intersection(benchmark_returns_is.index)
    excess_is = strategy_returns_is.reindex(common_is) - benchmark_returns_is.reindex(common_is)
    excess_is = excess_is.dropna()

    is_annual_excess = excess_is.mean() * TRADING_DAYS_PER_YEAR
    is_te = excess_is.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    is_ir = is_annual_excess / is_te if is_te > 0 else 0.0
    is_sharpe = (strategy_returns_is.mean() / strategy_returns_is.std(ddof=1)
                 * np.sqrt(TRADING_DAYS_PER_YEAR))

    # OOS metrics
    common_oos = strategy_returns_oos.index.intersection(benchmark_returns_oos.index)
    excess_oos = strategy_returns_oos.reindex(common_oos) - benchmark_returns_oos.reindex(common_oos)
    excess_oos = excess_oos.dropna()

    if len(excess_oos) < 20:
        return {
            "pass": False,
            "error": f"Insufficient OOS data ({len(excess_oos)} days)",
            "is_ir": is_ir,
        }

    oos_annual_excess = excess_oos.mean() * TRADING_DAYS_PER_YEAR
    oos_te = excess_oos.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    oos_ir = oos_annual_excess / oos_te if oos_te > 0 else 0.0
    oos_sharpe = (strategy_returns_oos.mean() / strategy_returns_oos.std(ddof=1)
                  * np.sqrt(TRADING_DAYS_PER_YEAR))

    ir_ratio = oos_ir / is_ir if is_ir != 0 else 0.0

    return {
        "is_annual_excess": float(is_annual_excess),
        "is_ir": float(is_ir),
        "is_sharpe": float(is_sharpe),
        "oos_annual_excess": float(oos_annual_excess),
        "oos_ir": float(oos_ir),
        "oos_sharpe": float(oos_sharpe),
        "ir_ratio": float(ir_ratio),
        "n_oos_days": len(excess_oos),
        "pass": oos_ir > 0.3 and ir_ratio > 0.3,
    }


# ===========================================================================
#  Final Report
# ===========================================================================

def generate_report(results):
    """Produce a final PASS/FAIL verdict."""
    lines = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("       STATISTICAL ROBUSTNESS VALIDATION REPORT")
    lines.append("       Strategy: IC-Linear (illiquidity + momentum)")
    lines.append("=" * 72)
    lines.append("")

    tests = [
        ("1. Bootstrap 95% CI", results.get("bootstrap")),
        ("2. Monte Carlo Permutation", results.get("monte_carlo")),
        ("3. Sub-sample Consistency", results.get("sub_sample")),
        ("4. Parameter Sensitivity", results.get("sensitivity")),
        ("5. Year-by-Year", results.get("yearly")),
        ("6. Out-of-Sample", results.get("oos")),
    ]

    n_pass = 0
    n_fail = 0

    for name, result in tests:
        lines.append(f"--- {name} ---")
        if result is None:
            lines.append("  [SKIPPED] No data available")
            n_fail += 1
            continue

        passed = result.get("pass", False)
        status = "PASS" if passed else "FAIL"
        if passed:
            n_pass += 1
        else:
            n_fail += 1

        lines.append(f"  Verdict: [{status}]")

        if "bootstrap" in name.lower() or "Bootstrap" in name:
            lines.append(f"  Observed annual excess: {result['observed_annual_excess']:+.2%}")
            lines.append(f"  95% CI: [{result['ci_lower']:+.2%}, {result['ci_upper']:+.2%}]")
            lines.append(f"  N days: {result['n_days']}, N bootstrap: {result['n_bootstrap']}")
            if not passed:
                lines.append(f"  CONCERN: CI lower bound < 0, excess may not be significant")

        elif "Monte Carlo" in name:
            lines.append(f"  Observed Sharpe: {result['observed_sharpe']:.3f}")
            lines.append(f"  p-value: {result['p_value']:.4f}")
            lines.append(f"  N simulations: {result['n_simulations']}")
            if not passed:
                lines.append(f"  CONCERN: Cannot reject H0 (no skill), p > 0.05")

        elif "Sub-sample" in name:
            lines.append(f"  Positive sub-periods: {result['n_positive']}/{result['n_splits']}")
            lines.append(f"  Std of sub-period excesses: {result['std_excess']:.2%}")
            for sp in result.get("sub_periods", []):
                lines.append(f"    P{sp['period']} ({sp['start_date']}~{sp['end_date']}): "
                           f"excess={sp['annual_excess']:+.2%}, IR={sp['ir']:.2f}")
            if not passed:
                lines.append(f"  CONCERN: Fewer than 3/4 sub-periods positive")

        elif "Sensitivity" in name:
            if "error" in result:
                lines.append(f"  Error: {result['error']}")
            else:
                lines.append(f"  Parameter combos tested: {result['n_combos']}")
                lines.append(f"  Mean Rank-IC: {result['mean_rank_ic']:.4f}")
                lines.append(f"  Std Rank-IC: {result['std_rank_ic']:.4f}")
                lines.append(f"  Min Rank-IC: {result['min_rank_ic']:.4f}")
                lines.append(f"  Max Rank-IC: {result['max_rank_ic']:.4f}")
                lines.append(f"  Mean IC-IR: {result['mean_ic_ir']:.3f}")
                lines.append(f"  All combos positive IC: {result['all_positive_ic']}")
                if not passed:
                    lines.append(f"  CONCERN: Some parameter combos yield negative IC (fragile)")

        elif "Year" in name:
            if "error" in result:
                lines.append(f"  Error: {result['error']}")
            else:
                lines.append(f"  Win rate: {result['n_positive']}/{result['n_years']} = {result['win_rate']:.0%}")
                for yr in result.get("yearly_returns", []):
                    lines.append(f"    {yr['year']}: excess={yr['annual_excess']:+.2%}")
                worst = result.get("worst_year", {})
                best = result.get("best_year", {})
                lines.append(f"  Worst year: {worst.get('year', '?')} ({worst.get('annual_excess', 0):+.2%})")
                lines.append(f"  Best year: {best.get('year', '?')} ({best.get('annual_excess', 0):+.2%})")
                if not passed:
                    lines.append(f"  CONCERN: Win rate < 67%")

        elif "Out-of-Sample" in name:
            if "error" in result:
                lines.append(f"  Error: {result['error']}")
                lines.append(f"  IS IR: {result.get('is_ir', 'N/A')}")
            else:
                lines.append(f"  IS: excess={result['is_annual_excess']:+.2%}, IR={result['is_ir']:.2f}, Sharpe={result['is_sharpe']:.2f}")
                lines.append(f"  OOS: excess={result['oos_annual_excess']:+.2%}, IR={result['oos_ir']:.2f}, Sharpe={result['oos_sharpe']:.2f}")
                lines.append(f"  IR ratio (OOS/IS): {result['ir_ratio']:.2f}")
                lines.append(f"  N OOS days: {result['n_oos_days']}")
                if not passed:
                    lines.append(f"  CONCERN: OOS IR < 0.3 or IR ratio < 0.3 (likely overfit)")

        lines.append("")

    # Final verdict
    lines.append("=" * 72)
    lines.append(f"  FINAL VERDICT: {n_pass} PASS / {n_fail} FAIL (out of 6 tests)")
    lines.append("")

    if n_fail == 0:
        lines.append("  CONCLUSION: Strategy passes ALL robustness checks.")
        lines.append("  The +20.4% excess return appears to be REAL ALPHA.")
        lines.append("  Confidence: HIGH")
    elif n_fail == 1:
        lines.append("  CONCLUSION: Strategy passes most checks with 1 concern.")
        lines.append("  The excess return is likely real but with caveats.")
        lines.append("  Confidence: MODERATE")
    elif n_fail == 2:
        lines.append("  CONCLUSION: Strategy shows 2 weaknesses.")
        lines.append("  The excess return may be partially inflated.")
        lines.append("  Confidence: LOW")
    else:
        lines.append("  CONCLUSION: Strategy FAILS multiple robustness checks.")
        lines.append("  The +20.4% excess is likely NOISE or OVERFITTING.")
        lines.append("  Confidence: VERY LOW - do not deploy.")

    lines.append("")
    lines.append("=" * 72)

    report = "\n".join(lines)
    print(report)
    return report


# ===========================================================================
#  Main
# ===========================================================================

def main():
    t0 = time.time()
    print("=" * 72)
    print("  ROBUSTNESS VALIDATION — IC-Linear Strategy")
    print("  Testing: Is +20.4% annual excess REAL or NOISE?")
    print("=" * 72)
    print()

    # -----------------------------------------------------------------------
    # Phase 1: Load data and compute in-sample results
    # -----------------------------------------------------------------------
    print("[Phase 1] Loading data...")
    all_data = load_all_stocks(max_stocks=None)
    print(f"  Loaded {len(all_data)} stocks from data_cache")

    if len(all_data) < 100:
        print("  ERROR: Need at least 100 stocks. Aborting.")
        return

    # Limit to a manageable subset for speed (use all if < 500)
    if len(all_data) > 500:
        # Use a random but reproducible subset
        rng = np.random.default_rng(42)
        symbols = sorted(all_data.keys())
        selected = rng.choice(symbols, size=500, replace=False).tolist()
        all_data = {s: all_data[s] for s in selected}
        print(f"  Subsampled to {len(all_data)} stocks for speed")

    print(f"  Computing factor panels (IS: {IS_START} ~ {IS_END})...")
    factor_panels_is = compute_factor_panels(all_data, ACTIVE_FACTORS, IS_START, IS_END)
    print(f"  Factor panels computed: {list(factor_panels_is.keys())}")

    # Build close panel for IS period
    print("  Building price panels...")
    price_panels = build_price_panels(all_data, IS_START, IS_END)
    close_panel_is = price_panels.get("close")
    if close_panel_is is None or close_panel_is.empty:
        print("  ERROR: No price data. Aborting.")
        return

    # Drop columns that are all NaN
    valid_cols = close_panel_is.columns[close_panel_is.notna().any()]
    close_panel_is = close_panel_is[valid_cols]
    print(f"  Close panel: {close_panel_is.shape[0]} days x {close_panel_is.shape[1]} symbols")

    # Generate IC-weighted scores
    print("  Generating IC-weighted scores (walk-forward)...")
    scores_panel = generate_ic_scores(
        factor_panels_is, close_panel_is,
        ic_lookback=DEFAULT_PARAMS["ic_lookback"],
        decay_halflife=DEFAULT_PARAMS["decay_halflife"],
        forward_horizon=20,
    )
    print(f"  Scores panel: {scores_panel.shape[0]} days x {scores_panel.shape[1]} symbols")

    # Run backtest
    print("  Running simplified backtest...")
    equity_curve, strategy_returns = run_simple_backtest(
        scores_panel, close_panel_is,
        top_k=DEFAULT_PARAMS["top_k"],
        rebalance_freq=DEFAULT_PARAMS["rebalance_freq"],
        cost_bps=DEFAULT_PARAMS["cost_bps"],
    )

    # Benchmark
    benchmark_curve, benchmark_returns = compute_benchmark_returns(close_panel_is)

    print(f"  Backtest complete: {len(strategy_returns)} trading days")
    print(f"  Strategy Sharpe: {strategy_returns.mean()/strategy_returns.std(ddof=1)*np.sqrt(244):.2f}")
    print()

    # -----------------------------------------------------------------------
    # Phase 2: Run robustness tests
    # -----------------------------------------------------------------------
    results = {}

    # Test 1: Bootstrap
    print("[Test 1] Bootstrap Confidence Interval...")
    results["bootstrap"] = bootstrap_excess_ci(
        strategy_returns, benchmark_returns,
        n_bootstrap=2000, confidence=0.95,
    )
    print(f"  Result: CI=[{results['bootstrap']['ci_lower']:+.2%}, {results['bootstrap']['ci_upper']:+.2%}] "
          f"-> {'PASS' if results['bootstrap']['pass'] else 'FAIL'}")

    # Test 2: Monte Carlo
    print("[Test 2] Monte Carlo Permutation Test...")
    results["monte_carlo"] = monte_carlo_pvalue(strategy_returns, n_simulations=2000)
    print(f"  Result: p={results['monte_carlo']['p_value']:.4f}, Sharpe={results['monte_carlo']['observed_sharpe']:.2f} "
          f"-> {'PASS' if results['monte_carlo']['pass'] else 'FAIL'}")

    # Test 3: Sub-sample
    print("[Test 3] Sub-sample Consistency...")
    results["sub_sample"] = sub_sample_analysis(equity_curve, benchmark_curve, n_splits=4)
    print(f"  Result: {results['sub_sample']['n_positive']}/4 positive "
          f"-> {'PASS' if results['sub_sample']['pass'] else 'FAIL'}")

    # Test 4: Parameter Sensitivity
    print("[Test 4] Parameter Sensitivity...")
    results["sensitivity"] = parameter_sensitivity(
        all_data, ACTIVE_FACTORS, IS_START, IS_END,
    )
    if "error" not in results["sensitivity"]:
        print(f"  Result: min_IC={results['sensitivity']['min_rank_ic']:.4f}, "
              f"all_positive={results['sensitivity']['all_positive_ic']} "
              f"-> {'PASS' if results['sensitivity']['pass'] else 'FAIL'}")
    else:
        print(f"  Result: ERROR - {results['sensitivity']['error']}")

    # Test 5: Year-by-Year
    print("[Test 5] Year-by-Year Analysis...")
    results["yearly"] = yearly_analysis(equity_curve, benchmark_curve)
    if "error" not in results["yearly"]:
        print(f"  Result: win_rate={results['yearly']['win_rate']:.0%} "
              f"-> {'PASS' if results['yearly']['pass'] else 'FAIL'}")
    else:
        print(f"  Result: ERROR - {results['yearly']['error']}")

    # Test 6: Out-of-Sample
    print("[Test 6] Out-of-Sample Validation...")
    print(f"  Loading OOS data ({OOS_START} ~ {OOS_END})...")

    # Compute OOS factor panels and backtest
    factor_panels_oos = compute_factor_panels(all_data, ACTIVE_FACTORS, OOS_START, OOS_END)
    if factor_panels_oos:
        price_panels_oos = build_price_panels(all_data, OOS_START, OOS_END)
        close_panel_oos = price_panels_oos.get("close")
        if close_panel_oos is not None and not close_panel_oos.empty:
            valid_cols_oos = close_panel_oos.columns[close_panel_oos.notna().any()]
            close_panel_oos = close_panel_oos[valid_cols_oos]

            scores_oos = generate_ic_scores(
                factor_panels_oos, close_panel_oos,
                ic_lookback=DEFAULT_PARAMS["ic_lookback"],
                decay_halflife=DEFAULT_PARAMS["decay_halflife"],
                forward_horizon=20,
            )
            equity_oos, strat_ret_oos = run_simple_backtest(
                scores_oos, close_panel_oos,
                top_k=DEFAULT_PARAMS["top_k"],
                rebalance_freq=DEFAULT_PARAMS["rebalance_freq"],
                cost_bps=DEFAULT_PARAMS["cost_bps"],
            )
            bench_oos, bench_ret_oos = compute_benchmark_returns(close_panel_oos)

            results["oos"] = oos_validation(
                strat_ret_oos, bench_ret_oos,
                strategy_returns, benchmark_returns,
            )
        else:
            results["oos"] = {"pass": False, "error": "No OOS price data",
                             "is_ir": results["bootstrap"]["observed_annual_excess"]}
    else:
        results["oos"] = {"pass": False, "error": "No OOS factor data (2026 H1 not in cache?)",
                         "is_ir": 0.0}

    if "error" not in results["oos"]:
        print(f"  Result: OOS IR={results['oos'].get('oos_ir', 'N/A')}, "
              f"ratio={results['oos'].get('ir_ratio', 'N/A')} "
              f"-> {'PASS' if results['oos']['pass'] else 'FAIL'}")
    else:
        print(f"  Result: {results['oos']['error']}")

    # -----------------------------------------------------------------------
    # Phase 3: Final Report
    # -----------------------------------------------------------------------
    print()
    generate_report(results)

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
