"""
Strategy Backtest v4 — Definitive optimized version

Optimizations over v3:
  1. Factor weight cap: no single factor > 0.15 (prevent illiquidity dominance)
  2. Liquidity-aware costs: slippage varies by avg daily amount
  3. Liquidity filter: exclude stocks with 20d avg daily amount < 50M
  4. Turnover control: buffer zone (top-80) + max 10 changes per rebalance
  5. Industry neutralization: no board group > 50% of portfolio
  6. Multi-strategy blend: IC-Linear (0.6) + Low Volatility (0.4)
  7. Full time range: 2018-07-01 to 2025-12-31 training/testing
  8. Proper benchmark: equal-weight ALL stocks on each date

Lessons respected:
  - NO survivorship bias: use ALL stocks available
  - Benchmark = full universe equal-weight
  - IC embargo = forward return horizon (63 trading days)
  - Cost model: stamp tax 10bp before 2023-08-28, 5bp after; commission 2.5bp
  - Data partition: 2026-01 to 2026-06 as FINAL OOS validation
"""
import sys, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from quant.factors.engine import FactorEngine
from quant.factors.library import get_all_factor_defs
from quant.model.linear import ICWeightedLinear

print("=" * 70)
print("  STRATEGY BACKTEST v4 — Definitive Optimized Version")
print("  8 optimizations applied on top of corrected v3")
print("=" * 70)
t0 = time.time()

# ============================================================
# Configuration
# ============================================================
START_DATE = pd.Timestamp("2018-07-01")
END_DATE = pd.Timestamp("2025-12-31")
OOS_START = pd.Timestamp("2026-01-01")
OOS_END = pd.Timestamp("2026-07-10")  # whatever data we have

FORWARD_HORIZON = 63  # quarterly holding period (trading days)
EMBARGO_DAYS = 63     # must match forward return horizon
TOP_K = 30            # portfolio size
BUFFER_K = 80         # buffer zone for turnover control
MAX_CHANGES = 10      # max stocks changed per rebalance
MIN_LIQUIDITY = 5e7   # 50M minimum 20d avg daily amount
WEIGHT_CAP = 0.15     # max weight for any single factor
BLEND_A = 0.6         # IC-Linear weight
BLEND_B = 0.4         # Low Volatility weight
MAX_INDUSTRY_PCT = 0.50  # max 50% from any board group

# ============================================================
# Cost Model (period-dependent + liquidity-aware)
# ============================================================
STAMP_TAX_CHANGE_DATE = pd.Timestamp("2023-08-28")
COMMISSION = 0.00025  # 2.5bp per side


def slippage_for_stock(avg_daily_amount):
    """Liquidity-aware slippage (Optimization #2)."""
    if avg_daily_amount > 5e8:
        return 0.0002   # 2bp for liquid
    elif avg_daily_amount > 1e8:
        return 0.0005   # 5bp for medium
    else:
        return 0.0010   # 10bp for illiquid


def cost_rate_per_side(date, avg_daily_amount):
    """One-way cost for a stock given its liquidity and the date."""
    stamp = 0.001 if date < STAMP_TAX_CHANGE_DATE else 0.0005
    slip = slippage_for_stock(avg_daily_amount)
    # Buy side: commission + slippage (no stamp tax on buy)
    # Sell side: commission + slippage + stamp tax
    # Average per side:
    buy = COMMISSION + slip
    sell = COMMISSION + slip + stamp
    return (buy + sell) / 2


def portfolio_cost_rate(date, holdings_amounts):
    """Average cost rate for a portfolio (weighted by equal position)."""
    if not holdings_amounts:
        return 0.001
    rates = [cost_rate_per_side(date, amt) for amt in holdings_amounts]
    return np.mean(rates)


# ============================================================
# [1/8] Load ALL stocks (no survivorship filter)
# ============================================================
print("\n[1/8] Loading ALL stocks (no survivorship filter)...")
cache_dir = Path(__file__).resolve().parent.parent / "data_cache"
all_parquets = sorted(cache_dir.glob("*.parquet"))
print(f"  Found {len(all_parquets)} parquet files")

all_data = {}
for f in all_parquets:
    sym = f.stem
    try:
        df = pd.read_parquet(f)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        if len(df) >= 120:
            all_data[sym] = df
    except Exception:
        pass

symbols = sorted(all_data.keys())
print(f"  Loaded {len(symbols)} stocks (>= 120 days history)")

# ============================================================
# Build trading calendar
# ============================================================
ref_sym = max(symbols, key=lambda s: len(all_data[s]))
ref_dates = sorted(all_data[ref_sym].index)
all_dates = [d for d in ref_dates if d >= START_DATE]
date_to_idx = {d: i for i, d in enumerate(all_dates)}
print(f"  Trading dates: {len(all_dates)} ({all_dates[0].date()} to {all_dates[-1].date()})")

# ============================================================
# [2/8] Compute factors
# ============================================================
print("\n[2/8] Computing factors...")
engine = FactorEngine()
factor_defs = get_all_factor_defs()
fast_factors = [f for f in factor_defs if f.lookback <= 60]
factor_names = [f.name for f in fast_factors]
print(f"  {len(factor_names)} factors (lookback <= 60)")

factor_panels = {name: {} for name in factor_names}
computed = 0
for sym in symbols:
    df = all_data[sym]
    for fdef in fast_factors:
        try:
            factor_panels[fdef.name][sym] = engine.compute(fdef.expression, df)
        except Exception:
            pass
    computed += 1
    if computed % 300 == 0:
        print(f"    {computed}/{len(symbols)}...")
print(f"  Done ({time.time()-t0:.0f}s)")

# ============================================================
# [3/8] Pre-compute returns and liquidity data
# ============================================================
print("\n[3/8] Pre-computing returns and liquidity...")
daily_returns = {}
avg_amount_20d = {}  # for liquidity filter and cost model

for sym in symbols:
    df = all_data[sym]
    c = df["close"]
    daily_returns[sym] = c.pct_change()
    # 20-day rolling average of daily amount
    avg_amount_20d[sym] = df["amount"].rolling(20, min_periods=10).mean()

# Forward returns (63-day)
returns_63d = {}
for sym in symbols:
    c = all_data[sym]["close"]
    returns_63d[sym] = c.shift(-FORWARD_HORIZON) / c - 1

print(f"  Done ({time.time()-t0:.0f}s)")

# ============================================================
# [4/8] Rebalance dates
# ============================================================
rebalance_dates = all_dates[::FORWARD_HORIZON]
# Only keep rebalance dates within our backtest window
rebalance_dates = [d for d in rebalance_dates if START_DATE <= d <= OOS_END]
print(f"\n[4/8] {len(rebalance_dates)} rebalance dates")

# ============================================================
# [5/8] IC-Linear with embargo + weight cap
# ============================================================
print("\n[5/8] Walk-forward IC-Linear (63d embargo, weight cap=0.15)...")
model = ICWeightedLinear(ic_lookback=252, min_ic=0.0, decay_halflife=126)

# Helper: get industry group from stock code
def get_board_group(sym):
    """Simplified industry neutralization by board (Optimization #5)."""
    if sym.startswith("6"):
        return "SH"  # Shanghai main
    elif sym.startswith("0"):
        return "SZ"  # Shenzhen main
    elif sym.startswith("3"):
        return "CY"  # ChiNext
    else:
        return "OTHER"


def apply_weight_cap(weights, cap=WEIGHT_CAP):
    """Cap individual factor weights (Optimization #1)."""
    total = sum(weights.values())
    if total <= 0:
        return weights
    capped = {}
    for k, v in weights.items():
        norm_v = v / total
        capped[k] = min(norm_v, cap) * total
    # Renormalize
    new_total = sum(capped.values())
    if new_total > 0:
        scale = total / new_total
        capped = {k: v * scale for k, v in capped.items()}
    return capped


def apply_liquidity_filter(scores_dict, rb_date):
    """Remove stocks below liquidity threshold (Optimization #3)."""
    filtered = {}
    for sym, score in scores_dict.items():
        if sym in avg_amount_20d and rb_date in avg_amount_20d[sym].index:
            liq = avg_amount_20d[sym].at[rb_date]
            if pd.notna(liq) and liq >= MIN_LIQUIDITY:
                filtered[sym] = score
    return filtered


def apply_industry_cap(ranked_syms, scores_dict, max_pct=MAX_INDUSTRY_PCT):
    """Ensure no board group exceeds max_pct of portfolio (Optimization #5)."""
    max_count = int(TOP_K * max_pct)
    group_counts = {"SH": 0, "SZ": 0, "CY": 0, "OTHER": 0}
    result = []
    for sym in ranked_syms:
        grp = get_board_group(sym)
        if group_counts[grp] < max_count:
            result.append(sym)
            group_counts[grp] += 1
        if len(result) >= TOP_K * 2:  # enough for buffer
            break
    return result


# Main IC-Linear prediction loop
ic_predictions = []  # (date, ranked_symbols, n_universe)

for rb_i, rb_date in enumerate(rebalance_dates):
    rb_idx = date_to_idx.get(rb_date)
    if rb_idx is None:
        continue

    # Embargo: hist ends at rb_date - 63 trading days
    embargo_cutoff_idx = rb_idx - EMBARGO_DAYS
    if embargo_cutoff_idx < 60:
        continue
    embargo_cutoff_date = all_dates[embargo_cutoff_idx]

    # Historical window for IC computation
    hist_start_idx = max(0, embargo_cutoff_idx - 252)
    hist_dates = all_dates[hist_start_idx:embargo_cutoff_idx + 1]
    if len(hist_dates) < 30:
        continue

    # Compute IC per factor (sample every 20 days for speed)
    ic_dict = {}
    for fname in factor_names:
        ics = []
        fp = factor_panels[fname]
        for hd in hist_dates[::20]:
            fv, rv = [], []
            for sym in symbols:
                if sym in fp and hd in fp[sym].index:
                    f = fp[sym].at[hd]
                    r = returns_63d[sym].get(hd, np.nan) if hd in returns_63d[sym].index else np.nan
                    if pd.notna(f) and pd.notna(r):
                        fv.append(f)
                        rv.append(r)
            if len(fv) > 50:
                ic, _ = spearmanr(fv, rv)
                if not np.isnan(ic):
                    ics.append(ic)
        if ics:
            ic_dict[fname] = np.mean(ics)

    model.update_ic_batch(ic_dict, rb_date)

    # Get weights with cap applied
    raw_weights = model.get_weights()
    capped_weights = apply_weight_cap(raw_weights)

    # Predict scores
    scores = {}
    for sym in symbols:
        if sym not in all_data or rb_date not in all_data[sym].index:
            continue
        feats = []
        ok = True
        for fname in factor_names:
            fp = factor_panels[fname]
            if sym in fp and rb_date in fp[sym].index:
                v = fp[sym].at[rb_date]
                feats.append(v if pd.notna(v) else 0)
            else:
                ok = False
                break
        if ok:
            scores[sym] = feats

    if len(scores) > 50:
        fm = np.array(list(scores.values()))
        m, s = fm.mean(0), fm.std(0)
        s[s == 0] = 1
        fn = (fm - m) / s

        # Use capped weights for prediction
        w = np.array([capped_weights.get(name, 0.0) for name in factor_names], dtype=float)
        weight_sum = w.sum()
        if weight_sum > 0:
            preds = (fn * w).sum(axis=1) / weight_sum
        else:
            preds = np.zeros(len(scores))

        pd_dict = {sym: float(preds[j]) for j, sym in enumerate(scores.keys())}

        # Apply liquidity filter
        pd_dict = apply_liquidity_filter(pd_dict, rb_date)

        if len(pd_dict) > 30:
            ranked = sorted(pd_dict.items(), key=lambda x: -x[1])
            ranked_syms = [s for s, _ in ranked]
            # Apply industry cap
            ranked_syms = apply_industry_cap(ranked_syms, pd_dict)
            ic_predictions.append((rb_date, ranked_syms, len(pd_dict)))

    if (rb_i + 1) % 5 == 0:
        print(f"    Period {rb_i+1}/{len(rebalance_dates)} done ({time.time()-t0:.0f}s)")

print(f"  IC-Linear predictions: {len(ic_predictions)} periods")
w = model.get_weights()
aw = {k: round(v, 4) for k, v in sorted(w.items(), key=lambda x: -x[1]) if v > 0}
print(f"  Active factors (raw): {aw}")

# ============================================================
# [6/8] Low Volatility scores
# ============================================================
print("\n[6/8] Computing Low Volatility scores...")
lowvol_predictions = []
vol_panel = factor_panels.get("vol_20d", {})

for rb_date in rebalance_dates:
    scores = {}
    for sym in symbols:
        if sym in vol_panel and rb_date in vol_panel[sym].index:
            v = vol_panel[sym].at[rb_date]
            if pd.notna(v):
                # Also apply liquidity filter
                if sym in avg_amount_20d and rb_date in avg_amount_20d[sym].index:
                    liq = avg_amount_20d[sym].at[rb_date]
                    if pd.notna(liq) and liq >= MIN_LIQUIDITY:
                        scores[sym] = v
    if len(scores) > 30:
        # Lower vol = better -> rank ascending
        ranked = sorted(scores.items(), key=lambda x: x[1])
        ranked_syms = [s for s, _ in ranked]
        ranked_syms = apply_industry_cap(ranked_syms, scores)
        lowvol_predictions.append((rb_date, ranked_syms, len(scores)))

print(f"  Low Vol predictions: {len(lowvol_predictions)} periods")

# ============================================================
# [7/8] Blend IC-Linear (0.6) + Low Vol (0.4)
# ============================================================
print("\n[7/8] Blending strategies (60% IC-Linear + 40% Low Vol)...")

# Build lookup dicts
ic_by_date = {d: (ranked, n) for d, ranked, n in ic_predictions}
lv_by_date = {d: (ranked, n) for d, ranked, n in lowvol_predictions}

blended_predictions = []
for rb_date in rebalance_dates:
    if rb_date not in ic_by_date or rb_date not in lv_by_date:
        continue

    ic_ranked, ic_n = ic_by_date[rb_date]
    lv_ranked, lv_n = lv_by_date[rb_date]

    # Convert ranks to scores (rank-based, normalized)
    # IC: rank 0 = best, so score = 1 - rank/n
    ic_scores = {}
    for i, sym in enumerate(ic_ranked):
        ic_scores[sym] = 1.0 - i / max(ic_n, 1)

    lv_scores = {}
    for i, sym in enumerate(lv_ranked):
        lv_scores[sym] = 1.0 - i / max(lv_n, 1)

    # Blend: only stocks that appear in both
    all_syms = set(ic_scores.keys()) | set(lv_scores.keys())
    blended = {}
    for sym in all_syms:
        s_ic = ic_scores.get(sym, 0.0)
        s_lv = lv_scores.get(sym, 0.0)
        blended[sym] = BLEND_A * s_ic + BLEND_B * s_lv

    ranked = sorted(blended.items(), key=lambda x: -x[1])
    ranked_syms = [s for s, _ in ranked]
    # Apply industry cap on blended result
    ranked_syms = apply_industry_cap(ranked_syms, blended)
    blended_predictions.append((rb_date, ranked_syms, len(blended)))

print(f"  Blended predictions: {len(blended_predictions)} periods")

# ============================================================
# [8/8] Backtest with turnover control + liquidity-aware costs
# ============================================================
print("\n[8/8] Running backtest with turnover control...")

# Pre-compute benchmark returns (full universe equal-weight per date)
print("  Computing full-universe benchmark...")
bench_daily = {}
for d in all_dates[1:]:
    rets = []
    for sym in symbols:
        sr = daily_returns[sym]
        if d in sr.index and pd.notna(sr.at[d]):
            rets.append(sr.at[d])
    bench_daily[d] = np.mean(rets) if rets else 0.0


def run_backtest_v4(holdings_per_period, label):
    """
    Backtest with:
    - Buffer zone turnover control (Optimization #4)
    - Max changes per rebalance (Optimization #4)
    - Liquidity-aware costs (Optimization #2)
    - Full universe benchmark (Lesson #2)
    """
    equity = [1_000_000.0]
    bench_equity = [1_000_000.0]
    dates_list = [holdings_per_period[0][0]]
    prev_holdings = set()
    total_turnover = 0.0
    n_rb = 0
    yearly_strat = {}
    yearly_bench = {}
    turnover_list = []

    for idx in range(len(holdings_per_period) - 1):
        rb_date, new_h_full = holdings_per_period[idx][0], holdings_per_period[idx][1]
        next_rb = holdings_per_period[idx + 1][0]

        # --- Turnover control (Optimization #4) ---
        # FIX: Buffer only prevents SELLING, doesn't block BUYING.
        # Old bug: keep filled all 30 slots, add was always truncated to 0.
        new_h = set(new_h_full[:TOP_K])

        if prev_holdings:
            buffered = set(new_h_full[:BUFFER_K])

            # Step 1: Identify forced sells (dropped below buffer)
            must_sell = prev_holdings - buffered  # below rank 80 → must go

            # Step 2: Identify desired sells (in buffer but not in new top-K)
            # These are "weak holds" — sell them to make room for fresh entries
            weak_holds = (prev_holdings & buffered) - new_h  # rank 31-80

            # Step 3: Total sells = must_sell + worst weak_holds (up to MAX_CHANGES)
            all_sells = list(must_sell)
            # Sort weak_holds by their rank (worst first = highest index in new_h_full)
            weak_ranked = [s for s in new_h_full if s in weak_holds]
            weak_ranked.reverse()  # worst rank first
            remaining_slots = MAX_CHANGES - len(all_sells)
            if remaining_slots > 0:
                all_sells += weak_ranked[:remaining_slots]
            all_sells = set(all_sells[:MAX_CHANGES])

            # Step 4: Identify buys (new top-K not in current holdings)
            all_buys = list(new_h - prev_holdings)
            all_buys = all_buys[:len(all_sells)]  # buy same number as sell

            # Step 5: Construct final portfolio
            final_h = list(prev_holdings - all_sells) + all_buys
            final_h = final_h[:TOP_K]
            new_h = set(final_h)

        # Compute turnover
        to = 0.0
        if prev_holdings:
            to = (len(prev_holdings - new_h) + len(new_h - prev_holdings)) / (2 * max(len(new_h), 1))
            total_turnover += to
            turnover_list.append(to)
        n_rb += 1
        prev_holdings = new_h

        # Get liquidity data for cost computation
        holdings_amounts = []
        for sym in new_h:
            if sym in avg_amount_20d and rb_date in avg_amount_20d[sym].index:
                amt = avg_amount_20d[sym].at[rb_date]
                if pd.notna(amt):
                    holdings_amounts.append(amt)

        avg_cost_rate = portfolio_cost_rate(rb_date, holdings_amounts)

        # Daily returns for this holding period
        period = [d for d in all_dates if rb_date < d <= next_rb]
        year = rb_date.year

        for di, pd_date in enumerate(period):
            # Strategy return
            ret = 0.0
            n = 0
            for sym in new_h:
                sr = daily_returns.get(sym)
                if sr is not None and pd_date in sr.index and pd.notna(sr.at[pd_date]):
                    ret += sr.at[pd_date]
                    n += 1
            if n > 0:
                ret /= n

            # Benchmark
            bench_ret = bench_daily.get(pd_date, 0.0)

            # Cost applied on rebalance day only
            cost = avg_cost_rate * to * 2 if di == 0 else 0.0  # round-trip

            strat_net = ret - cost
            equity.append(equity[-1] * (1 + strat_net))
            bench_equity.append(bench_equity[-1] * (1 + bench_ret))
            dates_list.append(pd_date)

            yearly_strat.setdefault(year, []).append(strat_net)
            yearly_bench.setdefault(year, []).append(bench_ret)

    # --- Metrics ---
    eq = pd.Series(equity)
    bm = pd.Series(bench_equity)
    total_ret = equity[-1] / equity[0] - 1
    bench_total = bench_equity[-1] / bench_equity[0] - 1
    years = len(equity) / 250
    ann_ret = (1 + total_ret) ** (1 / max(years, 0.01)) - 1
    ann_bench = (1 + bench_total) ** (1 / max(years, 0.01)) - 1
    dd = (eq / eq.cummax() - 1).min()
    daily_ret_series = eq.pct_change().dropna()
    sharpe = daily_ret_series.mean() / daily_ret_series.std() * np.sqrt(250) if daily_ret_series.std() > 0 else 0

    # Excess / IR
    excess_series = eq.pct_change().fillna(0) - bm.pct_change().fillna(0)
    te = excess_series.std() * np.sqrt(250)
    ir = (ann_ret - ann_bench) / te if te > 0 else 0

    avg_to = total_turnover / max(n_rb, 1)

    print(f"\n  {'='*60}")
    print(f"  {label}")
    print(f"  {'='*60}")
    print(f"    Strategy:  Total={total_ret*100:+.1f}%  Ann={ann_ret*100:+.1f}%  Sharpe={sharpe:.2f}  MaxDD={dd*100:.1f}%")
    print(f"    Benchmark: Total={bench_total*100:+.1f}%  Ann={ann_bench*100:+.1f}% (full universe EW)")
    print(f"    Excess:    Ann={ann_ret-ann_bench:+.1f}%  IR={ir:.3f}  TE={te*100:.1f}%")
    print(f"    Turnover:  {avg_to*100:.0f}%/rb  (target: ~35%)")

    # Year-by-year
    print(f"\n    Year-by-year:")
    print(f"    {'Year':<6} {'Strategy':>10} {'Benchmark':>10} {'Excess':>10}")
    print(f"    {'-'*38}")
    for y in sorted(yearly_strat.keys()):
        ys = np.prod([1 + r for r in yearly_strat[y]]) - 1
        yb = np.prod([1 + r for r in yearly_bench[y]]) - 1
        print(f"    {y:<6} {ys*100:>+9.1f}% {yb*100:>+9.1f}% {ys-yb:>+9.1f}%")

    return {
        "label": label, "total": total_ret, "ann": ann_ret,
        "bench_ann": ann_bench, "excess_ann": ann_ret - ann_bench,
        "sharpe": sharpe, "maxdd": float(dd), "ir": ir, "te": te,
        "avg_to": avg_to,
        "yearly": {y: {"strat": float(np.prod([1+r for r in rs])-1),
                       "bench": float(np.prod([1+r for r in yearly_bench[y]])-1)}
                   for y, rs in yearly_strat.items()}
    }


# --- Run backtests ---
results = []

# Main: Blended strategy
results.append(run_backtest_v4(blended_predictions, "V4 Blended (IC-Linear 60% + LowVol 40%)"))

# Component A: IC-Linear only (for comparison)
results.append(run_backtest_v4(ic_predictions, "V4 IC-Linear only (weight-capped)"))

# Component B: Low Vol only (for comparison)
results.append(run_backtest_v4(lowvol_predictions, "V4 Low Volatility only"))

# ============================================================
# OOS Validation (2026-01 to 2026-06)
# ============================================================
print("\n" + "=" * 70)
print("  OUT-OF-SAMPLE VALIDATION (2026-01 to 2026-06)")
print("  This period was NEVER used for parameter selection")
print("=" * 70)

oos_blended = [(d, r, n) for d, r, n in blended_predictions if d >= OOS_START]
oos_ic = [(d, r, n) for d, r, n in ic_predictions if d >= OOS_START]
oos_lv = [(d, r, n) for d, r, n in lowvol_predictions if d >= OOS_START]

if len(oos_blended) >= 2:
    results.append(run_backtest_v4(oos_blended, "OOS: Blended (2026)"))
else:
    print("  Insufficient OOS periods for blended")

if len(oos_ic) >= 2:
    results.append(run_backtest_v4(oos_ic, "OOS: IC-Linear (2026)"))
else:
    print("  Insufficient OOS periods for IC-Linear")

if len(oos_lv) >= 2:
    results.append(run_backtest_v4(oos_lv, "OOS: Low Vol (2026)"))
else:
    print("  Insufficient OOS periods for Low Vol")

# ============================================================
# Summary Table
# ============================================================
print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(f"  Period: {START_DATE.date()} -> {END_DATE.date()} (in-sample)")
print(f"  OOS:    {OOS_START.date()} -> {OOS_END.date()} (validation)")
print(f"  Universe: {len(symbols)} stocks (ALL, no survivorship filter)")
print(f"  Liquidity filter: 20d avg amount >= {MIN_LIQUIDITY/1e6:.0f}M")
print(f"  Benchmark: Full universe equal-weight")
print(f"  IC embargo: {EMBARGO_DAYS} trading days")
print(f"  Weight cap: {WEIGHT_CAP}")
print(f"  Turnover control: buffer={BUFFER_K}, max_changes={MAX_CHANGES}")
print(f"  Industry cap: {MAX_INDUSTRY_PCT*100:.0f}% per board")
print(f"  Blend: IC-Linear {BLEND_A*100:.0f}% + LowVol {BLEND_B*100:.0f}%")
print(f"  Costs: commission={COMMISSION*10000:.1f}bp + slippage(2-10bp) + stamp(5-10bp)")
print()
print(f"  {'Strategy':<45s} {'Ann':>7s} {'Excess':>8s} {'IR':>6s} {'Sharpe':>7s} {'MaxDD':>7s} {'TO/rb':>6s}")
print("  " + "-" * 88)
for r in results:
    print(f"  {r['label']:<45s} {r['ann']*100:>+6.1f}% {r['excess_ann']*100:>+7.1f}% {r['ir']:>5.3f} {r['sharpe']:>6.2f} {r['maxdd']*100:>6.1f}% {r['avg_to']*100:>5.0f}%")

print(f"\n  Benchmark (full universe EW): ann={results[0]['bench_ann']*100:+.1f}%")
print(f"\n  Total time: {time.time()-t0:.0f}s")
print("=" * 70)

# Save results
out = Path(__file__).resolve().parent.parent / "data" / "strategy_v4_results.json"
out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
print(f"  Saved: {out}")
