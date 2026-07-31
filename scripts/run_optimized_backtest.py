"""
Optimized backtest v2: Fix IC horizon + Reversal/LowVol combo + Turnover buffer
Key fixes:
  1. IC computed with 63-day forward returns (matches quarterly holding)
  2. Reversal + Low Volatility combined strategy
  3. Buffer zone: don't sell unless dropped below Top-50
  4. Year-by-year robustness check
"""
import sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from quant.factors.engine import FactorEngine
from quant.factors.library import get_all_factor_defs
from quant.model.linear import ICWeightedLinear
from quant.backtest.costs import CostModel

print("=" * 65)
print("  OPTIMIZED BACKTEST v2")
print("  Fixes: IC horizon match + Reversal/LowVol + Turnover buffer")
print("=" * 65)
t0 = time.time()

# ============================================================
# Load data
# ============================================================
print("\n[1/6] Loading data...")
cache_dir = Path(__file__).resolve().parent.parent / "data_cache"
file_sizes = [(f.stem, f.stat().st_size) for f in cache_dir.glob("*.parquet")]
file_sizes.sort(key=lambda x: -x[1])
symbols = [s for s, _ in file_sizes[:300]]  # expanded to 300

all_data = {}
for sym in symbols:
    df = pd.read_parquet(cache_dir / f"{sym}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    if len(df) > 500:
        all_data[sym] = df
symbols = list(all_data.keys())
print(f"  {len(symbols)} stocks loaded")

start_date = pd.Timestamp("2019-01-01")
end_date = pd.Timestamp("2024-12-31")
all_dates = [d for d in sorted(all_data[symbols[0]].index) if start_date <= d <= end_date]
print(f"  {all_dates[0].date()} -> {all_dates[-1].date()} ({len(all_dates)} days)")

# ============================================================
# Compute factors
# ============================================================
print("\n[2/6] Computing factors...")
engine = FactorEngine()
factor_defs = get_all_factor_defs()
fast_factors = [f for f in factor_defs if f.lookback <= 60]
factor_names = [f.name for f in fast_factors]
print(f"  {len(factor_names)} factors: {factor_names}")

factor_panels = {name: {} for name in factor_names}
for i, sym in enumerate(symbols):
    df = all_data[sym]
    for fdef in fast_factors:
        try:
            factor_panels[fdef.name][sym] = engine.compute(fdef.expression, df)
        except Exception:
            pass
    if (i + 1) % 100 == 0:
        print(f"    {i+1}/{len(symbols)}...")
print(f"  Done ({time.time()-t0:.0f}s)")

# ============================================================
# Forward returns: BOTH 5-day and 63-day
# ============================================================
print("\n[3/6] Computing forward returns...")
returns_5d = {}
returns_63d = {}
for sym in symbols:
    c = all_data[sym]["close"]
    returns_5d[sym] = c.shift(-5) / c - 1
    returns_63d[sym] = c.shift(-63) / c - 1  # KEY FIX: match holding period

# ============================================================
# Strategy implementations
# ============================================================
cm = CostModel()
top_k = 30
buffer_k = 50  # don't sell unless below top-50

rebalance_dates = all_dates[::63]  # quarterly
print(f"  {len(rebalance_dates)} rebalance dates")


def get_factor_cross_section(fname, date, syms):
    """Get factor values for all symbols on a date."""
    fp = factor_panels.get(fname, {})
    vals = {}
    for sym in syms:
        if sym in fp and date in fp[sym].index:
            v = fp[sym].at[date]
            if pd.notna(v):
                vals[sym] = v
    return vals


def run_strategy(holdings_per_period, label, verbose=True):
    """Run backtest with buffer-zone turnover control."""
    equity = [1_000_000.0]
    dates_list = [holdings_per_period[0][0]]
    prev_holdings = set()
    total_turnover = 0.0
    n_rb = 0
    yearly_returns = {}

    for idx in range(len(holdings_per_period) - 1):
        rb_date, new_h_list = holdings_per_period[idx]
        next_rb = holdings_per_period[idx + 1][0]
        new_h = set(new_h_list[:top_k])

        # Buffer zone: keep stocks that are still in top-K+buffer
        if prev_holdings:
            buffered = set(new_h_list[:buffer_k])
            # Keep old holdings that are still in buffer zone
            keep = prev_holdings & buffered
            # Add new entries from top-K
            add = new_h - prev_holdings
            # Final holdings = kept + added (up to top_k)
            final_h = list(keep) + list(add)
            final_h = final_h[:top_k]
            new_h = set(final_h)

        # Turnover
        to = 0.0
        if prev_holdings:
            to = (len(prev_holdings - new_h) + len(new_h - prev_holdings)) / (2 * max(len(new_h), 1))
            total_turnover += to
        n_rb += 1
        prev_holdings = new_h

        # Daily returns
        period = [d for d in all_dates if rb_date < d <= next_rb]
        year = rb_date.year
        if year not in yearly_returns:
            yearly_returns[year] = []

        for di, pd_date in enumerate(period):
            ret = 0.0
            n = 0
            for sym in new_h:
                df = all_data.get(sym)
                if df is not None and pd_date in df.index:
                    loc = df.index.get_loc(pd_date)
                    if loc > 0:
                        ret += df["close"].iloc[loc] / df["close"].iloc[loc - 1] - 1
                        n += 1
            if n > 0:
                ret /= n
            cost = cm.cost_rate() * to if di == 0 else 0.0
            net_ret = ret - cost
            equity.append(equity[-1] * (1 + net_ret))
            dates_list.append(pd_date)
            yearly_returns[year].append(net_ret)

    # Metrics
    eq = pd.Series(equity)
    total_ret = equity[-1] / equity[0] - 1
    years = len(equity) / 250
    ann_ret = (1 + total_ret) ** (1 / years) - 1
    dd = (eq / eq.cummax() - 1).min()
    daily_ret = eq.pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(250) if daily_ret.std() > 0 else 0
    avg_to = total_turnover / max(n_rb, 1)

    if verbose:
        print(f"  {label}")
        print(f"    Total={total_ret*100:+.1f}%  Ann={ann_ret*100:+.1f}%  Sharpe={sharpe:.2f}  MaxDD={dd*100:.1f}%  TO={avg_to*100:.0f}%/rb")
        # Year-by-year
        yr_str = "    "
        for y in sorted(yearly_returns.keys()):
            yr = np.prod([1 + r for r in yearly_returns[y]]) - 1
            yr_str += f"{y}:{yr*100:+.0f}%  "
        print(yr_str)

    return {
        "label": label, "total": total_ret, "ann": ann_ret,
        "sharpe": sharpe, "maxdd": dd, "avg_to": avg_to,
        "yearly": {y: float(np.prod([1+r for r in rs])-1) for y, rs in yearly_returns.items()}
    }


# ============================================================
# STRATEGY A: IC-Weighted Linear (FIXED: 63-day forward returns)
# ============================================================
print("\n[4/6] Strategy A: IC-Linear (63d forward IC)...")
model_63d = ICWeightedLinear(ic_lookback=252, min_ic=0.0, decay_halflife=126)
ic_preds_63d = []

for rb_date in rebalance_dates:
    hist = [d for d in all_dates if d < rb_date][-252:]
    if len(hist) < 60:
        continue
    ic_dict = {}
    for fname in factor_names:
        ics = []
        for hd in hist[::20]:
            fv, rv = [], []
            for sym in symbols:
                fp = factor_panels[fname]
                if sym in fp and hd in fp[sym].index and hd in returns_63d[sym].index:
                    f = fp[sym].at[hd]
                    r = returns_63d[sym].at[hd]
                    if pd.notna(f) and pd.notna(r):
                        fv.append(f)
                        rv.append(r)
            if len(fv) > 30:
                ic, _ = spearmanr(fv, rv)
                if not np.isnan(ic):
                    ics.append(ic)
        if ics:
            ic_dict[fname] = np.mean(ics)
    model_63d.update_ic_batch(ic_dict, rb_date)

    # Predict
    scores = {}
    for sym in symbols:
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
    if len(scores) > 30:
        fm = np.array(list(scores.values()))
        m, s = fm.mean(0), fm.std(0)
        s[s == 0] = 1
        fn = (fm - m) / s
        preds = model_63d.predict(fn, factor_names)
        pd_dict = {sym: float(preds[j]) for j, sym in enumerate(scores.keys())}
        ranked = sorted(pd_dict.items(), key=lambda x: -x[1])
        ic_preds_63d.append((rb_date, [s for s, _ in ranked]))

w63 = model_63d.get_weights()
aw63 = {k: round(v, 4) for k, v in sorted(w63.items(), key=lambda x: -x[1]) if v > 0}
print(f"  Active factors (63d IC): {aw63}")

# ============================================================
# STRATEGY B: Reversal + Low Vol (50/50 combo)
# ============================================================
print("\n[5/6] Strategy B: Reversal + LowVol combo...")
combo_preds = []
for rb_date in rebalance_dates:
    rev_scores = get_factor_cross_section("ret_20d", rb_date, symbols)
    vol_scores = get_factor_cross_section("vol_20d", rb_date, symbols)

    # Z-score normalize each
    common = set(rev_scores.keys()) & set(vol_scores.keys())
    if len(common) < 40:
        continue
    common = list(common)

    rev_arr = np.array([rev_scores[s] for s in common])
    vol_arr = np.array([vol_scores[s] for s in common])

    rev_z = (rev_arr - rev_arr.mean()) / (rev_arr.std() + 1e-8)
    vol_z = (vol_arr - vol_arr.mean()) / (vol_arr.std() + 1e-8)

    # Combined: LOW ret_20d (reversal) + LOW vol_20d (low volatility)
    # Both should be NEGATIVE direction (lower = better)
    combo_score = -0.6 * rev_z - 0.4 * vol_z  # 60% reversal + 40% low-vol

    ranked_idx = np.argsort(-combo_score)
    ranked_syms = [common[i] for i in ranked_idx]
    combo_preds.append((rb_date, ranked_syms))

# ============================================================
# STRATEGY C: Pure Reversal (with buffer)
# ============================================================
reversal_preds = []
for rb_date in rebalance_dates:
    rev_scores = get_factor_cross_section("ret_20d", rb_date, symbols)
    ranked = sorted(rev_scores.items(), key=lambda x: x[1])  # ascending = worst first
    reversal_preds.append((rb_date, [s for s, _ in ranked]))

# ============================================================
# STRATEGY D: Low Volatility only (with buffer)
# ============================================================
lowvol_preds = []
for rb_date in rebalance_dates:
    vol_scores = get_factor_cross_section("vol_20d", rb_date, symbols)
    ranked = sorted(vol_scores.items(), key=lambda x: x[1])  # ascending = lowest vol
    lowvol_preds.append((rb_date, [s for s, _ in ranked]))

# ============================================================
# STRATEGY E: Benchmark (equal weight, buy & hold)
# ============================================================
bench_preds = [(rb, symbols[:top_k]) for rb in rebalance_dates]

# ============================================================
# Run all strategies
# ============================================================
print("\n[6/6] Running backtests...")
print("=" * 65)
results = []

results.append(run_strategy(bench_preds, "E) Benchmark (buy & hold top-30)"))
print()
results.append(run_strategy(ic_preds_63d, "A) IC-Linear (63d IC, fixed horizon)"))
print()
results.append(run_strategy(reversal_preds, "C) Pure Reversal (ret_20d bottom)"))
print()
results.append(run_strategy(lowvol_preds, "D) Low Volatility (vol_20d bottom)"))
print()
results.append(run_strategy(combo_preds, "B) Reversal+LowVol (60/40 combo)"))

# ============================================================
# Summary table
# ============================================================
print("\n" + "=" * 65)
print("  FINAL COMPARISON TABLE")
print("=" * 65)
print(f"  {'Strategy':<35s} {'Ann':>6s} {'Sharpe':>7s} {'MaxDD':>7s} {'TO/rb':>6s}")
print("  " + "-" * 63)
for r in results:
    print(f"  {r['label']:<35s} {r['ann']*100:>+5.1f}% {r['sharpe']:>6.2f} {r['maxdd']*100:>6.1f}% {r['avg_to']*100:>5.0f}%")

# Excess vs benchmark
bench_ann = results[0]["ann"]
print(f"\n  Excess vs Benchmark (ann={bench_ann*100:.1f}%):")
for r in results[1:]:
    exc = r["ann"] - bench_ann
    print(f"    {r['label']:<35s} {exc*100:>+5.1f}%")

print(f"\n  Total time: {time.time()-t0:.0f}s")
print("=" * 65)

# Save
out = Path(__file__).resolve().parent.parent / "data" / "optimized_results_v2.json"
out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
print(f"  Saved: {out}")
