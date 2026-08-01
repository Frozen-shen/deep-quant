"""
End-to-end backtest: New architecture (IC-weighted linear + quarterly rebalance)
Uses existing cached data in data_cache/
"""
import sys, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quant.factors.engine import FactorEngine
from quant.factors.library import get_all_factor_defs
from quant.model.linear import ICWeightedLinear
from quant.backtest.costs import CostModel
from quant.backtest.metrics import compute_metrics

print("=" * 60)
print("  END-TO-END BACKTEST: NEW ARCHITECTURE")
print("  IC-Weighted Linear + Quarterly Rebalance")
print("=" * 60)
t0 = time.time()

# ============================================================
# STEP 1: Load data
# ============================================================
print("\n[Step 1] Loading cached data...")
cache_dir = Path(__file__).resolve().parent.parent / "data_cache"
file_sizes = [(f.stem, f.stat().st_size) for f in cache_dir.glob("*.parquet")]
file_sizes.sort(key=lambda x: -x[1])
symbols = [s for s, _ in file_sizes[:200]]  # top 200 by size (most liquid)

all_data = {}
for sym in symbols:
    df = pd.read_parquet(cache_dir / f"{sym}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    if len(df) > 500:
        all_data[sym] = df
symbols = list(all_data.keys())
print(f"  Loaded {len(symbols)} symbols with >500 days")

# Common dates
start_date = pd.Timestamp("2019-01-01")
end_date = pd.Timestamp("2024-12-31")
sample_dates = sorted(all_data[symbols[0]].index)
all_dates = [d for d in sample_dates if start_date <= d <= end_date]
print(f"  Date range: {all_dates[0].date()} -> {all_dates[-1].date()} ({len(all_dates)} days)")

# ============================================================
# STEP 2: Compute factors
# ============================================================
print("\n[Step 2] Computing factors...")
engine = FactorEngine()
factor_defs = get_all_factor_defs()
fast_factors = [f for f in factor_defs if f.lookback <= 60][:20]
factor_names = [f.name for f in fast_factors]
print(f"  Using {len(fast_factors)} factors: {factor_names[:5]}...")

factor_panels = {name: {} for name in factor_names}
for i, sym in enumerate(symbols):
    df = all_data[sym]
    for fdef in fast_factors:
        try:
            vals = engine.compute(fdef.expression, df)
            factor_panels[fdef.name][sym] = vals
        except Exception:
            pass
    if (i + 1) % 50 == 0:
        print(f"    {i+1}/{len(symbols)} symbols...")
print(f"  Done ({time.time()-t0:.1f}s)")

# ============================================================
# STEP 3: Forward returns
# ============================================================
returns_5d = {}
for sym in symbols:
    df = all_data[sym]
    returns_5d[sym] = df["close"].shift(-5) / df["close"] - 1

# ============================================================
# STEP 4: Walk-forward IC-weighted linear
# ============================================================
print("\n[Step 3] Walk-forward IC-weighted linear...")
model = ICWeightedLinear(ic_lookback=120, min_ic=0.02, decay_halflife=60)

rebalance_dates = all_dates[::63]  # quarterly
predictions_history = []

for rb_date in rebalance_dates:
    hist_dates = [d for d in all_dates if d < rb_date][-120:]
    if len(hist_dates) < 30:
        continue

    # Compute IC per factor
    ic_dict = {}
    for fname in factor_names:
        ics = []
        for hd in hist_dates[::10]:  # sample every 10 days
            fvals, rvals = [], []
            for sym in symbols:
                fp = factor_panels[fname]
                if sym in fp and hd in fp[sym].index:
                    fv = fp[sym].at[hd]
                    rv = returns_5d[sym].get(hd, np.nan) if hd in returns_5d[sym].index else np.nan
                    if pd.notna(fv) and pd.notna(rv):
                        fvals.append(fv)
                        rvals.append(rv)
            if len(fvals) > 30:
                ic, _ = spearmanr(fvals, rvals)
                if not np.isnan(ic):
                    ics.append(ic)
        if ics:
            ic_dict[fname] = np.mean(ics)

    model.update_ic_batch(ic_dict, rb_date)

    # Build feature matrix for prediction
    scores = {}
    for sym in symbols:
        feats = []
        valid = True
        for fname in factor_names:
            fp = factor_panels[fname]
            if sym in fp and rb_date in fp[sym].index:
                v = fp[sym].at[rb_date]
                feats.append(v if pd.notna(v) else 0.0)
            else:
                valid = False
                break
        if valid:
            scores[sym] = feats

    if len(scores) > 30:
        feat_matrix = np.array(list(scores.values()))
        m, s = feat_matrix.mean(axis=0), feat_matrix.std(axis=0)
        s[s == 0] = 1.0
        feat_norm = (feat_matrix - m) / s
        preds = model.predict(feat_norm, factor_names)
        pred_dict = {sym: float(preds[j]) for j, sym in enumerate(scores.keys())}
        predictions_history.append((rb_date, pred_dict))

print(f"  Predictions: {len(predictions_history)} periods")
weights = model.get_weights()
active_w = {k: round(v, 4) for k, v in sorted(weights.items(), key=lambda x: -x[1]) if v > 0}
print(f"  Active weights: {active_w}")

# ============================================================
# STEP 5: Backtest
# ============================================================
print("\n[Step 4] Backtesting...")
cm = CostModel()
top_k = 30
initial_capital = 1_000_000.0

equity_vals = [initial_capital]
equity_dates = [predictions_history[0][0]]
bench_vals = [initial_capital]

prev_holdings = set()
total_turnover = 0.0
n_rebalances = 0

for idx in range(len(predictions_history) - 1):
    rb_date, pred_dict = predictions_history[idx]
    next_rb_date = predictions_history[idx + 1][0]

    ranked = sorted(pred_dict.items(), key=lambda x: -x[1])
    new_holdings = set(sym for sym, _ in ranked[:top_k])

    turnover = 0.0
    if prev_holdings:
        sold = prev_holdings - new_holdings
        bought = new_holdings - prev_holdings
        turnover = (len(sold) + len(bought)) / (2 * top_k)
        total_turnover += turnover
    n_rebalances += 1
    prev_holdings = new_holdings

    period_dates = [d for d in all_dates if rb_date < d <= next_rb_date]

    for di, pd_date in enumerate(period_dates):
        # Portfolio return
        port_ret = 0.0
        n_valid = 0
        for sym in new_holdings:
            df = all_data.get(sym)
            if df is not None and pd_date in df.index:
                loc = df.index.get_loc(pd_date)
                if loc > 0:
                    ret = df["close"].iloc[loc] / df["close"].iloc[loc - 1] - 1
                    port_ret += ret
                    n_valid += 1
        if n_valid > 0:
            port_ret /= n_valid

        # Benchmark (equal weight all)
        bench_ret = 0.0
        n_b = 0
        for sym in symbols[:100]:
            df = all_data[sym]
            if pd_date in df.index:
                loc = df.index.get_loc(pd_date)
                if loc > 0:
                    bench_ret += df["close"].iloc[loc] / df["close"].iloc[loc - 1] - 1
                    n_b += 1
        if n_b > 0:
            bench_ret /= n_b

        # Cost on first day of period
        cost = cm.cost_rate() * turnover if di == 0 else 0.0

        equity_vals.append(equity_vals[-1] * (1 + port_ret - cost))
        bench_vals.append(bench_vals[-1] * (1 + bench_ret))
        equity_dates.append(pd_date)

# ============================================================
# STEP 6: Metrics
# ============================================================
print("\n[Step 5] Results...")
eq = pd.Series(equity_vals, index=pd.DatetimeIndex(equity_dates[: len(equity_vals)]))
bm = pd.Series(bench_vals, index=pd.DatetimeIndex(equity_dates[: len(bench_vals)]))

strat_m = compute_metrics(eq, bm)

total_ret = equity_vals[-1] / equity_vals[0] - 1
bench_total = bench_vals[-1] / bench_vals[0] - 1
excess = total_ret - bench_total
years = len(equity_vals) / 250
ann_strat = (1 + total_ret) ** (1 / years) - 1
ann_bench = (1 + bench_total) ** (1 / years) - 1
ann_excess = ann_strat - ann_bench
avg_turnover = total_turnover / max(n_rebalances, 1)

print(f"\n{'='*60}")
print(f"  RESULTS SUMMARY")
print(f"{'='*60}")
print(f"  Period:        {equity_dates[0].date()} -> {equity_dates[-1].date()} ({years:.1f}y)")
print(f"  Universe:      {len(symbols)} stocks")
print(f"  Rebalances:    {n_rebalances} (quarterly)")
print(f"  Avg turnover:  {avg_turnover*100:.0f}% per rebalance")
print(f"  Annual TO:     {avg_turnover * n_rebalances / years * 100:.0f}%")
print(f"  Cost/trade:    {cm.cost_rate()*10000:.1f}bp")
print(f"")
print(f"  --- Returns ---")
print(f"  Strategy:      {total_ret*100:+.1f}% total, {ann_strat*100:+.1f}% annualized")
print(f"  Benchmark:     {bench_total*100:+.1f}% total, {ann_bench*100:+.1f}% annualized")
print(f"  Excess:        {excess*100:+.1f}% total, {ann_excess*100:+.1f}% annualized")
print(f"")
print(f"  --- Risk ---")
print(f"  Sharpe:        {strat_m['sharpe']:.3f}")
print(f"  Max Drawdown:  {strat_m['max_drawdown']*100:.1f}%")
print(f"  CAGR:          {strat_m['cagr']*100:.1f}%")
if "information_ratio" in strat_m and strat_m["information_ratio"] is not None:
    print(f"  IR:            {strat_m['information_ratio']:.3f}")
if "sortino" in strat_m:
    print(f"  Sortino:       {strat_m['sortino']:.3f}")
print(f"")
print(f"  --- Model ---")
print(f"  Type:          IC-Weighted Linear (zero overfit)")
print(f"  Active factors: {len(active_w)}/{len(factor_names)}")
for k, v in list(active_w.items())[:8]:
    print(f"    {k:20s} weight={v:.4f}")
print(f"")
print(f"  Time: {time.time()-t0:.1f}s")
print(f"{'='*60}")

# Save results
import json
results = {
    "strategy": "ic_weighted_linear_quarterly",
    "period": f"{equity_dates[0].date()} to {equity_dates[-1].date()}",
    "n_stocks": len(symbols),
    "n_rebalances": n_rebalances,
    "total_return": round(total_ret, 4),
    "benchmark_return": round(bench_total, 4),
    "excess_return": round(excess, 4),
    "ann_excess": round(ann_excess, 4),
    "sharpe": round(strat_m["sharpe"], 3),
    "max_drawdown": round(strat_m["max_drawdown"], 4),
    "active_factors": active_w,
    "avg_turnover": round(avg_turnover, 4),
}
out_path = Path(__file__).resolve().parent.parent / "data" / "new_architecture_results.json"
out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\n  Saved to {out_path}")
