"""
Corrected backtest v3 — 修复全部5个方法论缺陷

Fixes:
  1. Universe: 使用全部1372只股票，不按文件大小筛选（消除幸存者偏差）
  2. Benchmark: 全池等权（不是top-30切片）
  3. IC embargo: hist截止于 rb_date - 63交易日（消除前视泄露）
  4. 数据分区: 回测期截止2024-06-30（尊重盲测边界）
  5. 成本分期: 印花税2023-08-28前10bp/后5bp
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

print("=" * 65)
print("  CORRECTED BACKTEST v3")
print("  5 methodological fixes applied")
print("=" * 65)
t0 = time.time()

# ============================================================
# FIX #5: Period-dependent cost model
# ============================================================
class CorrectedCostModel:
    """
    Realistic A-share cost model with period-dependent stamp tax.
    - Commission: 万2.5 (0.025%) per side, min 5 yuan
    - Slippage: 万3 (0.03%) market impact (conservative for liquid stocks)
    - Stamp tax: 千1 (0.1%) before 2023-08-28, 万5 (0.05%) after (sell-side only)
    - Transfer fee: 万0.1 (0.001%) both sides
    """
    STAMP_TAX_CHANGE_DATE = pd.Timestamp("2023-08-28")

    def __init__(self, commission=0.00025, slippage=0.0003, transfer_fee=0.00001):
        self.commission = commission
        self.slippage = slippage
        self.transfer_fee = transfer_fee

    def cost_rate(self, date=None):
        """One-way cost rate (buy or sell)."""
        stamp = 0.001 if (date is None or date < self.STAMP_TAX_CHANGE_DATE) else 0.0005
        # Buy: commission + slippage + transfer
        # Sell: commission + slippage + transfer + stamp
        # Average round-trip / 2 for per-side:
        buy_cost = self.commission + self.slippage + self.transfer_fee
        sell_cost = self.commission + self.slippage + self.transfer_fee + stamp
        return (buy_cost + sell_cost) / 2  # average per side

    def round_trip_rate(self, date=None):
        """Full round-trip cost (buy + sell)."""
        return self.cost_rate(date) * 2


cm = CorrectedCostModel()
print(f"\n[Cost Model]")
print(f"  Before 2023-08-28: {cm.round_trip_rate(pd.Timestamp('2023-01-01'))*10000:.1f}bp round-trip")
print(f"  After  2023-08-28: {cm.round_trip_rate(pd.Timestamp('2024-01-01'))*10000:.1f}bp round-trip")

# ============================================================
# FIX #1: Load ALL stocks (no file-size filtering)
# ============================================================
print("\n[1/6] Loading ALL stocks (no survivorship filter)...")
cache_dir = Path(__file__).resolve().parent.parent / "data_cache"
all_parquets = list(cache_dir.glob("*.parquet"))
print(f"  Found {len(all_parquets)} parquet files")

all_data = {}
for f in all_parquets:
    sym = f.stem
    try:
        df = pd.read_parquet(f)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        # Keep all stocks with at least 120 days of data
        if len(df) >= 120:
            all_data[sym] = df
    except Exception:
        pass

symbols = sorted(all_data.keys())
print(f"  Loaded {len(symbols)} stocks (>= 120 days history)")
print(f"  Includes short-history and potentially delisted stocks")

# ============================================================
# FIX #4: Test period ends at 2024-06-30 (blind test boundary)
# ============================================================
START_DATE = pd.Timestamp("2019-01-01")
END_DATE = pd.Timestamp("2024-06-30")  # FIX #4: respect blind test boundary
print(f"\n  Test period: {START_DATE.date()} -> {END_DATE.date()} (blind boundary)")

# Build common trading dates from the most complete stock
ref_sym = max(symbols, key=lambda s: len(all_data[s]))
all_dates = [d for d in sorted(all_data[ref_sym].index) if START_DATE <= d <= END_DATE]
print(f"  Trading dates: {len(all_dates)} (ref: {ref_sym})")

# ============================================================
# Compute factors (all stocks, all dates)
# ============================================================
print("\n[2/6] Computing factors...")
engine = FactorEngine()
factor_defs = get_all_factor_defs()
fast_factors = [f for f in factor_defs if f.lookback <= 60]
factor_names = [f.name for f in fast_factors]
print(f"  {len(factor_names)} factors")

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
    if computed % 200 == 0:
        print(f"    {computed}/{len(symbols)}...")
print(f"  Done ({time.time()-t0:.0f}s)")

# ============================================================
# Forward returns (63-day, for IC matching holding period)
# ============================================================
print("\n[3/6] Computing 63-day forward returns...")
returns_63d = {}
for sym in symbols:
    c = all_data[sym]["close"]
    returns_63d[sym] = c.shift(-63) / c - 1

# ============================================================
# Rebalance dates (quarterly)
# ============================================================
rebalance_dates = all_dates[::63]
print(f"  {len(rebalance_dates)} rebalance dates")

# ============================================================
# FIX #3: IC computation with EMBARGO
# The latest date used for IC must be rb_date - 63 trading days
# This ensures 63d forward returns don't peek past rb_date
# ============================================================
print("\n[4/6] IC-Linear with 63d embargo...")
model = ICWeightedLinear(ic_lookback=252, min_ic=0.0, decay_halflife=126)

# Build date index for embargo lookup
date_to_idx = {d: i for i, d in enumerate(all_dates)}
EMBARGO_DAYS = 63  # must match forward return horizon

ic_preds = []
for rb_i, rb_date in enumerate(rebalance_dates):
    # FIX #3: embargo — hist ends at rb_date - 63 trading days
    rb_idx = date_to_idx.get(rb_date, 0)
    embargo_cutoff_idx = rb_idx - EMBARGO_DAYS
    if embargo_cutoff_idx < 60:  # need at least 60 days of history
        continue
    embargo_cutoff_date = all_dates[embargo_cutoff_idx]

    # Historical dates for IC: from (embargo_cutoff - 252) to embargo_cutoff
    hist_start_idx = max(0, embargo_cutoff_idx - 252)
    hist_dates = all_dates[hist_start_idx:embargo_cutoff_idx + 1]

    if len(hist_dates) < 30:
        continue

    # Compute IC per factor (sample every 20 days for speed)
    ic_dict = {}
    for fname in factor_names:
        ics = []
        for hd in hist_dates[::20]:
            fv, rv = [], []
            for sym in symbols:
                fp = factor_panels[fname]
                if sym in fp and hd in fp[sym].index:
                    f = fp[sym].at[hd]
                    r = returns_63d[sym].get(hd, np.nan) if hd in returns_63d[sym].index else np.nan
                    if pd.notna(f) and pd.notna(r):
                        fv.append(f)
                        rv.append(r)
            if len(fv) > 50:  # need decent cross-section
                ic, _ = spearmanr(fv, rv)
                if not np.isnan(ic):
                    ics.append(ic)
        if ics:
            ic_dict[fname] = np.mean(ics)

    model.update_ic_batch(ic_dict, rb_date)

    # Predict: cross-sectional scores on rb_date
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
        preds = model.predict(fn, factor_names)
        pd_dict = {sym: float(preds[j]) for j, sym in enumerate(scores.keys())}
        ranked = sorted(pd_dict.items(), key=lambda x: -x[1])
        ic_preds.append((rb_date, [s for s, _ in ranked], len(scores)))

print(f"  Predictions: {len(ic_preds)} periods")
w = model.get_weights()
aw = {k: round(v, 4) for k, v in sorted(w.items(), key=lambda x: -x[1]) if v > 0}
print(f"  Active factors: {aw}")

# ============================================================
# Pre-compute daily returns matrix for speed
# ============================================================
print("\n[5/6] Pre-computing returns matrix...")
# Build a date-indexed returns DataFrame for all stocks
all_returns = {}
for sym in symbols:
    df = all_data[sym]
    c = df["close"]
    daily_ret = c.pct_change()
    all_returns[sym] = daily_ret

# Benchmark = equal weight all stocks (pre-computed per date)
print("  Computing full-universe benchmark returns...")
bench_daily = {}
for d in all_dates[1:]:  # skip first (no return)
    rets = []
    for sym in symbols:
        sr = all_returns[sym]
        if d in sr.index and pd.notna(sr.at[d]):
            rets.append(sr.at[d])
    bench_daily[d] = np.mean(rets) if rets else 0.0

# Strategy daily returns lookup (vectorized per symbol)
sym_returns = {}
for sym in symbols:
    sym_returns[sym] = all_returns[sym]

print(f"  Pre-computed {len(bench_daily)} benchmark daily returns")

# ============================================================
# FIX #2: Benchmark = FULL UNIVERSE equal weight
# ============================================================
top_k = 30
buffer_k = 50


def run_backtest_corrected(holdings_per_period, label):
    """Backtest with corrected costs and full-universe benchmark."""
    equity = [1_000_000.0]
    bench_equity = [1_000_000.0]
    dates_list = [holdings_per_period[0][0]]
    prev_holdings = set()
    total_turnover = 0.0
    n_rb = 0
    yearly_strat = {}
    yearly_bench = {}

    for idx in range(len(holdings_per_period) - 1):
        entry = holdings_per_period[idx]
        rb_date, new_h_full = entry[0], entry[1]
        next_rb = holdings_per_period[idx + 1][0]
        new_h = set(new_h_full[:top_k])

        # Buffer zone
        if prev_holdings:
            buffered = set(new_h_full[:buffer_k])
            keep = prev_holdings & buffered
            add = new_h - prev_holdings
            final_h = list(keep) + list(add)
            final_h = final_h[:top_k]
            new_h = set(final_h)

        to = 0.0
        if prev_holdings:
            to = (len(prev_holdings - new_h) + len(new_h - prev_holdings)) / (2 * max(len(new_h), 1))
            total_turnover += to
        n_rb += 1
        prev_holdings = new_h

        period = [d for d in all_dates if rb_date < d <= next_rb]
        year = rb_date.year

        for di, pd_date in enumerate(period):
            # Strategy return (use pre-computed returns)
            ret = 0.0
            n = 0
            for sym in new_h:
                sr = sym_returns.get(sym)
                if sr is not None and pd_date in sr.index and pd.notna(sr.at[pd_date]):
                    ret += sr.at[pd_date]
                    n += 1
            if n > 0:
                ret /= n

            # FIX #2: Benchmark = pre-computed full universe EW
            bench_ret = bench_daily.get(pd_date, 0.0)

            # FIX #5: Period-dependent cost
            cost = cm.round_trip_rate(pd_date) * to if di == 0 else 0.0

            strat_net = ret - cost
            equity.append(equity[-1] * (1 + strat_net))
            bench_equity.append(bench_equity[-1] * (1 + bench_ret))
            dates_list.append(pd_date)

            yearly_strat.setdefault(year, []).append(strat_net)
            yearly_bench.setdefault(year, []).append(bench_ret)

    # Metrics
    eq = pd.Series(equity)
    bm = pd.Series(bench_equity)
    total_ret = equity[-1] / equity[0] - 1
    bench_total = bench_equity[-1] / bench_equity[0] - 1
    years = len(equity) / 250
    ann_ret = (1 + total_ret) ** (1 / years) - 1
    ann_bench = (1 + bench_total) ** (1 / years) - 1
    dd = (eq / eq.cummax() - 1).min()
    daily_ret = eq.pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(250) if daily_ret.std() > 0 else 0

    # Excess
    excess_series = eq.pct_change().fillna(0) - bm.pct_change().fillna(0)
    te = excess_series.std() * np.sqrt(250)
    ir = (ann_ret - ann_bench) / te if te > 0 else 0

    avg_to = total_turnover / max(n_rb, 1)

    print(f"\n  {label}")
    print(f"    Strategy:  Total={total_ret*100:+.1f}%  Ann={ann_ret*100:+.1f}%  Sharpe={sharpe:.2f}  MaxDD={dd*100:.1f}%")
    print(f"    Benchmark: Total={bench_total*100:+.1f}%  Ann={ann_bench*100:+.1f}% (full universe EW)")
    print(f"    Excess:    Ann={ann_ret-ann_bench:+.1f}%  IR={ir:.3f}  TE={te*100:.1f}%")
    print(f"    Turnover:  {avg_to*100:.0f}%/rb  Annual={avg_to*n_rb/years*100:.0f}%")

    # Year-by-year
    print(f"    Year-by-year (strategy vs benchmark):")
    for y in sorted(yearly_strat.keys()):
        ys = np.prod([1 + r for r in yearly_strat[y]]) - 1
        yb = np.prod([1 + r for r in yearly_bench[y]]) - 1
        print(f"      {y}: strat={ys*100:+5.1f}%  bench={yb*100:+5.1f}%  excess={ys-yb:+5.1f}%")

    return {
        "label": label, "total": total_ret, "ann": ann_ret,
        "bench_ann": ann_bench, "excess_ann": ann_ret - ann_bench,
        "sharpe": sharpe, "maxdd": float(dd), "ir": ir, "te": te,
        "avg_to": avg_to,
        "yearly": {y: {"strat": float(np.prod([1+r for r in rs])-1),
                       "bench": float(np.prod([1+r for r in yearly_bench[y]])-1)}
                   for y, rs in yearly_strat.items()}
    }


# Strategy: IC-Linear with embargo
results = []
results.append(run_backtest_corrected(ic_preds, "IC-Linear (63d IC + embargo)"))

# Also run: Pure reversal (for comparison)
print("\n" + "-" * 65)
reversal_preds = []
for rb_date in rebalance_dates:
    fp = factor_panels.get("ret_20d", {})
    scores = {}
    for sym in symbols:
        if sym in fp and rb_date in fp[sym].index:
            v = fp[sym].at[rb_date]
            if pd.notna(v):
                scores[sym] = v
    ranked = sorted(scores.items(), key=lambda x: x[1])  # worst first
    reversal_preds.append((rb_date, [s for s, _ in ranked]))
results.append(run_backtest_corrected(reversal_preds, "Pure Reversal (ret_20d bottom)"))

# Low vol
print("\n" + "-" * 65)
lowvol_preds = []
for rb_date in rebalance_dates:
    fp = factor_panels.get("vol_20d", {})
    scores = {}
    for sym in symbols:
        if sym in fp and rb_date in fp[sym].index:
            v = fp[sym].at[rb_date]
            if pd.notna(v):
                scores[sym] = v
    ranked = sorted(scores.items(), key=lambda x: x[1])  # lowest vol first
    lowvol_preds.append((rb_date, [s for s, _ in ranked]))
results.append(run_backtest_corrected(lowvol_preds, "Low Volatility (vol_20d bottom)"))

# Reversal + LowVol combo
print("\n" + "-" * 65)
combo_preds = []
for rb_date in rebalance_dates:
    rev_fp = factor_panels.get("ret_20d", {})
    vol_fp = factor_panels.get("vol_20d", {})
    common = [s for s in symbols if s in rev_fp and rb_date in rev_fp[s].index
              and s in vol_fp and rb_date in vol_fp[s].index
              and pd.notna(rev_fp[s].at[rb_date]) and pd.notna(vol_fp[s].at[rb_date])]
    if len(common) < 50:
        continue
    rev_arr = np.array([rev_fp[s].at[rb_date] for s in common])
    vol_arr = np.array([vol_fp[s].at[rb_date] for s in common])
    rev_z = (rev_arr - rev_arr.mean()) / (rev_arr.std() + 1e-8)
    vol_z = (vol_arr - vol_arr.mean()) / (vol_arr.std() + 1e-8)
    combo = -0.6 * rev_z - 0.4 * vol_z
    ranked_idx = np.argsort(-combo)
    combo_preds.append((rb_date, [common[i] for i in ranked_idx]))
results.append(run_backtest_corrected(combo_preds, "Reversal+LowVol (60/40)"))

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 65)
print("  CORRECTED RESULTS (all 5 fixes applied)")
print("=" * 65)
print(f"  Period: {START_DATE.date()} -> {END_DATE.date()}")
print(f"  Universe: {len(symbols)} stocks (ALL, no filter)")
print(f"  Benchmark: Full universe equal-weight")
print(f"  IC embargo: 63 trading days")
print(f"  Costs: {cm.round_trip_rate(pd.Timestamp('2022-01-01'))*10000:.1f}bp (pre-2023) / {cm.round_trip_rate(pd.Timestamp('2024-01-01'))*10000:.1f}bp (post-2023)")
print()
print(f"  {'Strategy':<35s} {'Ann':>6s} {'Excess':>7s} {'IR':>6s} {'Sharpe':>7s} {'MaxDD':>7s}")
print("  " + "-" * 70)
for r in results:
    print(f"  {r['label']:<35s} {r['ann']*100:>+5.1f}% {r['excess_ann']*100:>+6.1f}% {r['ir']:>5.3f} {r['sharpe']:>6.2f} {r['maxdd']*100:>6.1f}%")

print(f"\n  Benchmark (full universe EW): ann={results[0]['bench_ann']*100:+.1f}%")
print(f"\n  Time: {time.time()-t0:.0f}s")
print("=" * 65)

# Save
out = Path(__file__).resolve().parent.parent / "data" / "corrected_results_v3.json"
out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
print(f"  Saved: {out}")
