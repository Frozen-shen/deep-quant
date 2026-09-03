"""
run_fold7_attribution.py — fold_7 (2026H1) 真实亏损归因诊断 (技术方案 3.4, P3)

背景: walkforward_results.json 中 fold_7 (滚动重训至 2025 底, 验证 2026H1)
绝对收益 -30.6% / 超额 -29.9% (基准 -0.7%)。本脚本对五条假设逐一给证据,
允许"当前证据不足以判断"作为诚实的中间结论, 每条假设都有明确排查方法与结果。

假设:
  H1 因子方向反转   — 对比 fold_7 每期调仓的因子"下注"方向 (weights_evolution
                      权重符号) 与 2026H1 验证窗内实现的前瞻收益 IC 方向；下注方向
                      是训练结果代理，不等同于独立重算的训练期 IC。
  H2 regime 误判    — 回放 RegimeDetector (CSI1000 基准) 在 2026H1 每日给出的
                      市场状态/仓位乘数, 对照该日之后 20 日实际市场 (基准与等权)。
  H3 持仓过度集中   — 由 trades 重建持仓, 检查单票/行业集中度是否频繁打满约束。
  H4 执行侧问题     — fold_7 成交价 vs 当日 VWAP (minute_5m) / 收盘价, 与全运行
                      执行质量 (execution_quality) 对照, 判断亏损是否执行放大。
  H5 样本量不足     — 6 次调仓 / 22 笔 / 116 日下, -30.6% 年化的 t 分布近似敏感性
                      区间 (含 21 日重叠标签的粗略有效样本数折算，不是正式 CI)。

用法 (离线, 只读本地):
  py scripts/active/run_fold7_attribution.py                # 全量 (H1 需预计算因子面板, 约 20-40 分钟)
  py scripts/active/run_fold7_attribution.py --only h2,h5    # 只跑指定假设 (H1/H2 需本地数据, 无网络)
  py scripts/active/run_fold7_attribution.py --h1-limit 200  # H1 抽样股票数 (冒烟)
  py scripts/active/run_fold7_attribution.py --json data/ic_validation/fold7_attribution.json

报告: docs/superpowers/plans/2026-09-03-fold7-attribution.md 由此脚本输出汇总。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

DEFAULT_RESULTS = os.path.join(BASE_DIR, "data", "ic_validation", "walkforward_results.json")
STORE_DIR = os.path.join(BASE_DIR, "data_store")
MINUTE_DIR = os.path.join(BASE_DIR, "data_store", "minute_5m")
BENCH_PATH = os.path.join(BASE_DIR, "data", "cache", "index_csi1000.parquet")
INDUSTRY_MAP = os.path.join(BASE_DIR, "data_store", "aux_industry", "industry_map.parquet")
LOOKBACK_START = "2024-01-01"   # 因子滚动窗 (~252 交易日) 前垫
VAL_START, VAL_END = "2026-01-05", "2026-06-30"
LABEL_HORIZON_DAYS = 21          # 与生产一致 (REBALANCE_DAYS+1)
_MIN_CS = 30                     # IC 截面最少股票数 (与生产 MIN_CROSS_SECTION 同量级)


# ════════════════════════ 公共加载 ════════════════════════

def _stock_files():
    """data_store 下 6 位代码 parquet 列表。"""
    out = []
    for fn in sorted(os.listdir(STORE_DIR)):
        stem = fn[:-8]  # 去掉 ".parquet"
        if fn.endswith(".parquet") and len(stem) == 6 and stem.isdigit():
            out.append(os.path.join(STORE_DIR, fn))
    return out


def load_closes(symbols, start=LOOKBACK_START, limit=0):
    """读 (date, close, amount) 子集, 返回 {sym: DataFrame}。过滤上市不足者。"""
    out = {}
    files = _stock_files()
    if symbols:
        wanted = set(symbols)
        files = [f for f in files if os.path.basename(f)[:6] in wanted]
    if limit:
        files = files[:limit]
    t0 = time.time()
    for i, path in enumerate(files):
        sym = os.path.basename(path)[:6]
        try:
            df = pd.read_parquet(path, columns=["date", "close", "amount"])
        except Exception:
            continue
        df = df[df["date"] >= pd.Timestamp(start)]
        if len(df) < 400:
            continue
        df = df.reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        out[sym] = df
        if (i + 1) % 1000 == 0:
            print(f"    [load] {i+1} 只, {len(out)} 有效 ({time.time()-t0:.0f}s)", flush=True)
    print(f"  [load] 有效股票 {len(out)}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    return out


def close_matrix(all_data, dates):
    """{sym: df} → DataFrame(date × sym) 收盘价 (仅 dates)。"""
    ser = {s: df.set_index("date")["close"].reindex(dates) for s, df in all_data.items()}
    return pd.DataFrame(ser)


def trading_days_in_window():
    from data.calendar import get_trading_days
    return [d for d in get_trading_days(VAL_START, VAL_END)]


def fwd_returns(cm: pd.DataFrame, horizon=LABEL_HORIZON_DAYS):
    """cm(date×sym) → 21 日后收益 DataFrame (对齐 cm 索引)。"""
    out = pd.DataFrame(index=cm.index, columns=cm.columns, dtype=float)
    idx = list(cm.index)
    for i, d in enumerate(idx):
        j = i + horizon
        if j >= len(idx):
            break
        out.loc[d] = cm.iloc[j] / cm.iloc[i] - 1.0
    return out


# ════════════════════════ H1: 因子方向反转 ════════════════════════

def _h1_bets(fold7):
    """weights_evolution → [(date, {fn: weight}), ...]。"""
    out = []
    for we in fold7.get("weights_evolution", []):
        w = {k: float(v) for k, v in (we.get("weights") or {}).items() if float(v) != 0}
        out.append((pd.Timestamp(we["date"]), w))
    return out


def compute_val_panels(symbols, val_dates, limit=0):
    """仅计算 fold_7 验证窗内因子值 (滚动窗前垫自 LOOKBACK_START)。

    返回 {factor_name: DataFrame(val_dates × sym, float32)} + 可算因子名列表。
    说明: fund_*/aux_*/ind_mom_60 等非 DSL 因子需外部面板 (生产由
    merge_surprise_panels 合并), 本脚本只重算 FactorScorer full_auto 的 DSL 因子,
    其余因子在结果中单独标注为"未纳入重算"。
    """
    from factor_scorer import FactorScorer
    scorer = FactorScorer.from_preset("full_auto")
    factor_names = sorted(scorer.factor_weights.keys())
    all_data = load_closes(symbols, start=LOOKBACK_START, limit=limit)
    t0 = time.time()
    cols = {fn: [] for fn in factor_names}  # 每因子存 (sym, Series)
    for i, (sym, df) in enumerate(all_data.items()):
        try:
            feats = scorer.compute_factors(df)
        except Exception:
            continue
        feats = feats.reset_index(drop=True)
        dts = pd.to_datetime(df["date"]).reset_index(drop=True)
        keep = dts.isin(val_dates)
        if not keep.any():
            continue
        fsub = feats[keep].select_dtypes(include=[np.number]).astype(np.float32)
        dsub = dts[keep]
        for fn in factor_names:
            if fn in fsub.columns:
                cols[fn].append(pd.Series(fsub[fn].values, index=dsub, name=sym))
        if (i + 1) % 1000 == 0:
            print(f"    [H1] 因子计算 {i+1}/{len(all_data)} ({time.time()-t0:.0f}s)", flush=True)
    panels = {}
    for fn in factor_names:
        if not cols[fn]:
            continue
        p = pd.concat(cols[fn], axis=1)
        panels[fn] = p.reindex(index=val_dates).astype(np.float32)
    print(f"  [H1] 面板完成: {len(panels)} 因子 × {len(val_dates)} 日 × "
          f"{len(all_data)} 只 ({time.time()-t0:.0f}s)", flush=True)
    return panels, all_data


def _spearman_cs(a, b):
    from scipy.stats import rankdata
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < _MIN_CS:
        return np.nan
    return float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])


def h1(fold7, val_dates, limit=0):
    """对比下注方向与验证窗实现 IC。返回统计 dict。

    股票池 = data_store 全市场 (≥400 行的样本, 近似 fold_7 PIT 口径, 未做
    流动性过滤 — 近似误差在报告中说明); 因子 = full_auto DSL 名单 ∩ 下注因子。
    """
    bets = _h1_bets(fold7)
    panels, all_data = compute_val_panels(None, val_dates, limit=limit)
    all_dates = val_dates + _extend_dates(val_dates)
    cm = close_matrix(all_data, all_dates)
    fwd = fwd_returns(cm)
    fwd_dates = [d for d in val_dates if d in fwd.index and fwd.loc[d].notna().sum() >= _MIN_CS]
    if not fwd_dates:
        return {"verdict": "insufficient", "note": "验证窗无足够前瞻收益样本"}

    # 每因子: 窗口内日均 IC (逐日截面 spearman, 逐日对齐面板/收益的股票交集)
    per_factor = {}
    for fn, panel in panels.items():
        ics = []
        for d in fwd_dates:
            if d not in panel.index:
                continue
            a = panel.loc[d].astype(float)
            b = fwd.loc[d].astype(float)
            common = a.index.intersection(b.index)
            if len(common) < _MIN_CS:
                continue
            ics.append(_spearman_cs(a[common].values, b[common].values))
        ics = [x for x in ics if not np.isnan(x)]
        if len(ics) >= 30:
            per_factor[fn] = {"ic_mean": float(np.mean(ics)), "n": len(ics)}

    # 下注方向 = 各调仓日权重符号 (取最后一次调仓为代表 + 逐期命中率)
    rows = []
    flipped_weight_share = {"flipped": 0.0, "total": 0.0}
    per_bet_hits = []
    last_date, last_w = bets[-1]
    for d, w in bets:
        hit = 0
        for fn, wt in w.items():
            if fn not in per_factor:
                continue
            sign_ok = (wt > 0) == (per_factor[fn]["ic_mean"] > 0)
            hit += 1 if sign_ok else 0
            if d == last_date:
                flipped_weight_share["total"] += abs(wt)
                if not sign_ok:
                    flipped_weight_share["flipped"] += abs(wt)
        per_bet_hits.append({"date": str(d.date()), "hit": hit, "n": sum(
            1 for fn in w if fn in per_factor)})

    flipped = []
    for fn, st in per_factor.items():
        if fn in last_w:
            ok = (last_w[fn] > 0) == (st["ic_mean"] > 0)
            flipped.append((fn, last_w[fn], st["ic_mean"], ok))
    flipped.sort(key=lambda x: -abs(x[1]))
    n_flip = sum(1 for _, _, _, ok in flipped if not ok)
    res = {
        "n_bets_dates": len(bets),
        "n_factors_bet": len(last_w),
        "n_factors_with_realized_ic": len(per_factor),
        "excluded_non_dsl": sorted(set(last_w) - set(per_factor)),
        "flip_count": n_flip,
        "flip_weight_share": (flipped_weight_share["flipped"] /
                              max(flipped_weight_share["total"], 1e-9)),
        "per_bet_hits": per_bet_hits,
        "top_flipped": [(fn, round(wt, 4), round(ic, 4))
                        for fn, wt, ic, ok in flipped if not ok][:15],
        "note": "IC 为 21 日前瞻收益逐日截面 spearman 均值 (重叠标签, 仅供方向判断)",
    }
    return res


def _extend_dates(val_dates):
    """val_dates 之后 ~25 个交易日 (前瞻收益实现期)。"""
    from data.calendar import get_trading_days
    if not val_dates:
        return []
    start = (val_dates[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return [d for d in get_trading_days(start, "2026-08-31")][:LABEL_HORIZON_DAYS + 5]


# ════════════════════════ H2: regime 误判回放 ════════════════════════

def h2(val_dates, all_data=None):
    """回放 RegimeDetector 2026H1 每日判断 vs 实际 20 日市场表现。"""
    from regime_detector import RegimeDetector
    if not os.path.exists(BENCH_PATH):
        return {"verdict": "insufficient", "note": f"缺基准文件 {BENCH_PATH}"}
    det = RegimeDetector.from_benchmark_parquet(BENCH_PATH, vol_source="daily")
    bench = pd.read_parquet(BENCH_PATH)
    bench["date"] = pd.to_datetime(bench["date"])
    bench = bench.set_index("date")["close"]
    bdays = [d for d in val_dates if d in bench.index]
    rows = []
    for i, d in enumerate(bdays):
        j = min(i + 20, len(bdays) - 1)
        bfwd = bench.iloc[j] / bench.iloc[i] - 1
        try:
            reg, vol_pct = det.detect_v2(str(d.date()))
        except Exception as e:
            rows.append({"date": str(d.date()), "error": str(e)})
            continue
        mults = det.get_weight_multipliers(str(d.date()))
        rows.append({
            "date": str(d.date()), "regime": str(reg), "vol_pct": round(vol_pct, 3),
            "mom_mult": mults.get("momentum"), "rev_mult": mults.get("reversal"),
            "bench_fwd20": round(bfwd, 4),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return {"verdict": "insufficient", "note": "基准数据无覆盖"}
    risk_off = df[df["mom_mult"] < 1.0]
    risk_on = df[df["mom_mult"] >= 1.0]
    df["risk_off"] = df["mom_mult"] < 1.0
    df["up_realized"] = df["bench_fwd20"] > 0
    # 系统性误判 = 风险关闭日里实际上涨占比 / 风险开启日里实际下跌占比
    out = {
        "n_days": len(df),
        "regime_counts": df["regime"].value_counts().to_dict(),
        "risk_off_days": int(len(risk_off)),
        "risk_off_mean_fwd20": float(risk_off["bench_fwd20"].mean()) if len(risk_off) else None,
        "risk_on_mean_fwd20": float(risk_on["bench_fwd20"].mean()) if len(risk_on) else None,
        "risk_off_but_up_share": float((risk_off["bench_fwd20"] > 0).mean()) if len(risk_off) else None,
        "risk_on_but_down_share": float((risk_on["bench_fwd20"] <= 0).mean()) if len(risk_on) else None,
        "vol_bucket_counts": {str(k): int(v) for k, v in df.groupby(
            pd.cut(df["vol_pct"], [0, .3, .7, 1.0])).size().items()},
        "daily": df.to_dict("records"),
        "note": "mom_mult<1 视为风险关闭 (下跌防护启用); 基准 = CSI1000 20 日实际收益",
    }
    return out


# ════════════════════════ H3: 持仓集中度 ════════════════════════

def h3(fold7, industry_map=INDUSTRY_MAP):
    """持仓集中度 — 用 positions_history 的 6 个调仓日 + trades 重建持仓近似。

    限制: 存档 positions_history 只含 symbol 列表, 权重按收盘价 × 股数估算
    (相对持仓内部), 未计现金; 用于集中度数量级判断而非精确仓位。
    """
    trades = sorted(fold7.get("trades", []),
                    key=lambda t: pd.Timestamp(t["date"]))
    # 按调仓日做事件时点回放；不能先累加所有成交再回填早期日期，
    # 否则后来的卖出/买入会污染此前的集中度。
    holdings = {}
    trade_i = 0
    rebal_dates = [pd.Timestamp(p["date"]) for p in
                   fold7.get("positions_history", [])]
    if not rebal_dates:
        rebal_dates = sorted({pd.Timestamp(t["date"]) for t in trades})
    syms = sorted({t["symbol"] for t in trades})
    closes = {}
    for s in syms:
        try:
            f = os.path.join(STORE_DIR, f"{s}.parquet")
            if not os.path.exists(f):
                continue
            sub = pd.read_parquet(f, columns=["date", "close"])
            closes[s] = sub.set_index("date")["close"]
        except Exception:
            continue
    imap = None
    if os.path.exists(industry_map):
        im = pd.read_parquet(industry_map)
        imap = {}
        for code, ind in zip(im["code"].astype(str), im["industry"].astype(str)):
            imap[code[-6:]] = ind  # sh600176 → 600176
    out_rows = []
    for d in rebal_dates:
        while trade_i < len(trades) and pd.Timestamp(trades[trade_i]["date"]) <= d:
            t = trades[trade_i]
            qty = int(t["qty"]) * (1 if t["action"] == "BUY" else -1)
            holdings[t["symbol"]] = holdings.get(t["symbol"], 0) + qty
            if holdings[t["symbol"]] == 0:
                holdings.pop(t["symbol"], None)
            trade_i += 1
        vals = {}
        for s in syms:
            if s not in closes:
                continue
            ser = closes[s]
            if not (ser.index <= d).any():
                continue
            px = float(ser[ser.index <= d].iloc[-1])
            if px > 0 and holdings.get(s, 0) > 0:
                vals[s] = holdings[s] * px
        tot = sum(vals.values())
        if tot <= 0:
            continue
        wts = {s: v / tot for s, v in vals.items()}
        top = max(wts.values())
        ind_w = {}
        if imap:
            for s, w in wts.items():
                ind = imap.get(s, "未知")
                ind_w[ind] = ind_w.get(ind, 0.0) + w
        ind_known = {k: v for k, v in ind_w.items() if k != "未知"}
        out_rows.append({
            "date": str(d.date()), "n_hold": len(vals),
            "max_single_pct": round(top * 100, 1),
            "max_industry_pct": round(max(ind_w.values(), default=0) * 100, 1),
            "max_known_industry_pct": round(max(ind_known.values(), default=0) * 100, 1),
            "unknown_map_pct": round(ind_w.get("未知", 0.0) * 100, 1),
            "industry": {k: round(v * 100, 1) for k, v in
                         sorted(ind_known.items(), key=lambda kv: -kv[1])[:3]},
        })
    if not out_rows:
        return {"verdict": "insufficient", "note": "收盘价重建失败"}
    o = pd.DataFrame(out_rows)
    return {
        "rows": out_rows,
        "n_dates": len(o),
        "max_single_pct_mean": float(o["max_single_pct"].mean()),
        "max_single_pct_max": float(o["max_single_pct"].max()),
        "max_industry_pct_mean": float(o["max_industry_pct"].mean()),
        "max_industry_pct_max": float(o["max_industry_pct"].max()),
        "max_known_industry_pct_mean": float(o["max_known_industry_pct"].mean()),
        "max_known_industry_pct_max": float(o["max_known_industry_pct"].max()),
        "unknown_map_pct_mean": float(o["unknown_map_pct"].mean()),
        "unknown_map_pct_max": float(o["unknown_map_pct"].max()),
        "hit_20pct_single": int((o["max_single_pct"] >= 19.9).sum()),
        "hit_25pct_known_industry": int((o["max_known_industry_pct"] >= 24.9).sum()),
        "note": ("权重=收盘价×股数 相对持仓内部估算 (无持仓市值存档, 未计现金); "
                 "行业 map 覆盖不全, '未知'桶为 map 缺口, 已知行业上限单独统计"),
    }


# ════════════════════════ H4: 执行侧 ════════════════════════

def _day_vwap(sym, d):
    """minute_5m 当日 VWAP = Σamount/Σvolume; 无分钟数据返回 None。"""
    path = os.path.join(MINUTE_DIR, f"{sym}.parquet")
    if not os.path.exists(path):
        return None
    try:
        m = pd.read_parquet(path, columns=["day", "amount", "volume"])
    except Exception:
        return None
    m = m[pd.to_datetime(m["day"]).dt.date == pd.Timestamp(d).date()]
    if len(m) == 0 or m["volume"].sum() <= 0:
        return None
    return float(m["amount"].sum() / m["volume"].sum())


def h4(fold7, exec_quality=None):
    """fold_7 成交 vs 当日 VWAP/收盘, 与全运行执行质量对照。

    数据校验: 600508 的 minute_5m amount/volume 与日线 OHLC 复权基准不一致
    (隐含价 12.01 vs 日线 8.46, 比值 ≈1.42), 其 vwap 残差标为数据质量候选，
    不在敏感性统计中使用。
    """
    trades = fold7.get("trades", [])
    if not trades:
        return {"verdict": "insufficient", "note": "无成交记录"}
    rows = []
    for t in trades:
        d = pd.Timestamp(t["date"])
        fill = float(t["price"])
        vwap = _day_vwap(t["symbol"], d)
        close_px = None
        try:
            sub = pd.read_parquet(os.path.join(STORE_DIR, f"{t['symbol']}.parquet"),
                                  columns=["date", "close"])
            sub = sub[sub["date"] == d]
            if len(sub):
                close_px = float(sub["close"].iloc[0])
        except Exception:
            pass
        rows.append({
            "date": str(d.date()), "sym": t["symbol"], "action": t["action"],
            "qty": t["qty"], "fill": round(fill, 4),
            "vwap": None if vwap is None else round(vwap, 4),
            "close": close_px,
            "fill_minus_vwap_bps": None if vwap is None else
            round((fill / vwap - 1) * 1e4, 1),
            "fill_minus_close_bps": None if close_px is None else
            round((fill / close_px - 1) * 1e4, 1),
            "n_fill_slices": len(t.get("fill_times", [])),
        })
    o = pd.DataFrame(rows)
    SUSPECT = 500.0  # |残差|>500bp 视为数据质量候选 (如 600508 复权基准不一致)
    o["suspect"] = o["fill_minus_vwap_bps"].abs() > SUSPECT

    def _side(sub):
        if not len(sub):
            return {"n": 0, "vs_vwap_bps_mean": None,
                    "vs_vwap_bps_median": None, "vs_close_bps_mean": None,
                    "suspect_rows": []}
        vv = sub["fill_minus_vwap_bps"].dropna()
        cc = sub["fill_minus_close_bps"].dropna()
        return {
            "n": int(len(sub)),
            "vs_vwap_bps_mean": float(vv.mean()) if len(vv) else None,
            "vs_vwap_bps_median": float(vv.median()) if len(vv) else None,
            "vs_close_bps_mean": float(cc.mean()) if len(cc) else None,
            "suspect_rows": sub[sub["suspect"]][["date", "sym", "fill",
                                                 "vwap", "fill_minus_vwap_bps"]]
            .to_dict("records"),
        }

    return {
        "n_trades": len(o),
        "n_vwap_covered": int(o["fill_minus_vwap_bps"].notna().sum()),
        "suspect_threshold_bps": SUSPECT,
        "by_side": {"BUY": _side(o[o["action"] == "BUY"]),
                    "SELL": _side(o[o["action"] == "SELL"])},
        # 把报告中用于排除数据质量候选行的敏感性统计一并落盘，避免人工
        # 计算出一个 JSON 中不存在、无法复核的“清洗后均值”。
        "excluding_suspect": {
            "n_trades": int((~o["suspect"]).sum()),
            "n_vwap_covered": int((~o["suspect"] &
                                    o["fill_minus_vwap_bps"].notna()).sum()),
            "by_side": {"BUY": _side(o[(~o["suspect"]) &
                                         (o["action"] == "BUY")]),
                        "SELL": _side(o[(~o["suspect"]) &
                                         (o["action"] == "SELL")])},
        },
        "all_run_execution_quality": exec_quality,
        "rows": o.to_dict("records"),
        "note": ("fill vs 当日分钟 VWAP (买方低于 VWAP 为有利); 生产 POV 残差预算 "
                 "10bp; |残差|>500bp 视为数据质量候选 (600508 分钟 amount/volume 与 "
                 "日线复权基准不一致, 隐含价 12.01 vs 日线 8.46, 比值≈1.42)"),
    }


# ════════════════════════ H5: 样本量与敏感性区间 ════════════════════════

def h5(fold7):
    from scipy import stats
    daily = fold7.get("daily_active_returns")
    n_reb = fold7.get("n_rebalances", 0)
    n_trades = len(fold7.get("trades", []))
    n_days = fold7.get("n_days", 0)
    if not daily:
        return {"verdict": "insufficient", "note": "无 daily_active_returns"}
    arr = np.asarray(daily, dtype=float)
    mean_d = float(arr.mean())   # daily_active_returns 为小数 (×100 = %)
    sd_d = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    # 21 日重叠标签 → 有效样本 ≈ 不重叠块数
    neff = max(1, int(len(arr) / LABEL_HORIZON_DAYS))
    se_d = sd_d / np.sqrt(len(arr))
    se_neff = sd_d / np.sqrt(neff)
    ci_full = stats.t.ppf(0.975, len(arr) - 1) * se_d * 252 * 100
    ci_neff = stats.t.ppf(0.975, neff - 1) * se_neff * 252 * 100
    ann = mean_d * 252 * 100  # pp (年化超额)
    return {
        "n_days": int(n_days), "n_rebalances": int(n_reb), "n_trades": int(n_trades),
        "daily_excess_mean_pct": round(mean_d * 100, 3),
        "daily_excess_std_pct": round(sd_d * 100, 3),
        "annualized_excess_pp": round(ann, 1),
        "ci_95_pp_full_obs": [round(ann - ci_full, 1), round(ann + ci_full, 1)],
        "ci_95_pp_neff21": [round(ann - ci_neff, 1), round(ann + ci_neff, 1)],
        "neff_21d": int(neff),
        "note": ("daily_active_returns 为小数; 年化超额 = 日均×252×100pp; "
                 "重叠标签按 21 日粗略折算有效样本数; 区间为 t 分布近似敏感性分析，"
                 "不替代基于自相关/块 bootstrap 的正式置信区间"),
    }


# ════════════════════════ main ════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--only", default="", help="逗号分隔 h1..h5; 默认全跑")
    ap.add_argument("--h1-limit", type=int, default=0, help="H1 抽样股票数上限 (0=全市场)")
    ap.add_argument("--json", default="", help="结果 JSON 输出路径")
    args = ap.parse_args()

    t0 = time.time()
    with open(args.results, encoding="utf-8") as f:
        doc = json.load(f)
    if "fold_7" not in (doc.get("results") or {}):
        print(f"✗ {args.results} 无 fold_7 记录"); sys.exit(1)
    fold7 = doc["results"]["fold_7"]
    print(f"fold_7: {fold7.get('period')} 总收益 {fold7.get('total_return')}% "
          f"年化 {fold7.get('annual_return')}% 超额 {fold7.get('excess_annual')}pp "
          f"基准 {fold7.get('benchmark_annual')}% ({fold7.get('n_rebalances')} 调仓, "
          f"{len(fold7.get('trades', []))} 笔)")

    val_dates = trading_days_in_window()
    only = [x.strip() for x in args.only.split(",") if x.strip()]
    todo = only or ["h1", "h2", "h3", "h4", "h5"]
    out = {"fold_7": fold7.get("period"), "hypotheses": {}}

    if "h1" in todo:
        print("\n=== H1 因子方向反转 ===", flush=True)
        out["hypotheses"]["h1"] = h1(fold7, val_dates, limit=args.h1_limit)
    if "h2" in todo:
        print("\n=== H2 regime 误判回放 ===", flush=True)
        out["hypotheses"]["h2"] = h2(val_dates)
    if "h3" in todo:
        print("\n=== H3 持仓集中度 ===", flush=True)
        out["hypotheses"]["h3"] = h3(fold7)
    if "h4" in todo:
        print("\n=== H4 执行侧 ===", flush=True)
        out["hypotheses"]["h4"] = h4(fold7, exec_quality=doc.get("execution_quality"))
    if "h5" in todo:
        print("\n=== H5 样本量敏感性区间 ===", flush=True)
        out["hypotheses"]["h5"] = h5(fold7)

    print(f"\n完成 ({time.time()-t0:.0f}s)")
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
        print(f"结果: {args.json}")


if __name__ == "__main__":
    main()
