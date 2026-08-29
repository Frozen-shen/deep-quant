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
        "report_date": pd.to_datetime(df["日期"], errors="coerce"),
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
    """公告漂移面板: 公告后 20 交易日市场调整累计收益。

    PIT-safe: 漂移窗口完成后 (公告日+20 交易日) 才生效, 持续到下一窗口完成;
    窗口内任何决策时点不暴露未来收益 (略滞后但保因果)。
    """
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
            events.append((idx[pos + 20], drift))
        out[s] = events
    return _to_panel(out, calendar)


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
        return pd.DataFrame(np.nan, index=idx, columns=sorted(symbols),
                            dtype=np.float32)
    ind_panel = pd.DataFrame(
        {k: pd.concat(v, axis=1).mean(axis=1) for k, v in ind_rets.items()})
    # 滚动 lookback 日累计收益 (对数近似, 避免复利偏差)
    # 缺日 (停牌/数据缺) 按 0 对数收益处理 — 建模假设, 非前视
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


_industry_map_cache: dict | None = None


def load_industry_map() -> dict:
    """行业映射 {6位代码: industry} (data_store/aux_industry 快照)。

    与回测 _load_industry_map 同一数据源与键规范 (去 sh/sz 前缀);
    缺失/损坏 → {} (调用方降级)。模块级缓存。"""
    global _industry_map_cache
    if _industry_map_cache is not None:
        return _industry_map_cache
    path = os.path.join(BASE_DIR, "data_store", "aux_industry",
                        "industry_map.parquet")
    if not os.path.exists(path):
        _industry_map_cache = {}
        return _industry_map_cache
    try:
        df = pd.read_parquet(path)
        codes = (df["code"].astype(str)
                 .str.replace("sh", "", regex=False)
                 .str.replace("sz", "", regex=False))
        _industry_map_cache = dict(zip(codes, df["industry"].astype(str)))
    except Exception:
        _industry_map_cache = {}
    return _industry_map_cache


def industry_mom_broadcast_at(all_data: dict, industry_map: dict,
                              as_of, lookback: int = 60) -> dict:
    """as_of 日各股行业动量广播值 (= industry_momentum_panel 最后一行)。

    与全历史面板逐值一致: rolling(lookback).sum 只依赖过去 lookback 个观测,
    取 as_of 前 ~90 个交易日窗口时, 窗口首行 pct_change 的截断伪影落在
    60 日求和窗之外。NaN (无映射/缺数据) 不入结果, 调用方按"不加分"处理
    (与回测 score_stocks 行业通道 nan→0 等价)。
    """
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=150)
    win = {}
    for s, df in all_data.items():
        try:
            d = pd.to_datetime(df["date"])
        except Exception:
            continue
        sub = df[d >= cutoff]
        if len(sub) >= 2:
            win[s] = sub
    if not win:
        return {}
    cal = sorted({d.date() for sub in win.values()
                  for d in pd.to_datetime(sub["date"])})
    if not cal:
        return {}
    panel = industry_momentum_panel(list(win), win, industry_map, cal,
                                    lookback=lookback)
    if panel.empty:
        return {}
    return panel.iloc[-1].dropna().to_dict()


def apply_industry_lambda(scores: dict, bcast: dict, lam: float) -> dict:
    """composite += λ × 截面z(行业动量广播值)。返回新 dict, 不改入参。

    与回测 score_stocks 行业通道数学一致 (run_walkforward_backtest.py
    L1477-1486): z 在打分域上取 nanmean/nanstd (ddof=0), 无值个股加 0,
    sd≤1e-9 / 有效值<2 时原样返回 (降级不阻断信号)。
    实盘 universe 是当日可评分股, 回测是 cross.index — 各自当日截面,
    机制一致即为 parity 的合理定义。
    """
    import math
    if lam <= 0 or not bcast or not scores:
        return dict(scores)
    vals = [bcast[s] for s in scores if s in bcast and bcast[s] == bcast[s]]
    if len(vals) < 2:
        return dict(scores)
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / len(vals)
    sd = math.sqrt(var)
    if sd <= 1e-9:
        return dict(scores)
    return {s: v + (lam * (bcast[s] - mu) / sd
                    if bcast.get(s) is not None and bcast[s] == bcast[s]
                    else 0.0)
            for s, v in scores.items()}
