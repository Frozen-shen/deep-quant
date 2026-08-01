"""
批量拉取事件源数据 → data/factor_cache/events/ (供 factors/event_signals.py 使用)

数据源 (akshare 1.18.64, 已实测):
  lockup : ak.stock_restricted_release_detail_em(start_date, end_date)
           单次调用返回区间内全市场解禁明细 (逐股票)。
  lhb    : ak.stock_lhb_detail_em(start_date, end_date)
           按"每个交易日一次调用"拉取龙虎榜明细 (支持断点续传)。
  insider: ak.stock_ggcg_em(symbol=YYYYMMDD)  (保留, 高管增减持)

输出:
  data/factor_cache/events/lockup.parquet   (全市场单文件)
      列: symbol, name, unlock_date, unlock_type, unlock_shares, actual_shares,
          actual_value, ratio_to_float, prev_close
  data/factor_cache/events/lhb.parquet      (全市场单文件)
      列: symbol, name, date, net_buy, buy_amount, sell_amount, lhb_turnover,
          market_turnover, turnover_rate, float_value, reason

用法:
  python scripts/fetch_events.py --source lockup          # 限售解禁 (未来120日)
  python scripts/fetch_events.py --source lhb             # 龙虎榜 (近60交易日)
  python scripts/fetch_events.py --source lhb --resume    # 断点续传
  python scripts/fetch_events.py --source all
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

EVENTS_DIR = os.path.join(BASE_DIR, "data", "factor_cache", "events")
LEGACY_DIR = os.path.join(BASE_DIR, "data", "event_cache")  # 旧缓存(insider)
os.makedirs(EVENTS_DIR, exist_ok=True)
os.makedirs(LEGACY_DIR, exist_ok=True)

LOCKUP_PATH = os.path.join(EVENTS_DIR, "lockup.parquet")
LHB_PATH = os.path.join(EVENTS_DIR, "lhb.parquet")

RATE_LIMIT = 0.5  # per-date 请求间隔 (秒)


def _fmt_eta(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"


# ════════════════════════════════════════
#  限售解禁 (lockup)
# ════════════════════════════════════════

_LOCKUP_COL_MAP = {
    "股票代码": "symbol", "股票简称": "name", "解禁时间": "unlock_date",
    "限售股类型": "unlock_type", "解禁数量": "unlock_shares",
    "实际解禁数量": "actual_shares", "实际解禁市值": "actual_value",
    "占解禁前流通市值比例": "ratio_to_float", "解禁前一交易日收盘价": "prev_close",
}


def fetch_lockup(days_ahead: int = 120, days_back: int = 30, resume: bool = True):
    """
    拉取限售解禁明细: 过去 days_back 日 ~ 未来 days_ahead 日。

    resume=True 时与已有 lockup.parquet 合并去重。
    """
    import akshare as ak
    import warnings
    warnings.filterwarnings("ignore")

    end = datetime.now() + timedelta(days=days_ahead)
    start = datetime.now() - timedelta(days=days_back)
    s_str, e_str = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

    print(f"[lockup] 拉取解禁明细 {s_str} ~ {e_str} ...", flush=True)
    t0 = time.time()
    try:
        raw = ak.stock_restricted_release_detail_em(start_date=s_str, end_date=e_str)
    except Exception as e:
        print(f"[lockup] 拉取失败: {e}", flush=True)
        return

    if raw is None or len(raw) == 0:
        print("[lockup] 区间内无解禁数据", flush=True)
        return

    df = raw.rename(columns={k: v for k, v in _LOCKUP_COL_MAP.items() if k in raw.columns})
    if "symbol" not in df.columns:
        print(f"[lockup] 列名不匹配: {list(raw.columns)}", flush=True)
        return

    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["unlock_date"] = pd.to_datetime(df["unlock_date"], errors="coerce")
    for c in ["unlock_shares", "actual_shares", "actual_value", "ratio_to_float", "prev_close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = ["symbol", "name", "unlock_date", "unlock_type", "unlock_shares",
            "actual_shares", "actual_value", "ratio_to_float", "prev_close"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].dropna(subset=["unlock_date"])

    # resume: 合并已有
    if resume and os.path.exists(LOCKUP_PATH):
        try:
            old = pd.read_parquet(LOCKUP_PATH)
            df = pd.concat([old, df], ignore_index=True)
            df = df.drop_duplicates(subset=["symbol", "unlock_date", "unlock_type"], keep="last")
        except Exception:
            pass

    df = df.sort_values("unlock_date").reset_index(drop=True)
    df.to_parquet(LOCKUP_PATH, index=False)
    print(f"[lockup] 完成: {len(df)} 条, {df['symbol'].nunique()} 只股票, "
          f"耗时 {time.time()-t0:.1f}s → {LOCKUP_PATH}", flush=True)


# ════════════════════════════════════════
#  龙虎榜 (lhb) — 每个交易日一次调用
# ════════════════════════════════════════

_LHB_COL_MAP = {
    "代码": "symbol", "名称": "name", "上榜日": "date",
    "龙虎榜净买额": "net_buy", "龙虎榜买入额": "buy_amount",
    "龙虎榜卖出额": "sell_amount", "龙虎榜成交额": "lhb_turnover",
    "市场总成交额": "market_turnover", "换手率": "turnover_rate",
    "流通市值": "float_value", "上榜原因": "reason",
}


def get_recent_trade_dates(n: int = 60) -> List[pd.Timestamp]:
    """获取最近 n 个交易日 (含今日)。优先 akshare 交易日历, 回退 data_store。"""
    import warnings
    warnings.filterwarnings("ignore")

    # 方式1: akshare 交易日历
    try:
        import akshare as ak
        cal = ak.tool_trade_date_hist_sina()
        col = cal.columns[0]
        dates = pd.to_datetime(cal[col])
        today = pd.Timestamp.now().normalize()
        dates = sorted([d for d in dates if d <= today])
        return dates[-n:]
    except Exception as e:
        print(f"[lhb] 交易日历获取失败 ({e}), 回退 data_store", flush=True)

    # 方式2: data_store 任一股的日期列
    ds = os.path.join(BASE_DIR, "data_store")
    if os.path.exists(ds):
        for f in os.listdir(ds):
            if f.endswith(".parquet") and f[0].isdigit():
                try:
                    ddf = pd.read_parquet(os.path.join(ds, f), columns=["date"])
                    dates = sorted(pd.to_datetime(ddf["date"]).unique())
                    today = pd.Timestamp.now().normalize()
                    dates = [d for d in dates if d <= today]
                    return dates[-n:]
                except Exception:
                    continue
    return []


def fetch_lhb(n_days: int = 60, resume: bool = True):
    """
    按交易日逐日拉取龙虎榜, 合并到 lhb.parquet。

    resume=True 时跳过已有日期 (按 date 列判断)。
    """
    import akshare as ak
    import warnings
    warnings.filterwarnings("ignore")

    trade_dates = get_recent_trade_dates(n_days)
    if not trade_dates:
        print("[lhb] 无法获取交易日历", flush=True)
        return

    # resume: 已覆盖的日期
    covered = set()
    old = None
    if resume and os.path.exists(LHB_PATH):
        try:
            old = pd.read_parquet(LHB_PATH)
            old["date"] = pd.to_datetime(old["date"], errors="coerce")
            covered = set(pd.to_datetime(old["date"]).dt.normalize().unique())
        except Exception:
            old = None

    todo = [d for d in trade_dates if pd.Timestamp(d).normalize() not in covered]
    print(f"[lhb] 交易日 {len(trade_dates)} 天, 待拉取 {len(todo)} 天 "
          f"(已覆盖 {len(covered)})", flush=True)

    frames = []
    done = empty = failed = 0
    t0 = time.time()

    for i, d in enumerate(todo):
        ds_str = d.strftime("%Y%m%d")
        try:
            raw = ak.stock_lhb_detail_em(start_date=ds_str, end_date=ds_str)
            if raw is not None and len(raw) > 0:
                df = raw.rename(columns={k: v for k, v in _LHB_COL_MAP.items() if k in raw.columns})
                if "symbol" in df.columns:
                    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
                    df["date"] = pd.Timestamp(d).normalize()
                    for c in ["net_buy", "buy_amount", "sell_amount",
                              "lhb_turnover", "market_turnover", "turnover_rate", "float_value"]:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors="coerce")
                    keep = ["symbol", "name", "date", "net_buy", "buy_amount", "sell_amount",
                            "lhb_turnover", "market_turnover", "turnover_rate", "float_value", "reason"]
                    keep = [c for c in keep if c in df.columns]
                    frames.append(df[keep])
                    done += 1
                else:
                    empty += 1
            else:
                empty += 1  # 非交易日或无上榜
        except Exception:
            failed += 1

        time.sleep(RATE_LIMIT)

        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(todo) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(todo)}] 有数据={done} 空={empty} 失败={failed} "
                  f"| 已用 {_fmt_eta(elapsed)} ETA {_fmt_eta(eta)}", flush=True)

    # 合并
    new_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if old is not None and len(old) > 0:
        if len(new_df) > 0:
            all_df = pd.concat([old, new_df], ignore_index=True)
        else:
            all_df = old
    else:
        all_df = new_df

    if len(all_df) > 0:
        all_df = all_df.drop_duplicates(subset=["symbol", "date", "reason"], keep="last")
        all_df = all_df.sort_values("date").reset_index(drop=True)
        all_df.to_parquet(LHB_PATH, index=False)
        print(f"[lhb] 完成: {len(all_df)} 条, {all_df['symbol'].nunique()} 只股票, "
              f"耗时 {time.time()-t0:.1f}s → {LHB_PATH}", flush=True)
    else:
        print("[lhb] 无数据写入", flush=True)


# ════════════════════════════════════════
#  高管增减持 (insider, 保留旧功能)
# ════════════════════════════════════════

def fetch_insider_trades():
    """拉取高管/股东增减持数据 (旧功能, 存到 data/event_cache)。"""
    import akshare as ak
    import warnings
    warnings.filterwarnings("ignore")

    print("[insider] 拉取高管增减持...", flush=True)
    all_data = []
    for year in [2022, 2023, 2024, 2025, 2026]:
        for quarter in ["0331", "0630", "0930", "1231"]:
            date_str = f"{year}{quarter}"
            cache_path = os.path.join(LEGACY_DIR, f"insider_{date_str}.parquet")
            if os.path.exists(cache_path):
                all_data.append(pd.read_parquet(cache_path))
                print(f"  {date_str}: 缓存", flush=True)
                continue
            try:
                df = ak.stock_ggcg_em(symbol=date_str)
                if df is not None and len(df) > 0:
                    df.to_parquet(cache_path, index=False)
                    all_data.append(df)
                    print(f"  {date_str}: {len(df)}条", flush=True)
            except Exception as e:
                print(f"  {date_str}: {e}", flush=True)

    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        out = os.path.join(LEGACY_DIR, "insider_all.parquet")
        df.to_parquet(out, index=False)
        print(f"[insider] 总计 {len(df)}条 → {out}", flush=True)


# ════════════════════════════════════════
#  主入口
# ════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="拉取事件源数据 (lockup/lhb/insider)")
    parser.add_argument("--source", choices=["lockup", "lhb", "insider", "all"],
                        default="all", help="数据源")
    parser.add_argument("--resume", action="store_true", help="断点续传 (lhb/lockup)")
    parser.add_argument("--days", type=int, default=60, help="lhb 回溯交易日数")
    parser.add_argument("--days-ahead", type=int, default=120, help="lockup 前瞻天数")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print(f"  事件数据拉取 (source={args.source}, resume={args.resume})", flush=True)
    print("=" * 60, flush=True)

    if args.source in ("lockup", "all"):
        fetch_lockup(days_ahead=args.days_ahead, resume=args.resume)
    if args.source in ("lhb", "all"):
        fetch_lhb(n_days=args.days, resume=args.resume)
    if args.source in ("insider", "all"):
        fetch_insider_trades()
