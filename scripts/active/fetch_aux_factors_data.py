"""
scripts/active/fetch_aux_factors_data.py — 辅助因子数据拉取 (2026-08-07)

补齐此前因数据不全而闲置的因子维度数据:
  - margin   两融明细 (深交所+上交所, 2015至今, 交易所直连)
  - lhb      龙虎榜 (东财, 按日全市场)
  - dzjy     大宗交易 (东财, 按日全市场)
  - restricted 限售解禁 (东财, 按股)
  - industry 东财行业分类 (一次性全市场) → 行业中性化必需
  - gdhs     股东户数 (东财, 按股)
  - analyst  分析师盈利预测 (东财, 按股)

缓存: data_store/aux_{source}/{code或日期}.parquet

用法:
  py scripts/active/fetch_aux_factors_data.py --source industry            # 行业分类(全市场一次性)
  py scripts/active/fetch_aux_factors_data.py --source margin --start 2015-01-01
  py scripts/active/fetch_aux_factors_data.py --source lhb --start 2015-01-01
  py scripts/active/fetch_aux_factors_data.py --source dzjy --start 2015-01-01
  py scripts/active/fetch_aux_factors_data.py --source restricted --offset 0 --max 600   # 按股分片
  py scripts/active/fetch_aux_factors_data.py --source gdhs --offset 0 --max 600
  py scripts/active/fetch_aux_factors_data.py --source analyst --offset 0 --max 600
  # 东财源被限流时加 --proxy 127.0.0.1:7897 (socks5h)
"""

import os
import sys
import time
import argparse
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

DEFAULT_START = "2015-01-01"
DEFAULT_END = datetime.now().strftime("%Y-%m-%d")


def get_cache_dir(source: str) -> str:
    return os.path.join(BASE_DIR, "data_store", f"aux_{source}")


def get_universe() -> list:
    store_dir = os.path.join(BASE_DIR, "data_store")
    files = [f.replace(".parquet", "") for f in os.listdir(store_dir)
             if f.endswith(".parquet") and not f.startswith("index_") and "minute" not in f]
    if len(files) > 100:
        return sorted(files)
    raise RuntimeError("data_store/ 中未找到日线数据")


def setup_proxy(proxy: str):
    """东财接口被限流时走 SOCKS5 (切换节点=换出口IP)。"""
    import socks
    import socket as _socket
    if ":" in proxy:
        host, port = proxy.rsplit(":", 1)
        port = int(port)
    else:
        host, port = proxy, 7897
    socks.set_default_proxy(socks.SOCKS5, host, port, rdns=True)
    _socket.socket = socks.socksocket
    print(f"  [PROXY] 走 SOCKS5 {host}:{port}")


# ── 两融 (交易所直连) ──────────────────────────────────────────────
def fetch_margin(date: str) -> pd.DataFrame:
    """某日全市场两融明细 (深交所 + 上交所合并)。"""
    import akshare as ak
    frames = []
    for fn, market in [(ak.stock_margin_detail_szse, "sz"), (ak.stock_margin_detail_sse, "sh")]:
        try:
            df = fn(date=date.replace("-", ""))
            if df is None or len(df) == 0:
                continue
            df["market"] = market
            df["date"] = date
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── 龙虎榜 (东财, 按日) ────────────────────────────────────────────
def fetch_lhb(date: str) -> pd.DataFrame:
    import akshare as ak
    try:
        df = ak.stock_lhb_detail_em(start_date=date.replace("-", ""),
                                    end_date=date.replace("-", ""))
        if df is not None and len(df) > 0:
            df["date"] = date
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ── 大宗交易 (东财, 按日) ───────────────────────────────────────────
def fetch_dzjy(date: str) -> pd.DataFrame:
    import akshare as ak
    try:
        df = ak.stock_dzjy_mrmx(symbol="A股", start_date=date.replace("-", ""),
                                end_date=date.replace("-", ""))
        if df is not None and len(df) > 0:
            df["date"] = date
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ── 限售解禁 (东财, 按股) ───────────────────────────────────────────
def fetch_restricted(code: str) -> pd.DataFrame:
    import akshare as ak
    try:
        df = ak.stock_restricted_release_queue_em(symbol=code)
        if df is not None and len(df) > 0:
            df["code"] = code
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ── 行业分类 (东财, 一次性全市场) ───────────────────────────────────
def fetch_industry() -> pd.DataFrame:
    import akshare as ak
    df = ak.stock_board_industry_name_em()  # 板块列表
    if df is None or len(df) == 0:
        return pd.DataFrame()
    # 逐板块取成分股
    rows = []
    for _, r in df.iterrows():
        try:
            cons = ak.stock_board_industry_cons_em(symbol=r["板块名称"])
            if cons is not None and len(cons) > 0:
                for _, c in cons.iterrows():
                    rows.append({"industry": r["板块名称"], "code": c["代码"]})
        except Exception:
            continue
        time.sleep(0.3)
    return pd.DataFrame(rows)


# ── 股东户数 (东财, 按股) ───────────────────────────────────────────
def fetch_gdhs(code: str) -> pd.DataFrame:
    import akshare as ak
    sym = code
    try:
        df = ak.stock_zh_a_gdhs_detail_em(symbol=sym)
        if df is not None and len(df) > 0:
            df["code"] = code
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ── 分析师盈利预测 (东财, 按股) ─────────────────────────────────────
def fetch_analyst(code: str) -> pd.DataFrame:
    import akshare as ak
    try:
        df = ak.stock_profit_forecast_em(symbol=code)
        if df is not None and len(df) > 0:
            df["code"] = code
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="辅助因子数据拉取")
    parser.add_argument("--source", required=True,
                        choices=["margin", "lhb", "dzjy", "restricted", "industry", "gdhs", "analyst"])
    parser.add_argument("--start", type=str, default=DEFAULT_START)
    parser.add_argument("--end", type=str, default=DEFAULT_END)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--proxy", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.proxy:
        setup_proxy(args.proxy)

    cache_dir = get_cache_dir(args.source)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"═══ 拉取 {args.source} ═══")
    print(f"  缓存: {cache_dir}")

    # ── industry: 一次性全市场 ──
    if args.source == "industry":
        out = os.path.join(cache_dir, "industry_map.parquet")
        if os.path.exists(out) and not args.force:
            print("  已存在, 跳过 (--force 重拉)")
            return
        df = fetch_industry()
        print(f"  板块数: {df['industry'].nunique() if len(df) else 0}, 成分股: {len(df)}")
        if len(df) > 1000:
            df.to_parquet(out, index=False)
            print(f"  ✅ 已保存 {out}")
        else:
            print("  ⚠️ 数据不足, 未保存")
        return

    # ── 按日拉取的源 ──
    if args.source in ("margin", "lhb", "dzjy"):
        dates = pd.bdate_range(args.start, args.end)
        done = 0
        for d in dates:
            ds = d.strftime("%Y-%m-%d")
            out = os.path.join(cache_dir, f"{ds}.parquet")
            if os.path.exists(out) and not args.force:
                done += 1
                continue
            if args.source == "margin":
                df = fetch_margin(ds)
            elif args.source == "lhb":
                df = fetch_lhb(ds)
            else:
                df = fetch_dzjy(ds)
            if len(df) > 0:
                df.to_parquet(out, index=False)
            done += 1
            if done % 50 == 0:
                print(f"  [{done}/{len(dates)}] {ds}")
            time.sleep(0.5)  # 节流
        print(f"  完成 {done}/{len(dates)} 天")
        return

    # ── 按股拉取的源 ──
    universe = get_universe()
    symbols = universe[args.offset:]
    if args.max:
        symbols = symbols[:args.max]
    fn = {"restricted": fetch_restricted, "gdhs": fetch_gdhs, "analyst": fetch_analyst}[args.source]
    ok = 0
    for i, code in enumerate(symbols):
        out = os.path.join(cache_dir, f"{code}.parquet")
        if os.path.exists(out) and not args.force:
            ok += 1
            continue
        try:
            df = fn(code)
            if len(df) > 0:
                df.to_parquet(out, index=False)
                ok += 1
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(symbols)}] 成功={ok}")
        time.sleep(0.5)
    print(f"  完成: 成功 {ok}/{len(symbols)}")


if __name__ == "__main__":
    main()
