"""
scripts/fetch_baostock_minute.py — 用 baostock 批量拉取全市场分钟线数据

数据源: baostock (免费, 无需注册, 历史深度 2022-至今)
  - 频率: 5分钟 / 15分钟
  - 列: date, time, code, open, high, low, close, volume, amount
  - 前复权

缓存: data_store/minute_15m/{symbol}.parquet 或 data_store/minute_5m/{symbol}.parquet

用法:
  py scripts/fetch_baostock_minute.py                          # 全市场 15分钟
  py scripts/fetch_baostock_minute.py --freq 5                 # 全市场 5分钟
  py scripts/fetch_baostock_minute.py --offset 0 --max 600     # 分片并行
  py scripts/fetch_baostock_minute.py --offset 600 --max 600
  py scripts/fetch_baostock_minute.py --offset 1200 --max 600
  py scripts/fetch_baostock_minute.py --offset 1800 --max 600
  py scripts/fetch_baostock_minute.py --offset 2400            # 剩余全部

预计耗时 (15分钟线, 2022-2026):
  单进程: ~11.5s/只 × 3000 ≈ 9.5 小时
  5进程并行: ~2 小时
"""

import os
import sys
import time
import argparse
from datetime import datetime

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

# 配置
START_DATE = "2022-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
CACHE_FIRST = True  # 跳过已有缓存的股票


def get_cache_dir(freq: str) -> str:
    return os.path.join(BASE_DIR, "data_store", f"minute_{freq}m")


def get_universe() -> list:
    """获取 A 股代码列表 — 用回测实际使用的宇宙 (data_store/)。"""
    store_dir = os.path.join(BASE_DIR, "data_store")
    if os.path.exists(store_dir):
        files = [f.replace(".parquet", "") for f in os.listdir(store_dir)
                 if f.endswith(".parquet") and not f.startswith("index_")
                 and "minute" not in f]
        if len(files) > 100:
            return sorted(files)
    raise RuntimeError("data_store/ 中未找到日线数据")


def to_baostock_symbol(code: str) -> str:
    """600519 -> sh.600519, 000001 -> sz.000001"""
    if code.startswith("6"):
        return f"sh.{code}"
    else:
        return f"sz.{code}"


def to_em_secid(code: str) -> str:
    """600519 -> 1.600519, 000001 -> 0.000001 (东财 secid)"""
    if code.startswith("6"):
        return f"1.{code}"
    else:
        return f"0.{code}"


# 东财历史行情镜像域名 (按 IP 限流时轮换)
EM_HOSTS = ["push2his.eastmoney.com", "92.push2his.eastmoney.com",
            "33.push2his.eastmoney.com", "11.push2his.eastmoney.com",
            "31.push2his.eastmoney.com", "63.push2his.eastmoney.com"]
_em_session = None


def _em_get(url: str, params: dict, host_idx: int = 0, retries: int = 4):
    """带重试 + 镜像轮换的东财请求; 返回 json 或 None."""
    global _em_session
    if _em_session is None:
        import requests
        _em_session = requests.Session()
        _em_session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "Chrome/125.0 Safari/537.36",
            "Referer": "https://quote.eastmoney.com/",
        })
    for attempt in range(retries):
        host = EM_HOSTS[(host_idx + attempt) % len(EM_HOSTS)]
        try:
            r = _em_session.get(f"https://{host}{url}", params=params, timeout=25)
            if r.status_code == 200:
                j = r.json()
                if (j.get("data") or {}).get("klines"):
                    return j
                # 空数据: 限流特征, 换镜像重试
        except Exception:
            pass
        time.sleep(2 + attempt * 2)
    return None


def fetch_one_em(code: str, freq: str, start: str, end: str) -> pd.DataFrame:
    """东财 5/15 分钟 K 线 (klt=5/15, fqt=1 前复权), 按年分请求.

    字段映射: 东财 klines 每行 "时间,开,收,高,低,量(手),额(元)"
    -> 统一列 datetime/day/open/high/low/close/volume(股)/amount
    """
    klt = "5" if freq == "5" else "15"
    chunks = []
    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cur <= end_ts:
        y = cur.year
        cend = min(cur + pd.offsets.YearEnd(0), end_ts)
        params = {"secid": to_em_secid(code), "fields1": "f1,f2,f3,f4,f5,f6",
                  "fields2": "f51,f52,f53,f54,f55,f56,f57",
                  "klt": klt, "fqt": "1",
                  "beg": cur.strftime("%Y%m%d"), "end": cend.strftime("%Y%m%d"),
                  "lmt": "100000"}
        j = _em_get("/api/qt/stock/kline/get", params)
        if j is None:
            return pd.DataFrame()
        kl = (j.get("data") or {}).get("klines") or []
        if kl:
            rows = []
            for line in kl:
                p = line.split(",")
                # p[0]=时间 p[1]=开 p[2]=收 p[3]=高 p[4]=低 p[5]=量(手) p[6]=额
                rows.append([p[0], p[1], p[3], p[4], p[2], p[5], p[6]])
            chunks.append(pd.DataFrame(rows, columns=[
                "datetime", "open", "high", "low", "close", "volume", "amount"]))
        cur = cend + pd.Timedelta(days=1)
        time.sleep(1.2)  # 节流, 防限流
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"].str[:19], errors="coerce")
    df["day"] = df["datetime"].dt.date.astype("datetime64[ns]")
    df["volume"] = df["volume"] * 100  # 东财单位=手 -> 对齐 baostock 单位=股
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    return df[["datetime", "day", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)


def setup_proxy(proxy: str):
    """让 baostock 的 TCP 直连走 SOCKS5 代理 (切换节点 = 换出口 IP)。

    baostock 用原生 socket 直连 public-api.baostock.com:10030,
    不走 HTTP 代理; 通过 PySocks 全局替换 socket.socket 使其走代理。
    用于黑名单 (10001011) 后换出口 IP。
    """
    import socks
    import socket as _socket
    if ":" in proxy:
        host, port = proxy.rsplit(":", 1)
        port = int(port)
    else:
        host, port = proxy, 7897
    socks.set_default_proxy(socks.SOCKS5, host, port, rdns=True)
    _socket.socket = socks.socksocket
    print(f"  [PROXY] baostock 走 SOCKS5 {host}:{port} (切换节点 = 换出口 IP)")


def fetch_one(bs, symbol: str, freq: str, start: str, end: str) -> pd.DataFrame:
    """拉取单只股票的分钟线数据。"""
    rs = bs.query_history_k_data_plus(
        symbol,
        "date,time,code,open,high,low,close,volume,amount",
        start_date=start,
        end_date=end,
        frequency=freq,
        adjustflag="2",  # 前复权
    )
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=rs.fields)

    # 类型转换
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 解析时间列: "20240102093500000" -> datetime
    df["datetime"] = pd.to_datetime(df["time"].str[:14], format="%Y%m%d%H%M%S")
    df["day"] = pd.to_datetime(df["date"])

    # 只保留有效行
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]

    return df[["datetime", "day", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Baostock 分钟线批量拉取")
    parser.add_argument("--freq", type=str, default="15", choices=["5", "15"],
                        help="K线频率: 5 或 15 (分钟)")
    parser.add_argument("--offset", type=int, default=0,
                        help="起始偏移 (用于并行分片)")
    parser.add_argument("--max", type=int, default=None,
                        help="最多拉取数量")
    parser.add_argument("--start", type=str, default=START_DATE,
                        help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=END_DATE,
                        help="结束日期 YYYY-MM-DD")
    parser.add_argument("--force", action="store_true",
                        help="强制重新拉取 (忽略缓存)")
    parser.add_argument("--proxy", type=str, default=None,
                        help="SOCKS5 代理 host:port (如 127.0.0.1:7897), 用于换出口 IP")
    parser.add_argument("--source", type=str, default="bs", choices=["bs", "em"],
                        help="数据源: bs=baostock (默认), em=东财 (走 --proxy 时生效)")
    args = parser.parse_args()

    if args.source == "em" and not args.proxy:
        print("--source em 需要 --proxy (东财直连被墙, 走 SOCKS5)")
        return

    if args.proxy:
        setup_proxy(args.proxy)

    freq = args.freq
    cache_dir = get_cache_dir(freq)
    os.makedirs(cache_dir, exist_ok=True)

    universe = get_universe()
    total = len(universe)

    # 分片
    symbols = universe[args.offset:]
    if args.max:
        symbols = symbols[:args.max]

    print(f"═══ Baostock {freq}分钟线拉取 ═══")
    print(f"  宇宙: {total} 只, 本次: {len(symbols)} 只 (offset={args.offset})")
    print(f"  日期: {args.start} ~ {args.end}")
    print(f"  缓存: {cache_dir}")
    print(f"  强制: {args.force}")
    print()

    bs = None
    if args.source != "em":
        import socket
        socket.setdefaulttimeout(120)  # 服务器挂起时单只最多等120s, 防 shard 卡死
        import baostock as bs
        lg = bs.login()
        if lg.error_code != "0":
            print(f"登录失败: {lg.error_msg}")
            return

    success = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    for i, code in enumerate(symbols):
        cache_path = os.path.join(cache_dir, f"{code}.parquet")

        # 缓存检查
        if not args.force and os.path.exists(cache_path):
            try:
                existing = pd.read_parquet(cache_path)
                if len(existing) > 100:  # 至少有几天数据
                    skipped += 1
                    continue
            except Exception:
                pass

        # 拉取
        try:
            if args.source == "em":
                df = fetch_one_em(code, freq, args.start, args.end)
            else:
                bs_sym = to_baostock_symbol(code)
                df = fetch_one(bs, bs_sym, freq, args.start, args.end)
            if len(df) > 0:
                df.to_parquet(cache_path, index=False)
                success += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  [ERROR] {code}: {e}")

        # 进度
        done = i + 1
        if done % 50 == 0 or done == len(symbols):
            elapsed = time.time() - t0
            speed = done / elapsed * 60
            eta = (len(symbols) - done) / max(speed / 60, 0.01) / 60
            print(f"  [{done}/{len(symbols)}] 成功={success} 跳过={skipped} "
                  f"失败={failed} | {speed:.0f}只/分 | ETA {eta:.0f}min")

    if bs is not None:
        bs.logout()

    elapsed = time.time() - t0
    print(f"\n═══ 完成 ═══")
    print(f"  耗时: {elapsed/60:.1f} 分钟")
    print(f"  成功: {success}, 跳过(缓存): {skipped}, 失败: {failed}")
    print(f"  缓存目录: {cache_dir}")

    # 统计
    files = [f for f in os.listdir(cache_dir) if f.endswith(".parquet")]
    total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in files)
    print(f"  总文件: {len(files)}, 总大小: {total_size/1024/1024:.0f}MB")


if __name__ == "__main__":
    main()
