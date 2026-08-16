"""
scripts/fetch_baostock_minute.py — 全市场分钟线批量拉取

数据源 (2026-08-16 起): 东财为主 (fetch_one_em, 经 eastmoney_proxy/cheapproxy
  网关转发, config.yaml eastmoney_proxy 段), baostock 保留为 legacy 选项
  (--source bs)。
  - 频率: 5分钟 / 15分钟
  - 列: datetime, day, open, high, low, close, volume(股), amount
  - 前复权 (东财 fqt=1 / baostock adjustflag=2), 两源 schema 统一可混存

缓存: data_store/minute_15m/{symbol}.parquet 或 data_store/minute_5m/{symbol}.parquet

用法:
  py scripts/fetch_baostock_minute.py                          # 全市场 15分钟
  py scripts/fetch_baostock_minute.py --freq 5                 # 全市场 5分钟
  py scripts/fetch_baostock_minute.py --offset 0 --max 600     # 分片并行
  py scripts/fetch_baostock_minute.py --offset 600 --max 600
  py scripts/fetch_baostock_minute.py --offset 1200 --max 600
  py scripts/fetch_baostock_minute.py --offset 1800 --max 600
  py scripts/fetch_baostock_minute.py --offset 2400            # 剩余全部
  py scripts/fetch_baostock_minute.py --since 5 --symbols 600519,000858
      # 增量: 只补最近 5 个交易日并合并去重 (daily_pipeline 盘后调用)

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
    """600519 -> sh.600519, 000001 -> sz.000001 (指数走 INDEX_SYMBOLS 映射)"""
    if code.startswith("6"):
        return f"sh.{code}"
    else:
        return f"sz.{code}"


def to_em_secid(code: str) -> str:
    """600519 -> 1.600519, 000001 -> 0.000001 (东财 secid; 指数走 INDEX_SYMBOLS)"""
    if code.startswith("6"):
        return f"1.{code}"
    else:
        return f"0.{code}"


# ── 指数分钟线支持 (路线A v20: 中证1000 5m 已实现波动率 → 风险层) ──
# 指数代码与股票代码冲突 (000001=平安银行 vs 上证指数), 必须显式映射。
INDEX_SYMBOLS = {
    "000852": {"bs": "sh.000852", "em": "1.000852", "name": "index_csi1000"},
    "000300": {"bs": "sh.000300", "em": "1.000300", "name": "index_csi300"},
    "000905": {"bs": "sh.000905", "em": "1.000905", "name": "index_csi500"},
}
INDEX_CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")


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


def fetch_one_em(code: str, freq: str, start: str, end: str,
                 secid: str | None = None, is_index: bool = False) -> pd.DataFrame:
    """东财 5/15 分钟 K 线 (klt=5/15, fqt=1 前复权), 按年分请求.

    字段映射: 东财 klines 每行 "时间,开,收,高,低,量(手),额(元)"
    -> 统一列 datetime/day/open/high/low/close/volume(股)/amount

    is_index=True: 用显式 secid (如 1.000852), fqt=0 不复权 (指数无复权概念)
    """
    klt = "5" if freq == "5" else "15"
    if secid is None:
        secid = to_em_secid(code)
    fqt = "0" if is_index else "1"
    chunks = []
    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cur <= end_ts:
        y = cur.year
        cend = min(cur + pd.offsets.YearEnd(0), end_ts)
        params = {"secid": secid, "fields1": "f1,f2,f3,f4,f5,f6",
                  "fields2": "f51,f52,f53,f54,f55,f56,f57",
                  "klt": klt, "fqt": fqt,
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


def fetch_one(bs, symbol: str, freq: str, start: str, end: str,
              adjustflag: str = "2") -> pd.DataFrame:
    """拉取单只股票/指数的分钟线数据。

    adjustflag: 2=前复权(股票), 3=不复权(指数, 指数无复权概念)
    """
    rs = bs.query_history_k_data_plus(
        symbol,
        "date,time,code,open,high,low,close,volume,amount",
        start_date=start,
        end_date=end,
        frequency=freq,
        adjustflag=adjustflag,
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


def merge_minute_incremental(existing, new) -> pd.DataFrame:
    """已有缓存 + 增量帧合并: 按 datetime 去重 (保留最新), 按时间排序。

    existing: 已落盘的分钟 parquet 内容 (统一列 datetime/day/open/...), 可为 None/空。
    new: 本次拉取的增量帧 (同 schema)。
    """
    if existing is None or len(existing) == 0:
        return new.reset_index(drop=True)
    df = pd.concat([existing, new], ignore_index=True)
    df = df.drop_duplicates(subset=["datetime"], keep="last")
    return df.sort_values("datetime").reset_index(drop=True)


def fetch_index(args, freq: str) -> int:
    """拉取单只指数分钟线 → data/cache/index_<name>_<freq>m.parquet.

    指数用 baostock sh.000852 (adjustflag=3 不复权) 主源;
    --source em 时走东财 secid (fqt=0) + SOCKS5 代理。
    """
    code = args.index
    if code not in INDEX_SYMBOLS:
        print(f"不支持的指数: {code}, 可选: {list(INDEX_SYMBOLS)}")
        return 1
    info = INDEX_SYMBOLS[code]
    name = info["name"]
    out_path = os.path.join(INDEX_CACHE_DIR, f"{name}_{freq}m.parquet")
    os.makedirs(INDEX_CACHE_DIR, exist_ok=True)

    if args.source == "em" and not args.proxy:
        print("--source em 需要 --proxy (东财直连被墙, 走 SOCKS5)")
        return 1
    if args.proxy:
        setup_proxy(args.proxy)

    # 东财源统一走 cheapproxy 网关 (config.yaml eastmoney_proxy 段)
    # install_patch 替换 requests.Session → _em_get 的懒加载 session 自动生效
    try:
        import eastmoney_proxy
        eastmoney_proxy.setup_from_config(BASE_DIR)
    except Exception as _e:
        print(f"  [WARN] 东财网关初始化失败: {_e}")

    print(f"═══ 指数 {freq}分钟线拉取: {code} ({info['bs']}) ═══")
    print(f"  日期: {args.start} ~ {args.end}")
    print(f"  输出: {out_path}")
    print()

    df = pd.DataFrame()
    if args.source != "em":
        import socket
        socket.setdefaulttimeout(120)
        import baostock as bs
        lg = bs.login()
        if lg.error_code != "0":
            print(f"登录失败: {lg.error_msg}")
            return 1
        df = fetch_one(bs, info["bs"], freq, args.start, args.end, adjustflag="3")
        bs.logout()
        if len(df) == 0:
            print("  baostock 返回空, 尝试东财源...")
    if len(df) == 0:
        df = fetch_one_em(code, freq, args.start, args.end,
                          secid=info["em"], is_index=True)
    if len(df) == 0:
        print("  ❌ 指数分钟线拉取失败 (两源均为空)")
        return 1

    df.to_parquet(out_path, index=False)
    days = pd.to_datetime(df["day"]).dt.date.unique()
    print(f"  ✅ 成功: {len(df)} 行, {len(days)} 个交易日, "
          f"{min(days)} ~ {max(days)}")
    print(f"  缓存: {out_path}")
    return 0


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
                        help="SOCKS5 代理 host:port (如 127.0.0.1:7897), "
                             "仅 baostock 源使用 (换出口 IP)")
    parser.add_argument("--source", type=str, default="em", choices=["bs", "em"],
                        help="数据源: em=东财 (默认, 走 eastmoney_proxy 网关), "
                             "bs=baostock (legacy)")
    parser.add_argument("--index", type=str, default=None,
                        help="拉取单只指数分钟线 (如 000852=中证1000), "
                             "保存到 data/cache/<name>_<freq>m.parquet")
    parser.add_argument("--since", type=int, default=None,
                        help="增量模式: 只补最近 N 个交易日 (与已有缓存合并去重; "
                             "无缓存股票回退拉最近 N*2 自然日)")
    parser.add_argument("--symbols", type=str, default=None,
                        help="指定股票列表 (逗号分隔, 如 600519,000858; "
                             "忽略 --offset/--max, 供管线盘后增量调用)")
    args = parser.parse_args()

    # ── 指数模式: 单只, 存 data/cache/index_*.parquet ──
    if args.index:
        return fetch_index(args, args.freq)

    if args.source == "em":
        # 东财源统一走 cheapproxy 网关 (config.yaml eastmoney_proxy 段);
        # install_patch 替换 requests.Session → _em_get 的懒加载 session 自动生效
        try:
            import eastmoney_proxy
            eastmoney_proxy.setup_from_config(BASE_DIR)
        except Exception as _e:
            print(f"  [WARN] 东财网关初始化失败: {_e}")

    if args.proxy:
        setup_proxy(args.proxy)

    freq = args.freq
    cache_dir = get_cache_dir(freq)
    os.makedirs(cache_dir, exist_ok=True)

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        total = len(symbols)
    else:
        universe = get_universe()
        total = len(universe)
        # 分片
        symbols = universe[args.offset:]
        if args.max:
            symbols = symbols[:args.max]

    mode = "增量" if args.since else "全量"
    print(f"═══ Baostock {freq}分钟线拉取 ({mode}) ═══")
    print(f"  宇宙: {total} 只, 本次: {len(symbols)} 只"
          f"{'' if args.symbols else f' (offset={args.offset})'}")
    print(f"  日期: {args.start} ~ {args.end}" + (f", 增量补 {args.since} 交易日" if args.since else ""))
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

        # 增量模式: 读已有缓存, 从本地最大日期往前缓冲 10 天续拉到今天
        existing_df = None
        start, end = args.start, args.end
        if args.since:
            if os.path.exists(cache_path):
                try:
                    existing_df = pd.read_parquet(cache_path)
                except Exception:
                    existing_df = None
            if existing_df is not None and len(existing_df) > 0 \
                    and "day" in existing_df.columns:
                max_day = pd.to_datetime(existing_df["day"]).max()
                start = (max_day - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            else:
                existing_df = None
                start = (pd.Timestamp.now()
                         - pd.Timedelta(days=args.since * 2)).strftime("%Y-%m-%d")
            end = pd.Timestamp.now().strftime("%Y-%m-%d")

        # 缓存检查 (全量模式: 已有足够数据跳过)
        if not args.force and not args.since and os.path.exists(cache_path):
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
                df = fetch_one_em(code, freq, start, end)
            else:
                bs_sym = to_baostock_symbol(code)
                df = fetch_one(bs, bs_sym, freq, start, end)
            if args.since:
                df = merge_minute_incremental(existing_df, df)
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
