"""
全量 A 股日线数据拉取脚本 — 目标 3000+ 只有效股票

数据源:
  股票列表: akshare stock_info_a_code_name()
  日线行情: akshare stock_zh_a_hist (eastmoney), 备用腾讯 API

过滤规则:
  - 剔除 ST / *ST 股票
  - 剔除上市不足 60 个交易日的股票
  - 剔除近 20 日日均成交额 < 500 万的股票

存储:
  前复权: data_store/{symbol}.parquet
  不复权: data_store/unadjusted/{symbol}.parquet  (用于涨跌停判断)
  元数据: data_store/_meta.json

用法:
  python scripts/fetch_full_universe.py                    # 全量拉取 (2+ 小时)
  python scripts/fetch_full_universe.py --resume           # 断点续传, 跳过已有文件
  python scripts/fetch_full_universe.py --check-only       # 仅统计, 不拉取
  python scripts/fetch_full_universe.py --limit 10         # 测试模式, 只拉 10 只
  python scripts/fetch_full_universe.py --force            # 强制重新拉取所有
  python scripts/fetch_full_universe.py --proxy http://127.0.0.1:7897
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

# ── 路径 ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/active/ → project root
DATA_STORE = os.path.join(BASE_DIR, "data_store")
UNADJ_DIR = os.path.join(DATA_STORE, "unadjusted")
META_FILE = os.path.join(DATA_STORE, "_meta.json")

# ── 参数 ─────────────────────────────────────────────────────────────
DEFAULT_START = "20180101"
DEFAULT_END = "20260731"
MIN_LIST_DAYS = 60          # 最少上市交易日
MIN_AVG_TURNOVER = 5_000_000  # 20日均成交额下限 (元)
RATE_LIMIT_SLEEP = 0.35     # 请求间隔 (~3 req/s)
PROGRESS_INTERVAL = 50      # 每 N 只打印进度

# ── 列名映射: akshare 中文 -> 英文 ───────────────────────────────────
COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover",
}

KEEP_COLS = ["date", "open", "high", "low", "close", "volume",
             "amount", "amplitude", "pct_change", "change", "turnover"]


def log(msg: str):
    """统一日志输出"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  代理处理
# ═══════════════════════════════════════════════════════════════════════

def setup_proxy(proxy_url: Optional[str] = None):
    """
    配置 HTTP 代理。

    Windows 下 requests 会读取系统代理 (如 Clash), 可能干扰 eastmoney 请求。
    此函数统一处理: 指定代理或绕过系统代理。
    """
    import requests
    import requests.sessions

    if proxy_url:
        _orig_merge = requests.Session.merge_environment_settings

        def _merge_with_proxy(self, url, proxies, stream, verify, cert):
            settings = _orig_merge(self, url, proxies, stream, verify, cert)
            settings["proxies"] = {"http": proxy_url, "https": proxy_url}
            return settings

        requests.Session.merge_environment_settings = _merge_with_proxy
        log(f"使用代理: {proxy_url}")
    else:
        # 绕过系统代理 (直连)
        _orig_merge = requests.Session.merge_environment_settings

        def _merge_no_proxy(self, url, proxies, stream, verify, cert):
            settings = _orig_merge(self, url, proxies, stream, verify, cert)
            settings["proxies"] = {}
            return settings

        requests.Session.merge_environment_settings = _merge_no_proxy

        _orig_get = requests.get

        def _get_no_proxy(url, **kwargs):
            with requests.Session() as s:
                s.trust_env = False
                return s.get(url, **kwargs)

        requests.get = _get_no_proxy
        log("已绕过系统代理 (直连)")


def _make_session():
    """创建不经过系统代理的 requests session"""
    import requests
    s = requests.Session()
    s.trust_env = False
    s.headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    return s


# ═══════════════════════════════════════════════════════════════════════
#  股票列表获取
# ═══════════════════════════════════════════════════════════════════════

def _get_stock_list_from_local() -> pd.DataFrame:
    """
    离线回退: 从本地 data_cache/ + data_store/ 目录扫描股票代码。
    当所有在线API不可用时使用。
    """
    from pathlib import Path

    codes = set()

    # 扫描 data_cache/ (1550只, 双交易所)
    dc_dir = Path(BASE_DIR) / "data_cache"
    if dc_dir.exists():
        for f in dc_dir.glob("*.parquet"):
            if len(f.stem) == 6 and f.stem.isdigit():
                codes.add(f.stem)

    # 扫描 data_store/ (2776只, 偏深交所)
    ds_dir = Path(DATA_STORE)
    if ds_dir.exists():
        for f in ds_dir.glob("*.parquet"):
            if len(f.stem) == 6 and f.stem.isdigit():
                codes.add(f.stem)

    if not codes:
        raise RuntimeError("本地无缓存数据, 无法获取股票列表")

    log(f"  本地缓存扫描: {len(codes)} 只 (data_cache + data_store 合并)")
    df = pd.DataFrame({"code": sorted(codes), "name": ""})
    return df

def get_stock_list() -> pd.DataFrame:
    """
    获取全部 A 股代码和名称。

    主数据源: ak.stock_info_a_code_name()
    备用1: ak.stock_zh_a_spot_em() (eastmoney 实时行情)
    备用2: 本地 data_cache/ + data_store/ 目录扫描 (离线模式)

    过滤:
      - 仅保留沪深主板/创业板/科创板 (6/0/3 开头)
      - 剔除 ST / *ST
    """
    import akshare as ak
    from pathlib import Path

    # 主数据源
    try:
        log("获取股票列表: ak.stock_info_a_code_name() ...")
        df = ak.stock_info_a_code_name()
        df = df.rename(columns={"code": "code", "name": "name"})
        # 确保列名统一
        if "code" not in df.columns:
            df.columns = ["code", "name"]
        source = "stock_info_a_code_name"
    except Exception as e:
        log(f"  stock_info_a_code_name 失败: {e}, 尝试 eastmoney ...")
        try:
            df = ak.stock_zh_a_spot_em()
            df = df[["代码", "名称"]].copy()
            df.columns = ["code", "name"]
            source = "stock_zh_a_spot_em"
        except Exception as e2:
            log(f"  eastmoney 也失败: {e2}, 使用本地缓存列表 ...")
            df = _get_stock_list_from_local()
            source = "local_cache_scan"

    total = len(df)
    log(f"  原始股票数 ({source}): {total}")

    # 标准化代码
    df["code"] = df["code"].astype(str).str.zfill(6)

    # 仅保留沪深 (6=沪, 0/3=深), 剔除北交所 (8/4开头)
    mask = df["code"].str.startswith(("6", "0", "3"))
    df = df[mask].copy()
    log(f"  沪深过滤后: {len(df)}")

    # 剔除 ST
    name = df["name"].astype(str)
    st_mask = name.str.contains("ST", case=False, na=False)
    n_st = st_mask.sum()
    df = df[~st_mask].copy()
    log(f"  剔除 ST ({n_st} 只) 后: {len(df)}")

    df = df.reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════
#  历史数据获取
# ═══════════════════════════════════════════════════════════════════════

def _fetch_hist_akshare(code: str, start_date: str, end_date: str,
                        adjust: str = "qfq") -> pd.DataFrame:
    """通过 akshare stock_zh_a_hist (eastmoney) 获取日线"""
    import akshare as ak
    df = ak.stock_zh_a_hist(
        symbol=code, period="daily",
        start_date=start_date, end_date=end_date, adjust=adjust,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns=COLUMN_MAP)
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _fetch_hist_tencent(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    备用数据源: 腾讯财经 API。

    单次最多返回 ~640 条, 需分段拉取。
    注意: 腾讯 API 不提供 amount/turnover, 这些列填 NaN。
    """
    from datetime import datetime as _dt, timedelta as _td

    prefix = "sh" if code.startswith("6") else "sz"
    symbol = f"{prefix}{code}"

    sd = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    ed = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

    session = _make_session()
    all_klines = []
    current_end = ed
    max_chunks = 10

    for _ in range(max_chunks):
        url = (
            f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
            f"?param={symbol},day,{sd},{current_end},2000,qfq"
        )
        resp = session.get(url, timeout=15)
        data = resp.json()

        payload = data.get("data")
        if not isinstance(payload, dict):
            break

        stock_data = payload.get(symbol)
        if not isinstance(stock_data, dict):
            break

        klines = stock_data.get("qfqday") or stock_data.get("day", [])
        if not klines:
            break

        all_klines = klines + all_klines

        first_date = _dt.strptime(klines[0][0], "%Y-%m-%d") - _td(days=1)
        current_end = first_date.strftime("%Y-%m-%d")

        if current_end < sd:
            break

    if not all_klines:
        return pd.DataFrame()

    # 腾讯格式: [date, open, close, high, low, volume]
    rows = []
    for k in all_klines:
        rows.append({
            "date": k[0],
            "open": float(k[1]),
            "close": float(k[2]),
            "high": float(k[3]),
            "low": float(k[4]),
            "volume": float(k[5]),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)

    # 计算衍生字段
    df["pct_change"] = df["close"].pct_change() * 100
    df["change"] = df["close"].diff()
    prev_close = df["close"].shift(1)
    df["amplitude"] = (df["high"] - df["low"]) / prev_close * 100

    # 腾讯不提供 amount/turnover
    df["amount"] = np.nan
    df["turnover"] = np.nan

    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].copy()
    return df


def fetch_stock(code: str, start_date: str, end_date: str,
                adjust: str = "qfq",
                max_retries: int = 3,
                use_tencent_fallback: bool = False) -> Optional[pd.DataFrame]:
    """
    获取单只股票日线数据, 带重试和备用数据源。

    返回 DataFrame (英文列名), 失败返回 None。
    """
    # 如果已知 eastmoney 不可用, 直接走腾讯
    if not use_tencent_fallback:
        last_err = None
        for attempt in range(max_retries):
            try:
                df = _fetch_hist_akshare(code, start_date, end_date, adjust)
                if df is not None and not df.empty:
                    return df
                if df is not None and df.empty:
                    return None  # 空数据不重试
            except Exception as e:
                last_err = e
                wait = 2 ** (attempt + 1)
                if attempt < max_retries - 1:
                    time.sleep(wait)

        # akshare 全部失败, 尝试腾讯 (仅前复权支持)
        if adjust == "qfq":
            try:
                df = _fetch_hist_tencent(code, start_date, end_date)
                if df is not None and not df.empty:
                    return df
            except Exception:
                pass

    return None


# ═══════════════════════════════════════════════════════════════════════
#  过滤: 上市天数 + 成交额
# ═══════════════════════════════════════════════════════════════════════

def check_listing_days(df: pd.DataFrame, min_days: int = MIN_LIST_DAYS) -> bool:
    """检查上市交易日是否 >= min_days"""
    return len(df) >= min_days


def check_avg_turnover(df: pd.DataFrame,
                       window: int = 20,
                       min_amount: float = MIN_AVG_TURNOVER) -> bool:
    """
    检查近 window 日日均成交额是否 >= min_amount。

    如果 amount 列全为 NaN (腾讯数据源), 则通过 (无法判断时放行)。
    """
    if "amount" not in df.columns:
        return True
    recent = df.tail(window)
    avg_amount = recent["amount"].mean()
    if pd.isna(avg_amount):
        return True  # 无数据时放行
    return avg_amount >= min_amount


# ═══════════════════════════════════════════════════════════════════════
#  数据验证
# ═══════════════════════════════════════════════════════════════════════

def validate_stock(df: pd.DataFrame, source: str = "akshare") -> Tuple[bool, str]:
    """
    验证数据质量。

    检查项:
      - 至少 100 行
      - 无全 NaN 列 (腾讯源的 amount/turnover 除外)
      - 日期单调递增
      - 价格为正
    """
    if len(df) < 100:
        return False, f"仅 {len(df)} 行 (需 >= 100)"

    skip_cols = {"amount", "turnover"} if source == "tencent" else set()
    all_nan = [c for c in df.columns[df.isna().all()].tolist()
               if c not in skip_cols]
    if all_nan:
        return False, f"全 NaN 列: {all_nan}"

    if not df["date"].is_monotonic_increasing:
        return False, "日期非单调递增"

    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            n_bad = int((df[col] <= 0).sum())
            if n_bad > 0:
                return False, f"{col} 有 {n_bad} 个非正值"

    return True, "ok"


# ═══════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════

def run(start_date: str = DEFAULT_START,
        end_date: str = DEFAULT_END,
        force: bool = False,
        resume: bool = False,
        check_only: bool = False,
        limit: Optional[int] = None):
    """
    主入口: 拉取全量 A 股日线数据。

    参数:
      start_date: 起始日期 YYYYMMDD
      end_date:   结束日期 YYYYMMDD
      force:      强制重新拉取 (忽略已有文件)
      resume:     断点续传 (跳过已有 parquet 文件)
      check_only: 仅统计, 不实际拉取
      limit:      限制拉取数量 (测试用)
    """
    os.makedirs(DATA_STORE, exist_ok=True)
    os.makedirs(UNADJ_DIR, exist_ok=True)

    # ── 获取股票列表 ──
    stocks = get_stock_list()
    codes = stocks["code"].tolist()

    if limit:
        codes = codes[:limit]
        log(f"测试模式: 限制为前 {limit} 只")

    total = len(codes)

    # ── check-only 模式 ──
    if check_only:
        existing = 0
        to_fetch = 0
        for code in codes:
            path = os.path.join(DATA_STORE, f"{code}.parquet")
            if os.path.exists(path):
                existing += 1
            else:
                to_fetch += 1
        log("=" * 60)
        log(f"[CHECK-ONLY] 股票池: {total} 只")
        log(f"  已有缓存: {existing} 只")
        log(f"  待拉取:   {to_fetch} 只")
        log(f"  预计耗时: {to_fetch * RATE_LIMIT_SLEEP / 60:.0f} 分钟 "
            f"(不含重试)")
        log("=" * 60)
        return None

    # ── 探测 eastmoney 可用性 ──
    use_tencent = False
    try:
        log("探测 eastmoney API ...")
        _fetch_hist_akshare("000001", "20240101", "20240105")
        log("  eastmoney: OK")
    except Exception:
        log("  eastmoney: 不可用, 将使用腾讯备用源")
        use_tencent = True

    # ── 确定跳过逻辑 ──
    # resume 或 默认(非force): 跳过已有文件
    skip_existing = resume or (not force)

    log(f"开始拉取: {total} 只, {start_date} -> {end_date}")
    log(f"  跳过已有: {skip_existing}, 数据源: "
        f"{'腾讯(备用)' if use_tencent else 'akshare/eastmoney'}")

    # ── 统计 ──
    done = 0
    skipped = 0
    filtered_days = 0       # 上市天数不足被过滤
    filtered_turnover = 0   # 成交额不足被过滤
    failed: List[str] = []
    invalid: List[str] = []
    valid_symbols: List[str] = []
    t0 = time.time()

    for i, code in enumerate(codes, 1):
        out_path = os.path.join(DATA_STORE, f"{code}.parquet")
        unadj_path = os.path.join(UNADJ_DIR, f"{code}.parquet")

        # 跳过已有
        if skip_existing and os.path.exists(out_path):
            skipped += 1
            valid_symbols.append(code)
            if i % PROGRESS_INTERVAL == 0 or i == total:
                _print_progress(i, total, code, "跳过(已缓存)",
                                done, skipped, t0)
            continue

        # 限速
        if done > 0:
            time.sleep(RATE_LIMIT_SLEEP)

        # 拉取前复权数据
        df = fetch_stock(code, start_date, end_date, adjust="qfq",
                         use_tencent_fallback=use_tencent)

        if df is None or df.empty:
            failed.append(code)
            if i % PROGRESS_INTERVAL == 0 or i == total:
                _print_progress(i, total, code, "失败", done, skipped, t0)
            continue

        # 过滤: 上市天数
        if not check_listing_days(df):
            filtered_days += 1
            continue

        # 过滤: 成交额
        if not check_avg_turnover(df):
            filtered_turnover += 1
            continue

        # 验证数据质量
        source_label = "tencent" if use_tencent else "akshare"
        is_valid, reason = validate_stock(df, source=source_label)
        if not is_valid:
            invalid.append(code)
            continue

        # 保存前复权
        df.to_parquet(out_path, index=False, engine="pyarrow")
        valid_symbols.append(code)
        done += 1

        # 拉取并保存不复权数据 (用于涨跌停判断)
        if not use_tencent:
            time.sleep(RATE_LIMIT_SLEEP)
            df_unadj = fetch_stock(code, start_date, end_date, adjust="",
                                   use_tencent_fallback=False)
            if df_unadj is not None and not df_unadj.empty:
                df_unadj.to_parquet(unadj_path, index=False, engine="pyarrow")

        # 进度
        if i % PROGRESS_INTERVAL == 0 or i == total:
            _print_progress(i, total, code, "完成", done, skipped, t0)

    # ── 汇总 ──
    elapsed = time.time() - t0
    log("=" * 60)
    log(f"拉取完成! 耗时 {elapsed:.0f}s ({elapsed/60:.1f} 分钟)")
    log(f"  成功:     {done}")
    log(f"  跳过:     {skipped} (已有缓存)")
    log(f"  过滤-天数: {filtered_days} (上市 < {MIN_LIST_DAYS} 日)")
    log(f"  过滤-额:  {filtered_turnover} (日均成交额 < {MIN_AVG_TURNOVER/1e4:.0f}万)")
    log(f"  失败:     {len(failed)}")
    log(f"  无效:     {len(invalid)}")
    log(f"  有效股票: {len(valid_symbols)} 只")

    if failed:
        log(f"  失败列表: {failed[:20]}{'...' if len(failed) > 20 else ''}")
    if invalid:
        log(f"  无效列表: {invalid[:20]}{'...' if len(invalid) > 20 else ''}")

    # ── 写入元数据 ──
    meta = {
        "symbols": sorted(valid_symbols),
        "count": len(valid_symbols),
        "date_range": [
            f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}",
            f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}",
        ],
        "last_update": datetime.now().isoformat(timespec="seconds"),
        "filters": {
            "min_list_days": MIN_LIST_DAYS,
            "min_avg_turnover": MIN_AVG_TURNOVER,
            "st_removed": True,
        },
        "stats": {
            "total_candidates": total,
            "fetched": done,
            "skipped": skipped,
            "filtered_days": filtered_days,
            "filtered_turnover": filtered_turnover,
            "failed": failed,
            "invalid": invalid,
        },
        "source": ("tencent_fallback" if use_tencent
                   else "akshare stock_zh_a_hist"),
        "adjust": "qfq",
    }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"元数据已写入: {META_FILE}")

    # 保存失败列表 (方便排查)
    if failed:
        fail_path = os.path.join(DATA_STORE, "_failures.json")
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump({"failed": failed, "invalid": invalid,
                       "timestamp": datetime.now().isoformat()},
                      f, ensure_ascii=False, indent=2)
        log(f"失败列表已写入: {fail_path}")

    return meta


def _print_progress(i: int, total: int, code: str, status: str,
                    done: int, skipped: int, t0: float):
    """打印进度信息"""
    elapsed = time.time() - t0
    pct = i / total * 100
    rate = (done + skipped) / elapsed if elapsed > 0 else 0
    eta = (total - i) / rate if rate > 0 else 0
    log(f"[{i}/{total}] {code} {status} | "
        f"进度 {pct:.1f}% | 有效 {done} | "
        f"耗时 {elapsed:.0f}s | 剩余 ~{eta:.0f}s")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="全量 A 股日线数据拉取 (目标 3000+ 只)")
    parser.add_argument("--force", action="store_true",
                        help="强制重新拉取所有 (忽略已有缓存)")
    parser.add_argument("--resume", action="store_true",
                        help="断点续传: 跳过已有 parquet 文件")
    parser.add_argument("--check-only", action="store_true",
                        help="仅统计待拉取数量, 不实际拉取")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制拉取数量 (测试用)")
    parser.add_argument("--start", type=str, default=DEFAULT_START,
                        help=f"起始日期 YYYYMMDD (默认: {DEFAULT_START})")
    parser.add_argument("--end", type=str, default=DEFAULT_END,
                        help=f"结束日期 YYYYMMDD (默认: {DEFAULT_END})")
    parser.add_argument("--proxy", type=str, default=None,
                        help="代理地址 (如 http://127.0.0.1:7897), "
                             "默认绕过系统代理")

    args = parser.parse_args()

    # 配置代理
    setup_proxy(args.proxy)

    meta = run(
        start_date=args.start,
        end_date=args.end,
        force=args.force,
        resume=args.resume,
        check_only=args.check_only,
        limit=args.limit,
    )

    # 根据失败率决定退出码
    if meta and meta.get("stats", {}).get("failed"):
        total_attempted = (len(meta["stats"]["failed"])
                           + meta["count"])
        if total_attempted > 0:
            fail_rate = len(meta["stats"]["failed"]) / total_attempted
            if fail_rate > 0.3:
                log(f"失败率过高: {fail_rate:.0%}")
                sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
