"""
增量数据更新脚本 — 每日收盘后拉取新日线数据并追加到 parquet 缓存

数据源: 东财 (efinance, 经 eastmoney_proxy 代理) 为主, 腾讯财经兜底
  (2026-08 腾讯源被封返回非 JSON, 东财直连被限流; 代理集成见 config.yaml eastmoney_proxy)

用法:
  py scripts/active/update_daily_data.py              # 增量更新所有缓存股票
  py scripts/active/update_daily_data.py --check-only # 仅检查数据落后情况
  py scripts/active/update_daily_data.py --symbols 600519,000858  # 更新指定股票
  py scripts/active/update_daily_data.py --force-full # 强制全量重新拉取

策略:
  - 对每只缓存股票, 读取 parquet 中最大日期
  - 只拉取 max_date + 1day 之后的新数据
  - append 写入 (去重: 按 date 去重后排序写入)
  - 同时更新未复权数据 (data_store/unadjusted/)
  - 更新完成后运行数据质量校验 (data/validator.py)
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

# 东财代理 (config.yaml eastmoney_proxy 段): 必须在 efinance 首次 import 前启用
import eastmoney_proxy
eastmoney_proxy.setup_from_config(BASE_DIR)

from data.calendar import is_trading_day

# ── 路径配置 ──
DATA_STORE = os.path.join(BASE_DIR, "data_store")
UNADJ_DIR = os.path.join(DATA_STORE, "unadjusted")
LOG_FILE = os.path.join(BASE_DIR, "data", "update_log.jsonl")

# ── 拉取参数 ──
DEFAULT_START = "20180101"
RATE_LIMIT_SLEEP = 0.35     # 请求间隔 (~3 req/s)
MAX_RETRIES = 3


def log(msg: str, level: str = "INFO"):
    """统一日志输出。"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(
            {"timestamp": ts, "level": level, "message": msg},
            ensure_ascii=False) + "\n")


def get_cached_symbols() -> List[str]:
    """获取 data_store/ 中已缓存的所有股票代码。"""
    if not os.path.exists(DATA_STORE):
        return []
    return sorted([
        f.replace(".parquet", "")
        for f in os.listdir(DATA_STORE)
        if f.endswith(".parquet") and len(f) == 14  # "XXXXXX.parquet" (6位代码 + .parquet)
    ])


def get_latest_date(parquet_path: str) -> Optional[pd.Timestamp]:
    """读取 parquet 文件中的最大日期。"""
    if not os.path.exists(parquet_path):
        return None
    try:
        df = pd.read_parquet(parquet_path)
        if len(df) == 0:
            return None
        df["date"] = pd.to_datetime(df["date"])
        return df["date"].max()
    except Exception as e:
        log(f"读取 {parquet_path} 失败: {e}", "WARN")
        return None


# ════════════════════════════════════════════════════════════
#  腾讯财经 API 数据拉取
# ════════════════════════════════════════════════════════════

def _make_session():
    """创建不经过系统代理的 requests session。"""
    import requests
    s = requests.Session()
    s.trust_env = False
    s.headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    return s


KEEP_COLS = ["date", "open", "high", "low", "close", "volume",
             "amount", "amplitude", "pct_change", "change", "turnover"]


def fetch_tencent(code: str, start_date: str, end_date: str,
                  adjust: str = "qfq") -> Optional[pd.DataFrame]:
    """
    从腾讯财经 API 拉取日线数据。

    单次最多返回 ~640 条, 需分段拉取。
    注意: 腾讯 API 不提供 amount/turnover, 这些列填 NaN。

    Args:
      code: 6位股票代码
      start_date: YYYYMMDD
      end_date: YYYYMMDD
      adjust: 'qfq' 前复权, '' 不复权

    Returns:
      DataFrame 或 None
    """
    from datetime import datetime as _dt, timedelta as _td

    prefix = "sh" if code.startswith("6") else "sz"
    symbol = f"{prefix}{code}"

    sd = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    ed = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

    # 腾讯 API 的复权参数
    fq_param = "qfq" if adjust == "qfq" else ""

    session = _make_session()
    all_klines = []
    current_end = ed
    max_chunks = 15  # 增量更新通常只需1-2段, 全量需要更多

    for _ in range(max_chunks):
        url = (
            f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
            f"?param={symbol},day,{sd},{current_end},2000,{fq_param}"
        )
        try:
            resp = session.get(url, timeout=15)
            data = resp.json()
        except Exception as e:
            log(f"  腾讯API请求失败 {code}: {e}", "WARN")
            return None

        payload = data.get("data")
        if not isinstance(payload, dict):
            break

        stock_data = payload.get(symbol)
        if not isinstance(stock_data, dict):
            break

        # 前复权用 qfqday, 不复权用 day
        klines = stock_data.get("qfqday") or stock_data.get("day", [])
        if not klines:
            break

        all_klines = klines + all_klines

        first_date = _dt.strptime(klines[0][0], "%Y-%m-%d") - _td(days=1)
        current_end = first_date.strftime("%Y-%m-%d")

        if current_end < sd:
            break

    if not all_klines:
        return None

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


def fetch_eastmoney(code: str, start_date: str, end_date: str,
                    adjust: str = "qfq") -> Optional[pd.DataFrame]:
    """
    从东财 API 拉取日线数据 (efinance, 经 eastmoney_proxy 代理)。

    2026-08 起为主数据源 (腾讯被封)。efinance 返回列比腾讯全 (含 amount/turnover)。

    Args:
      code: 6位股票代码
      start_date: YYYYMMDD
      end_date: YYYYMMDD
      adjust: 'qfq' 前复权, '' 不复权

    Returns:
      DataFrame 或 None
    """
    import efinance as ef

    fqt = 1 if adjust == "qfq" else 0
    df = ef.stock.get_quote_history(code, beg=start_date, end=end_date, klt=101, fqt=fqt)
    if df is None or len(df) == 0:
        return None

    df = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume", "成交额": "amount", "振幅": "amplitude",
        "涨跌幅": "pct_change", "涨跌额": "change", "换手率": "turnover",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)

    available = [c for c in KEEP_COLS if c in df.columns]
    return df[available].copy()


def fetch_incremental(symbol: str, start_date: str, end_date: str,
                      adjust: str = "qfq") -> Optional[pd.DataFrame]:
    """
    增量拉取单只股票的新数据。

    数据源优先级: 东财(代理) → 腾讯(兜底)。2026-08 腾讯被封后主走东财。

    Returns:
      DataFrame 或 None (无新数据/拉取失败)
    """
    sources = [fetch_eastmoney, fetch_tencent]
    for source in sources:
        for attempt in range(MAX_RETRIES):
            try:
                df = source(symbol, start_date, end_date, adjust)
                if df is not None and len(df) > 0:
                    return df
                break  # 空数据不重试, 换下一个源
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** (attempt + 1)
                    time.sleep(wait)
                else:
                    log(f"拉取 {symbol} ({source.__name__}, adjust={adjust}) 失败: {e}", "WARN")
    return None


# ════════════════════════════════════════════════════════════
#  增量追加
# ════════════════════════════════════════════════════════════

def append_to_parquet(existing_path: str, new_data: pd.DataFrame) -> int:
    """
    将新数据追加到 parquet 文件 (去重、排序)。

    Returns:
      新增行数
    """
    if os.path.exists(existing_path):
        existing = pd.read_parquet(existing_path)
        existing["date"] = pd.to_datetime(existing["date"])

        # 合并去重 (保留新数据)
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last")
        combined = combined.sort_values("date").reset_index(drop=True)

        added = len(combined) - len(existing)
    else:
        combined = new_data.sort_values("date").reset_index(drop=True)
        added = len(combined)

    if added > 0:
        combined.to_parquet(existing_path, index=False)

    return max(0, added)


def update_symbol(symbol: str, end_date: str,
                  update_unadjusted: bool = True) -> Dict:
    """
    更新单只股票的前复权和未复权数据。

    Returns:
      {"symbol": str, "qfq_added": int, "unadj_added": int, "error": str|None}
    """
    result = {"symbol": symbol, "qfq_added": 0, "unadj_added": 0, "error": None}

    # ── 前复权数据 (主缓存) ──
    qfq_path = os.path.join(DATA_STORE, f"{symbol}.parquet")
    latest_qfq = get_latest_date(qfq_path)

    if latest_qfq is None:
        # 无缓存: 全量拉取
        log(f"  {symbol}: 无缓存, 全量拉取前复权")
        new_data = fetch_incremental(symbol, DEFAULT_START, end_date, "qfq")
        if new_data is not None:
            added = append_to_parquet(qfq_path, new_data)
            result["qfq_added"] = added
    else:
        # 增量拉取
        next_day = (latest_qfq + timedelta(days=1)).strftime("%Y%m%d")
        if next_day <= end_date:
            new_data = fetch_incremental(symbol, next_day, end_date, "qfq")
            if new_data is not None and len(new_data) > 0:
                added = append_to_parquet(qfq_path, new_data)
                result["qfq_added"] = added

    # ── 未复权数据 (涨跌停判断用) ──
    if update_unadjusted:
        unadj_path = os.path.join(UNADJ_DIR, f"{symbol}.parquet")
        latest_unadj = get_latest_date(unadj_path)

        if latest_unadj is None:
            new_data = fetch_incremental(symbol, DEFAULT_START, end_date, "")
            if new_data is not None:
                os.makedirs(UNADJ_DIR, exist_ok=True)
                added = append_to_parquet(unadj_path, new_data)
                result["unadj_added"] = added
        else:
            next_day = (latest_unadj + timedelta(days=1)).strftime("%Y%m%d")
            if next_day <= end_date:
                new_data = fetch_incremental(symbol, next_day, end_date, "")
                if new_data is not None and len(new_data) > 0:
                    added = append_to_parquet(unadj_path, new_data)
                    result["unadj_added"] = added

    return result


# ════════════════════════════════════════════════════════════
#  批量更新
# ════════════════════════════════════════════════════════════

def update_all(end_date: str = None, symbols: List[str] = None,
               update_unadjusted: bool = True, max_failures: int = 20) -> Dict:
    """
    批量增量更新所有缓存股票。

    Args:
      end_date: 拉取截止日期 YYYYMMDD, None=今天
      symbols: 指定股票列表, None=全部缓存股票
      update_unadjusted: 是否同时更新未复权数据
      max_failures: 连续失败上限, 超过则中止

    Returns:
      统计字典
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    if symbols is None:
        symbols = get_cached_symbols()

    if not symbols:
        log("没有缓存的股票, 请先运行 scripts/active/fetch_daily_data.py", "ERROR")
        return {"total": 0, "updated": 0, "failed": 0, "total_added": 0,
                "qfq_added": 0, "unadj_added": 0, "failures": []}

    log(f"开始更新 {len(symbols)} 只股票, 截止日期: {end_date}")
    os.makedirs(DATA_STORE, exist_ok=True)
    if update_unadjusted:
        os.makedirs(UNADJ_DIR, exist_ok=True)

    stats = {"total": len(symbols), "updated": 0, "failed": 0,
             "total_added": 0, "qfq_added": 0, "unadj_added": 0,
             "failures": []}
    consecutive_failures = 0
    t0 = time.time()

    for i, sym in enumerate(symbols, 1):
        try:
            r = update_symbol(sym, end_date, update_unadjusted)
            stats["qfq_added"] += r["qfq_added"]
            stats["unadj_added"] += r["unadj_added"]

            if r["qfq_added"] > 0 or r["unadj_added"] > 0:
                stats["updated"] += 1
                stats["total_added"] += r["qfq_added"] + r["unadj_added"]
                log(f"  [{i}/{len(symbols)}] {sym}: "
                    f"+{r['qfq_added']}复权 +{r['unadj_added']}未复权")
                consecutive_failures = 0
            else:
                consecutive_failures += 1

            if r.get("error"):
                stats["failed"] += 1
                stats["failures"].append(f"{sym}: {r['error']}")

        except Exception as e:
            stats["failed"] += 1
            stats["failures"].append(f"{sym}: {str(e)}")
            consecutive_failures += 1
            log(f"  [{i}/{len(symbols)}] {sym}: ❌ {e}", "ERROR")

        # 进度报告
        if i % 50 == 0 or i == len(symbols):
            elapsed = time.time() - t0
            log(f"进度: {i}/{len(symbols)} | 更新:{stats['updated']} "
                f"失败:{stats['failed']} | {elapsed:.0f}s")

        # 连续失败保护
        if consecutive_failures >= max_failures:
            log(f"连续失败 {consecutive_failures} 次, 中止更新", "ERROR")
            break

        # 礼貌限速
        time.sleep(RATE_LIMIT_SLEEP)

    elapsed = time.time() - t0
    log(f"\n{'='*60}")
    log(f"更新完成: {stats['updated']}/{stats['total']} 只更新, "
        f"{stats['failed']} 只失败, 新增 {stats['total_added']} 行数据")
    log(f"耗时: {elapsed:.0f}s ({elapsed/60:.1f}分钟)")

    if stats["failures"]:
        log(f"失败详情 ({len(stats['failures'])}只):")
        for f in stats["failures"][:10]:
            log(f"  {f}")

    return stats


def check_only(symbols: List[str] = None) -> Dict:
    """
    Dry-run 模式: 检查所有缓存股票的数据落后情况, 不做实际拉取。

    Returns:
      统计字典 (含落后天数分布)
    """
    if symbols is None:
        symbols = get_cached_symbols()

    today = pd.Timestamp.now().normalize()
    lags = []
    no_cache = 0
    up_to_date = 0

    for sym in symbols:
        qfq_path = os.path.join(DATA_STORE, f"{sym}.parquet")
        latest = get_latest_date(qfq_path)
        if latest is None:
            no_cache += 1
            continue
        lag = (today - latest).days
        lags.append(lag)
        if lag <= 1:
            up_to_date += 1

    # 分布统计
    from collections import Counter
    dist = Counter()
    for lag in lags:
        if lag <= 1:
            dist["0-1天"] += 1
        elif lag <= 5:
            dist["2-5天"] += 1
        elif lag <= 10:
            dist["6-10天"] += 1
        elif lag <= 20:
            dist["11-20天"] += 1
        else:
            dist["20天以上"] += 1

    print(f"\n{'='*60}")
    print(f"数据状态检查 (截止 {today.date()})")
    print(f"{'='*60}")
    print(f"  总缓存股票: {len(symbols)}")
    print(f"  无缓存文件: {no_cache}")
    print(f"  数据更新到最近1天: {up_to_date}")
    print(f"  需要更新: {len(lags) - up_to_date}")
    if lags:
        print(f"  最大落后天数: {max(lags)}")
        print(f"  平均落后天数: {np.mean(lags):.1f}")
        print(f"\n  落后天数分布:")
        for k in ["0-1天", "2-5天", "6-10天", "11-20天", "20天以上"]:
            if k in dist:
                bar = "█" * min(50, dist[k])
                print(f"    {k:<10}: {dist[k]:>4} {bar}")

    return {"total": len(symbols), "need_update": len(lags) - up_to_date,
            "up_to_date": up_to_date, "no_cache": no_cache,
            "max_lag_days": max(lags) if lags else 0,
            "lag_distribution": dict(dist)}


def run_validation():
    """运行数据质量校验 (Phase 1.2)。"""
    try:
        from data.validator import validate_all
        report = validate_all()
        passed = report.passed if hasattr(report, 'passed') else True
        log(f"数据质量校验完成: {'通过' if passed else '发现问题'}")
        return report
    except ImportError:
        log("data.validator 模块不可用, 跳过校验", "WARN")
        return None
    except Exception as e:
        log(f"数据质量校验失败: {e}", "ERROR")
        return None


# ════════════════════════════════════════════════════════════
#  供 daily_pipeline.py 调用的入口
# ════════════════════════════════════════════════════════════

def run(start_date: str = DEFAULT_START, end_date: str = None,
        force: bool = False, resume: bool = True,
        check_only_mode: bool = False, limit: int = None) -> Optional[Dict]:
    """
    供 daily_pipeline.py 调用的统一入口。

    与 fetch_daily_data.run() 接口兼容, 但使用腾讯数据源做增量更新。
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    symbols = get_cached_symbols()
    if limit:
        symbols = symbols[:limit]

    if check_only_mode:
        check_only(symbols)
        return None

    stats = update_all(
        end_date=end_date,
        symbols=symbols,
        update_unadjusted=True,
    )

    # 数据质量校验
    if stats["total_added"] > 0:
        run_validation()

    # 更新数据版本号
    data_version = datetime.now().strftime("v%Y%m%d")
    try:
        import storage
        storage.init_db()
        storage.set_config("data_version", data_version)
        storage.set_config("last_data_update", datetime.now().isoformat())
        log(f"数据版本更新: {data_version}")
    except Exception as e:
        log(f"更新数据版本号失败: {e}", "WARN")

    return {
        "count": stats["updated"],
        "stats": {"failed": stats["failures"]},
    }


# ════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="增量更新股票日线数据 (腾讯数据源)")
    parser.add_argument("--check-only", action="store_true",
                       help="仅检查数据落后状态, 不拉取")
    parser.add_argument("--symbols", type=str, default=None,
                       help="指定股票代码, 逗号分隔 (默认全部)")
    parser.add_argument("--end-date", type=str, default=None,
                       help="拉取截止日期 YYYYMMDD (默认今天)")
    parser.add_argument("--force-full", action="store_true",
                       help="强制全量重新拉取 (删除现有缓存后重新拉取)")
    parser.add_argument("--skip-unadjusted", action="store_true",
                       help="跳过未复权数据更新")
    parser.add_argument("--skip-validation", action="store_true",
                       help="跳过数据质量校验")
    parser.add_argument("--max-failures", type=int, default=20,
                       help="连续失败上限 (默认20)")

    args = parser.parse_args()

    # 交易日检查
    today = pd.Timestamp.now().normalize()
    if not is_trading_day(today):
        log(f"今天 ({today.date()}) 不是交易日, 跳过数据更新")
        sys.exit(0)

    # 解析股票列表
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]

    # --check-only 模式
    if args.check_only:
        check_only(symbols)
        sys.exit(0)

    # --force-full 模式
    if args.force_full:
        log("⚠️ 强制全量重新拉取模式")
        if symbols is None:
            symbols = get_cached_symbols()
        for sym in symbols:
            for d in [DATA_STORE, UNADJ_DIR]:
                path = os.path.join(d, f"{sym}.parquet")
                if os.path.exists(path):
                    os.remove(path)
                    log(f"  已删除: {path}")
        log(f"已清空 {len(symbols)} 只股票的缓存")

    # 执行增量更新
    end_date = args.end_date or datetime.now().strftime("%Y%m%d")
    stats = update_all(
        end_date=end_date,
        symbols=symbols,
        update_unadjusted=not args.skip_unadjusted,
        max_failures=args.max_failures,
    )

    # 数据质量校验
    if not args.skip_validation and stats["total_added"] > 0:
        run_validation()

    # 更新数据版本号
    data_version = datetime.now().strftime("v%Y%m%d")
    try:
        import storage
        storage.init_db()
        storage.set_config("data_version", data_version)
        storage.set_config("last_data_update", datetime.now().isoformat())
        log(f"数据版本更新: {data_version}")
    except Exception as e:
        log(f"更新数据版本号失败: {e}", "WARN")

    # 根据结果退出
    if stats["failed"] >= stats["total"] * 0.3:
        log("⚠️ 超过30%股票更新失败, 请检查网络/数据源状态", "WARN")
        sys.exit(1)
    else:
        log("✅ 数据更新完成")
        sys.exit(0)
