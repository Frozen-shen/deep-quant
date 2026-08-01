"""
fetch_unadjusted_batch.py — 批量获取未复权日线数据

用途: 涨跌停检测需要未复权价格 (复权后10%/20%阈值失真)。
产出: data_cache/unadjusted/{code}.parquet

数据源: 腾讯财经 API (与 fetch_daily_data.py 相同的 tencent 源)
当前缺口: 仅176/1550只有未复权数据, 需补充到全覆盖。

用法:
  py scripts/fetch_unadjusted_batch.py --check-only   # 查看覆盖率
  py scripts/fetch_unadjusted_batch.py --resume       # 补拉缺失
  py scripts/fetch_unadjusted_batch.py --limit 10     # 测试
"""
import os
import sys
import time
import argparse
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_CACHE = BASE_DIR / "data_cache"
UNADJ_DIR = DATA_CACHE / "unadjusted"

RATE_LIMIT = 0.5  # 秒/请求


def _make_session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    s.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    return s


def get_cached_symbols() -> list:
    """获取 data_cache 中已有复权数据的股票列表。"""
    return sorted([
        f.stem for f in DATA_CACHE.glob("*.parquet")
        if len(f.stem) == 6 and f.stem.isdigit()
    ])


def get_existing_unadj() -> set:
    """获取已有未复权数据的股票。"""
    if not UNADJ_DIR.exists():
        return set()
    return {f.stem for f in UNADJ_DIR.glob("*.parquet") if len(f.stem) == 6}


def fetch_unadjusted_tencent(code: str, session: requests.Session,
                              start: str = "2018-01-01",
                              end: str = "2026-07-31") -> pd.DataFrame:
    """
    通过腾讯API获取未复权日线。
    关键: URL末尾不加 'qfq', 数据在 'day' 字段而非 'qfqday'。
    """
    from datetime import datetime, timedelta

    prefix = "sh" if code.startswith("6") else "sz"
    symbol = f"{prefix}{code}"

    all_klines = []
    current_end = end
    max_chunks = 10

    for _ in range(max_chunks):
        # 注意: 末尾逗号后为空 = 不复权
        url = (
            f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
            f"?param={symbol},day,{start},{current_end},2000,"
        )
        try:
            resp = session.get(url, timeout=15)
            data = resp.json()
        except Exception:
            break

        payload = data.get("data")
        if not isinstance(payload, dict):
            break

        stock_data = payload.get(symbol)
        if not isinstance(stock_data, dict):
            break

        # 未复权数据在 "day" 字段
        klines = stock_data.get("day", [])
        if not klines:
            break

        all_klines = klines + all_klines

        first_date = datetime.strptime(klines[0][0], "%Y-%m-%d") - timedelta(days=1)
        current_end = first_date.strftime("%Y-%m-%d")

        if current_end < start:
            break

        time.sleep(0.3)

    if not all_klines:
        return pd.DataFrame()

    rows = []
    for k in all_klines:
        rows.append({
            "date": k[0],
            "open": float(k[1]),
            "close": float(k[2]),
            "high": float(k[3]),
            "low": float(k[4]),
            "volume": float(k[5]) if len(k) > 5 else 0,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="批量获取未复权日线")
    parser.add_argument("--resume", action="store_true", help="跳过已有")
    parser.add_argument("--limit", type=int, default=0, help="限制数量")
    parser.add_argument("--check-only", action="store_true", help="只检查")
    args = parser.parse_args()

    UNADJ_DIR.mkdir(parents=True, exist_ok=True)

    all_symbols = get_cached_symbols()
    existing = get_existing_unadj()
    missing = [s for s in all_symbols if s not in existing]

    print(f"复权数据: {len(all_symbols)} 只")
    print(f"未复权已有: {len(existing)} 只")
    print(f"未复权缺失: {len(missing)} 只")
    print(f"覆盖率: {len(existing)/max(len(all_symbols),1)*100:.1f}%")

    if args.check_only:
        return

    to_fetch = missing if args.resume else all_symbols
    if args.limit > 0:
        to_fetch = to_fetch[:args.limit]

    print(f"\n开始获取: {len(to_fetch)} 只 (间隔{RATE_LIMIT}秒)")
    session = _make_session()
    success = 0
    failed = 0

    for i, code in enumerate(to_fetch):
        try:
            df = fetch_unadjusted_tencent(code, session)
            if df is not None and not df.empty:
                out_path = UNADJ_DIR / f"{code}.parquet"
                df.to_parquet(out_path, index=False)
                success += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  ✗ {code}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(to_fetch)} | 成功: {success} | 失败: {failed}",
                  flush=True)
        time.sleep(RATE_LIMIT)

    final = get_existing_unadj()
    print(f"\n完成。成功: {success}, 失败: {failed}")
    print(f"最终覆盖率: {len(final)/max(len(all_symbols),1)*100:.1f}%")


if __name__ == "__main__":
    main()
