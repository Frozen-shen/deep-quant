"""
scripts/active/fetch_missing_pit_stocks.py — 补拉 PIT universe 中缺失的股票数据

从 baostock 获取 2018-01-01 ~ 2026-07-31 日线数据，保存到 data_store/{symbol}.parquet
"""

import os
import sys
import json
import time
import warnings

import pandas as pd
import baostock as bs

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE_DIR = os.path.join(BASE_DIR, "data_store")
MISSING_PATH = os.path.join(BASE_DIR, "data", "cache", "missing_pit_stocks.json")

START_DATE = "2018-01-01"
END_DATE = "2026-07-31"

FIELDS = "date,code,open,high,low,close,volume,amount,turn"


def to_baostock_code(symbol: str) -> str:
    """000001 -> sz.000001, 600009 -> sh.600009"""
    if symbol.startswith("6"):
        return f"sh.{symbol}"
    else:
        return f"sz.{symbol}"


def fetch_one(symbol: str, start_date: str = START_DATE) -> pd.DataFrame | None:
    code = to_baostock_code(symbol)
    rs = bs.query_history_k_data_plus(
        code, FIELDS,
        start_date=start_date, end_date=END_DATE,
        frequency="d", adjustflag="2"  # qfq
    )
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=rs.fields)
    # Rename and convert
    df = df.rename(columns={"turn": "turnover"})
    for col in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop(columns=["code"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


BATCH_SIZE = 100  # Reconnect every N stocks to avoid session timeout


def main():
    import argparse
    parser = argparse.ArgumentParser(description="补拉 PIT 缺失股票")
    parser.add_argument("--missing-list", type=str, default=MISSING_PATH,
                        help="缺失股票 JSON 文件路径")
    parser.add_argument("--start", type=str, default=START_DATE,
                        help=f"起始日期 YYYY-MM-DD (默认: {START_DATE})")
    args = parser.parse_args()

    with open(args.missing_list) as f:
        missing = json.load(f)

    start_date = args.start

    print(f"待补拉: {len(missing)} 只")
    print(f"日期范围: {start_date} ~ {END_DATE}")
    print(f"保存到: {STORE_DIR}/")
    print()

    lg = bs.login()
    if lg.error_code != "0":
        print(f"baostock login failed: {lg.error_msg}")
        return

    success = 0
    failed = []
    fetched_in_batch = 0  # counts actual fetches since last login
    t0 = time.time()

    for i, sym in enumerate(missing):
        out_path = os.path.join(STORE_DIR, f"{sym}.parquet")
        if os.path.exists(out_path):
            success += 1
            continue

        # Batch reconnection: logout/login every BATCH_SIZE fetches
        if fetched_in_batch > 0 and fetched_in_batch % BATCH_SIZE == 0:
            bs.logout()
            time.sleep(1)
            lg = bs.login()
            if lg.error_code != "0":
                print(f"baostock re-login failed at {i}: {lg.error_msg}")
                failed.extend(missing[i:])
                break
            print(f"  [reconnected at index {i}]")

        try:
            df = fetch_one(sym, start_date=start_date)
            if df is not None and len(df) > 100:
                df.to_parquet(out_path, index=False)
                success += 1
            else:
                failed.append(sym)
        except Exception as e:
            failed.append(sym)
        fetched_in_batch += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(missing) - i - 1)
            print(f"  {i+1}/{len(missing)} done, {success} ok, {len(failed)} fail, "
                  f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

        time.sleep(0.3)  # Rate limit

    bs.logout()

    elapsed = time.time() - t0
    print(f"\n完成: {success} 成功, {len(failed)} 失败, 耗时 {elapsed:.0f}s")
    if failed:
        print(f"失败列表: {failed[:20]}...")
        with open(os.path.join(BASE_DIR, "data", "cache", "fetch_failed.json"), "w") as f:
            json.dump(failed, f)

    # Update _meta.json
    meta_path = os.path.join(STORE_DIR, "_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        all_syms = sorted(set(meta.get("symbols", [])) | set(
            os.path.basename(f).replace(".parquet", "")
            for f in os.listdir(STORE_DIR) if f.endswith(".parquet")
        ))
        meta["symbols"] = all_syms
        meta.setdefault("fix_history", []).append({
            "date": pd.Timestamp.now().isoformat(),
            "action": "fetch_missing_pit_stocks",
            "source": "baostock",
            "added": success,
            "failed": len(failed),
        })
        with open(meta_path, "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"_meta.json 更新: {len(all_syms)} 只")


if __name__ == "__main__":
    main()
