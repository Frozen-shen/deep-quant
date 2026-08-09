"""临时脚本: 15m 分钟线增量更新 (原3360只旧文件停在07-31, 增量合并到08-07)"""
# ⚠️ 代理初始化必须放在最顶部, 在 efinance 之前! (配置: config.yaml eastmoney_proxy)
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eastmoney_proxy
eastmoney_proxy.setup_from_config()

import time, glob
os.environ.pop("HTTP_PROXY", None); os.environ.pop("HTTPS_PROXY", None)
from efinance.shared.tickflow_prompt import session
session.trust_env = False
import efinance as ef
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data_store", "minute_15m")
END = "20260807"

COLS = {"日期": "datetime", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume", "成交额": "amount"}
OUT_COLS = ["datetime", "day", "open", "high", "low", "close", "volume", "amount"]


def fetch_inc(code: str, beg: str):
    df = ef.stock.get_quote_history(code, beg=beg, end=END, klt=15)
    if df is None or len(df) == 0:
        return None
    df = df.rename(columns=COLS)
    df = df[["datetime", "open", "close", "high", "low", "volume", "amount"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["day"] = pd.to_datetime(df["datetime"].dt.date)  # 与旧文件 day 列 dtype(datetime64) 一致
    return df[OUT_COLS]


def main():
    paths = sorted(glob.glob(os.path.join(OUT, "*.parquet")))
    ok, skip, fail = 0, 0, []
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        code = os.path.basename(p).replace(".parquet", "")
        try:
            old = pd.read_parquet(p)
            old["datetime"] = pd.to_datetime(old["datetime"])
            latest = old["datetime"].max()
            if str(latest)[:10] >= "2026-08-07":
                skip += 1
                continue
            beg = (latest + pd.Timedelta(days=1)).strftime("%Y%m%d")
            new = fetch_inc(code, beg)
            if new is None or len(new) == 0:
                fail.append(code)
                continue
            merged = pd.concat([old, new], ignore_index=True)
            merged = merged.drop_duplicates(subset="datetime", keep="last").sort_values("datetime")
            merged = merged[OUT_COLS]
            merged.to_parquet(p, index=False)
            ok += 1
        except Exception as e:
            print(f"  [EXC] {code}: {str(e)[:80]}", flush=True)
            fail.append(f"{code}:{str(e)[:40]}")
        if i % 100 == 0 or i == len(paths):
            el = (time.time() - t0) / 60
            print(f"  [{i}/{len(paths)}] ok={ok} skip={skip} fail={len(fail)} elapsed={el:.1f}min", flush=True)
        time.sleep(0.3)
    print(f"15m增量完成: ok={ok} skip={skip} fail={len(fail)}", flush=True)
    if fail:
        print("失败样例:", fail[:5], flush=True)


if __name__ == "__main__":
    main()
