"""临时脚本: 15m 分钟线补全 (1716只缺失, 断点续传, 后台)"""
import akshare_proxy_patch
akshare_proxy_patch.install_patch(
    "101.201.173.125",
    auth_token="[REDACTED]",
    retry=30,
    hook_domains=[
        "fund.eastmoney.com", "push2.eastmoney.com", "push2his.eastmoney.com",
        "push2ex.eastmoney.com", "datacenter-web.eastmoney.com",
        "emweb.securities.eastmoney.com", "searchapi.eastmoney.com/api/suggest/get",
    ],
    fast=True,
)
import os, time, glob
os.environ.pop("HTTP_PROXY", None); os.environ.pop("HTTPS_PROXY", None)
from efinance.shared.tickflow_prompt import session
session.trust_env = False
import efinance as ef
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data_store", "minute_15m")

COLS = {"日期": "datetime", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume", "成交额": "amount"}
OUT_COLS = ["datetime", "day", "open", "high", "low", "close", "volume", "amount"]


def fetch_15m(code: str):
    df = ef.stock.get_quote_history(code, beg="20220101", end="20500101", klt=15)
    if df is None or len(df) == 0:
        return None
    df = df.rename(columns=COLS)
    df = df[["datetime", "open", "close", "high", "low", "volume", "amount"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["day"] = pd.to_datetime(df["datetime"].dt.date)  # 与旧文件 day 列 dtype(datetime64) 一致
    return df[OUT_COLS]


def main():
    all_codes = set(os.path.basename(f).replace(".parquet", "")
                    for f in glob.glob(os.path.join(BASE, "data_store", "*.parquet")))
    done = set(os.path.basename(f).replace(".parquet", "")
               for f in glob.glob(os.path.join(OUT, "*.parquet")))
    todo = sorted(all_codes - done)
    print(f"15m: 总 {len(all_codes)}, 已有 {len(done)}, 待拉 {len(todo)}", flush=True)

    ok, fail = 0, []
    t0 = time.time()
    for i, code in enumerate(todo, 1):
        try:
            df = fetch_15m(code)
            if df is None:
                fail.append(code)
            else:
                df.to_parquet(os.path.join(OUT, f"{code}.parquet"), index=False)
                ok += 1
        except Exception:
            fail.append(code)
        if i % 100 == 0 or i == len(todo):
            el = (time.time() - t0) / 60
            print(f"  [{i}/{len(todo)}] ok={ok} fail={len(fail)} elapsed={el:.1f}min", flush=True)
        time.sleep(0.25)
    print(f"15m完成: ok={ok} fail={len(fail)}", flush=True)


if __name__ == "__main__":
    main()
