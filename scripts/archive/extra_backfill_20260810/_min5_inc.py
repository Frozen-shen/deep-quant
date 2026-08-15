"""临时脚本: 5m 分钟线增量更新 (拉多少算多少, 断点续传, 积分耗尽自然截断)"""
import sys, os, warnings, time, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eastmoney_proxy
eastmoney_proxy.setup_from_config()
os.environ.pop("HTTP_PROXY", None); os.environ.pop("HTTPS_PROXY", None)
from efinance.shared.tickflow_prompt import session
session.trust_env = False
warnings.filterwarnings("ignore")
import efinance as ef
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data_store", "minute_5m")
END = "20260807"

COLS = {"日期": "datetime", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume", "成交额": "amount"}
OUT_COLS = ["datetime", "day", "open", "high", "low", "close", "volume", "amount"]


def fetch_inc(code: str, beg: str):
    df = ef.stock.get_quote_history(code, beg=beg, end=END, klt=5)
    if df is None or len(df) == 0:
        return None
    df = df.rename(columns=COLS)
    df = df[["datetime", "open", "close", "high", "low", "volume", "amount"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["day"] = pd.to_datetime(df["datetime"].dt.date)
    return df[OUT_COLS]


def main():
    paths = sorted(glob.glob(os.path.join(OUT, "*.parquet")))
    # 缺失的 20 只也补 (全量拉)
    all_codes = set(os.path.basename(f).replace(".parquet", "")
                    for f in glob.glob(os.path.join(BASE, "data_store", "*.parquet")))
    have = {os.path.basename(f).replace(".parquet", "") for f in paths}
    missing = sorted(all_codes - have)
    paths += [os.path.join(OUT, f"{c}.parquet") for c in missing]
    print(f"5m: 已有 {len(have)}, 缺失补拉 {len(missing)}, 总处理 {len(paths)}", flush=True)

    ok, skip, fail = 0, 0, []
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        code = os.path.basename(p).replace(".parquet", "")
        try:
            if os.path.exists(p):
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
            else:
                # 缺失股票: 全量拉 2022 起
                df = ef.stock.get_quote_history(code, beg="20220101", end=END, klt=5)
                if df is None or len(df) == 0:
                    fail.append(code)
                    continue
                df = df.rename(columns=COLS)
                df = df[["datetime", "open", "close", "high", "low", "volume", "amount"]].copy()
                df["datetime"] = pd.to_datetime(df["datetime"])
                df["day"] = pd.to_datetime(df["datetime"].dt.date)
                df[OUT_COLS].to_parquet(p, index=False)
            ok += 1
        except Exception as e:
            fail.append(f"{code}:{str(e)[:40]}")
        if i % 100 == 0 or i == len(paths):
            el = (time.time() - t0) / 60
            print(f"  [{i}/{len(paths)}] ok={ok} skip={skip} fail={len(fail)} elapsed={el:.1f}min", flush=True)
        time.sleep(0.3)
    print(f"5m增量完成: ok={ok} skip={skip} fail={len(fail)}", flush=True)


if __name__ == "__main__":
    main()
