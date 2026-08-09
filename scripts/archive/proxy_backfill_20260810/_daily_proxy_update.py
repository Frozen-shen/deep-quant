"""
临时脚本: 日线增量代理版 (腾讯被封, 改用东财+代理; 落盘格式与现有一致)
"""
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
END_DATE = "20260807"
COLS = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
        "成交量": "volume", "成交额": "amount", "振幅": "amplitude",
        "涨跌幅": "pct_change", "涨跌额": "change", "换手率": "turnover"}
LOCAL_COLS = ["date", "open", "high", "low", "close", "volume", "amplitude",
              "pct_change", "change", "amount", "turnover", "outstanding_share"]


def fetch_inc(code: str, beg: str) -> pd.DataFrame:
    df = ef.stock.get_quote_history(code, beg=beg, end=END_DATE, klt=101, fqt=1)
    if df is None or len(df) == 0:
        print(f"  [DEBUG] {code} 空返回: type={type(df)} cols={list(df.columns) if df is not None else None}", flush=True)
        return None
    df = df.rename(columns=COLS)
    df["date"] = pd.to_datetime(df["date"])
    return df[list(COLS.values())]


def main():
    paths = sorted(glob.glob(os.path.join(BASE, "data_store", "*.parquet")))
    ok, skip, fail = 0, 0, []
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        code = os.path.basename(p).replace(".parquet", "")
        try:
            old = pd.read_parquet(p)
            old["date"] = pd.to_datetime(old["date"])
            latest = old["date"].max()
            if str(latest)[:10] >= "2026-08-07":
                skip += 1
                continue
            beg = (latest + pd.Timedelta(days=1)).strftime("%Y%m%d")
            new = fetch_inc(code, beg)
            if new is None or len(new) == 0:
                fail.append(f"{code}:无增量")
                continue
            merged = pd.concat([old, new], ignore_index=True)
            merged = merged.drop_duplicates(subset="date", keep="last").sort_values("date")
            for c in LOCAL_COLS:
                if c not in merged.columns:
                    merged[c] = None
            merged = merged[LOCAL_COLS].fillna({"outstanding_share": 0})
            merged.to_parquet(p, index=False)
            ok += 1
        except Exception as e:
            print(f"  [DEBUG-EXC] {code}: {str(e)[:80]}", flush=True)
            fail.append(f"{code}:{str(e)[:40]}")
        if i % 100 == 0 or i == len(paths):
            el = (time.time() - t0) / 60
            print(f"  [{i}/{len(paths)}] ok={ok} skip={skip} fail={len(fail)} elapsed={el:.1f}min", flush=True)
        time.sleep(0.6)  # 降速: 网关对快速连续请求限流 (2026-08-09 并发实验结论)
    print(f"日线增量完成: ok={ok} skip={skip} fail={len(fail)}", flush=True)
    if fail:
        print("失败样例:", fail[:5], flush=True)


if __name__ == "__main__":
    main()
