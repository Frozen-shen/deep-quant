"""临时脚本: 未复权代理版 (腾讯被封, 改东财 fqt=0 + 代理; 覆盖 data_store 全部股票)"""
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
OUT = os.path.join(BASE, "data_cache", "unadjusted")
os.makedirs(OUT, exist_ok=True)

COLS = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume", "成交额": "amount", "换手率": "turnover"}
OUT_COLS = ["date", "open", "high", "low", "close", "volume", "amount",
            "outstanding_share", "turnover"]


def fetch_unadj(code: str) -> pd.DataFrame:
    df = ef.stock.get_quote_history(code, beg="20180101", end="20260807", klt=101, fqt=0)
    if df is None or len(df) == 0:
        return None
    df = df.rename(columns=COLS)
    df["date"] = pd.to_datetime(df["date"])
    df["outstanding_share"] = 0.0  # 未复权不需要流通股本, 占位列
    return df[OUT_COLS]


def main():
    all_codes = set(os.path.basename(f).replace(".parquet", "")
                    for f in glob.glob(os.path.join(BASE, "data_store", "*.parquet")))
    done = set(os.path.basename(f).replace(".parquet", "")
               for f in glob.glob(os.path.join(OUT, "*.parquet")))
    todo = sorted(all_codes - done)
    print(f"未复权(代理): 总 {len(all_codes)}, 已有 {len(done)}, 待拉 {len(todo)}", flush=True)

    ok, fail = 0, []
    t0 = time.time()
    for i, code in enumerate(todo, 1):
        try:
            df = fetch_unadj(code)
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
        time.sleep(0.2)
    print(f"未复权完成: ok={ok} fail={len(fail)}", flush=True)


if __name__ == "__main__":
    main()
