"""临时脚本: 股东户数全市场拉取 (按市值降序, 断点续传, 后台)"""
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
import os, time, glob, warnings
os.environ.pop("HTTP_PROXY", None); os.environ.pop("HTTPS_PROXY", None)
warnings.filterwarnings("ignore")
import akshare as ak
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data_store", "aux_gdhs")
os.makedirs(OUT, exist_ok=True)


def main():
    # 按市值降序排序股票 (市值快照缺失的放最后)
    snap_path = os.path.join(BASE, "data_store", "aux_realtime_20260809.parquet")
    order = []
    if os.path.exists(snap_path):
        snap = pd.read_parquet(snap_path)
        col = "总市值" if "总市值" in snap.columns else snap.columns[3]
        snap = snap.copy()
        snap["_mktcap"] = pd.to_numeric(snap[col], errors="coerce").fillna(0)
        order = snap.sort_values("_mktcap", ascending=False)["代码"].astype(str).tolist()
    all_codes = set(os.path.basename(f).replace(".parquet", "")
                    for f in glob.glob(os.path.join(BASE, "data_store", "*.parquet")))
    order = [c for c in order if c in all_codes]
    order += sorted(all_codes - set(order))  # 市值快照没有的补在后面
    done = set(os.path.basename(f).replace(".parquet", "")
               for f in glob.glob(os.path.join(OUT, "*.parquet")))
    todo = [c for c in order if c not in done]
    print(f"股东户数: 总 {len(all_codes)}, 已有 {len(done)}, 待拉 {len(todo)} (市值降序)", flush=True)

    ok, fail = 0, []
    t0 = time.time()
    for i, code in enumerate(todo, 1):
        try:
            df = ak.stock_zh_a_gdhs_detail_em(symbol=code)
            if df is not None and len(df):
                df.to_parquet(os.path.join(OUT, f"{code}.parquet"), index=False)
                ok += 1
            else:
                fail.append(code)
        except Exception:
            fail.append(code)
        if i % 100 == 0 or i == len(todo):
            el = (time.time() - t0) / 60
            print(f"  [{i}/{len(todo)}] ok={ok} fail={len(fail)} elapsed={el:.1f}min", flush=True)
        time.sleep(0.4)
    print(f"股东户数完成: ok={ok} fail={len(fail)}", flush=True)


if __name__ == "__main__":
    main()
