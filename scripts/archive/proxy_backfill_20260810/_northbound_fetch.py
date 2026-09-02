"""临时脚本: 北向个股持仓补全 (缺失3137只, 代理, 断点续传)"""
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
import os, sys, time, glob, warnings
os.environ.pop("HTTP_PROXY", None); os.environ.pop("HTTPS_PROXY", None)
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from smart_money_fetcher import fetch_northbound_history

OUT = os.path.join(BASE, "data", "smart_money", "northbound")


def main():
    all_codes = set(os.path.basename(f).replace(".parquet", "")
                    for f in glob.glob(os.path.join(BASE, "data_store", "*.parquet")))
    done = set(os.path.basename(f).replace(".parquet", "")
               for f in glob.glob(os.path.join(OUT, "*.parquet")))
    todo = sorted(all_codes - done)
    print(f"北向: 总 {len(all_codes)}, 已有 {len(done)}, 待拉 {len(todo)}", flush=True)

    ok, fail = 0, []
    t0 = time.time()
    for i, code in enumerate(todo, 1):
        try:
            df = fetch_northbound_history(code)
            if df is None or len(df) == 0:
                fail.append(code)  # 无北向数据(非沪深港通标的)也跳过, 避免反复拉
            else:
                df.to_parquet(os.path.join(OUT, f"{code}.parquet"), index=False)
                ok += 1
        except Exception:
            fail.append(code)
        if i % 100 == 0 or i == len(todo):
            el = (time.time() - t0) / 60
            print(f"  [{i}/{len(todo)}] ok={ok} fail={len(fail)} elapsed={el:.1f}min", flush=True)
        time.sleep(0.35)
    print(f"北向完成: ok={ok} fail={len(fail)}", flush=True)


if __name__ == "__main__":
    main()
