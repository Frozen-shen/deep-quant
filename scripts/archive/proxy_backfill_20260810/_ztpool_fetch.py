"""临时脚本: 涨停池历史 (push2ex 域名, 已加入 hook)"""
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
import os, warnings, time, glob
os.environ.pop("HTTP_PROXY", None); os.environ.pop("HTTPS_PROXY", None)
warnings.filterwarnings("ignore")
import akshare as ak
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
zt_dir = os.path.join(BASE, "data_store", "aux_ztpool")
os.makedirs(zt_dir, exist_ok=True)

cal = pd.read_csv(os.path.join(BASE, "data", "cache", "trading_calendar.csv"))
cal["date"] = pd.to_datetime(cal["date"])
days = cal[(cal["date"] >= "2024-01-01") & (cal["date"] <= "2026-08-07")]["date"]
done = set(os.path.basename(f).replace(".parquet", "") for f in glob.glob(os.path.join(zt_dir, "*.parquet")))
todo = [d.strftime("%Y%m%d") for d in days if d.strftime("%Y%m%d") not in done]
print(f"交易日 {len(days)} 个, 已有 {len(done)}, 待拉 {len(todo)}", flush=True)

ok, fail = 0, []
for i, d in enumerate(todo, 1):
    try:
        df = ak.stock_zt_pool_em(date=d)
        if df is not None and len(df):
            df.to_parquet(os.path.join(zt_dir, f"{d}.parquet"), index=False)
            ok += 1
        else:
            fail.append(d)
    except Exception:
        fail.append(d)
    if i % 50 == 0 or i == len(todo):
        print(f"  [{i}/{len(todo)}] ok={ok} fail={len(fail)}", flush=True)
    time.sleep(0.3)
print(f"完成: ok={ok} fail={len(fail)}", flush=True)
