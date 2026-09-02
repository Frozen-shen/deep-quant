"""临时脚本: 依次拉取高管增减持/回购/市值快照/涨停池历史 (验证后删除)"""
import akshare_proxy_patch
akshare_proxy_patch.install_patch(
    "101.201.173.125",
    auth_token="[REDACTED]",
    retry=30,
    hook_domains=[
        "fund.eastmoney.com", "push2.eastmoney.com", "push2his.eastmoney.com",
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

def log(msg):
    print(msg, flush=True)

# 1. 高管增减持 (全市场全历史, 一次调用)
try:
    df = ak.stock_ggcg_em(symbol="全部")
    if df is not None and len(df):
        df.to_parquet(os.path.join(BASE, "data_store", "aux_ggcg.parquet"), index=False)
        log(f"[高管增减持] OK {len(df)} 行 -> data_store/aux_ggcg.parquet")
except Exception as e:
    log(f"[高管增减持] FAIL {str(e)[:80]}")

# 2. 回购 (全市场, 一次调用)
try:
    df = ak.stock_repurchase_em()
    if df is not None and len(df):
        df.to_parquet(os.path.join(BASE, "data_store", "aux_repurchase.parquet"), index=False)
        log(f"[回购] OK {len(df)} 行 -> data_store/aux_repurchase.parquet")
except Exception as e:
    log(f"[回购] FAIL {str(e)[:80]}")

# 3. 全市场市值/换手率快照 (一次调用 8-15 积分)
try:
    df = ak.stock_zh_a_spot_em()
    if df is not None and len(df):
        date = df["数据日期"].iloc[0] if "数据日期" in df.columns else time.strftime("%Y%m%d")
        path = os.path.join(BASE, "data_store", f"aux_realtime_{date}.parquet")
        df.to_parquet(path, index=False)
        log(f"[市值快照] OK {len(df)} 只 -> {path}")
except Exception as e:
    log(f"[市值快照] FAIL {str(e)[:80]}")

# 4. 涨停池历史: 2024-01-01 至今, 按交易日拉 (1积分/日), 断点续传
zt_dir = os.path.join(BASE, "data_store", "aux_ztpool")
os.makedirs(zt_dir, exist_ok=True)
cal = pd.read_csv(os.path.join(BASE, "data", "cache", "trading_calendar.csv"))
cal["date"] = pd.to_datetime(cal["date"])
days = cal[(cal["date"] >= "2024-01-01") & (cal["date"] <= "2026-08-07")]["date"]
done = set(os.path.basename(f).replace(".parquet", "") for f in glob.glob(os.path.join(zt_dir, "*.parquet")))
todo = [d.strftime("%Y%m%d") for d in days if d.strftime("%Y%m%d") not in done]
log(f"[涨停池] 交易日 {len(days)} 个, 已有 {len(done)}, 待拉 {len(todo)}")
ok, fail = 0, []
for i, d in enumerate(todo, 1):
    try:
        df = ak.stock_zt_pool_em(date=d)
        if df is not None and len(df):
            df.to_parquet(os.path.join(zt_dir, f"{d}.parquet"), index=False)
            ok += 1
        else:
            fail.append(d)  # 空数据也记录, 避免反复拉
    except Exception:
        fail.append(d)
    if i % 50 == 0 or i == len(todo):
        log(f"  [涨停池 {i}/{len(todo)}] ok={ok} fail={len(fail)}")
    time.sleep(0.3)
log(f"[涨停池] 完成: ok={ok} fail={len(fail)}")

log("全部完成")
