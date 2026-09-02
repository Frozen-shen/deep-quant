"""临时脚本: 补全业绩预告/分析师预期/ST列表 (验证后删除, 走 cheapproxy 代理)"""
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
import os, warnings, sys
os.environ.pop("HTTP_PROXY", None); os.environ.pop("HTTPS_PROXY", None)
warnings.filterwarnings("ignore")
import akshare as ak

BASE = os.path.dirname(os.path.abspath(__file__))

def log(msg):
    print(msg, flush=True)

# 1. 业绩预告: 补 2024Q1 ~ 2026Q2 共 10 个季度 -> data/pead_cache/forecast_{q}.parquet
pead_dir = os.path.join(BASE, "data", "pead_cache")
quarters = [f"{y}{m}" for y in (2024, 2025, 2026) for m in ("0331", "0630", "0930", "1231")]
quarters = [q for q in quarters if q <= "20260630"]
ok = 0
for q in quarters:
    path = os.path.join(pead_dir, f"forecast_{q}.parquet")
    if os.path.exists(path):
        continue
    try:
        df = ak.stock_yjyg_em(date=q)
        if df is not None and len(df):
            df.to_parquet(path, index=False)
            ok += 1
            log(f"[业绩预告 {q}] OK {len(df)} 行")
        else:
            log(f"[业绩预告 {q}] 空数据")
    except Exception as e:
        log(f"[业绩预告 {q}] FAIL {str(e)[:60]}")
log(f"业绩预告: 补 {ok} 个季度")

# 2. 分析师一致预期 (最新快照) -> data/smart_money/analyst_consensus_{date}.parquet
sm_dir = os.path.join(BASE, "data", "smart_money")
try:
    df = ak.stock_profit_forecast_em()
    if df is not None and len(df):
        df.to_parquet(os.path.join(sm_dir, "analyst_consensus_20260807.parquet"), index=False)
        log(f"[分析师预期] OK {len(df)} 行 -> analyst_consensus_20260807.parquet")
except Exception as e:
    log(f"[分析师预期] FAIL {str(e)[:60]}")

# 3. ST 名单 -> data/cache/st_list_20260807.parquet (universe 过滤用)
cache_dir = os.path.join(BASE, "data", "cache")
try:
    df = ak.stock_zh_a_st_em()
    if df is not None and len(df):
        df.to_parquet(os.path.join(cache_dir, "st_list_20260807.parquet"), index=False)
        log(f"[ST列表] OK {len(df)} 只 -> st_list_20260807.parquet")
except Exception as e:
    log(f"[ST列表] FAIL {str(e)[:60]}")

log("完成")
