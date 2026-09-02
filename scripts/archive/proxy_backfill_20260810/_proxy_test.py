"""临时测试脚本: 验证 akshare-proxy-patch 代理插件能否救活东财接口 (验证后删除)"""
# ⚠️ 插件引入必须放在最顶部, 在 efinance/akshare 之前!
import akshare_proxy_patch
akshare_proxy_patch.install_patch(
    "101.201.173.125",
    auth_token="[REDACTED]",
    retry=30,
    hook_domains=[
        "fund.eastmoney.com",
        "push2.eastmoney.com",
        "push2his.eastmoney.com",
        "emweb.securities.eastmoney.com",
        "searchapi.eastmoney.com/api/suggest/get",
    ],
    fast=True,
)

import os, time
os.environ.pop("HTTP_PROXY", None); os.environ.pop("HTTPS_PROXY", None)
from efinance.shared.tickflow_prompt import session
session.trust_env = False
import efinance as ef

# 1. 资金流 (之前稳定但现在断的接口)
try:
    df = ef.stock.get_history_bill("600519")
    print(f"[资金流] OK {len(df)}条 {df['日期'].iloc[0]}~{df['日期'].iloc[-1]}")
except Exception as e:
    print(f"[资金流] FAIL {str(e)[:100]}")

# 2. 日线 (kline 接口, 之前间歇断)
try:
    df = ef.stock.get_quote_history("600519", beg="20260701", end="20260731")
    print(f"[日线] OK {len(df)}条 最新{df['日期'].iloc[-1]}")
except Exception as e:
    print(f"[日线] FAIL {str(e)[:100]}")

# 3. 15分钟 (kline klt=15)
try:
    df = ef.stock.get_quote_history("600519", beg="20260728", end="20260731", klt=15)
    print(f"[15m] OK {len(df)}条")
except Exception as e:
    print(f"[15m] FAIL {str(e)[:100]}")
