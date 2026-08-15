"""验证第二批接口: 股权质押签名/情绪股池/基金持仓"""
import sys, os, warnings, inspect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eastmoney_proxy
eastmoney_proxy.setup_from_config()
warnings.filterwarnings("ignore")
import akshare as ak

# 1. 股权质押正确签名
try:
    print("股权质押签名:", str(inspect.signature(ak.stock_gpzy_pledge_ratio_em))[:80], flush=True)
    df = ak.stock_gpzy_pledge_ratio_em()
    print(f"[股权质押] OK {0 if df is None else len(df)}行", flush=True)
except Exception as e:
    print(f"[股权质押] FAIL {str(e)[:80]}", flush=True)

# 2. 情绪股池
for name, fn in [
    ("跌停池", lambda: ak.stock_zt_pool_dtgc_em(date="20260807")),
    ("强势池", lambda: ak.stock_zt_pool_strong_em(date="20260807")),
    ("昨日涨停", lambda: ak.stock_zt_pool_previous_em(date="20260807")),
]:
    try:
        df = fn()
        print(f"[{name}] OK {0 if df is None else len(df)}行", flush=True)
    except Exception as e:
        print(f"[{name}] FAIL {str(e)[:60]}", flush=True)

# 3. 基金持仓汇总
try:
    df = ak.stock_report_fund_hold(symbol="基金重仓股", date="20260630")
    print(f"[基金重仓股] OK {0 if df is None else len(df)}行 列:{list(df.columns)[:4] if df is not None else ''}", flush=True)
except Exception as e:
    print(f"[基金重仓股] FAIL {str(e)[:60]}", flush=True)
