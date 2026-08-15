"""验证东财高价值接口 (带代理)"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eastmoney_proxy
eastmoney_proxy.setup_from_config()
warnings.filterwarnings("ignore")
import akshare as ak


def t(name, fn):
    try:
        df = fn()
        n = 0 if df is None else len(df)
        cols = list(df.columns)[:5] if (df is not None and len(df)) else []
        print(f"[{name}] OK {n}行 列:{cols}", flush=True)
    except Exception as e:
        print(f"[{name}] FAIL {str(e)[:60]}", flush=True)


t("分红送配", lambda: ak.stock_fhps_em(date="20260630"))
t("股权质押", lambda: ak.stock_gpzy_pledge_ratio_em(symbol="全部股票"))
t("机构调研统计", lambda: ak.stock_jgdy_tj_em(date="20260807"))
t("板块资金流", lambda: ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流"))
t("北向资金历史", lambda: ak.stock_hsgt_hist_em(symbol="北向资金"))
t("业绩快报", lambda: ak.stock_yjkb_em(date="20260630"))
