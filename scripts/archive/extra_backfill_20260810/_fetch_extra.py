"""
临时脚本: 东财高价值数据全量拉取 (分红送配/北向历史/板块资金流/情绪池/质押/快报/调研)
落盘: data_store/aux_* 系列, 与 aux 面板格式一致
"""
import sys, os, warnings, time, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eastmoney_proxy
eastmoney_proxy.setup_from_config()
warnings.filterwarnings("ignore")
import akshare as ak
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(msg, flush=True)


def trading_days(start, end):
    cal = pd.read_csv(os.path.join(BASE, "data", "cache", "trading_calendar.csv"))
    cal["date"] = pd.to_datetime(cal["date"])
    return cal[(cal["date"] >= start) & (cal["date"] <= end)]["date"]


def fetch_series(name, out_dir, days, fn, fmt="%Y%m%d"):
    """按日拉取, 断点续传。"""
    os.makedirs(out_dir, exist_ok=True)
    done = set(os.path.basename(f).replace(".parquet", "") for f in glob.glob(os.path.join(out_dir, "*.parquet")))
    todo = [d for d in days if d.strftime(fmt) not in done]
    log(f"[{name}] 待拉 {len(todo)} 天")
    ok, fail = 0, []
    for i, d in enumerate(todo, 1):
        ds = d.strftime(fmt)
        try:
            df = fn(ds)
            if df is not None and len(df):
                df.to_parquet(os.path.join(out_dir, f"{ds}.parquet"), index=False)
                ok += 1
            else:
                fail.append(ds)
        except Exception:
            fail.append(ds)
        if i % 50 == 0 or i == len(todo):
            log(f"  [{name} {i}/{len(todo)}] ok={ok} fail={len(fail)}")
        time.sleep(0.3)
    log(f"[{name}] 完成 ok={ok} fail={len(fail)}")


def main():
    # ── 1. 北向资金历史 (一次全量 2015 至今) ──
    try:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
        if df is not None and len(df):
            df.to_parquet(os.path.join(BASE, "data", "smart_money", "northbound_hist.parquet"), index=False)
            log(f"[北向历史] OK {len(df)} 行 -> northbound_hist.parquet")
    except Exception as e:
        log(f"[北向历史] FAIL {str(e)[:60]}")

    # ── 2. 分红送配 (2016Q1 ~ 2026Q2) ──
    fh_dir = os.path.join(BASE, "data_store", "aux_fhps")
    os.makedirs(fh_dir, exist_ok=True)
    quarters = []
    for y in range(2016, 2027):
        for q in ("0331", "0630", "0930", "1231"):
            if f"{y}{q}" <= "20260630":
                quarters.append(f"{y}{q}")
    done_fh = set(os.path.basename(f).replace(".parquet", "") for f in glob.glob(os.path.join(fh_dir, "*.parquet")))
    ok_fh = 0
    for q in quarters:
        if q in done_fh:
            continue
        try:
            df = ak.stock_fhps_em(date=q)
            if df is not None and len(df):
                df.to_parquet(os.path.join(fh_dir, f"{q}.parquet"), index=False)
                ok_fh += 1
        except Exception:
            pass
        time.sleep(0.3)
    log(f"[分红送配] 完成 ok={ok_fh}/42 季度")

    # ── 3. 板块资金流 (近 250 交易日) ──
    days = trading_days("2025-07-01", "2026-08-07")
    fetch_series("板块资金流", os.path.join(BASE, "data_store", "aux_sector_flow"), days,
                 lambda ds: ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流"))

    # ── 4. 情绪池三件套 (近 2 周) ──
    days2w = trading_days("2026-07-20", "2026-08-07")
    for name, fn, sub in [
        ("跌停池", lambda ds: ak.stock_zt_pool_dtgc_em(date=ds), "aux_ztpool_dtgc"),
        ("强势池", lambda ds: ak.stock_zt_pool_strong_em(date=ds), "aux_ztpool_strong"),
        ("昨日涨停", lambda ds: ak.stock_zt_pool_previous_em(date=ds), "aux_ztpool_prev"),
    ]:
        fetch_series(name, os.path.join(BASE, "data_store", sub), days2w, fn)

    # ── 5. 股权质押 (近 24 个月末快照) ──
    gp_dir = os.path.join(BASE, "data_store", "aux_gpzy")
    os.makedirs(gp_dir, exist_ok=True)
    cal = pd.read_csv(os.path.join(BASE, "data", "cache", "trading_calendar.csv"))
    cal["date"] = pd.to_datetime(cal["date"])
    cal = cal[cal["date"] >= "2024-08-01"]
    month_last = cal.groupby(cal["date"].dt.to_period("M"))["date"].max()
    ok_gp = 0
    for d in month_last:
        ds = d.strftime("%Y%m%d")
        if os.path.exists(os.path.join(gp_dir, f"{ds}.parquet")):
            continue
        try:
            df = ak.stock_gpzy_pledge_ratio_em(date=ds)
            if df is not None and len(df):
                df.to_parquet(os.path.join(gp_dir, f"{ds}.parquet"), index=False)
                ok_gp += 1
        except Exception:
            pass
        time.sleep(0.3)
    log(f"[股权质押] 完成 ok={ok_gp}/{len(month_last)} 月")

    # ── 6. 业绩快报 (2016Q1~2026Q2) ──
    yj_dir = os.path.join(BASE, "data_store", "aux_yjkb")
    os.makedirs(yj_dir, exist_ok=True)
    ok_yj = 0
    for q in quarters:
        if os.path.exists(os.path.join(yj_dir, f"{q}.parquet")):
            continue
        try:
            df = ak.stock_yjkb_em(date=q)
            if df is not None and len(df):
                df.to_parquet(os.path.join(yj_dir, f"{q}.parquet"), index=False)
                ok_yj += 1
        except Exception:
            pass
        time.sleep(0.3)
    log(f"[业绩快报] 完成 ok={ok_yj}")

    # ── 7. 机构调研 (近 60 交易日) ──
    days60 = trading_days("2026-05-01", "2026-08-07")
    fetch_series("机构调研", os.path.join(BASE, "data_store", "aux_jgdy"), days60,
                 lambda ds: ak.stock_jgdy_tj_em(date=ds))

    log("全部完成!")


if __name__ == "__main__":
    main()
