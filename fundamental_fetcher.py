"""
fundamental_fetcher.py — 基本面数据批量获取 + 本地缓存

数据源: 同花顺 (THS) via akshare
  - stock_financial_abstract_ths: 财务摘要 (ROE/利润增速/营收增速/负债率/现金流)

缓存策略:
  - 每只股票一个 parquet: data/fundamental/{symbol}.parquet
  - 增量更新: 检查本地最新报告期, 只拉新数据
  - 全量刷新: --force 参数

用法:
  py fundamental_fetcher.py                # 增量更新所有缓存股票
  py fundamental_fetcher.py --force        # 全量刷新
  py fundamental_fetcher.py --check-only   # 仅检查状态
  py fundamental_fetcher.py --symbols 000001,600519  # 指定股票

输出:
  data/fundamental/{symbol}.parquet
  data/fundamental/_meta.json (更新日志)
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List, Optional

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FUND_DIR = os.path.join(BASE_DIR, "data", "fundamental")
META_PATH = os.path.join(FUND_DIR, "_meta.json")

# 限速: THS 接口每次请求间隔 (秒)
REQUEST_INTERVAL = 0.5
# 单次运行最大请求数 (防止被封)
MAX_REQUESTS_PER_RUN = 500


def ensure_dir():
    os.makedirs(FUND_DIR, exist_ok=True)


def get_cached_symbols() -> List[str]:
    """获取 data_store 中已有行情缓存的股票列表。"""
    from data_cache import get_cached_symbols as get_syms
    return get_syms()


def load_local(symbol: str) -> Optional[pd.DataFrame]:
    """加载本地缓存的基本面数据。"""
    path = os.path.join(FUND_DIR, f"{symbol}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        if len(df) == 0:
            return None
        return df
    except Exception:
        return None


def fetch_one(symbol: str) -> Optional[pd.DataFrame]:
    """
    拉取单只股票的 THS 财务摘要。

    Returns:
      DataFrame with columns: report_date, roe, profit_growth, revenue_growth,
      eps, bvps, ocf_ps, net_margin, debt_ratio, current_ratio, quick_ratio
    """
    from netgate import is_offline, OfflineViolation
    if is_offline():
        raise OfflineViolation("fundamental_fetcher.fetch_one: 离线模式禁止网络获取")
    import akshare as ak

    try:
        raw = ak.stock_financial_abstract_ths(symbol=symbol)
    except Exception as e:
        return None

    if raw is None or len(raw) == 0:
        return None

    # 标准化列名
    df = pd.DataFrame()
    df["report_date"] = pd.to_datetime(raw["报告期"])

    # 解析百分比列 (去掉 % 号)
    def parse_pct(series):
        return pd.to_numeric(
            series.astype(str).str.replace("%", "").replace("False", np.nan),
            errors="coerce"
        )

    def parse_num(series):
        return pd.to_numeric(
            series.astype(str).str.replace("亿", "").str.replace("万", "")
                  .replace("False", np.nan).str.replace(",", ""),
            errors="coerce"
        )

    # 核心因子字段
    if "净资产收益率" in raw.columns:
        df["roe"] = parse_pct(raw["净资产收益率"])
    if "净利润同比增长率" in raw.columns:
        df["profit_growth"] = parse_pct(raw["净利润同比增长率"])
    if "营业总收入同比增长率" in raw.columns:
        df["revenue_growth"] = parse_pct(raw["营业总收入同比增长率"])
    if "基本每股收益" in raw.columns:
        df["eps"] = parse_num(raw["基本每股收益"])
    if "每股净资产" in raw.columns:
        df["bvps"] = parse_num(raw["每股净资产"])
    if "每股经营现金流" in raw.columns:
        df["ocf_ps"] = parse_num(raw["每股经营现金流"])
    if "销售净利率" in raw.columns:
        df["net_margin"] = parse_pct(raw["销售净利率"])
    if "资产负债率" in raw.columns:
        df["debt_ratio"] = parse_pct(raw["资产负债率"])
    if "流动比率" in raw.columns:
        df["current_ratio"] = parse_num(raw["流动比率"])
    if "速动比率" in raw.columns:
        df["quick_ratio"] = parse_num(raw["速动比率"])
    if "扣非净利润同比增长率" in raw.columns:
        df["profit_growth_deducted"] = parse_pct(raw["扣非净利润同比增长率"])
    if "营业总收入" in raw.columns:
        df["revenue"] = parse_num(raw["营业总收入"])
    if "净利润" in raw.columns:
        df["net_profit"] = parse_num(raw["净利润"])

    df = df.sort_values("report_date").reset_index(drop=True)
    return df


def save_local(symbol: str, df: pd.DataFrame):
    """保存到本地 parquet。"""
    ensure_dir()
    path = os.path.join(FUND_DIR, f"{symbol}.parquet")
    df.to_parquet(path, index=False)


def get_latest_report_date(symbol: str) -> Optional[str]:
    """获取本地缓存的最新报告期。"""
    df = load_local(symbol)
    if df is None or "report_date" not in df.columns:
        return None
    return str(df["report_date"].max())


def batch_fetch(symbols: List[str], force: bool = False,
                check_only: bool = False, max_requests: int = MAX_REQUESTS_PER_RUN) -> dict:
    """
    批量获取基本面数据。

    Args:
      symbols: 股票代码列表
      force: 全量刷新 (忽略本地缓存)
      check_only: 仅检查状态, 不拉取
      max_requests: 单次运行最大请求数

    Returns:
      {"success": int, "skipped": int, "failed": int, "total": int}
    """
    ensure_dir()
    stats = {"success": 0, "skipped": 0, "failed": 0, "total": len(symbols)}
    request_count = 0

    print(f"基本面数据批量获取")
    print(f"  标的: {len(symbols)} 只")
    print(f"  模式: {'全量刷新' if force else '增量更新' if not check_only else '仅检查'}")
    print(f"  限速: {REQUEST_INTERVAL}s/请求, 上限: {max_requests}")
    print()

    for i, sym in enumerate(symbols):
        # 检查是否需要更新
        if not force:
            local_date = get_latest_report_date(sym)
            if local_date is not None:
                # 最新报告期在 6 个月内则跳过
                try:
                    latest = pd.Timestamp(local_date)
                    if (pd.Timestamp.now() - latest).days < 180:
                        stats["skipped"] += 1
                        continue
                except Exception:
                    pass

        if check_only:
            local_date = get_latest_report_date(sym)
            status = f"cached={local_date}" if local_date else "missing"
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(symbols)}] {status}")
            continue

        # 限速检查
        if request_count >= max_requests:
            print(f"\n  ⚠️ 达到单次运行请求上限 ({max_requests}), 剩余 {len(symbols) - i} 只下次继续")
            break

        # 拉取
        df = fetch_one(sym)
        request_count += 1

        if df is not None and len(df) > 0:
            save_local(sym, df)
            stats["success"] += 1
        else:
            stats["failed"] += 1

        # 进度
        if (i + 1) % 50 == 0 or i == len(symbols) - 1:
            elapsed = (i + 1) * REQUEST_INTERVAL
            eta = (len(symbols) - i - 1) * REQUEST_INTERVAL
            print(f"  [{i+1}/{len(symbols)}] "
                  f"✅{stats['success']} ⏭️{stats['skipped']} ❌{stats['failed']} "
                  f"| ETA: {eta/60:.0f}min", flush=True)

        time.sleep(REQUEST_INTERVAL)

    # 更新 meta
    meta = {
        "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "force" if force else "incremental",
        "stats": stats,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n完成: ✅{stats['success']} ⏭️{stats['skipped']} ❌{stats['failed']} / {stats['total']}")
    return stats


def load_fundamental_panel(symbols: List[str] = None) -> dict:
    """
    加载所有已缓存的基本面数据为 panel。

    Returns:
      {symbol: DataFrame} — 每个 DataFrame 含 report_date + 因子列
    """
    ensure_dir()
    if symbols is None:
        # 扫描目录
        symbols = [f.replace(".parquet", "") for f in os.listdir(FUND_DIR)
                   if f.endswith(".parquet") and not f.startswith("_")]

    panel = {}
    for sym in symbols:
        df = load_local(sym)
        if df is not None and len(df) >= 4:  # 至少4个季度
            panel[sym] = df

    return panel


def compute_fundamental_factors(panel: dict, price_data: dict,
                                as_of_date) -> dict:
    """
    计算截面基本面因子 (point-in-time, 避免前视偏差)。

    对每只股票, 取 as_of_date 之前最新已发布的财报数据。
    财报发布延迟: 一季报(4月底), 半年报(8月底), 三季报(10月底), 年报(4月底)
    保守估计: 报告期 + 2个月 作为可用日期。

    Args:
      panel: {symbol: fundamental_df}
      price_data: {symbol: DataFrame with close} — 用于计算 PB
      as_of_date: 截止日期

    Returns:
      {symbol: {"roe": float, "profit_growth": float, ...}}
    """
    as_of = pd.Timestamp(as_of_date)
    results = {}

    for sym, fund_df in panel.items():
        if "report_date" not in fund_df.columns:
            continue

        # Point-in-time: 报告期 + 2个月 <= as_of_date
        # (保守估计财报发布延迟)
        available = fund_df.copy()
        available["available_date"] = available["report_date"] + pd.DateOffset(months=2)
        mask = available["available_date"] <= as_of
        available = available[mask]

        if len(available) == 0:
            continue

        # 取最新一期
        latest = available.iloc[-1]
        factors = {}

        # ROE (年化: 如果是半年报则×2, 一季报×4)
        roe = latest.get("roe", np.nan)
        if not np.isnan(roe) if isinstance(roe, float) else pd.notna(roe):
            month = latest["report_date"].month
            if month == 3:
                roe = roe * 4  # 一季报年化
            elif month == 6:
                roe = roe * 2  # 半年报年化
            elif month == 9:
                roe = roe * 4 / 3  # 三季报年化
            factors["fund_roe"] = float(roe)

        # 利润增速
        pg = latest.get("profit_growth", np.nan)
        if pd.notna(pg):
            factors["fund_profit_growth"] = float(pg)

        # 营收增速
        rg = latest.get("revenue_growth", np.nan)
        if pd.notna(rg):
            factors["fund_revenue_growth"] = float(rg)

        # 负债率
        dr = latest.get("debt_ratio", np.nan)
        if pd.notna(dr):
            factors["fund_debt_ratio"] = float(dr)

        # 净利率
        nm = latest.get("net_margin", np.nan)
        if pd.notna(nm):
            factors["fund_net_margin"] = float(nm)

        # 每股经营现金流
        ocf = latest.get("ocf_ps", np.nan)
        if pd.notna(ocf):
            factors["fund_ocf_ps"] = float(ocf)

        # PB = price / bvps
        bvps = latest.get("bvps", np.nan)
        if pd.notna(bvps) and bvps > 0 and sym in price_data:
            pdf = price_data[sym]
            close_at_date = pdf[pdf["date"] <= as_of]
            if len(close_at_date) > 0:
                price = float(close_at_date["close"].iloc[-1])
                factors["fund_pb"] = price / bvps

        # 扣非利润增速
        pgd = latest.get("profit_growth_deducted", np.nan)
        if pd.notna(pgd):
            factors["fund_profit_growth_ded"] = float(pgd)

        # ===== Value Factors (price-relative) =====
        # Get close price on as_of_date for ratio computations
        close_price = np.nan
        if sym in price_data:
            pdf = price_data[sym]
            close_at_date = pdf[pdf["date"] <= as_of]
            if len(close_at_date) > 0:
                close_price = float(close_at_date["close"].iloc[-1])

        if pd.notna(close_price) and close_price > 0:
            # EP (Earnings Yield) = eps / price
            eps_val = latest.get("eps", np.nan)
            if pd.notna(eps_val):
                factors["fund_ep"] = float(eps_val) / close_price

            # BP (Book-to-Price) = bvps / price
            if pd.notna(bvps) and bvps > 0:
                factors["fund_bp"] = float(bvps) / close_price

            # SP (Sales-to-Price) = revenue_per_share / price
            # Derive revenue_ps from revenue / shares, where shares = net_profit / eps
            revenue_val = latest.get("revenue", np.nan)
            net_profit_val = latest.get("net_profit", np.nan)
            if pd.notna(revenue_val) and pd.notna(eps_val) and pd.notna(net_profit_val):
                if abs(eps_val) > 1e-9 and abs(net_profit_val) > 1e-9:
                    shares_est = net_profit_val / eps_val  # estimated shares outstanding
                    if shares_est > 0:
                        revenue_ps = revenue_val / shares_est
                        factors["fund_sp"] = float(revenue_ps) / close_price

            # OCF Yield = ocf_ps / price
            if pd.notna(ocf):
                factors["fund_ocf_yield"] = float(ocf) / close_price

            # Accruals = (eps - ocf_ps) / bvps (proxy: per-share basis)
            if pd.notna(eps_val) and pd.notna(ocf) and pd.notna(bvps) and bvps > 0:
                factors["fund_accruals"] = float(eps_val - ocf) / float(bvps)

        if factors:
            results[sym] = factors

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基本面数据批量获取")
    parser.add_argument("--force", action="store_true", help="全量刷新")
    parser.add_argument("--check-only", action="store_true", help="仅检查状态")
    parser.add_argument("--symbols", type=str, default="", help="指定股票(逗号分隔)")
    parser.add_argument("--max-requests", type=int, default=MAX_REQUESTS_PER_RUN,
                        help=f"单次最大请求数 (默认{MAX_REQUESTS_PER_RUN})")
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = get_cached_symbols()

    print(f"标的来源: {'指定' if args.symbols else 'data_store缓存'} ({len(symbols)} 只)")
    batch_fetch(symbols, force=args.force, check_only=args.check_only,
                max_requests=args.max_requests)
