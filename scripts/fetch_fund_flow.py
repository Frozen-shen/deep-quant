"""
批量拉取个股资金流数据 → data/factor_cache/money_flow/{symbol}.parquet

策略:
  - 主要方式: 逐股票调用 stock_individual_fund_flow 获取历史 (最完整)
  - 批量快照: stock_individual_fund_flow_rank 获取当日全市场排名 (仅当天)
  - 历史回填: 对每个交易日调用 rank API, 拼接为历史序列

数据列:
  date, main_net_inflow, large_net_inflow, medium_net_inflow, small_net_inflow, total_turnover

用法:
  python scripts/fetch_fund_flow.py                  # 默认拉取250个交易日
  python scripts/fetch_fund_flow.py --days 120       # 拉取120个交易日
  python scripts/fetch_fund_flow.py --resume         # 断点续传
  python scripts/fetch_fund_flow.py --mode rank      # 用rank API批量拉 (快但只有近期)
  python scripts/fetch_fund_flow.py --mode perstock  # 逐只拉取 (慢但历史完整)
  python scripts/fetch_fund_flow.py --symbols 600519 000858  # 指定股票
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

CACHE_DIR = os.path.join(BASE_DIR, "data", "factor_cache", "money_flow")
os.makedirs(CACHE_DIR, exist_ok=True)

RATE_LIMIT = 0.4  # 请求间隔 (秒)


def detect_market(symbol: str) -> str:
    """根据代码前缀判断市场: 6→sh, 0/3→sz, 其他→sh"""
    if symbol.startswith("6"):
        return "sh"
    elif symbol.startswith("0") or symbol.startswith("3"):
        return "sz"
    else:
        return "sh"


def get_universe_symbols() -> List[str]:
    """获取股票池: 优先从 data_cache 获取已缓存股票。"""
    data_cache_dir = os.path.join(BASE_DIR, "data_cache")
    if os.path.exists(data_cache_dir):
        syms = sorted([
            f.replace(".parquet", "") for f in os.listdir(data_cache_dir)
            if f.endswith(".parquet") and f[0].isdigit()
        ])
        if syms:
            return syms

    # 回退: 从 data_store 获取
    data_store = os.path.join(BASE_DIR, "data_store")
    if os.path.exists(data_store):
        syms = sorted([
            f.replace(".parquet", "") for f in os.listdir(data_store)
            if f.endswith(".parquet") and f[0].isdigit()
        ])
        if syms:
            return syms

    return []


def get_cached_symbols() -> List[str]:
    """获取已有资金流缓存的股票。"""
    return sorted([
        f.replace(".parquet", "") for f in os.listdir(CACHE_DIR)
        if f.endswith(".parquet")
    ])


# ════════════════════════════════════════
#  方式1: 逐只拉取 (stock_individual_fund_flow)
# ════════════════════════════════════════

def fetch_per_stock(symbol: str, max_retries: int = 2) -> Optional[pd.DataFrame]:
    """
    拉取单只股票的资金流历史。

    返回 DataFrame: date, main_net_inflow, large_net_inflow,
                     medium_net_inflow, small_net_inflow, total_turnover
    """
    import akshare as ak
    import warnings
    warnings.filterwarnings("ignore")

    market = detect_market(symbol)

    for attempt in range(max_retries + 1):
        try:
            df = ak.stock_individual_fund_flow(stock=symbol, market=market)
            if df is None or len(df) == 0:
                return None

            # 标准化列名 (akshare 返回中文列名)
            col_map = {
                "日期": "date",
                "主力净流入-净额": "main_net_inflow",
                "超大单净流入-净额": "super_large_net_inflow",
                "大单净流入-净额": "large_order_net_inflow",
                "中单净流入-净额": "medium_net_inflow",
                "小单净流入-净额": "small_net_inflow",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

            if "date" not in df.columns:
                # 尝试其他可能的日期列名
                for c in df.columns:
                    if "日期" in c or "date" in c.lower():
                        df = df.rename(columns={c: "date"})
                        break

            if "date" not in df.columns:
                return None

            # 计算 large_net_inflow = 超大单 + 大单
            if "super_large_net_inflow" in df.columns and "large_order_net_inflow" in df.columns:
                df["large_net_inflow"] = (
                    pd.to_numeric(df["super_large_net_inflow"], errors="coerce") +
                    pd.to_numeric(df["large_order_net_inflow"], errors="coerce")
                )
            elif "large_order_net_inflow" in df.columns:
                df["large_net_inflow"] = pd.to_numeric(df["large_order_net_inflow"], errors="coerce")
            else:
                df["large_net_inflow"] = np.nan

            # 数值转换
            for col in ["main_net_inflow", "medium_net_inflow", "small_net_inflow"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # 计算 total_turnover = 主力 + 中单 + 小单 (近似总成交额)
            # 更准确: 用绝对值之和 / 2, 但这里用净流入的绝对值之和作为近似
            # 实际上 total_turnover 应该从成交额字段获取
            # 尝试从原始数据中找成交额
            turnover_candidates = [c for c in df.columns if "成交额" in c or "turnover" in c.lower()]
            if turnover_candidates:
                df["total_turnover"] = pd.to_numeric(df[turnover_candidates[0]], errors="coerce")
            else:
                # 回退: 用 |主力净流入| + |中单| + |小单| 近似 (不精确但可用)
                parts = []
                for col in ["main_net_inflow", "medium_net_inflow", "small_net_inflow"]:
                    if col in df.columns:
                        parts.append(df[col].abs())
                if parts:
                    df["total_turnover"] = sum(parts)
                else:
                    df["total_turnover"] = np.nan

            # 选择最终列
            keep = ["date", "main_net_inflow", "large_net_inflow",
                    "medium_net_inflow", "small_net_inflow", "total_turnover"]
            keep = [c for c in keep if c in df.columns]
            df = df[keep].copy()

            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            return df

        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
            else:
                print(f"  [WARN] {symbol} 拉取失败: {e}", flush=True)
                return None

    return None


def run_perstock_mode(symbols: List[str], resume: bool = True):
    """逐只拉取模式: 每只股票获取完整历史。"""
    total = len(symbols)
    done = 0
    skipped = 0
    failed = 0

    print(f"[perstock] 开始拉取 {total} 只股票资金流...", flush=True)

    for i, sym in enumerate(symbols):
        cache_path = os.path.join(CACHE_DIR, f"{sym}.parquet")

        # 断点续传: 跳过已有缓存
        if resume and os.path.exists(cache_path):
            skipped += 1
            continue

        df = fetch_per_stock(sym)
        if df is not None and len(df) > 0:
            df.to_parquet(cache_path, index=False)
            done += 1
        else:
            failed += 1

        # 进度
        if (i + 1) % 20 == 0 or i == total - 1:
            print(f"  进度: {i+1}/{total}  成功={done} 跳过={skipped} 失败={failed}",
                  flush=True)

        time.sleep(RATE_LIMIT)

    print(f"\n[perstock] 完成: 成功={done}, 跳过={skipped}, 失败={failed}", flush=True)


# ════════════════════════════════════════
#  方式2: 批量排名 (stock_individual_fund_flow_rank)
# ════════════════════════════════════════

def fetch_rank_snapshot(indicator: str = "今日") -> Optional[pd.DataFrame]:
    """
    获取全市场资金流排名快照 (一次调用获取所有股票)。

    indicator: "今日", "3日", "5日", "10日"
    """
    import akshare as ak
    import warnings
    warnings.filterwarnings("ignore")

    try:
        df = ak.stock_individual_fund_flow_rank(indicator=indicator)
        if df is None or len(df) == 0:
            return None

        # 标准化
        col_map = {
            "代码": "symbol",
            "名称": "name",
            "主力净流入-净额": "main_net_inflow",
            "超大单净流入-净额": "super_large_net_inflow",
            "大单净流入-净额": "large_order_net_inflow",
            "中单净流入-净额": "medium_net_inflow",
            "小单净流入-净额": "small_net_inflow",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if "symbol" not in df.columns:
            return None

        df["symbol"] = df["symbol"].astype(str).str.zfill(6)

        # 计算 large_net_inflow
        if "super_large_net_inflow" in df.columns and "large_order_net_inflow" in df.columns:
            df["large_net_inflow"] = (
                pd.to_numeric(df["super_large_net_inflow"], errors="coerce") +
                pd.to_numeric(df["large_order_net_inflow"], errors="coerce")
            )
        else:
            df["large_net_inflow"] = np.nan

        for col in ["main_net_inflow", "medium_net_inflow", "small_net_inflow"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # total_turnover 近似
        turnover_candidates = [c for c in df.columns if "成交额" in c or "turnover" in c.lower()]
        if turnover_candidates:
            df["total_turnover"] = pd.to_numeric(df[turnover_candidates[0]], errors="coerce")
        else:
            parts = []
            for col in ["main_net_inflow", "medium_net_inflow", "small_net_inflow"]:
                if col in df.columns:
                    parts.append(df[col].abs())
            df["total_turnover"] = sum(parts) if parts else np.nan

        return df

    except Exception as e:
        print(f"  [WARN] rank API 失败: {e}", flush=True)
        return None


def run_rank_mode(days: int = 250, resume: bool = True):
    """
    批量排名模式: 用 rank API 获取当日快照, 按股票拆分存储。

    注意: rank API 只能获取近期数据 (今日/3日/5日/10日),
    无法回溯任意历史日期。适合每日增量更新。
    """
    print(f"[rank] 获取全市场资金流排名快照...", flush=True)

    # 获取今日数据
    df = fetch_rank_snapshot("今日")
    if df is None or len(df) == 0:
        print("[rank] 无法获取数据, 退出", flush=True)
        return

    today = datetime.now().strftime("%Y-%m-%d")
    n_stocks = 0
    n_updated = 0

    for _, row in df.iterrows():
        sym = row["symbol"]
        cache_path = os.path.join(CACHE_DIR, f"{sym}.parquet")

        new_row = {
            "date": today,
            "main_net_inflow": row.get("main_net_inflow", np.nan),
            "large_net_inflow": row.get("large_net_inflow", np.nan),
            "medium_net_inflow": row.get("medium_net_inflow", np.nan),
            "small_net_inflow": row.get("small_net_inflow", np.nan),
            "total_turnover": row.get("total_turnover", np.nan),
        }

        if os.path.exists(cache_path):
            # 追加到已有文件
            existing = pd.read_parquet(cache_path)
            existing["date"] = pd.to_datetime(existing["date"])

            # 检查是否已有今天的数据
            if today in existing["date"].dt.strftime("%Y-%m-%d").values:
                n_stocks += 1
                continue

            new_df = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
            new_df = new_df.sort_values("date").reset_index(drop=True)
        else:
            new_df = pd.DataFrame([new_row])

        new_df.to_parquet(cache_path, index=False)
        n_stocks += 1
        n_updated += 1

    print(f"[rank] 完成: {n_stocks} 只股票, 更新 {n_updated} 只", flush=True)
    print(f"[rank] 注意: rank API 仅获取当日快照, 历史数据请用 perstock 模式", flush=True)


# ════════════════════════════════════════
#  主入口
# ════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="拉取个股资金流数据")
    parser.add_argument("--days", type=int, default=250, help="回溯交易日数 (仅perstock模式)")
    parser.add_argument("--resume", action="store_true", help="断点续传, 跳过已有缓存")
    parser.add_argument("--mode", choices=["perstock", "rank"], default="perstock",
                        help="拉取模式: perstock=逐只完整历史, rank=批量当日快照")
    parser.add_argument("--symbols", nargs="*", default=None, help="指定股票代码")
    parser.add_argument("--limit", type=int, default=None, help="限制拉取数量 (调试用)")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("  资金流数据拉取", flush=True)
    print(f"  模式: {args.mode}  断点续传: {args.resume}", flush=True)
    print("=" * 60, flush=True)

    if args.mode == "perstock":
        # 获取股票列表
        if args.symbols:
            symbols = args.symbols
        else:
            symbols = get_universe_symbols()
            if args.limit:
                symbols = symbols[:args.limit]

        if not symbols:
            print("[ERROR] 未找到股票池, 请先运行 data_cache.py --fetch", flush=True)
            sys.exit(1)

        print(f"  股票池: {len(symbols)} 只", flush=True)
        run_perstock_mode(symbols, resume=args.resume)

    elif args.mode == "rank":
        run_rank_mode(days=args.days, resume=args.resume)
