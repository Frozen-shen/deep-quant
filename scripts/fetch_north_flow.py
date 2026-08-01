"""
拉取北向资金持仓数据 → data/factor_cache/north_flow/{symbol}.parquet

数据源:
  - ak.stock_hsgt_hold_stock_em(market="北向", indicator="今日排行") — 当日北向持仓排名
  - ak.stock_hsgt_individual_em(symbol) — 个股北向持仓历史

覆盖范围: 仅沪深港通标的 (~1500只), 非标的股票无数据

数据列:
  date, holding_shares, holding_ratio, holding_value

用法:
  python scripts/fetch_north_flow.py              # 拉取所有沪深港通股票
  python scripts/fetch_north_flow.py --resume     # 断点续传
  python scripts/fetch_north_flow.py --snapshot   # 仅获取当日快照 (快)
  python scripts/fetch_north_flow.py --symbols 600519 000858
  python scripts/fetch_north_flow.py --limit 50   # 限制数量 (调试)
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import List, Optional

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

CACHE_DIR = os.path.join(BASE_DIR, "data", "factor_cache", "north_flow")
os.makedirs(CACHE_DIR, exist_ok=True)

RATE_LIMIT = 0.4  # 请求间隔 (秒)


def get_cached_symbols() -> List[str]:
    """获取已有北向缓存的股票。"""
    return sorted([
        f.replace(".parquet", "") for f in os.listdir(CACHE_DIR)
        if f.endswith(".parquet")
    ])


# ════════════════════════════════════════
#  获取沪深港通股票列表
# ════════════════════════════════════════

def get_hsgt_stock_list() -> List[str]:
    """
    获取当前沪深港通标的列表。

    优先用 stock_hsgt_hold_stock_em 的排名列表 (自带代码),
    回退到 stock_shse_summary / stock_szse_summary。
    """
    import akshare as ak
    import warnings
    warnings.filterwarnings("ignore")

    symbols = set()

    # 方式1: 从北向持仓排名获取 (最直接)
    try:
        print("  获取北向持仓排名...", flush=True)
        df = ak.stock_hsgt_hold_stock_em(market="北向", indicator="今日排行")
        if df is not None and len(df) > 0:
            # 找到代码列
            code_col = None
            for c in df.columns:
                if "代码" in c or "code" in c.lower():
                    code_col = c
                    break
            if code_col:
                codes = df[code_col].astype(str).str.zfill(6).tolist()
                symbols.update(codes)
                print(f"  北向排名: {len(codes)} 只", flush=True)
    except Exception as e:
        print(f"  [WARN] 北向排名获取失败: {e}", flush=True)

    # 方式2: 沪股通 + 深股通分别获取
    if len(symbols) < 100:
        for market in ["沪股通", "深股通"]:
            try:
                df = ak.stock_hsgt_hold_stock_em(market=market, indicator="今日排行")
                if df is not None and len(df) > 0:
                    code_col = None
                    for c in df.columns:
                        if "代码" in c or "code" in c.lower():
                            code_col = c
                            break
                    if code_col:
                        codes = df[code_col].astype(str).str.zfill(6).tolist()
                        symbols.update(codes)
                        print(f"  {market}: {len(codes)} 只", flush=True)
                time.sleep(RATE_LIMIT)
            except Exception as e:
                print(f"  [WARN] {market} 获取失败: {e}", flush=True)

    # 方式3: 回退到 data_cache 中的股票 (过滤: 只取可能被纳入沪深港通的)
    if len(symbols) < 100:
        print("  [WARN] 沪深港通列表获取不完整, 回退到本地股票池", flush=True)
        data_cache_dir = os.path.join(BASE_DIR, "data_cache")
        if os.path.exists(data_cache_dir):
            local = [f.replace(".parquet", "") for f in os.listdir(data_cache_dir)
                     if f.endswith(".parquet") and f[0].isdigit()]
            symbols.update(local)

    return sorted(symbols)


# ════════════════════════════════════════
#  个股北向持仓历史
# ════════════════════════════════════════

def fetch_individual_history(symbol: str, max_retries: int = 2) -> Optional[pd.DataFrame]:
    """
    拉取单只股票的北向持仓历史。

    返回: date, holding_shares, holding_ratio, holding_value
    """
    import akshare as ak
    import warnings
    warnings.filterwarnings("ignore")

    for attempt in range(max_retries + 1):
        try:
            df = ak.stock_hsgt_individual_em(symbol=symbol)
            if df is None or len(df) == 0:
                return None

            # 标准化列名
            col_map = {
                "持股日期": "date",
                "持股数量": "holding_shares",
                "持股市值": "holding_value",
                "持股占A股百分比": "holding_ratio",
                "持股占总股本比例": "holding_ratio_total",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

            if "date" not in df.columns:
                # 尝试其他列名
                for c in df.columns:
                    if "日期" in c:
                        df = df.rename(columns={c: "date"})
                        break

            if "date" not in df.columns:
                return None

            # 如果没有 holding_ratio, 尝试用 holding_ratio_total
            if "holding_ratio" not in df.columns and "holding_ratio_total" in df.columns:
                df["holding_ratio"] = df["holding_ratio_total"]

            # 数值转换
            for col in ["holding_shares", "holding_ratio", "holding_value"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # 选择列
            keep = ["date", "holding_shares", "holding_ratio", "holding_value"]
            keep = [c for c in keep if c in df.columns]
            df = df[keep].copy()

            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            return df

        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
            else:
                # 不打印每只股票的错误 (太多了), 静默失败
                return None

    return None


def run_history_mode(symbols: List[str], resume: bool = True):
    """逐只拉取北向持仓历史。"""
    total = len(symbols)
    done = 0
    skipped = 0
    failed = 0

    print(f"[history] 开始拉取 {total} 只股票北向持仓历史...", flush=True)

    for i, sym in enumerate(symbols):
        cache_path = os.path.join(CACHE_DIR, f"{sym}.parquet")

        if resume and os.path.exists(cache_path):
            skipped += 1
            continue

        df = fetch_individual_history(sym)
        if df is not None and len(df) > 0:
            df.to_parquet(cache_path, index=False)
            done += 1
        else:
            failed += 1

        # 进度
        if (i + 1) % 50 == 0 or i == total - 1:
            print(f"  进度: {i+1}/{total}  成功={done} 跳过={skipped} 失败={failed}",
                  flush=True)

        time.sleep(RATE_LIMIT)

    print(f"\n[history] 完成: 成功={done}, 跳过={skipped}, 失败={failed}", flush=True)


# ════════════════════════════════════════
#  当日快照模式
# ════════════════════════════════════════

def run_snapshot_mode():
    """获取当日北向持仓快照, 追加到各股票缓存。"""
    import akshare as ak
    import warnings
    warnings.filterwarnings("ignore")

    print("[snapshot] 获取北向持仓今日排行...", flush=True)

    try:
        df = ak.stock_hsgt_hold_stock_em(market="北向", indicator="今日排行")
    except Exception as e:
        print(f"[snapshot] 获取失败: {e}", flush=True)
        return

    if df is None or len(df) == 0:
        print("[snapshot] 无数据", flush=True)
        return

    today = datetime.now().strftime("%Y-%m-%d")

    # 标准化列名
    col_map = {
        "代码": "symbol",
        "名称": "name",
        "今日持股股数": "holding_shares",
        "今日持股市值": "holding_value",
        "今日持股占A股比": "holding_ratio",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "symbol" not in df.columns:
        # 尝试找代码列
        for c in df.columns:
            if "代码" in c:
                df = df.rename(columns={c: "symbol"})
                break

    if "symbol" not in df.columns:
        print("[snapshot] 无法识别代码列", flush=True)
        return

    df["symbol"] = df["symbol"].astype(str).str.zfill(6)

    n_updated = 0
    for _, row in df.iterrows():
        sym = row["symbol"]
        cache_path = os.path.join(CACHE_DIR, f"{sym}.parquet")

        new_row = {
            "date": today,
            "holding_shares": pd.to_numeric(row.get("holding_shares"), errors="coerce")
                              if "holding_shares" in row.index else np.nan,
            "holding_ratio": pd.to_numeric(row.get("holding_ratio"), errors="coerce")
                             if "holding_ratio" in row.index else np.nan,
            "holding_value": pd.to_numeric(row.get("holding_value"), errors="coerce")
                             if "holding_value" in row.index else np.nan,
        }

        if os.path.exists(cache_path):
            existing = pd.read_parquet(cache_path)
            existing["date"] = pd.to_datetime(existing["date"])
            if today in existing["date"].dt.strftime("%Y-%m-%d").values:
                continue
            new_df = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
            new_df = new_df.sort_values("date").reset_index(drop=True)
        else:
            new_df = pd.DataFrame([new_row])

        new_df.to_parquet(cache_path, index=False)
        n_updated += 1

    print(f"[snapshot] 完成: 更新 {n_updated} 只股票", flush=True)


# ════════════════════════════════════════
#  主入口
# ════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="拉取北向资金持仓数据")
    parser.add_argument("--resume", action="store_true", help="断点续传")
    parser.add_argument("--snapshot", action="store_true", help="仅获取当日快照")
    parser.add_argument("--symbols", nargs="*", default=None, help="指定股票代码")
    parser.add_argument("--limit", type=int, default=None, help="限制数量 (调试)")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("  北向资金持仓数据拉取", flush=True)
    print(f"  模式: {'snapshot' if args.snapshot else 'history'}  断点续传: {args.resume}", flush=True)
    print("=" * 60, flush=True)

    if args.snapshot:
        run_snapshot_mode()
    else:
        # 获取股票列表
        if args.symbols:
            symbols = args.symbols
        else:
            symbols = get_hsgt_stock_list()
            if args.limit:
                symbols = symbols[:args.limit]

        if not symbols:
            print("[ERROR] 未获取到沪深港通股票列表", flush=True)
            sys.exit(1)

        print(f"  沪深港通股票: {len(symbols)} 只", flush=True)
        run_history_mode(symbols, resume=args.resume)
