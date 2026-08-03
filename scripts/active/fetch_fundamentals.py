"""
基本面数据批量拉取 v2 — 为盈利动量因子构建缓存

拉取所有已缓存股票的季度财务指标, 存储为标准parquet格式。
支持断点续传 (--resume) 和完整性检查 (--check-only)。

数据源: akshare stock_financial_analysis_indicator
存储: data/fundamental_cache/{symbol}.parquet
速率限制: 0.5s/请求 (避免被封)

用法:
  python scripts/fetch_fundamentals_v2.py                # 增量拉取 (跳过已有)
  python scripts/fetch_fundamentals_v2.py --resume       # 断点续传 (跳过已有, 同义)
  python scripts/fetch_fundamentals_v2.py --force        # 强制重新拉取所有
  python scripts/fetch_fundamentals_v2.py --check-only   # 只检查完整性, 不拉取
  python scripts/fetch_fundamentals_v2.py --limit 50     # 只拉取前50只 (测试)
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

CACHE_DIR = os.path.join(BASE_DIR, "data", "fundamental_cache")
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
DATA_STORE_DIR = os.path.join(BASE_DIR, "data_store")

# 速率限制 (秒)
RATE_LIMIT = 0.5

# 财报起始年份 (新浪接口必须传: 默认 "1900" 不在年份列表 → 返回空)
# 2018 起: 已有缓存覆盖 2015-2024, 缺失的主要是 2020+ 上市新股,
# 拉 2018-2026 (9年) 平衡覆盖与速度 (每只 ~7s vs 12年 ~15s)
START_YEAR = "2018"

# 标准输出列 (存储格式)
STANDARD_COLUMNS = [
    'report_date',      # 报告期 (日期)
    'announce_date',    # 公告日 (如有)
    'eps',              # 摊薄每股收益(元)
    'roe',              # 净资产收益率(%)
    'revenue_growth',   # 主营业务收入增长率(%)
    'profit_growth',    # 净利润增长率(%)
    'net_income',       # 净利润(万元)
    'operating_cash_flow',  # 每股经营性现金流(元)
    'total_assets',     # 总资产(万元)
    'debt_ratio',       # 资产负债率(%)
    'bvps',             # 每股净资产(元)
]

# akshare 原始列名 → 标准列名映射
COLUMN_MAP = {
    '日期': 'report_date',
    '摊薄每股收益(元)': 'eps',
    '净资产收益率(%)': 'roe',
    '主营业务收入增长率(%)': 'revenue_growth',
    '净利润增长率(%)': 'profit_growth',
    '净利润(万元)': 'net_income',
    '每股经营性现金流(元)': 'operating_cash_flow',
    '总资产(万元)': 'total_assets',
    '资产负债率(%)': 'debt_ratio',
    '每股净资产_调整前(元)': 'bvps',
}


def get_universe_symbols() -> List[str]:
    """
    获取需要拉取基本面的股票列表。
    优先从 data_store (全量) 获取, 回退到 data_cache (小池子)。
    """
    symbols = set()

    # 从 data_store 获取 (全量宇宙)
    if os.path.exists(DATA_STORE_DIR):
        for f in os.listdir(DATA_STORE_DIR):
            if f.endswith(".parquet") and not f.startswith("_"):
                symbols.add(f.replace(".parquet", ""))

    # 从 data_cache 获取 (小池子)
    if os.path.exists(DATA_CACHE_DIR):
        for f in os.listdir(DATA_CACHE_DIR):
            if f.endswith(".parquet") and not f.startswith("_"):
                symbols.add(f.replace(".parquet", ""))

    return sorted(symbols)


def fetch_single(symbol: str, retries: int = 3) -> Optional[pd.DataFrame]:
    """
    拉取单只股票的季度财务指标 (带退避重试)。

    Returns:
      标准化后的 DataFrame, 或 None (失败)
    """
    import akshare as ak
    import warnings
    import time as _time
    warnings.filterwarnings('ignore')

    for attempt in range(retries):
        try:
            # 必须传 start_year: 新浪接口默认 "1900" 不在年份列表 → 返回空
            df = ak.stock_financial_analysis_indicator(
                symbol=symbol, start_year=START_YEAR)
        except Exception:
            if attempt < retries - 1:
                _time.sleep(1.5 * (attempt + 1))  # 退避重试
                continue
            return None

        if df is None or len(df) == 0:
            if attempt < retries - 1:
                _time.sleep(1.5 * (attempt + 1))
                continue
            return None
        break

    # 标准化列名
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

    # 确保 report_date 存在
    if 'report_date' not in df.columns:
        return None

    # 转换日期
    df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
    df = df.dropna(subset=['report_date'])

    if len(df) == 0:
        return None

    # 排序
    df = df.sort_values('report_date').reset_index(drop=True)

    # 保留标准列 (如果存在)
    keep = [c for c in STANDARD_COLUMNS if c in df.columns]

    # 同时保留原始列 (兼容性: 老代码依赖中文列名)
    # 策略: 存储所有列, 但确保标准列存在
    if 'announce_date' not in df.columns:
        df['announce_date'] = pd.NaT

    return df


def check_integrity(symbols: List[str]) -> Tuple[int, int, List[str]]:
    """
    检查缓存完整性。

    Returns:
      (cached_count, missing_count, missing_symbols)
    """
    cached = []
    missing = []

    for sym in symbols:
        path = os.path.join(CACHE_DIR, f"{sym}.parquet")
        if os.path.exists(path):
            cached.append(sym)
        else:
            missing.append(sym)

    return len(cached), len(missing), missing


def run_fetch(symbols: List[str], force: bool = False, limit: Optional[int] = None):
    """
    主拉取逻辑。

    Args:
      symbols: 股票列表
      force: 强制重新拉取 (覆盖已有缓存)
      limit: 限制拉取数量
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    if limit:
        symbols = symbols[:limit]

    total = len(symbols)
    done = 0
    skipped = 0
    failed = []
    t0 = time.time()

    print(f"[FetchFundamentals] 开始拉取: {total} 只", flush=True)
    print(f"  缓存目录: {CACHE_DIR}", flush=True)
    print(f"  速率限制: {RATE_LIMIT}s/请求", flush=True)
    print(f"  模式: {'强制覆盖' if force else '增量'}", flush=True)

    for i, sym in enumerate(symbols, 1):
        out_path = os.path.join(CACHE_DIR, f"{sym}.parquet")

        # 增量模式: 跳过已有
        if not force and os.path.exists(out_path):
            skipped += 1
            if i % 100 == 0 or i == total:
                elapsed = time.time() - t0
                print(f"  [{i}/{total}] {sym} 跳过 (已缓存) "
                      f"[{elapsed:.0f}s]", flush=True)
            continue

        # 速率限制
        if done > 0:
            time.sleep(RATE_LIMIT)

        # 拉取
        df = fetch_single(sym)

        if df is None or len(df) == 0:
            failed.append(sym)
            if i % 100 == 0 or i == total:
                elapsed = time.time() - t0
                print(f"  [{i}/{total}] {sym} 失败 "
                      f"[{elapsed:.0f}s]", flush=True)
            continue

        # 保存
        try:
            df.to_parquet(out_path, index=False)
            done += 1
        except Exception as e:
            failed.append(sym)
            print(f"  [{i}/{total}] {sym} 写入失败: {e}", flush=True)
            continue

        # 进度
        if i % 50 == 0 or i == total:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remaining = total - i
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{i}/{total}] {sym} 完成 ({len(df)}行) "
                  f"[{elapsed:.0f}s, ~{eta:.0f}s remaining]", flush=True)

    # 汇总
    elapsed = time.time() - t0
    print(f"\n{'='*50}", flush=True)
    print(f"拉取完成: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"  新拉取: {done}", flush=True)
    print(f"  跳过:   {skipped}", flush=True)
    print(f"  失败:   {len(failed)}", flush=True)

    if failed:
        print(f"  失败列表: {failed[:30]}{'...' if len(failed) > 30 else ''}", flush=True)
        # 保存失败列表供重试
        fail_path = os.path.join(CACHE_DIR, "_failed.txt")
        with open(fail_path, 'w') as f:
            f.write('\n'.join(failed))
        print(f"  失败列表已保存: {fail_path}", flush=True)


def run_check(symbols: List[str]):
    """完整性检查模式"""
    cached, missing, missing_syms = check_integrity(symbols)
    total = len(symbols)

    print(f"[Check] 缓存完整性检查", flush=True)
    print(f"  宇宙总数: {total}", flush=True)
    print(f"  已缓存:   {cached} ({cached/total*100:.1f}%)", flush=True)
    print(f"  缺失:     {missing} ({missing/total*100:.1f}%)", flush=True)

    if missing_syms:
        print(f"  缺失样本: {missing_syms[:20]}{'...' if len(missing_syms) > 20 else ''}", flush=True)

    # 检查已有缓存的质量
    print(f"\n  缓存质量抽检:", flush=True)
    sample = [s for s in symbols if os.path.exists(os.path.join(CACHE_DIR, f"{s}.parquet"))][:10]
    for sym in sample:
        try:
            df = pd.read_parquet(os.path.join(CACHE_DIR, f"{sym}.parquet"))
            n_rows = len(df)
            has_std = all(c in df.columns for c in ['report_date', 'eps', 'roe'])
            print(f"    {sym}: {n_rows}行, 标准列={'OK' if has_std else 'MISSING'}", flush=True)
        except Exception as e:
            print(f"    {sym}: 读取失败 ({e})", flush=True)


def main():
    global RATE_LIMIT
    parser = argparse.ArgumentParser(description="基本面数据批量拉取 v2")
    parser.add_argument("--force", action="store_true",
                        help="强制重新拉取所有 (覆盖已有缓存)")
    parser.add_argument("--resume", action="store_true",
                        help="断点续传 (跳过已有, 与默认行为相同)")
    parser.add_argument("--check-only", action="store_true",
                        help="只检查完整性, 不拉取数据")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制拉取数量 (测试用)")
    parser.add_argument("--offset", type=int, default=0,
                        help="跳过前 N 只 (多进程分片并行用)")
    parser.add_argument("--rate", type=float, default=RATE_LIMIT,
                        help=f"请求间隔秒数 (默认{RATE_LIMIT})")
    args = parser.parse_args()

    RATE_LIMIT = args.rate

    # 获取股票列表
    symbols = get_universe_symbols()
    if args.offset:
        symbols = symbols[args.offset:]
    print(f"[FetchFundamentals] 股票池: {len(symbols)} 只 "
          f"(offset={args.offset})", flush=True)

    if not symbols:
        print("  错误: 未找到任何股票。请先运行 fetch_full_universe.py", flush=True)
        sys.exit(1)

    if args.check_only:
        run_check(symbols)
    else:
        run_fetch(symbols, force=args.force, limit=args.limit)


if __name__ == "__main__":
    main()
