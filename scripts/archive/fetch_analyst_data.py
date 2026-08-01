"""
批量拉取分析师盈利预测/评级数据 → data/factor_cache/analyst/

数据源现实 (akshare 1.18.64, 已实测):
  ✓ ak.stock_profit_forecast_em(symbol='')  — 单次批量调用返回全市场(~2800只)快照:
        代码/名称/研报数/机构评级(近六月 买入增持中性减持卖出)/各年预测每股收益
        ★ 只返回"当前快照", 无历史。本脚本按日存储 snapshot_YYYYMMDD.parquet,
          由 factors/analyst_revision.py 从历史快照计算"修正"类因子。
  ✗ ak.stock_profit_forecast_em(symbol='600519') — per-stock 调用已损坏 (NoneType)
  ✗ ak.stock_analyst_rating_em / stock_institute_recommend_detail_em — 不存在
  ✗ ak.stock_institute_recommend_detail(symbol) — 新浪源已失效 (返回HTML)

  因此: analyst_name / org / target_price 等"逐分析师明细"当前无可用接口,
        本脚本不产出这些列。target_deviation 因子依赖可选的外部
        data/factor_cache/analyst/target_price.parquet (symbol, target_price)。

模式:
  snapshot (默认) : 一次批量调用 → snapshot_YYYYMMDD.parquet (秒级)
  enrich-ths      : 可选, 逐只调用 stock_profit_forecast_ths 补充一致预期均值,
                    限速0.5s, 带ETA (用于小样本验证, 全量~23分钟)

用法:
  python scripts/fetch_analyst_data.py                 # 抓取今日快照
  python scripts/fetch_analyst_data.py --resume        # 今日已抓则跳过
  python scripts/fetch_analyst_data.py --check-only    # 仅报告覆盖率, 不抓取
  python scripts/fetch_analyst_data.py --enrich-ths --limit 50   # THS逐只补充(调试)
"""

import os
import sys
import time
import argparse
import glob
from datetime import datetime
from typing import List, Optional

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from factors.analyst_revision import standardize_snapshot, ANALYST_CACHE

os.makedirs(ANALYST_CACHE, exist_ok=True)

RATE_LIMIT = 0.5  # per-stock 请求间隔 (秒)


# ════════════════════════════════════════
#  快照模式 (推荐)
# ════════════════════════════════════════

def fetch_snapshot(force: bool = False) -> Optional[str]:
    """
    抓取全市场盈利预测快照 → snapshot_YYYYMMDD.parquet。

    返回保存路径; 若已存在且非 force 则跳过。
    """
    import akshare as ak
    import warnings
    warnings.filterwarnings("ignore")

    today = datetime.now().strftime("%Y%m%d")
    out_path = os.path.join(ANALYST_CACHE, f"snapshot_{today}.parquet")

    if os.path.exists(out_path) and not force:
        df = pd.read_parquet(out_path)
        print(f"[snapshot] 今日快照已存在 ({len(df)} 只), 跳过。用 --force 覆盖。", flush=True)
        return out_path

    print("[snapshot] 拉取全市场盈利预测 (stock_profit_forecast_em)...", flush=True)
    t0 = time.time()
    try:
        raw = ak.stock_profit_forecast_em(symbol="")
    except Exception as e:
        print(f"[snapshot] 拉取失败: {e}", flush=True)
        return None

    if raw is None or len(raw) == 0:
        print("[snapshot] 返回空数据", flush=True)
        return None

    df = standardize_snapshot(raw)
    if len(df) == 0:
        print("[snapshot] 标准化后为空 (列名可能已变化)", flush=True)
        print(f"  原始列: {list(raw.columns)}", flush=True)
        return None

    df["snapshot_date"] = pd.Timestamp(today)
    df.to_parquet(out_path, index=False)

    n_covered = int((df["report_count"] >= 3).sum()) if "report_count" in df.columns else 0
    print(f"[snapshot] 完成: {len(df)} 只, 覆盖>=3研报 {n_covered} 只 "
          f"({n_covered/len(df)*100:.1f}%), 耗时 {time.time()-t0:.1f}s", flush=True)
    print(f"[snapshot] → {out_path}", flush=True)
    return out_path


# ════════════════════════════════════════
#  THS 逐只补充模式 (带限速+ETA)
# ════════════════════════════════════════

def _fmt_eta(seconds: float) -> str:
    """秒数 → 'H:MM:SS'。"""
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def get_universe_symbols() -> List[str]:
    """从 data_store 获取股票池。"""
    ds = os.path.join(BASE_DIR, "data_store")
    if not os.path.exists(ds):
        return []
    return sorted([
        f.replace(".parquet", "") for f in os.listdir(ds)
        if f.endswith(".parquet") and f[0].isdigit()
    ])


def enrich_ths(symbols: List[str], resume: bool = True):
    """
    逐只调用 stock_profit_forecast_ths 补充一致预期均值/覆盖。

    存储: data/factor_cache/analyst/ths/{symbol}.parquet
    列: fiscal_year, n_analysts, mean_eps, min_eps, max_eps, industry_avg

    限速 RATE_LIMIT, 带进度与ETA。全量~2800只约需23分钟。
    """
    import akshare as ak
    import warnings
    warnings.filterwarnings("ignore")

    ths_dir = os.path.join(ANALYST_CACHE, "ths")
    os.makedirs(ths_dir, exist_ok=True)

    total = len(symbols)
    done = skipped = failed = 0
    t0 = time.time()

    print(f"[enrich-ths] 开始逐只拉取 {total} 只 (限速{RATE_LIMIT}s, 带ETA)...", flush=True)

    for i, sym in enumerate(symbols):
        cache_path = os.path.join(ths_dir, f"{sym}.parquet")
        if resume and os.path.exists(cache_path):
            skipped += 1
        else:
            try:
                df = ak.stock_profit_forecast_ths(symbol=sym, indicator="预测年报每股收益")
                if df is not None and len(df) > 0:
                    col_map = {
                        "年度": "fiscal_year", "预测机构数": "n_analysts",
                        "最小值": "min_eps", "均值": "mean_eps",
                        "最大值": "max_eps", "行业平均数": "industry_avg",
                    }
                    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                    for c in ["n_analysts", "min_eps", "mean_eps", "max_eps", "industry_avg"]:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors="coerce")
                    df["symbol"] = sym
                    df.to_parquet(cache_path, index=False)
                    done += 1
                else:
                    failed += 1
            except Exception:
                failed += 1
            time.sleep(RATE_LIMIT)

        # 进度 + ETA
        if (i + 1) % 25 == 0 or i == total - 1:
            elapsed = time.time() - t0
            processed = done + failed  # 实际发起请求的数量
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = total - (i + 1)
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{i+1}/{total}] 成功={done} 跳过={skipped} 失败={failed} "
                  f"| 已用 {_fmt_eta(elapsed)} ETA {_fmt_eta(eta)}", flush=True)

    print(f"\n[enrich-ths] 完成: 成功={done}, 跳过={skipped}, 失败={failed}, "
          f"总耗时 {_fmt_eta(time.time()-t0)}", flush=True)


# ════════════════════════════════════════
#  覆盖率检查
# ════════════════════════════════════════

def check_only():
    """报告最新快照的覆盖率统计, 不发起抓取。"""
    files = sorted(glob.glob(os.path.join(ANALYST_CACHE, "snapshot_*.parquet")))
    print("=" * 60, flush=True)
    print("  分析师数据覆盖率检查", flush=True)
    print("=" * 60, flush=True)
    print(f"  快照数量: {len(files)}", flush=True)

    if not files:
        print("  [!] 无快照。请先运行: python scripts/fetch_analyst_data.py", flush=True)
        return

    print(f"  日期范围: {os.path.basename(files[0])} ~ {os.path.basename(files[-1])}", flush=True)

    if len(files) < 2:
        print("  [!] 仅1个快照: revision_30d / coverage_change 暂不可用 (需>=2个快照)", flush=True)

    df = standardize_snapshot(pd.read_parquet(files[-1]))
    n = len(df)
    if n == 0 or "report_count" not in df.columns:
        print("  [!] 最新快照无效", flush=True)
        return

    for thr in (1, 3, 5, 10):
        cnt = int((df["report_count"] >= thr).sum())
        print(f"  研报数 >= {thr:2d}: {cnt:5d} 只 ({cnt/n*100:5.1f}%)", flush=True)

    # 各年EPS覆盖
    for yr in range(2024, 2029):
        col = f"eps_{yr}"
        if col in df.columns:
            cnt = int(df[col].notna().sum())
            print(f"  {yr}预测EPS 有效: {cnt:5d} 只 ({cnt/n*100:5.1f}%)", flush=True)


# ════════════════════════════════════════
#  主入口
# ════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="拉取分析师盈利预测/评级数据")
    parser.add_argument("--resume", action="store_true", help="断点续传/跳过已存在")
    parser.add_argument("--force", action="store_true", help="强制覆盖今日快照")
    parser.add_argument("--check-only", action="store_true", help="仅报告覆盖率, 不抓取")
    parser.add_argument("--enrich-ths", action="store_true", help="THS逐只补充一致预期(慢)")
    parser.add_argument("--limit", type=int, default=None, help="enrich-ths 限制数量(调试)")
    parser.add_argument("--symbols", nargs="*", default=None, help="enrich-ths 指定股票")
    args = parser.parse_args()

    if args.check_only:
        check_only()
        sys.exit(0)

    print("=" * 60, flush=True)
    print("  分析师数据拉取", flush=True)
    print("=" * 60, flush=True)

    # 1. 快照 (核心)
    if not args.enrich_ths:
        fetch_snapshot(force=args.force or not args.resume)

    # 2. 可选 THS 逐只补充
    if args.enrich_ths:
        if args.symbols:
            symbols = args.symbols
        else:
            symbols = get_universe_symbols()
            if args.limit:
                symbols = symbols[:args.limit]
        if not symbols:
            print("[ERROR] 未获取到股票池", flush=True)
            sys.exit(1)
        print(f"  股票池: {len(symbols)} 只", flush=True)
        enrich_ths(symbols, resume=args.resume)
