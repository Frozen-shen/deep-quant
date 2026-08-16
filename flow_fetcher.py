"""
资金流因子获取器
数据源: 新浪财经资金流排名 (via akshare)
缓存: data/flow/

数据特性:
  - SNAPSHOT 数据源 — 每次获取的是当前排名快照, 非历史序列
  - 每日运行一次, 逐日积累快照, 最终形成时间序列
  - IC 验证需要至少 60 天快照积累

接口:
  ak.stock_fund_flow_individual(symbol) — 个股资金流排名
    symbol: '即时', '3日排行', '5日排行', '10日排行', '20日排行'

    '即时' 返回列: 序号, 股票代码, 股票简称, 最新价, 涨跌幅, 换手率, 流入资金, 流出资金, 净额, 成交额
    N日排行 返回列: 序号, 股票代码, 股票简称, 最新价, 阶段涨跌幅, 连续换手率, 资金流入净额

用法:
  py flow_fetcher.py                    # 缓存当日所有周期快照
  py flow_fetcher.py --period 20日排行   # 仅缓存指定周期
  py flow_fetcher.py --status           # 查看缓存状态
"""

import os
import sys
import json
import time
import glob
import argparse
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FLOW_DIR = os.path.join(BASE_DIR, "data", "flow")
META_PATH = os.path.join(FLOW_DIR, "_meta.json")

# 限速: 接口请求间隔 (秒)
REQUEST_INTERVAL = 1.0

# 所有可用周期
ALL_PERIODS = ["即时", "3日排行", "5日排行", "10日排行", "20日排行"]

# 列名映射: 中文 → 英文
COLUMN_MAP = {
    "股票代码": "symbol",
    "股票简称": "name",
    "最新价": "price",
    "今日涨跌幅": "pct_change",
    "涨跌幅": "pct_change",
    "换手率": "turnover",
    "连续换手率": "turnover_nd",
    "阶段涨跌幅": "pct_change_nd",
    "流入资金": "inflow",
    "流出资金": "outflow",
    "净额": "net_flow",
    "成交额": "amount",
    "资金流入净额": "net_flow_nd",
}


def ensure_dir():
    os.makedirs(FLOW_DIR, exist_ok=True)


def _parse_chinese_amount(val) -> Optional[float]:
    """
    解析带中文单位的金额字符串, 统一转为万元。
    例: '287.23亿' → 2872300.0 (万元)
        '-8306.28万' → -8306.28 (万元)
        '3.35亿' → 33500.0 (万元)
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if np.isnan(val):
            return None
        return float(val)

    s = str(val).strip().replace(",", "")
    if not s or s in ("-", "--", "False", "None", "nan"):
        return None

    multiplier = 1.0
    if s.endswith("亿"):
        multiplier = 10000.0  # 1亿 = 10000万
        s = s[:-1]
    elif s.endswith("万"):
        multiplier = 1.0
        s = s[:-1]

    try:
        return float(s) * multiplier
    except ValueError:
        return None


def _parse_pct(val) -> Optional[float]:
    """解析百分比字符串, 返回浮点数。例: '561.95%' → 561.95"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if np.isnan(val):
            return None
        return float(val)
    s = str(val).strip().replace(",", "").replace("%", "")
    if not s or s in ("-", "--", "False", "None", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将中文列名映射为英文, 未映射的列保留原名。"""
    rename_map = {}
    for col in df.columns:
        if col in COLUMN_MAP:
            rename_map[col] = COLUMN_MAP[col]
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _parse_amount_columns(df: pd.DataFrame) -> pd.DataFrame:
    """解析金额列 (带中文单位) 为数值型 (万元)。"""
    amount_cols = ["inflow", "outflow", "net_flow", "amount", "net_flow_nd"]
    for col in amount_cols:
        if col in df.columns:
            df[col] = df[col].apply(_parse_chinese_amount)
    return df


def _parse_pct_columns(df: pd.DataFrame) -> pd.DataFrame:
    """解析百分比列为数值型。"""
    pct_cols = ["pct_change", "turnover", "turnover_nd", "pct_change_nd"]
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].apply(_parse_pct)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. fetch_money_flow_snapshot
# ─────────────────────────────────────────────────────────────────────────────

def fetch_money_flow_snapshot(period: str = "20日排行") -> pd.DataFrame:
    """
    获取全市场资金流排名快照。

    Args:
        period: '即时', '3日排行', '5日排行', '10日排行', '20日排行'

    Returns:
        DataFrame with flow data (empty on failure)
        Columns (after normalization):
          symbol, name, price, + period-specific flow columns
    """
    from netgate import is_offline, OfflineViolation
    if is_offline():
        raise OfflineViolation("flow_fetcher.fetch_money_flow_snapshot: 离线模式禁止网络获取")
    import akshare as ak
    import socket

    # ★ 全局 socket 超时: akshare/requests 默认无超时, 网络挂起会永久阻塞。
    #    设 60s 超时让挂起请求快速失败 → 走 except 分支继续后续周期。
    _prev_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(60)
    try:
        df = ak.stock_fund_flow_individual(symbol=period)
    except Exception as e:
        print(f"  [WARN] fetch_money_flow_snapshot('{period}') 失败: {e}")
        return pd.DataFrame()
    finally:
        socket.setdefaulttimeout(_prev_timeout)

    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = _normalize_columns(df)
    df = _parse_amount_columns(df)
    df = _parse_pct_columns(df)

    # 确保 symbol 为字符串
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.zfill(6)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. cache_flow_data
# ─────────────────────────────────────────────────────────────────────────────

def cache_flow_data(period: str = "20日排行") -> Optional[str]:
    """
    获取当日快照并缓存到 data/flow/。

    文件命名: flow_{period}_{YYYYMMDD}.parquet

    Returns:
        保存的文件路径, 失败返回 None
    """
    ensure_dir()

    df = fetch_money_flow_snapshot(period)
    if df is None or len(df) == 0:
        print(f"  [WARN] {period} 无数据, 跳过缓存")
        return None

    today = datetime.now().strftime("%Y%m%d")
    filename = f"flow_{period}_{today}.parquet"
    path = os.path.join(FLOW_DIR, filename)

    # 添加快照日期列
    df["snapshot_date"] = today

    df.to_parquet(path, index=False)
    print(f"  OK 已缓存: {filename} ({len(df)} 行)")
    return path


def cache_all_periods() -> dict:
    """
    缓存所有周期的当日快照。

    Returns:
        {period: file_path_or_None}
    """
    results = {}
    for i, period in enumerate(ALL_PERIODS):
        if i > 0:
            time.sleep(REQUEST_INTERVAL)
        print(f"  [{i+1}/{len(ALL_PERIODS)}] 获取 {period}...", flush=True)
        results[period] = cache_flow_data(period)

    # 更新 meta
    meta = {
        "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "periods_cached": {k: bool(v) for k, v in results.items()},
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. load_flow_history
# ─────────────────────────────────────────────────────────────────────────────

def load_flow_history(symbol: str) -> pd.DataFrame:
    """
    加载某只股票的所有历史快照, 合并为时间序列。

    Args:
        symbol: 6位股票代码, 如 '000001'

    Returns:
        DataFrame indexed by snapshot_date, 含各周期资金流指标
        若无缓存数据则返回空 DataFrame
    """
    ensure_dir()

    # 查找所有快照文件
    all_files = glob.glob(os.path.join(FLOW_DIR, "flow_*.parquet"))
    if not all_files:
        return pd.DataFrame()

    symbol = symbol.zfill(6)
    records = []

    for fpath in sorted(all_files):
        try:
            df = pd.read_parquet(fpath)
        except Exception:
            continue

        if "symbol" not in df.columns:
            if "股票代码" in df.columns:
                df = df.rename(columns={"股票代码": "symbol"})
            else:
                continue

        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
        row = df[df["symbol"] == symbol]

        if len(row) > 0:
            r = row.iloc[0].to_dict()
            # 从文件名提取周期信息
            basename = os.path.basename(fpath)
            parts = basename.replace(".parquet", "").split("_")
            # parts: ['flow', period..., 'YYYYMMDD']
            if len(parts) >= 3:
                r["_period"] = "_".join(parts[1:-1])
            records.append(r)

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)
    if "snapshot_date" in result.columns:
        result["snapshot_date"] = pd.to_datetime(result["snapshot_date"], format="%Y%m%d")
        result = result.sort_values("snapshot_date").reset_index(drop=True)

    return result


def load_all_snapshots(period: str = "20日排行") -> pd.DataFrame:
    """
    加载某周期的所有历史快照, 合并为全市场时间序列。

    Args:
        period: 周期名称

    Returns:
        DataFrame with snapshot_date + symbol + flow columns
    """
    ensure_dir()
    pattern = os.path.join(FLOW_DIR, f"flow_{period}_*.parquet")
    files = sorted(glob.glob(pattern))

    if not files:
        return pd.DataFrame()

    dfs = []
    for fpath in files:
        try:
            df = pd.read_parquet(fpath)
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    if "snapshot_date" in combined.columns:
        combined["snapshot_date"] = pd.to_datetime(combined["snapshot_date"], format="%Y%m%d")
        combined = combined.sort_values(["snapshot_date", "symbol"]).reset_index(drop=True)

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 4. compute_flow_factors
# ─────────────────────────────────────────────────────────────────────────────

def compute_flow_factors(all_data: dict, as_of_date=None) -> dict:
    """
    计算资金流因子 (point-in-time)。

    Args:
        all_data: {symbol: DataFrame} — 每只股票的 load_flow_history() 结果
                  或者 {symbol: {period: latest_row_dict}}
        as_of_date: 截止日期 (str 或 Timestamp), 仅使用此日期之前的快照

    Returns:
        {symbol: {factor_name: value}}

    Factors:
        flow_main_net_20d   — 20日主力净流入净额 (万元)
        flow_main_pct_20d   — 20日阶段涨跌幅 (作为资金流强度的代理)
        flow_main_net_5d    — 5日主力净流入净额 (万元)
        flow_main_pct_5d    — 5日阶段涨跌幅
        flow_momentum       — 资金流动量加速度 (5日净额 - 20日净额/4)
        flow_consistency    — 连续净流入一致性 (正值天数比例)
    """
    if as_of_date is not None:
        as_of = pd.Timestamp(as_of_date)
    else:
        as_of = pd.Timestamp.now()

    results = {}

    for sym, data in all_data.items():
        factors = _compute_single_stock_factors(sym, data, as_of)
        if factors:
            results[sym] = factors

    return results


def _compute_single_stock_factors(sym: str, data, as_of: pd.Timestamp) -> dict:
    """计算单只股票的资金流因子。"""
    if isinstance(data, pd.DataFrame) and len(data) > 0:
        return _factors_from_dataframe(data, as_of)
    elif isinstance(data, dict):
        return _factors_from_dict(data)
    return {}


def _factors_from_dataframe(df: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    """从历史快照 DataFrame 计算因子。"""
    factors = {}

    # 过滤截止日期
    if "snapshot_date" in df.columns:
        df = df[df["snapshot_date"] <= as_of].copy()

    if len(df) == 0:
        return {}

    # 按周期分组取最新
    period_groups = {}
    if "_period" in df.columns:
        for period, grp in df.groupby("_period"):
            latest = grp.sort_values("snapshot_date").iloc[-1]
            period_groups[period] = latest
    else:
        period_groups["unknown"] = df.iloc[-1]

    # 提取 20日排行数据
    row_20d = period_groups.get("20日排行")
    if row_20d is not None:
        net_nd = _safe_float(row_20d, "net_flow_nd")
        pct_nd = _safe_float(row_20d, "pct_change_nd")
        if net_nd is not None:
            factors["flow_main_net_20d"] = net_nd
        if pct_nd is not None:
            factors["flow_main_pct_20d"] = pct_nd

    # 提取 5日排行数据
    row_5d = period_groups.get("5日排行")
    if row_5d is not None:
        net_nd = _safe_float(row_5d, "net_flow_nd")
        pct_nd = _safe_float(row_5d, "pct_change_nd")
        if net_nd is not None:
            factors["flow_main_net_5d"] = net_nd
        if pct_nd is not None:
            factors["flow_main_pct_5d"] = pct_nd

    # flow_momentum: 5日净流入 - 20日净流入/4 (加速度)
    net_5d = factors.get("flow_main_net_5d")
    net_20d = factors.get("flow_main_net_20d")
    if net_5d is not None and net_20d is not None:
        factors["flow_momentum"] = net_5d - net_20d / 4.0

    # flow_consistency: 多日快照中净流入为正的天数比例
    if "_period" in df.columns:
        instant_df = df[df["_period"] == "即时"]
        if len(instant_df) >= 3:
            net_col = "net_flow" if "net_flow" in instant_df.columns else None
            if net_col:
                positive_ratio = (instant_df[net_col] > 0).mean()
                factors["flow_consistency"] = float(positive_ratio)

    return factors


def _factors_from_dict(data: dict) -> dict:
    """从字典形式的最新数据计算因子 (用于无历史时)。"""
    factors = {}

    row_20d = data.get("20日排行")
    if row_20d is not None and isinstance(row_20d, dict):
        net = _safe_float(row_20d, "net_flow_nd")
        pct = _safe_float(row_20d, "pct_change_nd")
        if net is not None:
            factors["flow_main_net_20d"] = net
        if pct is not None:
            factors["flow_main_pct_20d"] = pct

    row_5d = data.get("5日排行")
    if row_5d is not None and isinstance(row_5d, dict):
        net = _safe_float(row_5d, "net_flow_nd")
        pct = _safe_float(row_5d, "pct_change_nd")
        if net is not None:
            factors["flow_main_net_5d"] = net
        if pct is not None:
            factors["flow_main_pct_5d"] = pct

    net_5d = factors.get("flow_main_net_5d")
    net_20d = factors.get("flow_main_net_20d")
    if net_5d is not None and net_20d is not None:
        factors["flow_momentum"] = net_5d - net_20d / 4.0

    return factors


def _safe_float(row, key: str) -> Optional[float]:
    """安全提取浮点值。"""
    try:
        if isinstance(row, dict):
            val = row.get(key, None)
        elif hasattr(row, "get"):
            val = row.get(key, None)
        else:
            val = getattr(row, key, None)

        if val is None:
            return None
        if isinstance(val, float) and np.isnan(val):
            return None
        if pd.isna(val):
            return None
        return float(val)
    except (KeyError, TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. 便捷函数: 一次性获取当日全市场因子 (用于快速验证)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_today_factors() -> dict:
    """
    获取当日各周期快照, 直接计算截面因子 (无需历史积累)。

    适用于首次运行或快速验证。

    Returns:
        {symbol: {factor_name: value}}
    """
    all_data = {}

    for i, period in enumerate(ALL_PERIODS):
        if i > 0:
            time.sleep(REQUEST_INTERVAL)
        print(f"  [{i+1}/{len(ALL_PERIODS)}] 获取 {period}...", flush=True)
        df = fetch_money_flow_snapshot(period)
        if df is None or len(df) == 0:
            continue

        if "symbol" not in df.columns:
            continue

        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
        for _, row in df.iterrows():
            sym = row["symbol"]
            if sym not in all_data:
                all_data[sym] = {}
            all_data[sym][period] = row.to_dict()

    factors = compute_flow_factors(all_data)
    print(f"  计算完成: {len(factors)} 只股票有因子")
    return factors


# ─────────────────────────────────────────────────────────────────────────────
# 6. 状态检查
# ─────────────────────────────────────────────────────────────────────────────

def show_status():
    """显示缓存状态。"""
    ensure_dir()
    files = glob.glob(os.path.join(FLOW_DIR, "flow_*.parquet"))
    if not files:
        print("  无缓存数据。运行 'py flow_fetcher.py' 开始积累。")
        return

    # 按周期分组统计
    from collections import defaultdict
    period_counts = defaultdict(list)
    for f in files:
        basename = os.path.basename(f)
        parts = basename.replace(".parquet", "").split("_")
        if len(parts) >= 3:
            period = "_".join(parts[1:-1])
            date = parts[-1]
            period_counts[period].append(date)

    print(f"  缓存目录: {FLOW_DIR}")
    print(f"  总文件数: {len(files)}")
    print()
    for period, dates in sorted(period_counts.items()):
        dates_sorted = sorted(dates)
        print(f"  {period}: {len(dates)} 天 ({dates_sorted[0]} ~ {dates_sorted[-1]})")

    # meta
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"\n  上次运行: {meta.get('last_run', 'unknown')}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="资金流因子获取器")
    parser.add_argument("--period", type=str, default="", help="指定周期 (默认全部)")
    parser.add_argument("--status", action="store_true", help="查看缓存状态")
    parser.add_argument("--factors", action="store_true", help="直接计算当日因子 (不缓存)")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.factors:
        factors = fetch_today_factors()
        # 打印前5个示例
        for i, (sym, f) in enumerate(factors.items()):
            if i >= 5:
                break
            print(f"  {sym}: {f}")
    else:
        print("资金流快照缓存")
        print(f"  缓存目录: {FLOW_DIR}")
        print(f"  限速: {REQUEST_INTERVAL}s/请求")
        print()

        if args.period:
            if args.period not in ALL_PERIODS:
                print(f"  [ERROR] 无效周期: {args.period}")
                print(f"  可选: {ALL_PERIODS}")
                sys.exit(1)
            cache_flow_data(args.period)
        else:
            cache_all_periods()

        print("\n完成。")
