"""
smart_money_fetcher.py — 聪明钱因子: 北向资金持仓 + 分析师预测

数据源: 东方财富 via akshare
  - stock_hsgt_individual_em: 北向资金个股持仓历史
  - stock_profit_forecast_em: 全市场分析师一致预期

缓存策略:
  - 北向: data/smart_money/northbound/{symbol}.parquet
  - 分析师: data/smart_money/analyst_consensus_{date}.parquet

用法:
  from smart_money_fetcher import compute_smart_money_factors
  factors = compute_smart_money_factors(price_data, as_of_date="2025-06-30")

输出因子:
  北向: nb_holding_pct, nb_change_5d, nb_change_20d, nb_momentum, nb_new_high
  分析师: analyst_rating_score, analyst_target_upside, analyst_coverage, analyst_eps_revision
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SMART_MONEY_DIR = os.path.join(BASE_DIR, "data", "smart_money")
NORTHBOUND_DIR = os.path.join(SMART_MONEY_DIR, "northbound")

# 限速: 每次请求间隔 (秒)
REQUEST_INTERVAL = 0.5
# 单次运行最大请求数
MAX_REQUESTS_PER_RUN = 500


def ensure_dirs():
    """确保缓存目录存在。"""
    os.makedirs(NORTHBOUND_DIR, exist_ok=True)


# ============================================================
# Part 1: 北向资金持仓
# ============================================================

def fetch_northbound_history(symbol: str) -> pd.DataFrame:
    """
    拉取单只股票的北向资金持仓历史。

    Args:
      symbol: 股票代码, 如 "600519"

    Returns:
      DataFrame with columns: 持股日期, 持股数量, 持股市值, 持股数量占A股百分比, etc.
      如果该股票无北向数据, 返回空 DataFrame。
    """
    import akshare as ak

    try:
        df = ak.stock_hsgt_individual_em(symbol=symbol)
        if df is None or len(df) == 0:
            return pd.DataFrame()
        # 确保日期列为 datetime (实际列名: 持股日期)
        date_col = "持股日期" if "持股日期" in df.columns else "日期"
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def batch_fetch_northbound(symbols: List[str],
                           max_stocks: int = MAX_REQUESTS_PER_RUN) -> Dict[str, pd.DataFrame]:
    """
    批量拉取北向资金持仓数据并缓存。

    Args:
      symbols: 股票代码列表
      max_stocks: 单次运行最大请求数 (默认500)

    Returns:
      {symbol: DataFrame} — 成功获取的数据
    """
    ensure_dirs()
    results = {}
    request_count = 0

    symbols = symbols[:max_stocks]
    total = len(symbols)
    print(f"北向资金批量获取: {total} 只, 限速 {REQUEST_INTERVAL}s")

    for i, sym in enumerate(symbols):
        # 检查本地缓存
        cache_path = os.path.join(NORTHBOUND_DIR, f"{sym}.parquet")
        if os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                if len(df) > 0:
                    results[sym] = df
                    continue
            except Exception:
                pass

        # 拉取
        df = fetch_northbound_history(sym)
        request_count += 1

        if len(df) > 0:
            df.to_parquet(cache_path, index=False)
            results[sym] = df

        # 进度
        if (i + 1) % 50 == 0 or i == total - 1:
            print(f"  [{i+1}/{total}] 已获取 {len(results)} 只", flush=True)

        time.sleep(REQUEST_INTERVAL)

    print(f"完成: {len(results)}/{total} 只成功")
    return results


def compute_northbound_factors(northbound_data: Dict[str, pd.DataFrame],
                               as_of_date) -> Dict[str, dict]:
    """
    计算北向资金因子 (point-in-time)。

    因子:
      - nb_holding_pct: 北向持股占比 (最新可用)
      - nb_change_5d: 5日持股变化
      - nb_change_20d: 20日持股变化
      - nb_momentum: nb_change_5d / (abs(nb_change_20d) + 1e-9) — 加速度
      - nb_new_high: 持股数量是否为近60日新高 (1.0 or 0.0)

    Args:
      northbound_data: {symbol: DataFrame} — 北向持仓历史
      as_of_date: 截止日期 (只用 <= 此日期的数据)

    Returns:
      {symbol: {factor_name: value}}
    """
    as_of = pd.Timestamp(as_of_date)
    results = {}

    for sym, df in northbound_data.items():
        if len(df) == 0:
            continue

        # 确定日期列名 (实际为 "持股日期", 兼容 "日期")
        date_col = "持股日期" if "持股日期" in df.columns else "日期"
        if date_col not in df.columns:
            continue

        # Point-in-time: 只用 as_of_date 之前的数据
        hist = df[df[date_col] <= as_of].copy()
        if len(hist) == 0:
            continue

        hist = hist.sort_values(date_col).reset_index(drop=True)
        factors = {}

        # 持股数量列
        holding_col = "持股数量"
        # 持股比例列 (实际为 "持股数量占A股百分比", 兼容旧名)
        pct_col = None
        for candidate in ["持股数量占A股百分比", "持股数量占总股本百分比", "持股数量占发行股百分比"]:
            if candidate in hist.columns:
                pct_col = candidate
                break

        if holding_col not in hist.columns:
            continue

        latest = hist.iloc[-1]
        latest_holding = latest[holding_col]

        # nb_holding_pct: 持股比例
        if pct_col is not None and pd.notna(latest[pct_col]):
            factors["nb_holding_pct"] = float(latest[pct_col])

        # nb_change_5d: 5日持股变化
        if len(hist) >= 6:
            holding_5d_ago = hist.iloc[-6][holding_col]
            if pd.notna(latest_holding) and pd.notna(holding_5d_ago):
                factors["nb_change_5d"] = float(latest_holding - holding_5d_ago)

        # nb_change_20d: 20日持股变化
        if len(hist) >= 21:
            holding_20d_ago = hist.iloc[-21][holding_col]
            if pd.notna(latest_holding) and pd.notna(holding_20d_ago):
                factors["nb_change_20d"] = float(latest_holding - holding_20d_ago)

        # nb_momentum: 加速度
        change_5d = factors.get("nb_change_5d")
        change_20d = factors.get("nb_change_20d")
        if change_5d is not None and change_20d is not None:
            factors["nb_momentum"] = change_5d / (abs(change_20d) + 1e-9)

        # nb_new_high: 近60日新高
        window = hist.tail(60)
        if pd.notna(latest_holding) and len(window) > 0:
            max_holding = window[holding_col].max()
            factors["nb_new_high"] = 1.0 if latest_holding >= max_holding else 0.0

        if factors:
            results[sym] = factors

    return results


# ============================================================
# Part 2: 分析师预测
# ============================================================

def fetch_analyst_consensus() -> pd.DataFrame:
    """
    拉取全市场分析师一致预期快照。

    Returns:
      DataFrame with columns: 代码, 名称, 评级, 目标价, 预测当年每股收益, etc.
    """
    import akshare as ak

    try:
        df = ak.stock_profit_forecast_em()
        if df is None or len(df) == 0:
            return pd.DataFrame()
        # 缓存到本地
        ensure_dirs()
        today = datetime.now().strftime("%Y%m%d")
        cache_path = os.path.join(SMART_MONEY_DIR, f"analyst_consensus_{today}.parquet")
        df.to_parquet(cache_path, index=False)
        return df
    except Exception as e:
        print(f"  ⚠️ 分析师预期拉取失败: {e}")
        return pd.DataFrame()


def _rating_to_score(rating: str) -> Optional[float]:
    """将评级文本转换为分数 (兼容单列评级格式)。"""
    if not isinstance(rating, str):
        return None
    rating = rating.strip()
    mapping = {
        "买入": 5.0,
        "增持": 4.0,
        "推荐": 4.0,
        "强烈推荐": 5.0,
        "中性": 3.0,
        "持有": 3.0,
        "观望": 3.0,
        "减持": 2.0,
        "卖出": 1.0,
    }
    return mapping.get(rating)


def _compute_weighted_rating(row) -> Optional[float]:
    """
    从分类评级计数列计算加权平均评分。

    实际数据格式: 机构投资评级(近六个月)-买入/增持/中性/减持/卖出 各有多少家
    """
    score_map = {"买入": 5.0, "增持": 4.0, "中性": 3.0, "减持": 2.0, "卖出": 1.0}
    total_count = 0
    weighted_sum = 0.0

    for label, score in score_map.items():
        col = f"机构投资评级(近六个月)-{label}"
        if col in row.index:
            count = pd.to_numeric(row[col], errors="coerce")
            if pd.notna(count) and count > 0:
                weighted_sum += count * score
                total_count += count

    if total_count > 0:
        return weighted_sum / total_count
    return None


def _get_eps_columns(columns) -> tuple:
    """
    从列名中提取当年和次年的 EPS 预测列。

    实际格式: "2025预测每股收益", "2026预测每股收益", etc.
    Returns: (current_year_col, next_year_col) or (None, None)
    """
    import re
    eps_cols = []
    for col in columns:
        m = re.match(r"(\d{4})预测每股收益", col)
        if m:
            eps_cols.append((int(m.group(1)), col))

    eps_cols.sort()
    if len(eps_cols) >= 2:
        return eps_cols[0][1], eps_cols[1][1]
    return None, None


def compute_analyst_factors(consensus_df: pd.DataFrame,
                            price_data: Dict[str, pd.DataFrame],
                            as_of_date) -> Dict[str, dict]:
    """
    计算分析师因子。

    实际数据格式 (stock_profit_forecast_em):
      - 代码, 名称, 研报数
      - 机构投资评级(近六个月)-买入/增持/中性/减持/卖出 (各有多少家)
      - 2025预测每股收益, 2026预测每股收益, ... (年度 EPS 预测)

    因子:
      - analyst_rating_score: 加权评级分 (买入=5, 增持=4, 中性=3, 减持=2, 卖出=1)
      - analyst_target_upside: (目标价均值 - 当前价) / 当前价 (如有目标价列)
      - analyst_coverage: 覆盖分析师数量 (研报数)
      - analyst_eps_revision: (次年EPS - 当年EPS) / abs(当年EPS) — 增长预期

    Args:
      consensus_df: 分析师一致预期 DataFrame
      price_data: {symbol: DataFrame with close, date} — 用于计算上行空间
      as_of_date: 截止日期 (用于获取当前价格)

    Returns:
      {symbol: {factor_name: value}}
    """
    if consensus_df is None or len(consensus_df) == 0:
        return {}

    as_of = pd.Timestamp(as_of_date)
    results = {}

    # 确定列名
    code_col = "代码" if "代码" in consensus_df.columns else None
    if code_col is None:
        return {}

    # 检测是否有目标价列
    target_col = "目标价" if "目标价" in consensus_df.columns else None

    # 检测 EPS 列
    eps_current_col, eps_next_col = _get_eps_columns(consensus_df.columns)

    # 逐行处理 (每行一只股票)
    for _, row in consensus_df.iterrows():
        sym = str(row[code_col])
        factors = {}

        # analyst_rating_score: 加权评级
        # 优先用分类计数列, 兼容单列 "评级" 格式
        rating_score = _compute_weighted_rating(row)
        if rating_score is not None:
            factors["analyst_rating_score"] = float(rating_score)
        elif "评级" in row.index:
            score = _rating_to_score(row["评级"])
            if score is not None:
                factors["analyst_rating_score"] = score

        # analyst_target_upside: 目标价上行空间
        if target_col is not None and sym in price_data:
            target = pd.to_numeric(row.get(target_col), errors="coerce")
            if pd.notna(target) and target > 0:
                pdf = price_data[sym]
                if "date" in pdf.columns:
                    price_at_date = pdf[pdf["date"] <= as_of]
                else:
                    price_at_date = pdf
                if len(price_at_date) > 0:
                    current_price = float(price_at_date["close"].iloc[-1])
                    if current_price > 0:
                        factors["analyst_target_upside"] = (target - current_price) / current_price

        # analyst_coverage: 研报数 (覆盖数量)
        if "研报数" in row.index:
            coverage = pd.to_numeric(row["研报数"], errors="coerce")
            if pd.notna(coverage):
                factors["analyst_coverage"] = float(coverage)
        else:
            # 兼容: 用评级总数作为覆盖
            total_ratings = 0
            for label in ["买入", "增持", "中性", "减持", "卖出"]:
                col = f"机构投资评级(近六个月)-{label}"
                if col in row.index:
                    cnt = pd.to_numeric(row[col], errors="coerce")
                    if pd.notna(cnt):
                        total_ratings += cnt
            if total_ratings > 0:
                factors["analyst_coverage"] = float(total_ratings)

        # analyst_eps_revision: EPS 增长预期
        if eps_current_col and eps_next_col:
            eps_current = pd.to_numeric(row.get(eps_current_col), errors="coerce")
            eps_next = pd.to_numeric(row.get(eps_next_col), errors="coerce")
            if pd.notna(eps_current) and pd.notna(eps_next) and abs(eps_current) > 1e-9:
                factors["analyst_eps_revision"] = float(
                    (eps_next - eps_current) / abs(eps_current)
                )

        if factors:
            results[sym] = factors

    return results


# ============================================================
# Part 3: 组合入口
# ============================================================

_nb_cache: Optional[Dict[str, pd.DataFrame]] = None


def load_smart_money_data(use_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """
    加载所有已缓存的北向资金数据。

    Args:
      use_cache: 是否使用模块级缓存 (回测循环中避免重复IO)

    Returns:
      {symbol: DataFrame}
    """
    global _nb_cache
    if use_cache and _nb_cache is not None:
        return _nb_cache

    ensure_dirs()
    data = {}
    for fname in os.listdir(NORTHBOUND_DIR):
        if fname.endswith(".parquet"):
            sym = fname.replace(".parquet", "")
            try:
                df = pd.read_parquet(os.path.join(NORTHBOUND_DIR, fname))
                if len(df) > 0:
                    data[sym] = df
            except Exception:
                pass

    if use_cache:
        _nb_cache = data
    return data


def load_latest_analyst_consensus() -> Optional[pd.DataFrame]:
    """
    加载最新的分析师一致预期缓存。

    Returns:
      DataFrame or None
    """
    ensure_dirs()
    # 找最新的 analyst_consensus_*.parquet
    files = [f for f in os.listdir(SMART_MONEY_DIR)
             if f.startswith("analyst_consensus_") and f.endswith(".parquet")]
    if not files:
        return None
    files.sort()
    latest = files[-1]
    try:
        return pd.read_parquet(os.path.join(SMART_MONEY_DIR, latest))
    except Exception:
        return None


def compute_smart_money_factors(price_data: Dict[str, pd.DataFrame],
                                as_of_date) -> Dict[str, dict]:
    """
    主入口: 计算所有聪明钱因子 (北向 + 分析师)。

    Args:
      price_data: {symbol: DataFrame with close, date}
      as_of_date: 截止日期

    Returns:
      {symbol: {factor_name: value, ...}}
    """
    results = {}

    # 1. 北向资金因子
    northbound_data = load_smart_money_data()
    if northbound_data:
        nb_factors = compute_northbound_factors(northbound_data, as_of_date)
        for sym, factors in nb_factors.items():
            if sym not in results:
                results[sym] = {}
            results[sym].update(factors)

    # 2. 分析师因子
    consensus_df = load_latest_analyst_consensus()
    if consensus_df is not None and len(consensus_df) > 0:
        analyst_factors = compute_analyst_factors(consensus_df, price_data, as_of_date)
        for sym, factors in analyst_factors.items():
            if sym not in results:
                results[sym] = {}
            results[sym].update(factors)

    return results
