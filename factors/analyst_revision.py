"""
分析师修正/预期因子 — 捕捉卖方一致预期的边际变化

设计原则:
  分析师数据历史短 (~1年快照)、覆盖窄 (仅被充分覆盖的个股),
  与资金流因子类似, 不适合作为独立 alpha, 而是作为"实时叠加层"。
  无数据时返回 NaN, 不影响基础打分 (与 PEAD / MoneyFlow 模式一致)。

数据来源:
  ak.stock_profit_forecast_em(symbol='') — 全市场盈利预测快照 (单次批量调用)
    列: 代码, 名称, 研报数, 机构投资评级(近六个月)-买入/增持/中性/减持/卖出,
        2025预测每股收益, 2026预测每股收益, 2027预测每股收益, 2028预测每股收益

  ★ 该接口只返回"当前快照", 无历史。因此 fetch_analyst_data.py 会按日存储快照
    到 data/factor_cache/analyst/snapshot_YYYYMMDD.parquet, 本因子从历史快照
    计算"修正" (当前 vs N天前)。数据积累越久, revision 类因子越可靠。

因子列表:
  1. revision_30d     — 30日一致预期EPS修正 (当前 vs 30天前), coverage>=3
  2. upgrade_ratio    — 近6月评级中 (买入+增持) 占比, coverage>=3
  3. target_deviation — 目标价空间 mean(target)/price - 1, coverage>=3
                        (★ 批量接口不提供目标价, 需 per-stock 缓存; 无则 NaN)
  4. coverage_change  — 覆盖变化 report_count(当前) - report_count(90天前)

用法:
  from factors.analyst_revision import AnalystRevision

  ar = AnalystRevision()
  factors = ar.compute_factors(as_of_date="2026-08-01")
  # → {"600519": {"revision_30d": 0.03, ...}, ...}
"""

import os
import sys
import glob
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# 分析师快照缓存目录
ANALYST_CACHE = os.path.join(BASE_DIR, "data", "factor_cache", "analyst")

# 因子名列表
FACTOR_NAMES = [
    "revision_30d",
    "upgrade_ratio",
    "target_deviation",
    "coverage_change",
]

# 评级方向: 看多 = 买入 + 增持
BULLISH_RATINGS = ["buy", "add"]

# 快照中可能的盈利预测年份列 → 标准化名
_EPS_YEAR_COLS = {
    "2024预测每股收益": "eps_2024",
    "2025预测每股收益": "eps_2025",
    "2026预测每股收益": "eps_2026",
    "2027预测每股收益": "eps_2027",
    "2028预测每股收益": "eps_2028",
}


def _ensure_dirs():
    os.makedirs(ANALYST_CACHE, exist_ok=True)


def _snapshot_files() -> List[str]:
    """返回所有快照文件路径, 按日期升序。"""
    if not os.path.exists(ANALYST_CACHE):
        return []
    files = glob.glob(os.path.join(ANALYST_CACHE, "snapshot_*.parquet"))
    return sorted(files)


def _snapshot_date_from_path(path: str) -> pd.Timestamp:
    """从文件名提取快照日期: snapshot_20260801.parquet → Timestamp。"""
    base = os.path.basename(path)
    token = base.replace("snapshot_", "").replace(".parquet", "")
    return pd.Timestamp(token)


def standardize_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 stock_profit_forecast_em 原始列标准化。

    输出列:
      symbol, name, report_count, buy, add, neutral, reduce, sell,
      eps_2024..eps_2028 (存在的年份)
    """
    col_map = {
        "代码": "symbol",
        "名称": "name",
        "研报数": "report_count",
        "机构投资评级(近六个月)-买入": "buy",
        "机构投资评级(近六个月)-增持": "add",
        "机构投资评级(近六个月)-中性": "neutral",
        "机构投资评级(近六个月)-减持": "reduce",
        "机构投资评级(近六个月)-卖出": "sell",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df = df.rename(columns={k: v for k, v in _EPS_YEAR_COLS.items() if k in df.columns})

    if "symbol" not in df.columns:
        return pd.DataFrame()

    df["symbol"] = df["symbol"].astype(str).str.zfill(6)

    num_cols = ["report_count", "buy", "add", "neutral", "reduce", "sell"]
    num_cols += [c for c in _EPS_YEAR_COLS.values() if c in df.columns]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = ["symbol", "name"] + [c for c in num_cols if c in df.columns]
    return df[keep].copy()


class AnalystRevision:
    """
    分析师修正因子计算器。

    从 data/factor_cache/analyst/snapshot_YYYYMMDD.parquet 加载历史快照,
    计算4个分析师情绪/修正因子。
    """

    def __init__(self, min_coverage: int = 3):
        """
        Args:
          min_coverage: 最小研报数 (低于此值视为覆盖不足, 因子置 NaN)
        """
        _ensure_dirs()
        self.min_coverage = min_coverage
        # 内存缓存: {snapshot_date: DataFrame(index=symbol)}
        self._snap_cache: Dict[pd.Timestamp, pd.DataFrame] = {}
        self._snap_index: Optional[List[pd.Timestamp]] = None
        # 可选的目标价缓存 {symbol: target_price}
        self._target_price: Optional[Dict[str, float]] = None

    # ════════════════════════════════════════
    #  数据加载
    # ════════════════════════════════════════

    def _build_index(self):
        """构建快照日期索引 (懒加载)。"""
        if self._snap_index is not None:
            return
        dates = []
        for path in _snapshot_files():
            try:
                dates.append(_snapshot_date_from_path(path))
            except Exception:
                continue
        self._snap_index = sorted(dates)

    def _load_snapshot(self, date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """加载指定日期的快照, 返回 index=symbol 的 DataFrame。"""
        if date in self._snap_cache:
            return self._snap_cache[date]

        path = os.path.join(ANALYST_CACHE, f"snapshot_{date.strftime('%Y%m%d')}.parquet")
        if not os.path.exists(path):
            self._snap_cache[date] = None
            return None

        try:
            df = standardize_snapshot(pd.read_parquet(path))
            if len(df) == 0:
                self._snap_cache[date] = None
                return None
            df = df.drop_duplicates("symbol", keep="last").set_index("symbol")
            self._snap_cache[date] = df
            return df
        except Exception:
            self._snap_cache[date] = None
            return None

    def _snapshots_in_range(self, as_of: pd.Timestamp,
                            lookback_days: int) -> List[pd.Timestamp]:
        """返回 as_of 之前 lookback_days 窗口内的快照日期 (升序)。"""
        self._build_index()
        if not self._snap_index:
            return []
        lo = as_of - pd.Timedelta(days=lookback_days)
        return [d for d in self._snap_index if lo <= d <= as_of]

    def _latest_snapshot_date(self, as_of: pd.Timestamp,
                              within_days: int = None) -> Optional[pd.Timestamp]:
        """返回 <= as_of 的最新快照日期; within_days 限制最大回溯。"""
        self._build_index()
        if not self._snap_index:
            return None
        cands = [d for d in self._snap_index if d <= as_of]
        if within_days is not None:
            lo = as_of - pd.Timedelta(days=within_days)
            cands = [d for d in cands if d >= lo]
        if not cands:
            return None
        return cands[-1]

    def _nearest_snapshot_date(self, target: pd.Timestamp,
                               tolerance_days: int = 15) -> Optional[pd.Timestamp]:
        """返回最接近 target 的快照日期 (误差 <= tolerance_days)。"""
        self._build_index()
        if not self._snap_index:
            return None
        best, best_gap = None, None
        for d in self._snap_index:
            gap = abs((d - target).days)
            if best_gap is None or gap < best_gap:
                best, best_gap = d, gap
        if best is not None and best_gap <= tolerance_days:
            return best
        return None

    def _prior_snapshot_date(self, reference: pd.Timestamp,
                             min_days: int, max_days: int = None) -> Optional[pd.Timestamp]:
        """
        返回 reference 之前、间隔落在 [min_days, max_days] 的最新快照日期。

        自校准: 快照逐日积累时, reference-30d 附近通常有快照; 若积累稀疏
        (如早期), 退而使用更早的有效快照, 只要间隔 >= min_days 即视为可比较。
        无满足条件的快照 → None。
        """
        self._build_index()
        if not self._snap_index:
            return None
        lo_gap, hi_gap = min_days, max_days if max_days is not None else 10**9
        cands = []
        for d in self._snap_index:
            gap = (reference - d).days
            if lo_gap <= gap <= hi_gap:
                cands.append(d)
        if not cands:
            return None
        return cands[-1]  # 最新 (间隔最小) 的满足条件快照

    def _load_target_prices(self) -> Dict[str, float]:
        """
        加载可选的目标价缓存 {symbol: target_price}。

        批量盈利预测接口不提供目标价。若存在
        data/factor_cache/analyst/target_price.parquet (列: symbol, target_price),
        则加载; 否则返回空 dict。
        """
        if self._target_price is not None:
            return self._target_price

        path = os.path.join(ANALYST_CACHE, "target_price.parquet")
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                df["symbol"] = df["symbol"].astype(str).str.zfill(6)
                df["target_price"] = pd.to_numeric(df["target_price"], errors="coerce")
                df = df.dropna(subset=["target_price"])
                self._target_price = dict(zip(df["symbol"], df["target_price"]))
            except Exception:
                self._target_price = {}
        else:
            self._target_price = {}
        return self._target_price

    # ════════════════════════════════════════
    #  一致预期EPS提取
    # ════════════════════════════════════════

    @staticmethod
    def _consensus_eps(row: pd.Series, fiscal_year: int) -> Optional[float]:
        """
        从快照行提取指定财年的一致预期EPS。

        优先取 eps_{fiscal_year}; 若缺失, 取该年之后最近的非空年份。
        """
        for yr in range(fiscal_year, fiscal_year + 4):
            col = f"eps_{yr}"
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    return float(val)
        return None

    # ════════════════════════════════════════
    #  单只股票因子计算
    # ════════════════════════════════════════

    def _compute_single(self, symbol: str, as_of: pd.Timestamp) -> Dict[str, float]:
        """计算单只股票的4个分析师因子。"""
        result = {f: np.nan for f in FACTOR_NAMES}
        fiscal_year = as_of.year

        # 最新快照 (允许回溯60天, 超过则视为数据过期)
        latest_date = self._latest_snapshot_date(as_of, within_days=60)
        if latest_date is None:
            return result
        latest = self._load_snapshot(latest_date)
        if latest is None or symbol not in latest.index:
            return result

        row = latest.loc[symbol]
        report_count = row.get("report_count", np.nan)
        if pd.isna(report_count) or report_count < self.min_coverage:
            return result  # 覆盖不足, 全部置 NaN

        # ── 1. upgrade_ratio: (买入+增持) / 全部评级 ──
        buy = row.get("buy", 0) or 0
        add = row.get("add", 0) or 0
        neutral = row.get("neutral", 0) or 0
        reduce = row.get("reduce", 0) or 0
        sell = row.get("sell", 0) or 0
        total_ratings = buy + add + neutral + reduce + sell
        if total_ratings > 0:
            result["upgrade_ratio"] = float((buy + add) / total_ratings)

        # ── 2. coverage_change: 当前研报数 - 90天前研报数 ──
        d90 = self._prior_snapshot_date(latest_date, min_days=45, max_days=150)
        if d90 is not None and d90 != latest_date:
            snap90 = self._load_snapshot(d90)
            if snap90 is not None and symbol in snap90.index:
                rc90 = snap90.loc[symbol].get("report_count", np.nan)
                if pd.notna(rc90):
                    result["coverage_change"] = float(report_count - rc90)

        # ── 3. revision_30d: 当前一致预期EPS vs 30天前 ──
        eps_now = self._consensus_eps(row, fiscal_year)
        d30 = self._prior_snapshot_date(latest_date, min_days=15, max_days=75)
        if eps_now is not None and d30 is not None and d30 != latest_date:
            snap30 = self._load_snapshot(d30)
            if snap30 is not None and symbol in snap30.index:
                eps_old = self._consensus_eps(snap30.loc[symbol], fiscal_year)
                if eps_old is not None and abs(eps_old) > 1e-6:
                    rev = (eps_now - eps_old) / abs(eps_old)
                    result["revision_30d"] = float(np.clip(rev, -2.0, 2.0))

        # ── 4. target_deviation: 目标价 / 现价 - 1 ──
        #     现价从 data_store 最新收盘价取; 目标价从可选缓存取
        targets = self._load_target_prices()
        if symbol in targets:
            price = self._latest_price(symbol, as_of)
            if price is not None and price > 0:
                dev = targets[symbol] / price - 1
                result["target_deviation"] = float(np.clip(dev, -1.0, 5.0))

        return result

    def _latest_price(self, symbol: str, as_of: pd.Timestamp) -> Optional[float]:
        """获取 as_of 当日(或之前最近)收盘价, 用于 target_deviation。"""
        for sub in ("data_store", "data_cache"):
            path = os.path.join(BASE_DIR, sub, f"{symbol}.parquet")
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_parquet(path, columns=["date", "close"])
                df["date"] = pd.to_datetime(df["date"])
                df = df[df["date"] <= as_of]
                if len(df) == 0:
                    continue
                return float(df.iloc[-1]["close"])
            except Exception:
                continue
        return None

    # ════════════════════════════════════════
    #  批量计算
    # ════════════════════════════════════════

    @staticmethod
    def _consensus_eps_series(snap: pd.DataFrame, fiscal_year: int) -> pd.Series:
        """
        向量化提取快照中每只股票指定财年的一致预期EPS。

        优先 eps_{fiscal_year}, 缺失则取之后最近非空年份。返回 index=symbol 的 Series。
        """
        cols = [f"eps_{yr}" for yr in range(fiscal_year, fiscal_year + 4)
                if f"eps_{yr}" in snap.columns]
        if not cols:
            return pd.Series(np.nan, index=snap.index)
        # bfill 沿年份列方向: 取第一个非空
        return snap[cols].bfill(axis=1).iloc[:, 0]

    def compute_factors(self, as_of_date: str = None,
                        symbols: list = None) -> Dict[str, Dict[str, float]]:
        """
        批量计算所有可用股票的分析师因子 (向量化)。

        Args:
          as_of_date: 截止日期 '2026-08-01', None=今天
          symbols: 指定股票列表, None=取最新快照中的所有股票

        Returns:
          {symbol: {factor_name: value, ...}, ...}  仅保留至少一个非NaN的股票
        """
        if as_of_date is None:
            as_of = pd.Timestamp.now().normalize()
        else:
            as_of = pd.Timestamp(as_of_date)

        latest_date = self._latest_snapshot_date(as_of, within_days=60)
        if latest_date is None:
            return {}
        latest = self._load_snapshot(latest_date)
        if latest is None or len(latest) == 0:
            return {}

        if symbols is not None:
            latest = latest.loc[latest.index.intersection(symbols)]
            if len(latest) == 0:
                return {}

        fiscal_year = as_of.year
        # 覆盖过滤
        rc = latest.get("report_count", pd.Series(np.nan, index=latest.index))
        covered = latest[rc >= self.min_coverage]

        # 初始化结果矩阵 (全 NaN)
        res = pd.DataFrame(np.nan, index=latest.index, columns=FACTOR_NAMES)

        if len(covered) > 0:
            # ── upgrade_ratio ──
            for c in ["buy", "add", "neutral", "reduce", "sell"]:
                if c not in covered.columns:
                    covered[c] = 0.0
            total = (covered["buy"] + covered["add"] + covered["neutral"]
                     + covered["reduce"] + covered["sell"])
            with np.errstate(invalid="ignore", divide="ignore"):
                ur = np.where(total > 0, (covered["buy"] + covered["add"]) / total, np.nan)
            res.loc[covered.index, "upgrade_ratio"] = ur

            # ── revision_30d ──
            d30 = self._prior_snapshot_date(latest_date, min_days=15, max_days=75)
            if d30 is not None and d30 != latest_date:
                snap30 = self._load_snapshot(d30)
                if snap30 is not None and len(snap30) > 0:
                    eps_now = self._consensus_eps_series(covered, fiscal_year)
                    eps_old = self._consensus_eps_series(snap30, fiscal_year)
                    eps_old = eps_old.reindex(covered.index)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        rev = (eps_now - eps_old) / eps_old.abs()
                    rev = rev.where(eps_old.abs() > 1e-6).clip(-2.0, 2.0)
                    res.loc[covered.index, "revision_30d"] = rev

            # ── coverage_change ──
            d90 = self._prior_snapshot_date(latest_date, min_days=45, max_days=150)
            if d90 is not None and d90 != latest_date:
                snap90 = self._load_snapshot(d90)
                if snap90 is not None and "report_count" in snap90.columns:
                    rc90 = snap90["report_count"].reindex(covered.index)
                    cc = covered["report_count"] - rc90
                    res.loc[covered.index, "coverage_change"] = cc

        # ── target_deviation (可选, 需外部目标价缓存) ──
        targets = self._load_target_prices()
        if targets:
            for sym in res.index.intersection(list(targets.keys())):
                price = self._latest_price(sym, as_of)
                if price and price > 0:
                    res.loc[sym, "target_deviation"] = np.clip(targets[sym] / price - 1, -1.0, 5.0)

        # 转为 dict, 仅保留至少一个非NaN的股票
        res = res.dropna(how="all")
        return {sym: {f: (float(row[f]) if pd.notna(row[f]) else np.nan)
                      for f in FACTOR_NAMES}
                for sym, row in res.iterrows()}

    def compute_factor_matrix(self, as_of_date: str = None,
                              symbols: list = None) -> pd.DataFrame:
        """返回因子矩阵 DataFrame (index=symbol, columns=FACTOR_NAMES)。"""
        factors = self.compute_factors(as_of_date, symbols)
        if not factors:
            return pd.DataFrame(columns=FACTOR_NAMES)
        df = pd.DataFrame.from_dict(factors, orient="index")
        df.index.name = "symbol"
        for c in FACTOR_NAMES:
            if c not in df.columns:
                df[c] = np.nan
        return df[FACTOR_NAMES]

    # ════════════════════════════════════════
    #  分数增强 (叠加层, 与 PEAD / MoneyFlow 模式一致)
    # ════════════════════════════════════════

    def enhance_scores(self, base_scores: Dict[str, float],
                       analyst_scores: Dict[str, float] = None,
                       weight: float = 0.08) -> Dict[str, float]:
        """
        将分析师综合分数叠加到基础分数上。

        Args:
          base_scores: 原始因子分数 {symbol: score}
          analyst_scores: 分析师综合分数, None=自动计算
          weight: 权重 (默认8%, 因历史短、覆盖窄, 权重应低)
        """
        if analyst_scores is None:
            analyst_scores = self.compute_composite_score()

        if not analyst_scores:
            return base_scores

        vals = list(analyst_scores.values())
        if len(vals) < 2:
            return base_scores
        mu, sigma = np.mean(vals), np.std(vals)
        if sigma < 1e-8:
            return base_scores

        enhanced = dict(base_scores)
        for sym, av in analyst_scores.items():
            if sym in enhanced:
                enhanced[sym] += weight * (av - mu) / sigma
        return enhanced

    def compute_composite_score(self, as_of_date: str = None,
                                symbols: list = None) -> Dict[str, float]:
        """
        将4个因子合成为单一分数 (截面z-score等权, 方向对齐)。

        方向约定 (正值=看多):
          revision_30d:     + (正=预期上调)
          upgrade_ratio:    + (正=评级偏多)
          target_deviation: + (正=有上行空间)
          coverage_change:  + (正=关注度上升)
        """
        matrix = self.compute_factor_matrix(as_of_date, symbols)
        if matrix.empty:
            return {}

        scores = pd.Series(0.0, index=matrix.index)
        n_factors = 0
        for col in FACTOR_NAMES:
            s = matrix[col].dropna()
            if len(s) < 10:
                continue
            mu, sigma = s.mean(), s.std()
            if sigma < 1e-8:
                continue
            z = (matrix[col] - mu) / sigma
            scores += z.fillna(0)
            n_factors += 1

        if n_factors == 0:
            return {}
        scores /= n_factors
        return scores.to_dict()

    # ════════════════════════════════════════
    #  统计信息
    # ════════════════════════════════════════

    def get_data_stats(self) -> dict:
        """获取分析师数据覆盖情况。"""
        self._build_index()
        n_snaps = len(self._snap_index) if self._snap_index else 0

        stats = {
            "n_snapshots": n_snaps,
            "snapshot_dates": (
                f"{self._snap_index[0].date()} ~ {self._snap_index[-1].date()}"
                if n_snaps else "N/A"
            ),
            "min_coverage": self.min_coverage,
        }

        if n_snaps:
            latest = self._load_snapshot(self._snap_index[-1])
            if latest is not None:
                n_total = len(latest)
                n_covered = int((latest["report_count"] >= self.min_coverage).sum())
                stats["stocks_in_latest"] = n_total
                stats["stocks_covered_ge3"] = n_covered
                stats["coverage_pct"] = round(n_covered / max(n_total, 1), 4)

        stats["limitation"] = (
            "分析师快照无历史(需逐日积累), revision类因子需>=2个快照; "
            "批量接口不提供目标价, target_deviation 需额外 per-stock 缓存"
        )
        return stats

    def clear_cache(self):
        """清除内存缓存 (数据更新后调用)。"""
        self._snap_cache.clear()
        self._snap_index = None
        self._target_price = None


# ════════════════════════════════════════
#  快速测试 / CLI
# ════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="分析师修正因子")
    parser.add_argument("--stats", action="store_true", help="显示数据覆盖统计")
    parser.add_argument("--compute", type=str, default=None, help="计算指定日期的因子")
    parser.add_argument("--top", type=int, default=20, help="显示前N只")
    args = parser.parse_args()

    ar = AnalystRevision()

    if args.stats or not args.compute:
        stats = ar.get_data_stats()
        print("分析师数据覆盖:", flush=True)
        for k, v in stats.items():
            print(f"  {k}: {v}", flush=True)

    if args.compute:
        print(f"\n计算 {args.compute} 分析师因子...", flush=True)
        scores = ar.compute_composite_score(as_of_date=args.compute)
        print(f"  有效股票: {len(scores)}", flush=True)
        top = sorted(scores.items(), key=lambda x: -x[1])[:args.top]
        print(f"\n  Top-{args.top} (综合分):", flush=True)
        for sym, s in top:
            print(f"    {sym}: {s:+.3f}", flush=True)
