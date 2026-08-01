"""
事件信号因子 — 限售解禁压力 / 龙虎榜机构行为 / 业绩预告惊喜(增强版)

设计原则:
  事件类因子多为"稀有/二元"信号 (多数股票为0), 且部分数据历史短,
  与 PEAD / MoneyFlow 一致, 作为"实时叠加层"使用, 无数据时透明透传。

数据来源 (由 scripts/fetch_events.py 预拉取):
  - data/factor_cache/events/lockup.parquet  ← ak.stock_restricted_release_detail_em
      列: symbol, name, unlock_date, unlock_type, unlock_shares, actual_shares,
          actual_value, ratio_to_float (占解禁前流通市值比例), prev_close
  - data/factor_cache/events/lhb.parquet     ← ak.stock_lhb_detail_em
      列: symbol, name, date, net_buy, buy_amount, sell_amount, lhb_turnover,
          market_turnover, turnover_rate, float_value, reason
  - data/pead_cache/forecast_*.parquet       ← ak.stock_yjyg_em (业绩预告)
      列含: symbol, ann_date, profit_change_pct (预告净利润变动幅度)

因子列表:
  1. lockup_pressure  — 未来30日解禁市值占流通市值比例之和 (高=抛压, 负向信号)
  2. lhb_institutional— 近5日龙虎榜净买额/当日总成交额 (未上榜=0, 正向信号)
  3. preview_surprise — 业绩预告变动幅度 - 同期截面中位数 (相对惊喜, 正向)

用法:
  from factors.event_signals import EventSignals

  es = EventSignals()
  factors = es.compute_factors(as_of_date="2026-08-01")
  # → {"600519": {"lockup_pressure": 0.0, "lhb_institutional": 0.02, ...}, ...}
"""

import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# 事件缓存目录
EVENTS_CACHE = os.path.join(BASE_DIR, "data", "factor_cache", "events")
PEAD_CACHE = os.path.join(BASE_DIR, "data", "pead_cache")

LOCKUP_PATH = os.path.join(EVENTS_CACHE, "lockup.parquet")
LHB_PATH = os.path.join(EVENTS_CACHE, "lhb.parquet")

# 因子名列表
FACTOR_NAMES = [
    "lockup_pressure",
    "lhb_institutional",
    "preview_surprise",
]


def _ensure_dirs():
    os.makedirs(EVENTS_CACHE, exist_ok=True)


class EventSignals:
    """
    事件信号因子计算器。

    从 data/factor_cache/events/ 加载预拉取的解禁/龙虎榜数据,
    从 data/pead_cache/ 加载业绩预告, 计算3个事件因子。
    """

    def __init__(self, lockup_window_days: int = 30, lhb_lookback_days: int = 5,
                 preview_lookback_days: int = 30):
        """
        Args:
          lockup_window_days: 解禁压力前瞻窗口 (未来N日)
          lhb_lookback_days: 龙虎榜回溯窗口 (近N日上榜视为有效)
          preview_lookback_days: 业绩预告回溯窗口 (近N日公告视为有效)
        """
        _ensure_dirs()
        self.lockup_window_days = lockup_window_days
        self.lhb_lookback_days = lhb_lookback_days
        self.preview_lookback_days = preview_lookback_days
        self._lockup: Optional[pd.DataFrame] = None
        self._lhb: Optional[pd.DataFrame] = None
        self._preview: Optional[pd.DataFrame] = None

    # ════════════════════════════════════════
    #  数据加载
    # ════════════════════════════════════════

    def _load_lockup(self) -> pd.DataFrame:
        """加载限售解禁明细 (全市场单文件)。"""
        if self._lockup is not None:
            return self._lockup
        if not os.path.exists(LOCKUP_PATH):
            self._lockup = pd.DataFrame()
            return self._lockup
        try:
            df = pd.read_parquet(LOCKUP_PATH)
            df["symbol"] = df["symbol"].astype(str).str.zfill(6)
            df["unlock_date"] = pd.to_datetime(df["unlock_date"], errors="coerce")
            for c in ["unlock_shares", "actual_shares", "actual_value",
                      "ratio_to_float", "prev_close"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            self._lockup = df.dropna(subset=["unlock_date"])
        except Exception as e:
            print(f"[EventSignals] 加载 lockup 失败: {e}", flush=True)
            self._lockup = pd.DataFrame()
        return self._lockup

    def _load_lhb(self) -> pd.DataFrame:
        """加载龙虎榜明细 (全市场单文件)。"""
        if self._lhb is not None:
            return self._lhb
        if not os.path.exists(LHB_PATH):
            self._lhb = pd.DataFrame()
            return self._lhb
        try:
            df = pd.read_parquet(LHB_PATH)
            df["symbol"] = df["symbol"].astype(str).str.zfill(6)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            for c in ["net_buy", "buy_amount", "sell_amount",
                      "lhb_turnover", "market_turnover", "turnover_rate"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            self._lhb = df.dropna(subset=["date"])
        except Exception as e:
            print(f"[EventSignals] 加载 lhb 失败: {e}", flush=True)
            self._lhb = pd.DataFrame()
        return self._lhb

    def _load_preview(self) -> pd.DataFrame:
        """加载业绩预告 (复用 pead_cache)。"""
        if self._preview is not None:
            return self._preview
        frames = []
        if os.path.exists(PEAD_CACHE):
            for fname in os.listdir(PEAD_CACHE):
                if fname.startswith("forecast_") and fname.endswith(".parquet"):
                    try:
                        fdf = pd.read_parquet(os.path.join(PEAD_CACHE, fname))
                        if len(fdf) > 0:
                            frames.append(fdf)
                    except Exception:
                        continue
        if not frames:
            self._preview = pd.DataFrame()
            return self._preview

        df = pd.concat(frames, ignore_index=True)
        col_map = {
            "股票代码": "symbol", "公告日期": "ann_date",
            "预告净利润变动幅度": "profit_change_pct",
            "业绩变动幅度": "profit_change_pct",  # akshare 实际列名
            "预测数值": "forecast_net", "上年同期值": "prev_year_net",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.zfill(6)
        if "ann_date" in df.columns:
            df["ann_date"] = pd.to_datetime(df["ann_date"], errors="coerce")
        if "profit_change_pct" in df.columns:
            df["profit_change_pct"] = pd.to_numeric(df["profit_change_pct"], errors="coerce")
        # 回退: 缺失变动幅度时, 用 (预测值 - 上年同期) / |上年同期| × 100 推导
        if "forecast_net" in df.columns and "prev_year_net" in df.columns:
            df["forecast_net"] = pd.to_numeric(df["forecast_net"], errors="coerce")
            df["prev_year_net"] = pd.to_numeric(df["prev_year_net"], errors="coerce")
            if "profit_change_pct" not in df.columns:
                df["profit_change_pct"] = np.nan
            need = df["profit_change_pct"].isna() & df["prev_year_net"].abs().gt(1e-8)
            df.loc[need, "profit_change_pct"] = (
                (df.loc[need, "forecast_net"] - df.loc[need, "prev_year_net"])
                / df.loc[need, "prev_year_net"].abs() * 100.0
            )
        self._preview = df
        return self._preview

    # ════════════════════════════════════════
    #  单只股票因子计算
    # ════════════════════════════════════════

    def _lockup_pressure(self, symbol: str, as_of: pd.Timestamp) -> float:
        """
        未来 lockup_window_days 日解禁压力:
          sum(ratio_to_float) for unlock_date in [as_of, as_of+window]

        ratio_to_float = 解禁市值 / 解禁前流通市值, 直接衡量抛压占比。
        无解禁事件 → 0。
        """
        df = self._load_lockup()
        if len(df) == 0:
            return 0.0
        end = as_of + pd.Timedelta(days=self.lockup_window_days)
        mask = ((df["symbol"] == symbol) &
                (df["unlock_date"] >= as_of) &
                (df["unlock_date"] <= end))
        sub = df[mask]
        if len(sub) == 0:
            return 0.0
        if "ratio_to_float" in sub.columns:
            vals = sub["ratio_to_float"].dropna()
            if len(vals) > 0:
                return float(vals.sum())
        # 回退: 无比例列时用解禁市值粗略归一 (除以流通市值需外部数据, 这里返回原始占比和)
        return 0.0

    def _lhb_institutional(self, symbol: str, as_of: pd.Timestamp) -> float:
        """
        近 lhb_lookback_days 日龙虎榜净买入强度:
          net_buy / market_turnover (取窗口内最近一次上榜)

        未上榜 → 0 (二元/稀有事件, 多数股票为0)。
        """
        df = self._load_lhb()
        if len(df) == 0:
            return 0.0
        start = as_of - pd.Timedelta(days=self.lhb_lookback_days)
        mask = ((df["symbol"] == symbol) &
                (df["date"] >= start) &
                (df["date"] <= as_of))
        sub = df[mask]
        if len(sub) == 0:
            return 0.0
        # 取最近一次上榜
        latest = sub.sort_values("date").iloc[-1]
        net = latest.get("net_buy", np.nan)
        mkt = latest.get("market_turnover", np.nan)
        if pd.notna(net) and pd.notna(mkt) and mkt > 0:
            return float(np.clip(net / mkt, -1.0, 1.0))
        return 0.0

    def _preview_surprise(self, symbol: str, as_of: pd.Timestamp,
                          period_median: Optional[float] = None) -> Optional[float]:
        """
        业绩预告相对惊喜:
          preview_surprise = (announced_change_pct - expected_change_pct) / 100

        announced  = 预告净利润变动幅度 (stock_yjyg_em, %)
        expected   = 同期截面中位数 (相对惊喜基准; 默认0=绝对惊喜)

        取 preview_lookback_days 内最新一条预告。无预告 → None。
        """
        df = self._load_preview()
        if len(df) == 0 or "profit_change_pct" not in df.columns:
            return None
        start = as_of - pd.Timedelta(days=self.preview_lookback_days)
        mask = ((df["symbol"] == symbol) &
                (df["ann_date"] >= start) &
                (df["ann_date"] <= as_of))
        sub = df[mask]
        if len(sub) == 0:
            return None
        latest = sub.sort_values("ann_date").iloc[-1]
        announced = latest["profit_change_pct"]
        if pd.isna(announced):
            return None
        expected = period_median if period_median is not None else 0.0
        surprise = (announced - expected) / 100.0
        return float(np.clip(surprise, -5.0, 5.0))

    def _compute_period_median(self, as_of: pd.Timestamp) -> Optional[float]:
        """计算 preview 窗口内全市场预告变动幅度的截面中位数 (相对惊喜基准)。"""
        df = self._load_preview()
        if len(df) == 0 or "profit_change_pct" not in df.columns:
            return None
        start = as_of - pd.Timedelta(days=self.preview_lookback_days)
        mask = (df["ann_date"] >= start) & (df["ann_date"] <= as_of)
        vals = df.loc[mask, "profit_change_pct"].dropna()
        if len(vals) < 5:
            return None
        return float(vals.median())

    def _compute_single(self, symbol: str, as_of: pd.Timestamp,
                        period_median: Optional[float] = None) -> Dict[str, float]:
        """计算单只股票的3个事件因子。缺失值统一为 np.nan (二元信号 lockup/lhb 默认0)。"""
        preview = self._preview_surprise(symbol, as_of, period_median)
        return {
            "lockup_pressure": self._lockup_pressure(symbol, as_of),
            "lhb_institutional": self._lhb_institutional(symbol, as_of),
            "preview_surprise": np.nan if preview is None else preview,
        }

    # ════════════════════════════════════════
    #  向量化批量计算 (O(events) per day, 而非 O(symbols×events))
    # ════════════════════════════════════════

    def _lockup_vector(self, as_of: pd.Timestamp) -> Dict[str, float]:
        """全部股票的解禁压力 {symbol: pressure} (仅含有事件的股票)。"""
        df = self._load_lockup()
        if len(df) == 0 or "ratio_to_float" not in df.columns:
            return {}
        end = as_of + pd.Timedelta(days=self.lockup_window_days)
        mask = (df["unlock_date"] >= as_of) & (df["unlock_date"] <= end)
        sub = df[mask].dropna(subset=["ratio_to_float"])
        if len(sub) == 0:
            return {}
        return sub.groupby("symbol")["ratio_to_float"].sum().to_dict()

    def _lhb_vector(self, as_of: pd.Timestamp) -> Dict[str, float]:
        """全部股票的龙虎榜净买强度 {symbol: net_buy/market_turnover} (取窗口内最近一次)。"""
        df = self._load_lhb()
        if len(df) == 0:
            return {}
        start = as_of - pd.Timedelta(days=self.lhb_lookback_days)
        mask = (df["date"] >= start) & (df["date"] <= as_of)
        sub = df[mask].copy()
        if len(sub) == 0:
            return {}
        # 每只股票取最近一次上榜
        sub = sub.sort_values("date").groupby("symbol", as_index=False).tail(1)
        out = {}
        for _, r in sub.iterrows():
            net, mkt = r.get("net_buy", np.nan), r.get("market_turnover", np.nan)
            if pd.notna(net) and pd.notna(mkt) and mkt > 0:
                out[r["symbol"]] = float(np.clip(net / mkt, -1.0, 1.0))
        return out

    def _preview_vector(self, as_of: pd.Timestamp,
                        period_median: Optional[float]) -> Dict[str, float]:
        """全部股票的预告相对惊喜 {symbol: surprise} (取窗口内最新一条)。"""
        df = self._load_preview()
        if len(df) == 0 or "profit_change_pct" not in df.columns:
            return {}
        start = as_of - pd.Timedelta(days=self.preview_lookback_days)
        mask = (df["ann_date"] >= start) & (df["ann_date"] <= as_of)
        sub = df[mask].dropna(subset=["profit_change_pct"]).copy()
        if len(sub) == 0:
            return {}
        sub = sub.sort_values("ann_date").groupby("symbol", as_index=False).tail(1)
        expected = period_median if period_median is not None else 0.0
        out = {}
        for _, r in sub.iterrows():
            s = (r["profit_change_pct"] - expected) / 100.0
            out[r["symbol"]] = float(np.clip(s, -5.0, 5.0))
        return out

    # ════════════════════════════════════════
    #  批量计算
    # ════════════════════════════════════════

    def compute_factors(self, as_of_date: str = None,
                        symbols: list = None) -> Dict[str, Dict[str, float]]:
        """
        批量计算事件因子 (向量化, 每日 O(events))。

        Args:
          as_of_date: 截止日期 '2026-08-01', None=今天
          symbols: 指定股票列表, None=自动取 (有事件 ∪ data_store) 的并集

        Returns:
          {symbol: {factor_name: value, ...}, ...}
          lockup/lhb 无事件时为 0.0; preview 无预告时为 NaN。
        """
        if as_of_date is None:
            as_of = pd.Timestamp.now().normalize()
        else:
            as_of = pd.Timestamp(as_of_date)

        if symbols is None:
            symbols = self._default_symbols(as_of)

        period_median = self._compute_period_median(as_of)
        lockup_v = self._lockup_vector(as_of)
        lhb_v = self._lhb_vector(as_of)
        preview_v = self._preview_vector(as_of, period_median)

        results = {}
        for sym in symbols:
            results[sym] = {
                "lockup_pressure": float(lockup_v.get(sym, 0.0)),
                "lhb_institutional": float(lhb_v.get(sym, 0.0)),
                "preview_surprise": preview_v.get(sym, np.nan),
            }
        return results

    def _default_symbols(self, as_of: pd.Timestamp) -> list:
        """默认股票池: 有事件的股票 ∪ data_store 全部。"""
        syms = set()
        # 有解禁/龙虎榜事件的股票
        lockup = self._load_lockup()
        if len(lockup) > 0:
            end = as_of + pd.Timedelta(days=self.lockup_window_days)
            m = (lockup["unlock_date"] >= as_of - pd.Timedelta(days=90)) & (lockup["unlock_date"] <= end)
            syms.update(lockup.loc[m, "symbol"].tolist())
        lhb = self._load_lhb()
        if len(lhb) > 0:
            m = lhb["date"] >= as_of - pd.Timedelta(days=self.lhb_lookback_days)
            syms.update(lhb.loc[m, "symbol"].tolist())
        # data_store 全量 (便于截面IC)
        ds = os.path.join(BASE_DIR, "data_store")
        if os.path.exists(ds):
            for f in os.listdir(ds):
                if f.endswith(".parquet") and f[0].isdigit():
                    syms.add(f.replace(".parquet", ""))
        return sorted(syms)

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
    #  分数增强 (叠加层)
    # ════════════════════════════════════════

    def enhance_scores(self, base_scores: Dict[str, float],
                       event_scores: Dict[str, float] = None,
                       weight: float = 0.08) -> Dict[str, float]:
        """将事件综合分数叠加到基础分数上 (lockup 方向取反)。"""
        if event_scores is None:
            event_scores = self.compute_composite_score()
        if not event_scores:
            return base_scores
        vals = list(event_scores.values())
        if len(vals) < 2:
            return base_scores
        mu, sigma = np.mean(vals), np.std(vals)
        if sigma < 1e-8:
            return base_scores
        enhanced = dict(base_scores)
        for sym, ev in event_scores.items():
            if sym in enhanced:
                enhanced[sym] += weight * (ev - mu) / sigma
        return enhanced

    def compute_composite_score(self, as_of_date: str = None,
                                symbols: list = None) -> Dict[str, float]:
        """
        合成单一事件分数 (截面z-score等权, 方向对齐)。

        方向约定 (正值=看多):
          lockup_pressure:   - (高抛压=负面, 取反)
          lhb_institutional: + (机构净买入=正面)
          preview_surprise:  + (相对惊喜=正面)
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
            if col == "lockup_pressure":
                z = -z  # 抛压取反
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
        """获取事件数据覆盖情况。"""
        lockup = self._load_lockup()
        lhb = self._load_lhb()
        preview = self._load_preview()

        stats = {
            "lockup_events": len(lockup),
            "lockup_stocks": int(lockup["symbol"].nunique()) if len(lockup) > 0 else 0,
            "lockup_date_range": (
                f"{lockup['unlock_date'].min().date()} ~ {lockup['unlock_date'].max().date()}"
                if len(lockup) > 0 else "N/A"
            ),
            "lhb_events": len(lhb),
            "lhb_stocks": int(lhb["symbol"].nunique()) if len(lhb) > 0 else 0,
            "lhb_date_range": (
                f"{lhb['date'].min().date()} ~ {lhb['date'].max().date()}"
                if len(lhb) > 0 else "N/A"
            ),
            "preview_events": len(preview),
        }
        stats["limitation"] = (
            "lockup/lhb 为稀有事件, 多数股票因子值为0; 历史长度取决于fetch范围"
        )
        return stats

    def clear_cache(self):
        """清除内存缓存。"""
        self._lockup = None
        self._lhb = None
        self._preview = None


# ════════════════════════════════════════
#  快速测试 / CLI
# ════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="事件信号因子")
    parser.add_argument("--stats", action="store_true", help="显示数据覆盖统计")
    parser.add_argument("--compute", type=str, default=None, help="计算指定日期的因子")
    parser.add_argument("--top", type=int, default=20, help="显示前N只")
    args = parser.parse_args()

    es = EventSignals()

    if args.stats or not args.compute:
        stats = es.get_data_stats()
        print("事件数据覆盖:", flush=True)
        for k, v in stats.items():
            print(f"  {k}: {v}", flush=True)

    if args.compute:
        print(f"\n计算 {args.compute} 事件因子...", flush=True)
        scores = es.compute_composite_score(as_of_date=args.compute)
        print(f"  有效股票: {len(scores)}", flush=True)
        top = sorted(scores.items(), key=lambda x: -x[1])[:args.top]
        print(f"\n  Top-{args.top} (综合分):", flush=True)
        for sym, s in top:
            print(f"    {sym}: {s:+.3f}", flush=True)
