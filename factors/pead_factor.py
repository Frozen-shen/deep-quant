"""
PEAD 事件因子 — 业绩预告惊喜度打分

作为现有因子系统的增强叠加层:
  - 有业绩预告事件时: 惊喜度 ± 调整分数
  - 无事件时: 不影响原始打分, 完全回退到价量因子

设计原则:
  PEAD 因子不替代价量因子, 而是作为"催化剂"层
  ——有事件时增强/抑制信号, 无事件时透明透传。

用法:
  from factors.pead_factor import PEADFactor

  pead = PEADFactor()
  scores = pead.compute_surprise_scores(as_of_date="2026-08-03")
  # → {"600519": 0.12, "000858": -0.05, ...}  (只有有事件预告的股票非零)

  # 在 FactorScorer 中集成:
  adjusted = pead.enhance_scores(base_scores, surprise_scores, weight=0.15)
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# PEAD 事件缓存目录
PEAD_CACHE_DIR = os.path.join(BASE_DIR, "data", "pead_cache")


def _ensure_cache_dir():
    os.makedirs(PEAD_CACHE_DIR, exist_ok=True)


class PEADFactor:
    """
    PEAD (Post-Earnings Announcement Drift) 因子。

    每日扫描 akshare stock_yjyg_em 获取最新业绩预告,
    计算 surprise = (预测净利润 - 上年同期净利润) / |上年同期净利润|,
    正惊喜 → 买入加分, 负惊喜 → 卖出减分。
    """

    def __init__(self, decay_days: int = 30, max_surprise_abs: float = 5.0):
        """
        Args:
          decay_days: 事件影响衰减天数 (超过N天后surprise归零)
          max_surprise_abs: 惊喜度绝对值上限 (截断极端值)
        """
        self.decay_days = decay_days
        self.max_surprise_abs = max_surprise_abs
        self._cache: Optional[pd.DataFrame] = None
        self._cache_date: Optional[str] = None
        _ensure_cache_dir()

    # ════════════════════════════════════════
    #  数据拉取
    # ════════════════════════════════════════

    def _fetch_forecasts(self, year: int, period: str = "1231") -> Optional[pd.DataFrame]:
        """获取指定年份/报告期的业绩预告。"""
        cache_path = os.path.join(PEAD_CACHE_DIR, f"forecast_{year}{period}.parquet")

        # 先读缓存
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            if len(df) > 0:
                return df

        # 从 akshare 拉取
        try:
            import akshare as ak
            import warnings
            warnings.filterwarnings('ignore')
            df = ak.stock_yjyg_em(date=f"{year}{period}")
            if df is not None and len(df) > 0:
                df.to_parquet(cache_path, index=False)
                return df
        except Exception as e:
            print(f"[PEAD] 拉取 {year}{period} 失败: {e}")

        return None

    def _load_all_forecasts(self) -> pd.DataFrame:
        """加载所有已缓存的业绩预告数据。"""
        forecasts = []
        # 拉取 2020-2026 各期
        for year in range(2020, 2027):
            for period in ["0331", "0630", "0930", "1231"]:
                df = self._fetch_forecasts(year, period)
                if df is not None and len(df) > 0:
                    forecasts.append(df)

        if not forecasts:
            return pd.DataFrame()

        df = pd.concat(forecasts, ignore_index=True)
        return self._clean_forecasts(df)

    def _clean_forecasts(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗业绩预告数据, 计算惊喜度。"""
        df = df.copy()

        # 标准化列名 (akshare 可能返回不同列名)
        col_map = {
            '股票代码': 'symbol', '股票简称': 'name',
            '公告日期': 'ann_date', '预测数值': 'forecast_net',
            '上年同期值': 'prev_year_net', '业绩变动': 'change_type',
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if 'symbol' not in df.columns or 'ann_date' not in df.columns:
            print("[PEAD] 数据列名不匹配, 无法计算惊喜度")
            return pd.DataFrame()

        # 补零
        df['symbol'] = df['symbol'].astype(str).str.zfill(6)
        df['ann_date'] = pd.to_datetime(df['ann_date'])

        # 数值转换
        if 'forecast_net' in df.columns:
            df['forecast_net'] = pd.to_numeric(df['forecast_net'], errors='coerce')
        if 'prev_year_net' in df.columns:
            df['prev_year_net'] = pd.to_numeric(df['prev_year_net'], errors='coerce')

        # 计算惊喜度
        valid = (df['forecast_net'].notna() &
                df['prev_year_net'].notna() &
                (df['prev_year_net'] != 0))
        df = df[valid].copy()

        df['surprise'] = (df['forecast_net'] - df['prev_year_net']) / df['prev_year_net'].abs()
        df['surprise'] = df['surprise'].clip(-self.max_surprise_abs, self.max_surprise_abs)

        # 分类
        df['surprise_sign'] = np.where(df['surprise'] > 0, 'positive', 'negative')

        return df[['symbol', 'ann_date', 'surprise', 'surprise_sign', 'forecast_net', 'prev_year_net']]

    def _get_cached_forecasts(self) -> pd.DataFrame:
        """获取缓存的业绩预告（带日期缓存避免重复加载）。"""
        today = datetime.now().strftime("%Y%m%d")
        if self._cache is not None and self._cache_date == today:
            return self._cache

        self._cache = self._load_all_forecasts()
        self._cache_date = today
        return self._cache

    # ════════════════════════════════════════
    #  惊喜度计算
    # ════════════════════════════════════════

    def compute_surprise_scores(self, as_of_date: str = None) -> Dict[str, float]:
        """
        计算截至指定日期的 PEAD 惊喜度分数。

        Args:
          as_of_date: '2026-08-03' 或 None=今天

        Returns:
          {symbol: surprise_score}  — 只有有事件且未过期的股票非零
        """
        if as_of_date is None:
            as_of_dt = pd.Timestamp.now().normalize()
        else:
            as_of_dt = pd.Timestamp(as_of_date)

        forecasts = self._get_cached_forecasts()
        if len(forecasts) == 0:
            return {}

        # 筛选: 公告日在 as_of_date 之前, 且在 decay_days 窗口内
        cutoff = as_of_dt - timedelta(days=self.decay_days)
        recent = forecasts[
            (forecasts['ann_date'] <= as_of_dt) &
            (forecasts['ann_date'] >= cutoff)
        ]

        if len(recent) == 0:
            return {}

        # 按股票聚合 (同一股票短期内可能有多期预告, 取最新)
        scores = {}
        for sym, group in recent.groupby('symbol'):
            # 取最新一条预告的惊喜度
            latest = group.sort_values('ann_date').iloc[-1]
            scores[sym] = float(latest['surprise'])

        return scores

    def compute_latest_events(self, lookback_days: int = 7) -> pd.DataFrame:
        """
        获取最近N天新发布的业绩预告列表 (用于实时监控)。

        Returns:
          DataFrame with columns [symbol, ann_date, surprise, surprise_sign]
        """
        forecasts = self._get_cached_forecasts()
        if len(forecasts) == 0:
            return pd.DataFrame()

        cutoff = pd.Timestamp.now().normalize() - timedelta(days=lookback_days)
        recent = forecasts[forecasts['ann_date'] >= cutoff]
        return recent.sort_values('ann_date', ascending=False)

    # ════════════════════════════════════════
    #  分数增强 (叠加层)
    # ════════════════════════════════════════

    def enhance_scores(self, base_scores: Dict[str, float],
                       surprise_scores: Dict[str, float] = None,
                       weight: float = 0.15) -> Dict[str, float]:
        """
        将 PEAD 惊喜度叠加到基分数上。

        Args:
          base_scores: 原始因子分数 {symbol: score} (如 ic_auto 的输出)
          surprise_scores: PEAD惊喜度 {symbol: surprise}, None=自动计算
          weight: PEAD 权重 (0~1, 默认15%)

        Returns:
          增强后的分数 {symbol: score}
        """
        if surprise_scores is None:
            surprise_scores = self.compute_surprise_scores()

        if not surprise_scores:
            return base_scores  # 无事件, 透传

        # 对 surprise 做 z-score 标准化 (跨所有有事件股票)
        surprises = list(surprise_scores.values())
        if len(surprises) < 2:
            # 只有1个事件, 直接用符号
            mean_s, std_s = 0.0, 1.0
        else:
            mean_s = np.mean(surprises)
            std_s = np.std(surprises) if np.std(surprises) > 0 else 1.0

        enhanced = {}
        for sym, base in base_scores.items():
            if sym in surprise_scores:
                # z-score 标准化 surprise + 权重叠加
                z_surprise = (surprise_scores[sym] - mean_s) / std_s
                enhanced[sym] = base + weight * z_surprise
            else:
                enhanced[sym] = base

        return enhanced

    def get_event_stats(self) -> dict:
        """获取PEAD事件统计信息。"""
        forecasts = self._get_cached_forecasts()
        if len(forecasts) == 0:
            return {"has_data": False, "total_events": 0}

        pos = len(forecasts[forecasts['surprise_sign'] == 'positive'])
        neg = len(forecasts[forecasts['surprise_sign'] == 'negative'])
        return {
            "has_data": True,
            "total_events": len(forecasts),
            "positive": pos,
            "negative": neg,
            "date_range": f"{forecasts['ann_date'].min().date()} ~ {forecasts['ann_date'].max().date()}",
            "mean_surprise": float(forecasts['surprise'].mean()),
        }


# ════════════════════════════════════════
#  快速测试 / CLI
# ════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PEAD事件因子")
    parser.add_argument("--stats", action="store_true", help="显示事件统计")
    parser.add_argument("--latest", type=int, default=7, help="显示最近N天新事件")
    parser.add_argument("--scores", type=str, default=None, help="计算指定日期的惊喜度")
    args = parser.parse_args()

    pead = PEADFactor()

    if args.stats:
        stats = pead.get_event_stats()
        print(f"PEAD 事件统计:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    if args.latest:
        events = pead.compute_latest_events(args.latest)
        print(f"\n最近 {args.latest} 天新事件: {len(events)} 条")
        for _, e in events.head(20).iterrows():
            sgn = "📈" if e['surprise'] > 0 else "📉"
            print(f"  {sgn} {e['symbol']} {e['ann_date'].date()} "
                  f"surprise={e['surprise']:+.2f}")

    if args.scores:
        scores = pead.compute_surprise_scores(args.scores)
        print(f"\n{args.scores} 有效PEAD事件: {len(scores)} 只")
        for sym, s in sorted(scores.items(), key=lambda x: -abs(x[1]))[:20]:
            print(f"  {sym}: {s:+.3f}")
