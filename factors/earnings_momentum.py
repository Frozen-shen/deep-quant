"""
盈利动量因子 — 基于变化/惊喜的Alpha信号

与现有基本面因子(fund_roe等LEVEL值, ICIR<0.13)不同,
本模块关注的是"变化量"和"超预期", 学术上被证明有更强预测力:

因子列表:
  1. em_sue          — 标准化超预期盈余 (Standardized Unexpected Earnings)
  2. em_roe_accel    — ROE加速度 (单季度同比变化)
  3. em_rev_surprise — 营收惊喜 (增速的二阶导)
  4. em_preview      — 业绩预告惊喜 (立即可用, 无45天延迟)
  5. em_accrual      — 应计比率 (盈余质量, 负向信号)

PIT规则:
  - 季报: 报告期结束 + 45天后可用 (Q1→5/15, Q2→8/14, Q3→11/14, Q4→2/14)
  - 业绩预告: 公告日当天即可用 (无延迟)
  - 绝不使用未来数据

用法:
  from factors.earnings_momentum import EarningsMomentum

  em = EarningsMomentum()
  scores = em.compute_all(as_of_date="2023-06-01")
  # → DataFrame[symbol, em_sue, em_roe_accel, em_rev_surprise, em_preview, em_accrual]
"""

import os
import sys
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

CACHE_DIR = os.path.join(BASE_DIR, "data", "fundamental_cache")
PEAD_CACHE_DIR = os.path.join(BASE_DIR, "data", "pead_cache")

# PIT 延迟天数
PIT_LAG_DAYS = 45

# 因子名称列表
EM_FACTOR_NAMES = [
    'em_sue', 'em_roe_accel', 'em_rev_surprise', 'em_preview', 'em_accrual',
]


def safe_float(val):
    """安全转换为float"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _quarter_label(date: pd.Timestamp) -> str:
    """返回季度标签, 如 '2023Q1'"""
    q = (date.month - 1) // 3 + 1
    return f"{date.year}Q{q}"


def _same_quarter_last_year(date: pd.Timestamp) -> str:
    """返回去年同期季度标签, 如 date='2023Q2' → '2022Q2'"""
    q = (date.month - 1) // 3 + 1
    return f"{date.year - 1}Q{q}"


class EarningsMomentum:
    """
    盈利动量因子引擎。

    基于 fundamental_cache 中的季度财务数据计算5个动量/惊喜因子,
    严格遵循PIT规则, 绝不使用未来数据。
    """

    def __init__(self, cache_dir: str = None, pead_cache_dir: str = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.pead_cache_dir = pead_cache_dir or PEAD_CACHE_DIR
        self._fund_cache: Dict[str, pd.DataFrame] = {}
        self._preview_cache: Optional[pd.DataFrame] = None

    # ════════════════════════════════════════
    #  数据加载
    # ════════════════════════════════════════

    def _load_fundamentals(self, symbol: str) -> Optional[pd.DataFrame]:
        """加载单只股票的季度财务数据 (带内存缓存)"""
        if symbol in self._fund_cache:
            return self._fund_cache[symbol]

        path = os.path.join(self.cache_dir, f"{symbol}.parquet")
        if not os.path.exists(path):
            self._fund_cache[symbol] = None
            return None

        try:
            df = pd.read_parquet(path)
            if '日期' in df.columns:
                df['日期'] = pd.to_datetime(df['日期'])
                df = df.sort_values('日期').reset_index(drop=True)
            self._fund_cache[symbol] = df
            return df
        except Exception:
            self._fund_cache[symbol] = None
            return None

    def _load_previews(self) -> pd.DataFrame:
        """加载所有业绩预告缓存数据"""
        if self._preview_cache is not None:
            return self._preview_cache

        frames = []
        if not os.path.exists(self.pead_cache_dir):
            self._preview_cache = pd.DataFrame()
            return self._preview_cache

        for fname in os.listdir(self.pead_cache_dir):
            if fname.startswith("forecast_") and fname.endswith(".parquet"):
                try:
                    fdf = pd.read_parquet(os.path.join(self.pead_cache_dir, fname))
                    if len(fdf) > 0:
                        frames.append(fdf)
                except Exception:
                    continue

        if not frames:
            self._preview_cache = pd.DataFrame()
            return self._preview_cache

        df = pd.concat(frames, ignore_index=True)

        # 标准化列名
        col_map = {
            '股票代码': 'symbol', '股票简称': 'name',
            '公告日期': 'ann_date', '预测数值': 'forecast_net',
            '上年同期值': 'prev_year_net', '业绩预告类型': 'change_type',
            '预告净利润变动幅度': 'profit_change_pct',
            '预告净利润下限': 'profit_lower', '预告净利润上限': 'profit_upper',
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if 'symbol' in df.columns:
            df['symbol'] = df['symbol'].astype(str).str.zfill(6)
        if 'ann_date' in df.columns:
            df['ann_date'] = pd.to_datetime(df['ann_date'], errors='coerce')

        self._preview_cache = df
        return self._preview_cache

    # ════════════════════════════════════════
    #  PIT 过滤
    # ════════════════════════════════════════

    def _pit_available(self, df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        """
        PIT过滤: 只返回 as_of 日期可用的财报行。

        规则: 报告期('日期'列) + 45天 <= as_of
        即: 报告期 <= as_of - 45天
        """
        cutoff = as_of - timedelta(days=PIT_LAG_DAYS)
        if '日期' not in df.columns:
            return pd.DataFrame()
        return df[df['日期'] <= cutoff].copy()

    def _pit_available_previews(self, df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        """
        业绩预告PIT: 公告日 <= as_of 即可用 (无延迟)。
        """
        if 'ann_date' not in df.columns or len(df) == 0:
            return pd.DataFrame()
        return df[df['ann_date'] <= as_of].copy()

    # ════════════════════════════════════════
    #  因子计算
    # ════════════════════════════════════════

    def _compute_sue(self, symbol: str, as_of: pd.Timestamp) -> Optional[float]:
        """
        SUE (标准化超预期盈余):
          sue = (actual_eps - expected_eps) / std(eps, last_4_quarters)

        朴素预期: expected = 去年同期同季度EPS
        """
        df = self._load_fundamentals(symbol)
        if df is None or len(df) < 5:
            return None

        available = self._pit_available(df, as_of)
        if len(available) < 5:
            return None

        # 提取EPS列
        eps_col = '摊薄每股收益(元)'
        if eps_col not in available.columns:
            return None

        eps_series = available[eps_col].apply(safe_float)
        available = available.copy()
        available['_eps'] = eps_series
        available = available.dropna(subset=['_eps'])

        if len(available) < 5:
            return None

        latest = available.iloc[-1]
        latest_date = latest['日期']
        latest_eps = latest['_eps']

        # 找去年同期
        target_label = _same_quarter_last_year(latest_date)
        available['_qlabel'] = available['日期'].apply(_quarter_label)
        same_q_ly = available[available['_qlabel'] == target_label]

        if len(same_q_ly) == 0:
            return None

        expected_eps = same_q_ly.iloc[-1]['_eps']

        # 用最近4个季度的EPS计算标准差
        recent_4 = available.tail(4)['_eps'].values
        std_eps = np.std(recent_4, ddof=1)

        if std_eps < 1e-8:
            # 标准差为0 → 无法标准化, 用绝对变化
            surprise = latest_eps - expected_eps
            if abs(surprise) < 1e-8:
                return 0.0
            return None  # 方差为0但有surprise, 不可靠

        sue = (latest_eps - expected_eps) / std_eps
        return float(np.clip(sue, -5.0, 5.0))

    def _compute_roe_accel(self, symbol: str, as_of: pd.Timestamp) -> Optional[float]:
        """
        ROE加速度:
          roe_accel = roe_this_quarter - roe_same_quarter_last_year

        使用单季度ROE (非累计TTM)
        """
        df = self._load_fundamentals(symbol)
        if df is None or len(df) < 5:
            return None

        available = self._pit_available(df, as_of)
        if len(available) < 5:
            return None

        roe_col = '净资产收益率(%)'
        if roe_col not in available.columns:
            return None

        available = available.copy()
        available['_roe'] = available[roe_col].apply(safe_float)
        available = available.dropna(subset=['_roe'])

        if len(available) < 5:
            return None

        latest = available.iloc[-1]
        latest_date = latest['日期']
        latest_roe = latest['_roe']

        # 找去年同期
        target_label = _same_quarter_last_year(latest_date)
        available['_qlabel'] = available['日期'].apply(_quarter_label)
        same_q_ly = available[available['_qlabel'] == target_label]

        if len(same_q_ly) == 0:
            return None

        roe_ly = same_q_ly.iloc[-1]['_roe']
        accel = latest_roe - roe_ly

        # 截断极端值 (ROE变动超过±50%视为异常)
        return float(np.clip(accel, -50.0, 50.0))

    def _compute_rev_surprise(self, symbol: str, as_of: pd.Timestamp) -> Optional[float]:
        """
        营收惊喜 (增速加速度):
          rev_surprise = rev_yoy_this_q - rev_yoy_last_q

        即: 本季度营收同比增速 - 上季度营收同比增速
        正值 = 营收增长在加速
        """
        df = self._load_fundamentals(symbol)
        if df is None or len(df) < 6:
            return None

        available = self._pit_available(df, as_of)
        if len(available) < 6:
            return None

        rev_col = '主营业务收入增长率(%)'
        if rev_col not in available.columns:
            return None

        available = available.copy()
        available['_rev_g'] = available[rev_col].apply(safe_float)
        available = available.dropna(subset=['_rev_g'])

        if len(available) < 2:
            return None

        # 本季度增速 vs 上季度增速
        rev_this_q = available.iloc[-1]['_rev_g']
        rev_last_q = available.iloc[-2]['_rev_g']

        surprise = rev_this_q - rev_last_q

        # 截断 (增速变动超过±200pp视为异常)
        return float(np.clip(surprise, -200.0, 200.0))

    def _compute_preview_surprise(self, symbol: str, as_of: pd.Timestamp) -> Optional[float]:
        """
        业绩预告惊喜:
          preview_surprise = (预告净利润 - 上年同期) / |上年同期|

        使用 stock_yjyg_em 数据, 公告日当天即可用 (无45天延迟)。
        取最近30天内的最新预告。
        """
        previews = self._load_previews()
        if len(previews) == 0:
            return None

        # PIT: 公告日 <= as_of
        avail = self._pit_available_previews(previews, as_of)
        if len(avail) == 0:
            return None

        # 筛选该股票
        sym_previews = avail[avail['symbol'] == symbol]
        if len(sym_previews) == 0:
            return None

        # 取最近30天内的最新预告
        cutoff = as_of - timedelta(days=30)
        recent = sym_previews[sym_previews['ann_date'] >= cutoff]
        if len(recent) == 0:
            return None

        latest = recent.sort_values('ann_date').iloc[-1]

        # 优先使用预告净利润变动幅度
        if 'profit_change_pct' in latest.index:
            pct = safe_float(latest.get('profit_change_pct'))
            if pct is not None:
                # 变动幅度已经是百分比形式, 转为小数
                return float(np.clip(pct / 100.0, -5.0, 5.0))

        # 回退: 用预测数值 vs 上年同期
        forecast = safe_float(latest.get('forecast_net'))
        prev_year = safe_float(latest.get('prev_year_net'))

        if forecast is not None and prev_year is not None and abs(prev_year) > 1e-8:
            surprise = (forecast - prev_year) / abs(prev_year)
            return float(np.clip(surprise, -5.0, 5.0))

        return None

    def _compute_accrual(self, symbol: str, as_of: pd.Timestamp) -> Optional[float]:
        """
        应计比率 (盈余质量):
          accrual = (净利润 - 经营现金流) / 总资产

        高应计 = 低质量盈余 → 负向信号 (值越小越好)
        """
        df = self._load_fundamentals(symbol)
        if df is None or len(df) < 1:
            return None

        available = self._pit_available(df, as_of)
        if len(available) == 0:
            return None

        latest = available.iloc[-1]

        # 尝试多种列名 (不同版本akshare返回不同列名)
        net_income = safe_float(latest.get('净利润(万元)')) or \
                     safe_float(latest.get('净利润(元)'))
        ocf = safe_float(latest.get('每股经营性现金流(元)'))
        total_assets = safe_float(latest.get('总资产(万元)')) or \
                       safe_float(latest.get('总资产(元)'))

        # 如果有每股数据, 尝试用EPS和每股净资产近似
        if net_income is None or ocf is None or total_assets is None:
            # 使用简化版: 用已有列近似
            eps = safe_float(latest.get('摊薄每股收益(元)'))
            bvps = safe_float(latest.get('每股净资产_调整前(元)'))
            ocf_ps = safe_float(latest.get('每股经营性现金流(元)'))

            if eps is not None and ocf_ps is not None and bvps is not None and abs(bvps) > 1e-8:
                # 近似: (EPS - 每股OCF) / 每股净资产
                accrual = (eps - ocf_ps) / abs(bvps)
                return float(np.clip(accrual, -2.0, 2.0))
            return None

        if abs(total_assets) < 1e-8:
            return None

        accrual = (net_income - ocf) / abs(total_assets)
        return float(np.clip(accrual, -2.0, 2.0))

    # ════════════════════════════════════════
    #  批量计算
    # ════════════════════════════════════════

    def compute_single(self, symbol: str, as_of: pd.Timestamp) -> Dict[str, Optional[float]]:
        """
        计算单只股票在指定日期的所有盈利动量因子。

        Returns:
          {factor_name: value_or_None}
        """
        return {
            'em_sue': self._compute_sue(symbol, as_of),
            'em_roe_accel': self._compute_roe_accel(symbol, as_of),
            'em_rev_surprise': self._compute_rev_surprise(symbol, as_of),
            'em_preview': self._compute_preview_surprise(symbol, as_of),
            'em_accrual': self._compute_accrual(symbol, as_of),
        }

    def compute_all(self, as_of_date: str, symbols: List[str] = None) -> pd.DataFrame:
        """
        批量计算所有股票的盈利动量因子。

        Args:
          as_of_date: '2023-06-01' 格式日期
          symbols: 股票代码列表, None=自动从缓存目录获取

        Returns:
          DataFrame[symbol, em_sue, em_roe_accel, em_rev_surprise, em_preview, em_accrual]
        """
        as_of = pd.Timestamp(as_of_date)

        if symbols is None:
            symbols = self._get_cached_symbols()

        print(f"[EarningsMomentum] 计算 {len(symbols)} 只 @ {as_of_date}", flush=True)

        rows = []
        for i, sym in enumerate(symbols):
            factors = self.compute_single(sym, as_of)
            factors['symbol'] = sym
            rows.append(factors)

            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(symbols)}", flush=True)

        result = pd.DataFrame(rows)
        cols = ['symbol'] + EM_FACTOR_NAMES
        result = result[cols]

        # 统计覆盖率
        for col in EM_FACTOR_NAMES:
            valid = result[col].notna().sum()
            print(f"  {col}: {valid}/{len(result)} 有效 ({valid/len(result)*100:.1f}%)", flush=True)

        return result

    def _get_cached_symbols(self) -> List[str]:
        """获取fundamental_cache中所有已缓存的股票代码"""
        if not os.path.exists(self.cache_dir):
            return []
        return sorted([
            f.replace(".parquet", "")
            for f in os.listdir(self.cache_dir)
            if f.endswith(".parquet")
        ])

    # ════════════════════════════════════════
    #  预计算接口 (与 FactorCache 对齐)
    # ════════════════════════════════════════

    def precompute(self, all_data: dict, sample_freq: int = 5) -> dict:
        """
        为所有股票预计算盈利动量因子 (用于回测pipeline)。

        与 fundamental_cache_builder.precompute_fundamental_factors 接口对齐。
        每隔 sample_freq 个交易日重新计算一次 (因子更新频率低, 无需每日计算)。

        Args:
          all_data: {symbol: DataFrame(date, open, close, ...)}
          sample_freq: 每隔N天计算一次 (默认5=每周)

        Returns:
          {symbol: DataFrame(date, em_sue, em_roe_accel, ...)}
        """
        em_cache = {}
        total = len(all_data)

        print(f"[EarningsMomentum] 预计算 {total} 只...", flush=True)

        for i, (sym, df) in enumerate(all_data.items()):
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{total}", flush=True)

            dates = pd.to_datetime(df['date'].tolist()).sort_values()
            # 每隔N天采样计算
            sample_dates = dates.iloc[::sample_freq]

            factor_rows = []
            for d in sample_dates:
                factors = self.compute_single(sym, d)
                factors['date'] = d
                factor_rows.append(factors)

            if not factor_rows:
                fdf = pd.DataFrame({'date': dates.tolist()})
                for name in EM_FACTOR_NAMES:
                    fdf[name] = np.nan
                em_cache[sym] = fdf
                continue

            # 构建因子时间序列, ffill到每个交易日
            fdf = pd.DataFrame(factor_rows).sort_values('date')
            fdf['date'] = pd.to_datetime(fdf['date'])

            tdf = pd.DataFrame({'date': dates.tolist()})
            tdf['date'] = tdf['date'].astype('datetime64[us]')
            fdf['date'] = fdf['date'].astype('datetime64[us]')

            merged = pd.merge_asof(
                tdf, fdf, on='date', direction='backward'
            )

            # 补充缺失列
            for name in EM_FACTOR_NAMES:
                if name not in merged.columns:
                    merged[name] = np.nan

            em_cache[sym] = merged[['date'] + EM_FACTOR_NAMES]

        print(f"  完成: {len(em_cache)} 只", flush=True)
        return em_cache

    def clear_cache(self):
        """清除内存缓存 (用于数据更新后)"""
        self._fund_cache.clear()
        self._preview_cache = None


# ════════════════════════════════════════
#  快速测试 / CLI
# ════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="盈利动量因子")
    parser.add_argument("--symbol", type=str, default="000001", help="测试股票代码")
    parser.add_argument("--date", type=str, default="2024-06-30", help="计算日期")
    parser.add_argument("--batch", type=int, default=0, help="批量计算N只 (0=全部)")
    args = parser.parse_args()

    em = EarningsMomentum()

    if args.batch > 0:
        symbols = em._get_cached_symbols()[:args.batch]
        result = em.compute_all(args.date, symbols)
        print(f"\n结果 ({len(result)} 只):")
        print(result.dropna(how='all', subset=EM_FACTOR_NAMES).head(20).to_string())
    else:
        print(f"\n{args.symbol} @ {args.date}:")
        factors = em.compute_single(args.symbol, pd.Timestamp(args.date))
        for k, v in factors.items():
            if v is not None:
                print(f"  {k}: {v:+.4f}")
            else:
                print(f"  {k}: N/A")
