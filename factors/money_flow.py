"""
资金流因子 — 捕捉机构/聪明钱行为信号

设计原则:
  资金流数据历史短 (1-2年), 不适合作为独立策略的alpha来源,
  而是作为"实时叠加层"叠加在长历史价量因子之上。

  与 PEAD 因子类似, 无数据时返回 NaN, 不影响基础打分。

因子列表:
  1. net_inflow_5d   — 5日主力净流入率 (主力净买入 / 总成交额)
  2. north_change_5d — 5日北向持仓变动 (仅沪深港通, 其余 NaN)
  3. smart_money_20d — 20日大单净买入率均值
  4. flow_momentum   — 资金流动量 (5日净流入率 - 20日净流入率)
  5. flow_vol        — 20日资金流波动率 (高波动=分歧, 负面信号)

用法:
  from factors.money_flow import MoneyFlowFactor

  mf = MoneyFlowFactor()
  factors = mf.compute_factors(as_of_date="2026-08-01")
  # → {"600519": {"net_inflow_5d": 0.03, ...}, ...}
"""

import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# 资金流缓存目录
FUND_FLOW_CACHE = os.path.join(BASE_DIR, "data", "factor_cache", "money_flow")
NORTH_FLOW_CACHE = os.path.join(BASE_DIR, "data", "factor_cache", "north_flow")

# 因子名列表
FACTOR_NAMES = [
    "net_inflow_5d",
    "north_change_5d",
    "smart_money_20d",
    "flow_momentum",
    "flow_vol",
]


def _ensure_dirs():
    os.makedirs(FUND_FLOW_CACHE, exist_ok=True)
    os.makedirs(NORTH_FLOW_CACHE, exist_ok=True)


class MoneyFlowFactor:
    """
    资金流因子计算器。

    从 data/factor_cache/money_flow/{symbol}.parquet 加载历史资金流,
    从 data/factor_cache/north_flow/{symbol}.parquet 加载北向持仓,
    计算5个行为因子。
    """

    def __init__(self):
        _ensure_dirs()
        self._fund_cache: Dict[str, pd.DataFrame] = {}
        self._north_cache: Dict[str, pd.DataFrame] = {}

    # ════════════════════════════════════════
    #  数据加载
    # ════════════════════════════════════════

    def _load_fund_flow(self, symbol: str) -> Optional[pd.DataFrame]:
        """加载个股资金流数据 (带内存缓存)。"""
        if symbol in self._fund_cache:
            return self._fund_cache[symbol]

        path = os.path.join(FUND_FLOW_CACHE, f"{symbol}.parquet")
        if not os.path.exists(path):
            self._fund_cache[symbol] = None
            return None

        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        self._fund_cache[symbol] = df
        return df

    def _load_north_flow(self, symbol: str) -> Optional[pd.DataFrame]:
        """加载北向持仓数据 (带内存缓存)。"""
        if symbol in self._north_cache:
            return self._north_cache[symbol]

        path = os.path.join(NORTH_FLOW_CACHE, f"{symbol}.parquet")
        if not os.path.exists(path):
            self._north_cache[symbol] = None
            return None

        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        self._north_cache[symbol] = df
        return df

    # ════════════════════════════════════════
    #  单只股票因子计算
    # ════════════════════════════════════════

    def _compute_single(self, symbol: str, as_of: pd.Timestamp) -> Dict[str, float]:
        """
        计算单只股票的5个资金流因子。

        要求数据列:
          fund_flow: date, main_net_inflow, large_net_inflow, total_turnover
          north_flow: date, holding_ratio (北向持仓占比)
        """
        result = {f: np.nan for f in FACTOR_NAMES}

        # ── 资金流因子 (4个) ──
        fund_df = self._load_fund_flow(symbol)
        if fund_df is not None and len(fund_df) > 0:
            # 截取 as_of 之前的数据
            mask = fund_df["date"] <= as_of
            df = fund_df[mask].copy()

            if len(df) >= 5:
                # 1. net_inflow_5d: 5日主力净流入 / 5日总成交额
                recent5 = df.tail(5)
                turnover_5d = recent5["total_turnover"].sum()
                if turnover_5d > 0:
                    result["net_inflow_5d"] = (
                        recent5["main_net_inflow"].sum() / turnover_5d
                    )

                # 3. smart_money_20d: 20日 (超大单+大单净买入) / 总成交额 的均值
                if len(df) >= 20:
                    recent20 = df.tail(20)
                    # large_net_inflow = 超大单净买入 + 大单净买入
                    daily_ratio = np.where(
                        recent20["total_turnover"].values > 0,
                        recent20["large_net_inflow"].values / recent20["total_turnover"].values,
                        np.nan
                    )
                    valid_ratios = daily_ratio[~np.isnan(daily_ratio)]
                    if len(valid_ratios) > 0:
                        result["smart_money_20d"] = float(np.mean(valid_ratios))

                # 4. flow_momentum: net_inflow_5d - net_inflow_20d
                if len(df) >= 20 and not np.isnan(result["net_inflow_5d"]):
                    recent20 = df.tail(20)
                    turnover_20d = recent20["total_turnover"].sum()
                    if turnover_20d > 0:
                        net_inflow_20d = recent20["main_net_inflow"].sum() / turnover_20d
                        result["flow_momentum"] = result["net_inflow_5d"] - net_inflow_20d

                # 5. flow_vol: 20日 (daily_net_inflow / daily_turnover) 的标准差
                if len(df) >= 20:
                    recent20 = df.tail(20)
                    daily_flow = np.where(
                        recent20["total_turnover"].values > 0,
                        recent20["main_net_inflow"].values / recent20["total_turnover"].values,
                        np.nan
                    )
                    valid_flow = daily_flow[~np.isnan(daily_flow)]
                    if len(valid_flow) >= 10:
                        result["flow_vol"] = float(np.std(valid_flow, ddof=1))

        # ── 北向因子 ──
        north_df = self._load_north_flow(symbol)
        if north_df is not None and len(north_df) >= 6:
            mask = north_df["date"] <= as_of
            ndf = north_df[mask]

            if len(ndf) >= 6:
                # north_change_5d: 今日持仓占比 - 5日前持仓占比
                today_ratio = ndf.iloc[-1]["holding_ratio"]
                # 找5个交易日前的记录
                if len(ndf) >= 6:
                    five_ago_ratio = ndf.iloc[-6]["holding_ratio"]
                    result["north_change_5d"] = today_ratio - five_ago_ratio

        return result

    # ════════════════════════════════════════
    #  批量计算
    # ════════════════════════════════════════

    def compute_factors(self, as_of_date: str = None,
                        symbols: list = None) -> Dict[str, Dict[str, float]]:
        """
        批量计算所有可用股票的资金流因子。

        Args:
          as_of_date: 截止日期 '2026-08-01', None=今天
          symbols: 指定股票列表, None=自动扫描缓存目录

        Returns:
          {symbol: {factor_name: value, ...}, ...}
        """
        if as_of_date is None:
            as_of = pd.Timestamp.now().normalize()
        else:
            as_of = pd.Timestamp(as_of_date)

        if symbols is None:
            # 自动扫描缓存目录中的所有股票
            symbols = self._get_cached_symbols()

        results = {}
        for sym in symbols:
            factors = self._compute_single(sym, as_of)
            # 只保留至少有一个非NaN值的股票
            if any(not np.isnan(v) for v in factors.values()):
                results[sym] = factors

        return results

    def compute_factor_matrix(self, as_of_date: str = None,
                              symbols: list = None) -> pd.DataFrame:
        """
        返回因子矩阵 DataFrame (index=symbol, columns=factor_names)。

        便于直接用于排名和相关性分析。
        """
        factors = self.compute_factors(as_of_date, symbols)
        if not factors:
            return pd.DataFrame(columns=FACTOR_NAMES)

        df = pd.DataFrame.from_dict(factors, orient="index")
        df.index.name = "symbol"
        return df

    def _get_cached_symbols(self) -> list:
        """扫描缓存目录获取可用股票列表 (fund_flow + north_flow 并集)。"""
        symbols = set()
        if os.path.exists(FUND_FLOW_CACHE):
            for f in os.listdir(FUND_FLOW_CACHE):
                if f.endswith(".parquet"):
                    symbols.add(f.replace(".parquet", ""))
        if os.path.exists(NORTH_FLOW_CACHE):
            for f in os.listdir(NORTH_FLOW_CACHE):
                if f.endswith(".parquet"):
                    symbols.add(f.replace(".parquet", ""))
        return sorted(symbols)

    # ════════════════════════════════════════
    #  分数增强 (叠加层, 与 PEAD 模式一致)
    # ════════════════════════════════════════

    def enhance_scores(self, base_scores: Dict[str, float],
                       flow_scores: Dict[str, float] = None,
                       weight: float = 0.10) -> Dict[str, float]:
        """
        将资金流因子叠加到基础分数上。

        Args:
          base_scores: 原始因子分数 {symbol: score}
          flow_scores: 资金流综合分数 {symbol: flow_score}, None=自动计算
          weight: 资金流权重 (默认10%, 因为历史短、可靠性低于价量因子)

        Returns:
          增强后的分数 {symbol: score}
        """
        if flow_scores is None:
            flow_scores = self.compute_composite_score()

        if not flow_scores:
            return base_scores

        # z-score 标准化
        vals = list(flow_scores.values())
        if len(vals) < 2:
            return base_scores
        mu = np.mean(vals)
        sigma = np.std(vals)
        if sigma < 1e-8:
            return base_scores

        enhanced = dict(base_scores)
        for sym, fv in flow_scores.items():
            if sym in enhanced:
                z = (fv - mu) / sigma
                enhanced[sym] += weight * z

        return enhanced

    def compute_composite_score(self, as_of_date: str = None,
                                symbols: list = None) -> Dict[str, float]:
        """
        将5个因子合成为单一分数 (等权, 方向对齐后)。

        方向约定 (正值=看多):
          net_inflow_5d:   + (正=机构买入)
          north_change_5d: + (正=北向加仓)
          smart_money_20d: + (正=大单持续买入)
          flow_momentum:   + (正=加速流入)
          flow_vol:        - (高波动=分歧, 负面)
        """
        matrix = self.compute_factor_matrix(as_of_date, symbols)
        if matrix.empty:
            return {}

        # 对每个因子做截面 z-score
        scores = pd.Series(0.0, index=matrix.index)
        n_factors = 0

        for col in FACTOR_NAMES:
            if col not in matrix.columns:
                continue
            s = matrix[col].dropna()
            if len(s) < 10:
                continue
            mu, sigma = s.mean(), s.std()
            if sigma < 1e-8:
                continue
            z = (matrix[col] - mu) / sigma
            # flow_vol 方向取反
            if col == "flow_vol":
                z = -z
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
        """获取资金流数据覆盖情况。"""
        fund_syms = []
        north_syms = []

        if os.path.exists(FUND_FLOW_CACHE):
            fund_syms = [f.replace(".parquet", "") for f in os.listdir(FUND_FLOW_CACHE)
                         if f.endswith(".parquet")]
        if os.path.exists(NORTH_FLOW_CACHE):
            north_syms = [f.replace(".parquet", "") for f in os.listdir(NORTH_FLOW_CACHE)
                          if f.endswith(".parquet")]

        # 检查数据长度
        fund_days = []
        for sym in fund_syms[:20]:  # 抽样
            path = os.path.join(FUND_FLOW_CACHE, f"{sym}.parquet")
            try:
                df = pd.read_parquet(path, columns=["date"])
                fund_days.append(len(df))
            except Exception:
                pass

        return {
            "fund_flow_stocks": len(fund_syms),
            "north_flow_stocks": len(north_syms),
            "fund_flow_sample_days": (
                f"min={min(fund_days)}, max={max(fund_days)}, median={int(np.median(fund_days))}"
                if fund_days else "N/A"
            ),
            "limitation": "资金流数据仅1-2年历史, 建议作为实时叠加层使用",
        }


# ════════════════════════════════════════
#  快速测试 / CLI
# ════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="资金流因子")
    parser.add_argument("--stats", action="store_true", help="显示数据覆盖统计")
    parser.add_argument("--compute", type=str, default=None, help="计算指定日期的因子")
    parser.add_argument("--top", type=int, default=20, help="显示前N只")
    args = parser.parse_args()

    mf = MoneyFlowFactor()

    if args.stats:
        stats = mf.get_data_stats()
        print("资金流数据覆盖:")
        for k, v in stats.items():
            print(f"  {k}: {v}", flush=True)

    if args.compute:
        print(f"\n计算 {args.compute} 资金流因子...", flush=True)
        scores = mf.compute_composite_score(as_of_date=args.compute)
        print(f"  有效股票: {len(scores)}", flush=True)
        top = sorted(scores.items(), key=lambda x: -x[1])[:args.top]
        print(f"\n  Top-{args.top} (综合分):")
        for sym, s in top:
            print(f"    {sym}: {s:+.3f}", flush=True)

    if not args.stats and not args.compute:
        # 默认: 显示统计
        stats = mf.get_data_stats()
        print("资金流数据覆盖:")
        for k, v in stats.items():
            print(f"  {k}: {v}", flush=True)
