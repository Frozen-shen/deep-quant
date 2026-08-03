"""
walk_forward.py — Walk-Forward 动态 IC 权重计算

核心思想:
  静态 IC 权重在 research 期训练后固定不变, 但因子有效性会随市场
  风格轮动而衰减。Walk-forward 在每个调仓日用 trailing window
  重新估计 IC/ICIR, 使权重自适应当前市场环境。

方法:
  1. 取 trailing N 个月 (默认12个月) 的调仓日
  2. 对每个因子, 计算截面 IC (Spearman rank corr with forward return)
  3. ICIR = mean(IC_series) / std(IC_series)
  4. 用 ICIR 作为动态权重 (带衰减: 近期IC权重更高)

用法:
    from walk_forward import WalkForwardICWeighter
    weighter = WalkForwardICWeighter(factor_cache, all_data, lookback_months=12)
    adapted_factors = weighter.update_weights(factors, as_of_date)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from scipy import stats

from logger import get_logger

log = get_logger("walk_forward")


class WalkForwardICWeighter:
    """
    Walk-Forward 动态 IC 权重计算器。

    在每个调仓日, 用 trailing window 重新估计各因子的 ICIR,
    替代静态权重, 实现因子权重的自适应更新。
    """

    def __init__(self, factor_cache, all_data: Dict[str, pd.DataFrame],
                 lookback_months: int = 12,
                 min_ic_obs: int = 6,
                 decay_halflife: int = 60,
                 winsorize_pct: float = 0.01):
        """
        参数:
          factor_cache: FactorCache 实例, 提供历史因子值
          all_data: {symbol: DataFrame} 全量行情数据
          lookback_months: trailing window 长度 (月)
          min_ic_obs: 最少 IC 观测数 (低于此数用静态权重)
          decay_halflife: IC 时间衰减半衰期 (天), 近期 IC 权重更高
          winsorize_pct: 极值缩尾比例
        """
        self.factor_cache = factor_cache
        self.all_data = all_data
        self.lookback_months = lookback_months
        self.min_ic_obs = min_ic_obs
        self.decay_halflife = decay_halflife
        self.winsorize_pct = winsorize_pct

        # 缓存: {date_str: {factor_name: icir}}
        self._icir_cache = {}
        # 预计算交易日列表
        self._trading_dates = self._build_trading_calendar()

    def _build_trading_calendar(self) -> List:
        """从 all_data 构建交易日历。"""
        all_dates = set()
        for df in self.all_data.values():
            if "date" in df.columns:
                all_dates.update(pd.to_datetime(df["date"]).dt.date.tolist())
        return sorted(all_dates)

    def _get_trailing_rebalance_dates(self, as_of_date, rebalance_days: int = 20) -> List:
        """获取 trailing window 内的调仓日列表。"""
        as_of_ts = pd.Timestamp(as_of_date)
        start_ts = as_of_ts - pd.DateOffset(months=self.lookback_months)
        start_date = start_ts.date()
        as_of_date_only = as_of_ts.date()

        # 找到 window 内的交易日
        window_dates = [d for d in self._trading_dates
                        if start_date <= d <= as_of_date_only]

        # 每隔 rebalance_days 取一个调仓日
        rebalance_dates = window_dates[::rebalance_days]
        return rebalance_dates

    def _compute_forward_returns(self, as_of_date, horizon: int = 20) -> Dict[str, float]:
        """计算 as_of_date 之后 horizon 天的前瞻收益率。"""
        as_of_ts = pd.Timestamp(as_of_date)
        fwd_returns = {}

        for sym, df in self.all_data.items():
            dates = pd.to_datetime(df["date"])
            mask = dates >= as_of_ts
            future = df[mask].head(horizon + 1)
            if len(future) < 2:
                continue
            ret = future["close"].iloc[-1] / future["close"].iloc[0] - 1
            if not np.isnan(ret):
                fwd_returns[sym] = ret

        return fwd_returns

    def _compute_cross_sectional_ic(self, factor_values: Dict[str, float],
                                     fwd_returns: Dict[str, float]) -> float:
        """计算单个调仓日的截面 IC (Spearman rank correlation)。"""
        common = set(factor_values.keys()) & set(fwd_returns.keys())
        if len(common) < 30:  # 至少30只股票才有意义
            return np.nan

        fv = np.array([factor_values[s] for s in common])
        fr = np.array([fwd_returns[s] for s in common])

        # Winsorize
        if self.winsorize_pct > 0:
            lo_f, hi_f = np.nanpercentile(fv, [self.winsorize_pct * 100, (1 - self.winsorize_pct) * 100])
            lo_r, hi_r = np.nanpercentile(fr, [self.winsorize_pct * 100, (1 - self.winsorize_pct) * 100])
            fv = np.clip(fv, lo_f, hi_f)
            fr = np.clip(fr, lo_r, hi_r)

        # 去除 NaN
        valid = ~(np.isnan(fv) | np.isnan(fr))
        if valid.sum() < 30:
            return np.nan

        ic, _ = stats.spearmanr(fv[valid], fr[valid])
        return ic

    def compute_trailing_icir(self, factor_name: str, as_of_date,
                               rebalance_days: int = 20) -> Optional[float]:
        """
        计算单个因子在 trailing window 内的 ICIR。

        Returns:
          ICIR 值, 或 None (数据不足)
        """
        rebalance_dates = self._get_trailing_rebalance_dates(as_of_date, rebalance_days)
        if len(rebalance_dates) < self.min_ic_obs:
            return None

        ic_series = []
        dates_used = []

        for rb_date in rebalance_dates:
            # 获取该调仓日的因子截面值
            factor_values = {}
            for sym in self.all_data:
                feats = self.factor_cache.get(sym, rb_date)
                if feats is not None and factor_name in feats:
                    val = feats[factor_name]
                    if not np.isnan(val):
                        factor_values[sym] = val

            if len(factor_values) < 30:
                continue

            # 计算前瞻收益
            fwd_returns = self._compute_forward_returns(rb_date, horizon=rebalance_days)

            # 计算 IC
            ic = self._compute_cross_sectional_ic(factor_values, fwd_returns)
            if not np.isnan(ic):
                ic_series.append(ic)
                dates_used.append(rb_date)

        if len(ic_series) < self.min_ic_obs:
            return None

        ic_arr = np.array(ic_series)

        # 时间衰减加权: 近期 IC 权重更高
        if self.decay_halflife > 0 and len(ic_arr) > 1:
            n = len(ic_arr)
            # 用位置做衰减 (最近的权重最高, dates_used 按时间升序)
            positions = np.arange(n)[::-1].astype(float)  # 最近的=0, 最远的=n-1
            decay_weights = np.exp(-np.log(2) * positions / self.decay_halflife)
            decay_weights /= decay_weights.sum()

            weighted_mean = np.sum(ic_arr * decay_weights)
            # 加权标准差
            weighted_var = np.sum(decay_weights * (ic_arr - weighted_mean) ** 2)
            weighted_std = np.sqrt(weighted_var)
        else:
            weighted_mean = np.mean(ic_arr)
            weighted_std = np.std(ic_arr)

        if weighted_std < 1e-9:
            return 0.0

        icir = weighted_mean / weighted_std
        return icir

    def update_weights(self, factors: List[Dict], as_of_date,
                       rebalance_days: int = 20) -> List[Dict]:
        """
        用 walk-forward ICIR 更新因子权重。

        参数:
          factors: 原始因子列表 (含静态 icir)
          as_of_date: 当前调仓日
          rebalance_days: 调仓间隔

        Returns:
          更新后的因子列表 (icir 字段被替换为动态值)
        """
        as_of_ts = pd.Timestamp(as_of_date).date()
        cache_key = str(as_of_ts)

        # 检查缓存
        if cache_key in self._icir_cache:
            cached = self._icir_cache[cache_key]
            adapted = []
            for f in factors:
                new_f = f.copy()
                if f["name"] in cached:
                    new_f["icir"] = cached[f["name"]]
                    new_f["weight_source"] = "walk_forward"
                adapted.append(new_f)
            return adapted

        log.info("[WF] 计算动态IC权重: %s (trailing %d months)",
                 as_of_ts, self.lookback_months)

        dynamic_icir = {}
        n_updated = 0
        n_fallback = 0

        for f in factors:
            name = f["name"]
            category = f.get("category", "price_volume")

            # 只对 price_volume 类因子做 walk-forward
            # 基本面/分钟因子数据不够长, 用静态权重
            if category != "price_volume":
                dynamic_icir[name] = f["icir"]
                continue

            icir = self.compute_trailing_icir(name, as_of_ts, rebalance_days)
            if icir is not None:
                dynamic_icir[name] = icir
                n_updated += 1
            else:
                # 数据不足, 用静态权重
                dynamic_icir[name] = f["icir"]
                n_fallback += 1

        # 缓存结果
        self._icir_cache[cache_key] = dynamic_icir

        log.info("[WF] 动态权重: %d 更新, %d 回退静态", n_updated, n_fallback)

        # 构建更新后的因子列表
        adapted = []
        for f in factors:
            new_f = f.copy()
            if f["name"] in dynamic_icir:
                new_f["icir"] = dynamic_icir[f["name"]]
                new_f["weight_source"] = "walk_forward" if f.get("category") == "price_volume" else "static"
            adapted.append(new_f)

        return adapted

    def get_weight_report(self, factors: List[Dict], as_of_date) -> Dict:
        """生成权重变化报告 (用于诊断)。"""
        as_of_ts = pd.Timestamp(as_of_date).date()
        cache_key = str(as_of_ts)

        if cache_key not in self._icir_cache:
            return {"error": "No cached weights for this date"}

        dynamic = self._icir_cache[cache_key]
        report = {
            "date": str(as_of_ts),
            "factors": []
        }

        for f in factors:
            name = f["name"]
            static_icir = f.get("icir", 0)
            dyn_icir = dynamic.get(name, static_icir)
            change = dyn_icir - static_icir
            report["factors"].append({
                "name": name,
                "static_icir": round(static_icir, 4),
                "dynamic_icir": round(dyn_icir, 4),
                "change": round(change, 4),
                "direction": "up" if change > 0.01 else ("down" if change < -0.01 else "stable"),
            })

        # 排序: 变化最大的在前
        report["factors"].sort(key=lambda x: abs(x["change"]), reverse=True)
        return report
