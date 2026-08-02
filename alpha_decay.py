"""
alpha_decay.py — 因子衰减监控与动态权重管理系统

解决核心问题: 因子权重一旦确定就永远不变，导致BLIND期IR暴跌。
本模块提供:
  1. RollingICMonitor   — 滚动IC监控，实时检测因子衰减
  2. estimate_ic_half_life / compute_effective_sample_size — IC统计工具
  3. DynamicWeightManager — 动态权重调整，可直接集成到回测循环
  4. AlphaGraveyard     — Alpha墓地，记录已失效因子
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import date, datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# 1. 滚动IC监控
# ---------------------------------------------------------------------------

class RollingICMonitor:
    """滚动IC监控器，跟踪每个因子的每日IC值并检测衰减。

    Parameters
    ----------
    window : int
        滚动窗口大小（交易日数）。
    min_periods : int
        计算滚动统计量所需的最小观测数。
    """

    def __init__(self, window: int = 60, min_periods: int = 30) -> None:
        self.window = window
        self.min_periods = min_periods
        # factor_name -> list of (date, ic_value)
        self._records: Dict[str, List[tuple]] = defaultdict(list)

    def update(self, factor_name: str, date: Union[str, datetime, pd.Timestamp], ic_value: float) -> None:
        """记录某因子在某日的IC值。

        Parameters
        ----------
        factor_name : str
            因子名称。
        date : str | datetime | pd.Timestamp
            日期。
        ic_value : float
            当日IC值。
        """
        ts = pd.Timestamp(date)
        self._records[factor_name].append((ts, ic_value))

    def _to_series(self, factor_name: str) -> pd.Series:
        """将原始记录转为按日期索引的pd.Series。"""
        if factor_name not in self._records:
            return pd.Series(dtype=float)
        records = self._records[factor_name]
        dates = [r[0] for r in records]
        values = [r[1] for r in records]
        s = pd.Series(values, index=pd.DatetimeIndex(dates), name=factor_name)
        s = s.sort_index()
        return s

    def get_rolling_ic(self, factor_name: str) -> pd.Series:
        """获取滚动均值IC序列。

        Parameters
        ----------
        factor_name : str
            因子名称。

        Returns
        -------
        pd.Series
            以日期为索引的滚动均值IC序列。
        """
        s = self._to_series(factor_name)
        if s.empty:
            return s
        return s.rolling(window=self.window, min_periods=self.min_periods).mean()

    def get_current_icir(self, factor_name: str) -> float:
        """计算最近window天的ICIR (IC均值 / IC标准差)。

        Parameters
        ----------
        factor_name : str
            因子名称。

        Returns
        -------
        float
            ICIR值。若数据不足则返回0.0。
        """
        s = self._to_series(factor_name)
        if s.empty:
            return 0.0
        recent = s.iloc[-self.window:]
        if len(recent) < self.min_periods:
            return 0.0
        mean_ic = recent.mean()
        std_ic = recent.std(ddof=1)
        if std_ic == 0 or np.isnan(std_ic):
            return 0.0
        return float(mean_ic / std_ic)

    def is_decayed(self, factor_name: str, threshold: float = 0.0) -> bool:
        """判断因子是否衰减。

        如果最近window天的IC均值 < threshold，判定为衰减。

        Parameters
        ----------
        factor_name : str
            因子名称。
        threshold : float
            衰减阈值，默认0.0。

        Returns
        -------
        bool
            True表示因子已衰减。
        """
        s = self._to_series(factor_name)
        if s.empty:
            return True
        recent = s.iloc[-self.window:]
        if len(recent) < self.min_periods:
            return False  # 数据不足，不做判定
        return float(recent.mean()) < threshold

    def get_decay_alerts(self) -> List[dict]:
        """返回所有衰减因子的告警列表。

        Returns
        -------
        list of dict
            每个dict包含: factor_name, rolling_ic_mean, current_icir, status。
        """
        alerts: List[dict] = []
        for factor_name in self._records:
            s = self._to_series(factor_name)
            if s.empty:
                continue
            recent = s.iloc[-self.window:]
            if len(recent) < self.min_periods:
                continue
            ic_mean = float(recent.mean())
            icir = self.get_current_icir(factor_name)
            if ic_mean < 0.0:
                alerts.append({
                    "factor_name": factor_name,
                    "rolling_ic_mean": round(ic_mean, 6),
                    "current_icir": round(icir, 4),
                    "status": "decayed",
                    "n_obs": len(recent),
                })
        return alerts


# ---------------------------------------------------------------------------
# 2. IC半衰期估计
# ---------------------------------------------------------------------------

def estimate_ic_half_life(ic_series: pd.Series, max_lag: int = 120) -> float:
    """用IC自相关函数拟合指数衰减，估计IC半衰期。

    模型: autocorr(lag) ≈ exp(-lag / half_life)

    Parameters
    ----------
    ic_series : pd.Series
        IC时间序列。
    max_lag : int
        最大滞后期数。

    Returns
    -------
    float
        估计的半衰期（天）。若拟合失败则返回np.inf。
    """
    ic_series = ic_series.dropna()
    n = len(ic_series)
    if n < max_lag + 1:
        max_lag = n - 1
    if max_lag < 5:
        return np.inf

    # 计算自相关
    lags = np.arange(1, max_lag + 1)
    autocorrs = np.array([ic_series.autocorr(lag=int(l)) for l in lags])

    # 过滤掉NaN和非正值（对数拟合需要正值）
    valid = np.isfinite(autocorrs) & (autocorrs > 0)
    if valid.sum() < 3:
        return np.inf

    lags_valid = lags[valid].astype(float)
    ac_valid = autocorrs[valid]

    # 拟合 log(autocorr) = -lag / half_life
    # 即 log(autocorr) = (-1/half_life) * lag
    log_ac = np.log(ac_valid)

    # 线性回归: log_ac = slope * lag, slope = -1/half_life
    try:
        slope, _ = np.polyfit(lags_valid, log_ac, 1)
    except (np.linalg.LinAlgError, ValueError):
        return np.inf

    if slope >= 0:
        # 没有衰减趋势
        return np.inf

    half_life = -1.0 / slope
    return float(half_life)


def compute_effective_sample_size(ic_series: pd.Series) -> float:
    """考虑自相关后的有效样本量。

    公式: n_eff = n * (1 - rho1) / (1 + rho1)
    其中 rho1 是lag-1自相关系数。

    用于修正ICIR的标准误，避免高估显著性。

    Parameters
    ----------
    ic_series : pd.Series
        IC时间序列。

    Returns
    -------
    float
        有效样本量。
    """
    ic_series = ic_series.dropna()
    n = len(ic_series)
    if n < 3:
        return float(n)

    rho1 = ic_series.autocorr(lag=1)
    if rho1 is None or np.isnan(rho1):
        return float(n)

    # 边界保护
    if rho1 >= 1.0:
        return 1.0
    if rho1 <= -1.0:
        return float(n)

    n_eff = n * (1.0 - rho1) / (1.0 + rho1)
    return max(1.0, float(n_eff))


# ---------------------------------------------------------------------------
# 3. 动态权重调整
# ---------------------------------------------------------------------------

class DynamicWeightManager:
    """动态因子权重管理器。

    根据滚动ICIR动态调整因子权重，可直接集成到回测循环中。

    核心逻辑:
      - rolling_ICIR 与 base_ICIR 同号: weight = rolling_ICIR (用最新估计)
      - rolling_ICIR 与 base_ICIR 异号: weight = 0 (因子失效，清零)
      - rolling_ICIR 不显著 (|t| < 1.5): weight = base_weight * 0.5 (降权)
      - 最终归一化: weights / sum(|weights|)

    Parameters
    ----------
    base_weights : dict
        初始ICIR权重，{factor_name: icir_weight}。
    lookback : int
        滚动窗口大小。
    decay_method : str
        衰减方法: "exponential" / "half_life" / "rolling"。
    """

    def __init__(
        self,
        base_weights: Dict[str, float],
        lookback: int = 60,
        decay_method: str = "exponential",
    ) -> None:
        self.base_weights = dict(base_weights)
        self.lookback = lookback
        self.decay_method = decay_method
        self._monitor = RollingICMonitor(window=lookback, min_periods=max(10, lookback // 2))

    def update_ic(self, factor_name: str, date: Union[str, datetime, pd.Timestamp], ic_value: float) -> None:
        """更新某因子的IC记录。

        Parameters
        ----------
        factor_name : str
            因子名称。
        date : str | datetime | pd.Timestamp
            日期。
        ic_value : float
            当日IC值。
        """
        self._monitor.update(factor_name, date, ic_value)

    def _compute_rolling_icir(self, factor_name: str) -> tuple:
        """计算滚动ICIR及其t统计量。

        Returns
        -------
        tuple of (rolling_icir, t_stat)
        """
        s = self._monitor._to_series(factor_name)
        if s.empty:
            return 0.0, 0.0
        recent = s.iloc[-self.lookback:]
        n = len(recent)
        if n < 5:
            return 0.0, 0.0

        mean_ic = recent.mean()
        std_ic = recent.std(ddof=1)
        if std_ic == 0 or np.isnan(std_ic):
            return 0.0, 0.0

        icir = mean_ic / std_ic

        # 有效样本量修正
        n_eff = compute_effective_sample_size(recent)
        # t统计量: ICIR * sqrt(n_eff)
        t_stat = icir * np.sqrt(n_eff)

        return float(icir), float(t_stat)

    def _compute_exponential_decay_weight(self, factor_name: str, rolling_icir: float) -> float:
        """指数衰减方法: 根据IC半衰期对base_weight进行折扣。"""
        s = self._monitor._to_series(factor_name)
        if s.empty or len(s) < 20:
            return self.base_weights.get(factor_name, 0.0)

        half_life = estimate_ic_half_life(s, max_lag=min(120, len(s) - 1))
        base_w = self.base_weights.get(factor_name, 0.0)

        if np.isinf(half_life) or half_life <= 0:
            # 没有衰减，直接用rolling
            return rolling_icir

        # 用最近lookback天的"年龄"来折扣
        # 衰减因子 = exp(-lookback / (2 * half_life))
        decay_factor = np.exp(-self.lookback / (2.0 * half_life))
        return rolling_icir * decay_factor

    def get_dynamic_weights(self, date: Union[str, datetime, pd.Timestamp, None] = None) -> Dict[str, float]:
        """计算当前动态权重。

        Parameters
        ----------
        date : optional
            当前日期（目前未使用，预留接口）。

        Returns
        -------
        dict
            归一化后的因子权重 {factor_name: weight}。
        """
        raw_weights: Dict[str, float] = {}

        for factor_name, base_w in self.base_weights.items():
            rolling_icir, t_stat = self._compute_rolling_icir(factor_name)

            # 数据不足时使用base_weight
            s = self._monitor._to_series(factor_name)
            if s.empty or len(s) < 10:
                raw_weights[factor_name] = base_w
                continue

            # 判断符号
            same_sign = (rolling_icir * base_w) > 0

            if not same_sign:
                # 异号: 因子失效，清零
                raw_weights[factor_name] = 0.0
            elif abs(t_stat) < 1.5:
                # 不显著: 降权
                raw_weights[factor_name] = base_w * 0.5
            else:
                # 同号且显著: 使用最新估计
                if self.decay_method == "exponential":
                    raw_weights[factor_name] = self._compute_exponential_decay_weight(factor_name, rolling_icir)
                elif self.decay_method == "half_life":
                    # 半衰期方法: 类似exponential但更激进
                    s_full = self._monitor._to_series(factor_name)
                    hl = estimate_ic_half_life(s_full, max_lag=min(120, len(s_full) - 1))
                    if np.isinf(hl) or hl <= 0:
                        raw_weights[factor_name] = rolling_icir
                    else:
                        decay_factor = np.exp(-self.lookback / hl)
                        raw_weights[factor_name] = rolling_icir * decay_factor
                else:
                    # "rolling": 纯滚动ICIR
                    raw_weights[factor_name] = rolling_icir

        # 归一化: weights / sum(|weights|)
        total_abs = sum(abs(w) for w in raw_weights.values())
        if total_abs == 0:
            # 所有因子都失效了，回退到base_weights
            total_abs = sum(abs(w) for w in self.base_weights.values())
            if total_abs == 0:
                return {}
            return {k: v / total_abs for k, v in self.base_weights.items()}

        return {k: v / total_abs for k, v in raw_weights.items()}

    def get_factor_status(self) -> Dict[str, dict]:
        """返回每个因子的当前状态。

        Returns
        -------
        dict
            {factor_name: {"status": str, "rolling_icir": float, "weight": float, "t_stat": float}}
            status: "active" | "decayed" | "weakened"
        """
        weights = self.get_dynamic_weights()
        status_map: Dict[str, dict] = {}

        for factor_name, base_w in self.base_weights.items():
            rolling_icir, t_stat = self._compute_rolling_icir(factor_name)
            w = weights.get(factor_name, 0.0)

            if w == 0.0:
                status = "decayed"
            elif abs(t_stat) < 1.5:
                status = "weakened"
            else:
                status = "active"

            status_map[factor_name] = {
                "status": status,
                "rolling_icir": round(rolling_icir, 4),
                "base_icir": round(base_w, 4),
                "t_stat": round(t_stat, 4),
                "weight": round(w, 6),
            }

        return status_map

    def load_from_p5_report(self, path: str) -> None:
        """从P5报告加载base_weights。

        P5报告的 selected_factors 数组中每个元素有 name 和 icir 字段。

        Parameters
        ----------
        path : str
            P5报告JSON文件路径。
        """
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)

        factors = report.get("selected_factors", [])
        self.base_weights = {}
        for factor in factors:
            name = factor["name"]
            icir = factor["icir"]
            # 考虑 weight_multiplier（regime调整）
            multiplier = factor.get("weight_multiplier", 1.0)
            self.base_weights[name] = icir * multiplier

    @classmethod
    def from_p5_report(cls, path: str, lookback: int = 60, decay_method: str = "exponential") -> "DynamicWeightManager":
        """从P5报告创建DynamicWeightManager的工厂方法。

        Parameters
        ----------
        path : str
            P5报告JSON文件路径。
        lookback : int
            滚动窗口。
        decay_method : str
            衰减方法。

        Returns
        -------
        DynamicWeightManager
        """
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)

        factors = report.get("selected_factors", [])
        base_weights: Dict[str, float] = {}
        for factor in factors:
            name = factor["name"]
            icir = factor["icir"]
            multiplier = factor.get("weight_multiplier", 1.0)
            base_weights[name] = icir * multiplier

        return cls(base_weights=base_weights, lookback=lookback, decay_method=decay_method)


# ---------------------------------------------------------------------------
# 4. Alpha墓地
# ---------------------------------------------------------------------------

class AlphaGraveyard:
    """Alpha墓地 — 记录已失效的因子，防止僵尸因子复活。

    Parameters
    ----------
    path : str
        持久化JSON文件路径。
    """

    def __init__(self, path: str = "data/alpha_graveyard.json") -> None:
        self.path = path
        self._dead: Dict[str, dict] = {}
        self.load()

    def bury(self, factor_name: str, reason: str, burial_date: str) -> None:
        """记录一个死亡因子。

        Parameters
        ----------
        factor_name : str
            因子名称。
        reason : str
            死亡原因描述。
        burial_date : str
            埋葬日期 (ISO格式字符串)。
        """
        self._dead[factor_name] = {
            "factor_name": factor_name,
            "reason": reason,
            "burial_date": burial_date,
        }
        self.save()

    def is_dead(self, factor_name: str) -> bool:
        """检查因子是否已死亡。

        Parameters
        ----------
        factor_name : str
            因子名称。

        Returns
        -------
        bool
        """
        return factor_name in self._dead

    def list_dead(self) -> List[dict]:
        """列出所有死亡因子。

        Returns
        -------
        list of dict
        """
        return list(self._dead.values())

    def save(self) -> None:
        """持久化到JSON文件。"""
        os.makedirs(os.path.dirname(self.path) if os.path.dirname(self.path) else ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._dead, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        """从JSON文件加载。"""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._dead = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._dead = {}
        else:
            self._dead = {}
