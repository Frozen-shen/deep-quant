"""
高级市场状态检测 (Advanced Regime Detection)

解决原始 RegimeDetector 的四个核心问题:
  1. 滞后严重 → HMM 概率模型 + 变点检测提前识别转折
  2. 硬切换 → 渐进过渡 (概率加权混合权重)
  3. 单尺度 → 多尺度 Regime (5d / 20d / 60d)
  4. 无过渡概率 → HMM 输出状态概率矩阵

组件:
  - HMMRegimeDetector: 基于 Gaussian HMM 的三状态 (bull/bear/sideways) 检测
  - detect_changepoints: CUSUM 变点检测
  - bayesian_online_changepoint: 在线贝叶斯变点检测 (BOCPD)
  - MultiScaleRegime: 多尺度动量/波动率/趋势综合判断
  - compute_transition_weights: 概率加权渐进过渡

依赖: numpy, pandas, scipy, hmmlearn
"""

import warnings
import concurrent.futures
from typing import Dict, List, Optional

import numpy as np
from scipy import stats


# ═══════════════════════════════════════════════════════════
# 1. HMM Regime 检测
# ═══════════════════════════════════════════════════════════


class HMMRegimeDetector:
    """
    基于 Gaussian Hidden Markov Model 的市场状态检测器。

    三个隐状态:
      - bull: 高均值收益、低/中波动
      - bear: 低均值收益(负)、高波动
      - sideways: 接近零均值、中等波动

    特征工程:
      - daily_return: 日收益率
      - 5d_volatility: 5日滚动波动率
      - 20d_momentum: 20日累计收益 (动量)

    用法:
        detector = HMMRegimeDetector(n_states=3, lookback=252)
        detector.fit(returns)
        result = detector.get_current_regime(returns)
        # {"state": "bull", "probabilities": {"bull": 0.7, ...}, "confidence": 0.7}
    """

    def __init__(self, n_states: int = 3, lookback: int = 252):
        """
        初始化 HMM Regime 检测器。

        Args:
            n_states: 隐状态数量 (默认3: bull/bear/sideways)
            lookback: 训练窗口长度 (交易日数, 默认252≈1年)
        """
        self.n_states = n_states
        self.lookback = lookback
        self._model = None
        self._state_names: Optional[List[str]] = None
        self._fitted = False

    def _build_features(self, returns: np.ndarray) -> np.ndarray:
        """
        从收益率序列构建特征矩阵。

        Args:
            returns: 一维日收益率数组

        Returns:
            (n_samples, 3) 特征矩阵: [daily_return, 5d_vol, 20d_momentum]
            前19个样本因窗口不足被丢弃。
        """
        returns = np.asarray(returns, dtype=np.float64)
        n = len(returns)

        # 日收益率
        daily_ret = returns.copy()

        # 5日滚动波动率
        vol_5d = np.full(n, np.nan)
        for i in range(4, n):
            vol_5d[i] = np.std(returns[i - 4:i + 1], ddof=1)

        # 20日累计收益 (动量)
        mom_20d = np.full(n, np.nan)
        cum = np.cumsum(returns)
        for i in range(19, n):
            mom_20d[i] = cum[i] - cum[i - 19] if i >= 20 else cum[i]

        # 丢弃前19个不完整样本
        start = 19
        features = np.column_stack([
            daily_ret[start:],
            vol_5d[start:],
            mom_20d[start:]
        ])

        # 处理残余 NaN (用0填充)
        features = np.nan_to_num(features, nan=0.0)
        return features

    def _name_states(self, model) -> List[str]:
        """
        根据各状态的收益率均值和波动率命名。

        规则:
          - 最高均值 → bull
          - 最低均值 → bear
          - 其余 → sideways
        """
        means = model.means_[:, 0]  # 第一列是 daily_return
        sorted_idx = np.argsort(means)

        names = ["sideways"] * self.n_states
        names[sorted_idx[-1]] = "bull"
        names[sorted_idx[0]] = "bear"
        return names

    def fit(self, returns: np.ndarray, timeout: float = 60.0):
        """
        用 Gaussian HMM 拟合收益率序列。

        Args:
            returns: 一维日收益率数组 (如 CSI1000 的日收益率)
            timeout: 最大拟合时间 (秒), 超时抛出 TimeoutError

        Raises:
            TimeoutError: 拟合超时
            ValueError: 数据不足
        """
        from hmmlearn.hmm import GaussianHMM

        returns = np.asarray(returns, dtype=np.float64)
        if len(returns) < 50:
            raise ValueError(f"数据不足: 需要至少50个样本, 实际 {len(returns)}")

        # 只用最近 lookback 个样本训练
        if len(returns) > self.lookback:
            train_returns = returns[-self.lookback:]
        else:
            train_returns = returns

        features = self._build_features(train_returns)

        def _do_fit():
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=200,
                random_state=42,
                tol=1e-4,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(features)
            return model

        # 超时保护
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_do_fit)
            try:
                self._model = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"HMM 拟合超时 ({timeout}s)。数据长度={len(train_returns)}, "
                    f"特征维度={features.shape}"
                )

        self._state_names = self._name_states(self._model)
        self._fitted = True

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """
        预测隐状态序列。

        Args:
            returns: 一维日收益率数组

        Returns:
            一维整数数组, 每个元素是状态索引 (0..n_states-1)
            长度 = len(returns) - 19 (丢弃预热期)
        """
        self._check_fitted()
        features = self._build_features(returns)
        return self._model.predict(features)

    def get_state_probabilities(self, returns: np.ndarray) -> np.ndarray:
        """
        返回每个时间点的状态概率矩阵。

        Args:
            returns: 一维日收益率数组

        Returns:
            (n_days, n_states) 概率矩阵, 每行之和为1
            行对应时间步 (已丢弃前19个预热样本)
        """
        self._check_fitted()
        features = self._build_features(returns)
        return self._model.predict_proba(features)

    def get_current_regime(self, returns: np.ndarray) -> dict:
        """
        获取当前 (最新一天) 的 regime 判断。

        Args:
            returns: 一维日收益率数组 (包含到最新日期的完整序列)

        Returns:
            {
                "state": "bull" / "bear" / "sideways",
                "probabilities": {"bull": 0.7, "bear": 0.1, "sideways": 0.2},
                "confidence": 0.7  # 最高概率值
            }
        """
        self._check_fitted()
        proba = self.get_state_probabilities(returns)
        last_proba = proba[-1]  # 最新一天的概率

        # 构建 name -> probability 映射
        prob_dict = {}
        for i, name in enumerate(self._state_names):
            prob_dict[name] = float(last_proba[i])

        # 当前状态 = 最高概率
        current_idx = int(np.argmax(last_proba))
        current_state = self._state_names[current_idx]
        confidence = float(last_proba[current_idx])

        return {
            "state": current_state,
            "probabilities": prob_dict,
            "confidence": confidence,
        }

    def _check_fitted(self):
        """检查模型是否已拟合。"""
        if not self._fitted or self._model is None:
            raise RuntimeError("模型尚未拟合, 请先调用 fit()")


# ═══════════════════════════════════════════════════════════
# 2. 变点检测 (Change-Point Detection)
# ═══════════════════════════════════════════════════════════


def detect_changepoints(
    returns: np.ndarray,
    method: str = "cusum",
    threshold: float = None,
) -> List[int]:
    """
    检测收益率序列中的结构性变点。

    Args:
        returns: 一维日收益率数组
        method: 检测方法, 目前支持 "cusum"
        threshold: CUSUM 阈值。None 时自动设为收益率标准差的 4 倍。

    Returns:
        变点索引列表 (相对于输入数组的整数索引)

    算法 (CUSUM):
        对去均值后的收益率做累积和:
          S_t = S_{t-1} + (x_t - mean)
        当 |S_t - running_min| > threshold 或 |S_t - running_max| > threshold 时
        标记为变点, 并重置累积和。
    """
    returns = np.asarray(returns, dtype=np.float64)
    n = len(returns)
    if n < 10:
        return []

    if method != "cusum":
        raise ValueError(f"不支持的方法: {method}, 目前仅支持 'cusum'")

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    if threshold is None:
        threshold = 4.0 * std_ret

    # 双向 CUSUM
    changepoints = []
    s_pos = 0.0  # 正向累积和 (检测均值上移)
    s_neg = 0.0  # 负向累积和 (检测均值下移)

    for i in range(n):
        deviation = returns[i] - mean_ret
        s_pos = max(0.0, s_pos + deviation)
        s_neg = max(0.0, s_neg - deviation)

        if s_pos > threshold or s_neg > threshold:
            changepoints.append(i)
            # 重置
            s_pos = 0.0
            s_neg = 0.0

    return changepoints


def bayesian_online_changepoint(
    returns: np.ndarray,
    hazard_rate: float = 1 / 252,
) -> np.ndarray:
    """
    简化版 Bayesian Online Changepoint Detection (BOCPD)。

    基于 Adams & MacKay (2007) 的简化实现:
      - 使用 Gaussian 观测模型 (已知方差, 用样本方差估计)
      - 共轭先验: Normal-Inverse-Gamma 简化为固定方差 Gaussian
      - 返回每个时间点的"运行长度" (run length)

    Run length 解读:
      - run_length 大 → 距离上次变点很久, regime 稳定
      - run_length 突然变小 (接近0) → 刚发生 regime 变化

    Args:
        returns: 一维日收益率数组
        hazard_rate: 先验变点概率 (默认 1/252 ≈ 每年一次)

    Returns:
        一维整数数组 (与输入同长), 每个元素是该时间点的最大后验运行长度。
        第一个元素为 0。
    """
    returns = np.asarray(returns, dtype=np.float64)
    n = len(returns)

    if n == 0:
        return np.array([], dtype=np.int32)

    # 估计观测方差 (用全局方差作为固定方差)
    obs_var = np.var(returns, ddof=1) if n > 1 else 1e-8
    obs_var = max(obs_var, 1e-12)

    # 先验参数
    mu0 = np.mean(returns) if n > 0 else 0.0
    kappa0 = 1.0  # 先验等效样本量

    # 运行长度概率: R[t] = P(run_length = r | x_{1:t})
    # 用列表存储, R[r] 表示当前 run_length=r 的概率
    R = np.array([1.0])  # 初始: run_length=0 概率为1

    # 充分统计量: 每个 run_length 对应的 (sum_x, count)
    # 用于计算预测分布
    sum_x = np.array([0.0])
    count = np.array([0])

    run_lengths = np.zeros(n, dtype=np.int32)

    for t in range(n):
        x = returns[t]

        # 预测概率: P(x_t | run_length = r)
        # 使用每个 run_length 的后验均值和方差
        n_r = count + kappa0
        mu_r = (sum_x + kappa0 * mu0) / n_r
        pred_var = obs_var * (1.0 + 1.0 / n_r)

        # Gaussian 预测似然
        pred_probs = stats.norm.pdf(x, loc=mu_r, scale=np.sqrt(pred_var))

        # 增长概率 (不变点): R_growth[r+1] = R[r] * (1-H) * P(x|r)
        growth = R * (1.0 - hazard_rate) * pred_probs

        # 变点概率: R_cp[0] = sum(R[r] * H * P(x|r))
        cp = np.sum(R * hazard_rate * pred_probs)

        # 新的运行长度分布
        new_R = np.zeros(len(R) + 1)
        new_R[0] = cp
        new_R[1:] = growth

        # 归一化
        evidence = np.sum(new_R)
        if evidence > 0:
            new_R /= evidence
        else:
            new_R = np.zeros(len(R) + 1)
            new_R[0] = 1.0

        # 更新充分统计量
        new_sum_x = np.zeros(len(new_R))
        new_count = np.zeros(len(new_R), dtype=np.int64)
        new_sum_x[0] = 0.0
        new_count[0] = 0
        new_sum_x[1:] = sum_x + x
        new_count[1:] = count + 1

        R = new_R
        sum_x = new_sum_x
        count = new_count

        # 截断过长的运行长度 (防止数值问题)
        max_rl = min(len(R), 500)
        if len(R) > max_rl:
            R = R[:max_rl]
            sum_x = sum_x[:max_rl]
            count = count[:max_rl]
            # 重新归一化
            s = np.sum(R)
            if s > 0:
                R /= s

        # MAP run length
        run_lengths[t] = int(np.argmax(R))

    return run_lengths


# ═══════════════════════════════════════════════════════════
# 3. 多尺度 Regime
# ═══════════════════════════════════════════════════════════


class MultiScaleRegime:
    """
    多尺度市场状态综合判断。

    在多个时间尺度 (5日/20日/60日) 上分别计算:
      - 动量方向 (累计收益)
      - 波动率水平 (年化)
      - 趋势强度 (收益率 / 波动率, 类似 Sharpe)

    然后综合多尺度判断:
      - 多数尺度同向 → 高置信度 regime
      - 尺度冲突 → "transition" 状态

    用法:
        msr = MultiScaleRegime(scales=[5, 20, 60])
        result = msr.get_composite_regime(returns)
        # {"regime": "bull", "confidence": 0.9, "scales_agree": 3/3}
    """

    def __init__(self, scales: List[int] = None):
        """
        初始化多尺度 Regime 检测器。

        Args:
            scales: 时间尺度列表 (交易日数)。
                    默认 [5, 20, 60] 对应周度/月度/季度。
        """
        if scales is None:
            scales = [5, 20, 60]
        self.scales = sorted(scales)

    def compute(self, returns: np.ndarray) -> dict:
        """
        对每个尺度计算动量、波动率和趋势方向。

        Args:
            returns: 一维日收益率数组

        Returns:
            {
                "5d": {"momentum": +0.02, "vol": 0.015, "trend": "up"},
                "20d": {"momentum": +0.05, "vol": 0.018, "trend": "up"},
                "60d": {"momentum": -0.01, "vol": 0.022, "trend": "down"},
            }
        """
        returns = np.asarray(returns, dtype=np.float64)
        n = len(returns)
        result = {}

        for scale in self.scales:
            key = f"{scale}d"

            if n < scale:
                result[key] = {"momentum": 0.0, "vol": 0.0, "trend": "flat"}
                continue

            window = returns[-scale:]

            # 动量: 窗口累计收益
            momentum = float(np.sum(window))

            # 波动率: 窗口内日收益标准差
            vol = float(np.std(window, ddof=1)) if scale > 1 else 0.0

            # 趋势方向: 用 动量/波动率 (类似 t-stat) 判断
            if vol > 1e-10:
                t_stat = momentum / (vol * np.sqrt(scale))
            else:
                t_stat = 0.0

            if t_stat > 0.3:
                trend = "up"
            elif t_stat < -0.3:
                trend = "down"
            else:
                trend = "flat"

            result[key] = {
                "momentum": momentum,
                "vol": vol,
                "trend": trend,
            }

        return result

    def get_composite_regime(self, returns: np.ndarray) -> dict:
        """
        综合多尺度判断当前 regime。

        逻辑:
          - 统计各尺度的趋势方向 (up/down/flat)
          - 如果 >= 2/3 尺度为 "up" → "bull"
          - 如果 >= 2/3 尺度为 "down" → "bear"
          - 如果全部 "flat" → "sideways"
          - 如果方向冲突 (有up有down) → "transition"

        置信度:
          - 所有尺度一致: confidence = 1.0
          - 多数一致: confidence = agree_count / total
          - transition: confidence = 0.5 (不确定性高)

        Args:
            returns: 一维日收益率数组

        Returns:
            {
                "regime": "bull" / "bear" / "sideways" / "transition",
                "confidence": 0.8,
                "scales_agree": 0.67,  # 同意比例
                "detail": {各尺度详细结果}
            }
        """
        detail = self.compute(returns)
        n_scales = len(self.scales)

        # 统计方向
        trends = [detail[f"{s}d"]["trend"] for s in self.scales]
        n_up = trends.count("up")
        n_down = trends.count("down")
        n_flat = trends.count("flat")

        # 判断
        majority_threshold = n_scales / 2.0

        if n_up > majority_threshold and n_down == 0:
            regime = "bull"
            confidence = n_up / n_scales
            scales_agree = n_up / n_scales
        elif n_down > majority_threshold and n_up == 0:
            regime = "bear"
            confidence = n_down / n_scales
            scales_agree = n_down / n_scales
        elif n_up > 0 and n_down > 0:
            # 方向冲突
            regime = "transition"
            confidence = 0.5
            scales_agree = max(n_up, n_down) / n_scales
        elif n_flat == n_scales:
            regime = "sideways"
            confidence = 1.0
            scales_agree = 1.0
        else:
            # 有 flat + 一个方向, 但没有多数
            if n_up > n_down:
                regime = "bull"
                confidence = n_up / n_scales
                scales_agree = n_up / n_scales
            elif n_down > n_up:
                regime = "bear"
                confidence = n_down / n_scales
                scales_agree = n_down / n_scales
            else:
                regime = "sideways"
                confidence = n_flat / n_scales
                scales_agree = n_flat / n_scales

        return {
            "regime": regime,
            "confidence": float(confidence),
            "scales_agree": float(scales_agree),
            "detail": detail,
        }


# ═══════════════════════════════════════════════════════════
# 4. 渐进过渡 (Gradual Transition)
# ═══════════════════════════════════════════════════════════


def compute_transition_weights(
    regime_probs: Dict[str, float],
    factor_weights_by_regime: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    根据 regime 概率计算概率加权混合因子权重。

    核心思想:
      不再硬切换 (regime变了 → 因子权重立刻全换),
      而是按概率混合:
        w_final[factor] = Σ P(regime) × w(factor | regime)

      这样当 P(bull) 从 0.8 缓慢降到 0.5 时,
      因子权重也是渐进调整的, 避免仓位剧烈变动。

    Args:
        regime_probs: 各 regime 的概率, 如 {"bull": 0.6, "bear": 0.1, "sideways": 0.3}
                      概率之和应为1 (内部会归一化)
        factor_weights_by_regime: 每个 regime 对应的因子权重字典, 如:
            {
                "bull": {"mom_20d": 0.3, "vol_20d": -0.1, "size": 0.2},
                "bear": {"mom_20d": -0.1, "vol_20d": -0.3, "size": 0.1},
                "sideways": {"mom_20d": 0.1, "vol_20d": -0.2, "size": 0.15},
            }

    Returns:
        混合后的因子权重字典, 如:
            {"mom_20d": 0.18, "vol_20d": -0.21, "size": 0.165}

    Example:
        >>> probs = {"bull": 0.6, "bear": 0.1, "sideways": 0.3}
        >>> weights = {
        ...     "bull": {"mom_20d": 0.3, "vol_20d": -0.1},
        ...     "bear": {"mom_20d": -0.1, "vol_20d": -0.3},
        ...     "sideways": {"mom_20d": 0.1, "vol_20d": -0.2},
        ... }
        >>> result = compute_transition_weights(probs, weights)
        >>> # result["mom_20d"] = 0.6*0.3 + 0.1*(-0.1) + 0.3*0.1 = 0.20
    """
    # 归一化概率
    total_prob = sum(regime_probs.values())
    if total_prob <= 0:
        # 等权 fallback
        n = len(regime_probs)
        normalized = {k: 1.0 / n for k in regime_probs}
    else:
        normalized = {k: v / total_prob for k, v in regime_probs.items()}

    # 收集所有因子名
    all_factors = set()
    for regime_weights in factor_weights_by_regime.values():
        all_factors.update(regime_weights.keys())

    # 概率加权混合
    blended = {}
    for factor in all_factors:
        weighted_sum = 0.0
        for regime, prob in normalized.items():
            w = factor_weights_by_regime.get(regime, {}).get(factor, 0.0)
            weighted_sum += prob * w
        blended[factor] = float(weighted_sum)

    return blended
