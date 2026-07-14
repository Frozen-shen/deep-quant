"""
策略模块 — 双均线交叉策略 (MA Crossover)

经典趋势跟踪策略：
- 金叉 (Golden Cross): 短均线上穿长均线 → 买入信号
- 死叉 (Death Cross):  短均线下穿长均线 → 卖出信号
"""

import pandas as pd
import numpy as np


class MACrossoverStrategy:
    """
    双均线交叉策略。

    参数
    ----
    short_window : int
        短期均线窗口（默认 5 日）
    long_window : int
        长期均线窗口（默认 20 日）
    """

    def __init__(self, short_window: int = 5, long_window: int = 20):
        if short_window >= long_window:
            raise ValueError("短均线窗口必须小于长均线窗口")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据收盘价计算均线并生成交易信号。

        参数
        ----
        df : pd.DataFrame
            必须包含 'close' 列

        返回
        ----
        pd.DataFrame
            原 DataFrame 加上以下列:
            - ma_short : 短期均线
            - ma_long  : 长期均线
            - signal   : 1(买入), -1(卖出), 0(无操作)
            - position : 当前持仓状态 0 或 1
        """
        df = df.copy()

        # 计算双均线
        df["ma_short"] = df["close"].rolling(window=self.short_window).mean()
        df["ma_long"] = df["close"].rolling(window=self.long_window).mean()

        # ---------- 金叉/死叉检测 ----------
        # 条件: 今昨两天均线都有效（非 NaN）
        valid = (df["ma_short"].notna() & df["ma_long"].notna() &
                 df["ma_short"].shift(1).notna() & df["ma_long"].shift(1).notna())

        # 金叉: 昨天 short <= long，今天 short > long
        golden_cross = (
            valid &
            (df["ma_short"].shift(1) <= df["ma_long"].shift(1)) &
            (df["ma_short"] > df["ma_long"])
        )

        # 死叉: 昨天 short >= long，今天 short < long
        death_cross = (
            valid &
            (df["ma_short"].shift(1) >= df["ma_long"].shift(1)) &
            (df["ma_short"] < df["ma_long"])
        )

        # 信号列: 1 买入, -1 卖出, 0 无操作
        df["signal"] = 0
        df.loc[golden_cross, "signal"] = 1
        df.loc[death_cross, "signal"] = -1

        # 持仓状态: 根据信号累加 (买入=持仓, 卖出=空仓)
        df["position"] = df["signal"].replace(-1, 0)  # 死叉 → 空仓
        # 用 ffill 传播持仓状态（金叉后一直持有，直到死叉）
        # 先将 signal==1 的位置设为 1，signal==-1 的位置设为 0，其余 NaN，然后 ffill
        df["position"] = np.where(df["signal"] == 1, 1,
                          np.where(df["signal"] == -1, 0, np.nan))
        df["position"] = df["position"].ffill().fillna(0).astype(int)

        # 统计
        buy_count = (df["signal"] == 1).sum()
        sell_count = (df["signal"] == -1).sum()
        print(f"[Strategy] MA{self.short_window}×MA{self.long_window} "
              f"金叉 {buy_count} 次, 死叉 {sell_count} 次")

        return df


def apply_llm_filter(
    df_signals: pd.DataFrame,
    df_llm_daily: pd.DataFrame,
    sentiment_threshold: float = 0.0,
    filter_mode: str = "block",
) -> pd.DataFrame:
    """
    用 LLM 每日聚合因子对交易信号做二次过滤。

    这个方法对回测引擎完全透明 — 只修改 signal 列，
    backtest.py 不需要任何改动。

    参数
    ----
    df_signals : pd.DataFrame
        strategy.generate_signals() 的输出，含 signal, position, date 列
    df_llm_daily : pd.DataFrame
        LLMFactorEngine.aggregate_to_daily() 的输出，
        含 date, llm_sentiment_daily 列
    sentiment_threshold : float
        情感阈值，只有 sentiment > threshold 时才允许买入
    filter_mode : str
        - "block":  情感不佳时阻止买入信号（signal 置 0），但保留卖出信号
        - "scale":  情感不佳时降低仓位（暂未实现，保留扩展）

    返回
    ----
    pd.DataFrame
        修改后的 df_signals（signal 列被过滤）
    """
    if df_llm_daily.empty:
        print("[LLM-Filter] 无 LLM 因子数据，跳过过滤")
        return df_signals

    df = df_signals.copy()

    # 确保日期对齐
    df_llm = df_llm_daily.copy()
    df_llm["date"] = pd.to_datetime(df_llm["date"])
    df["date"] = pd.to_datetime(df["date"])

    # 合并 LLM 情感分数
    df = df.merge(
        df_llm[["date", "llm_sentiment_daily", "llm_news_count"]],
        on="date",
        how="left",
    )

    # 无 LLM 数据的日期填充默认值
    df["llm_sentiment_daily"] = df["llm_sentiment_daily"].fillna(0)
    df["llm_news_count"] = df["llm_news_count"].fillna(0)

    # 标记有 LLM 数据的日期（news_count > 0）
    has_llm_data = df["llm_news_count"] > 0

    # 记录原始信号数量
    original_buy_count = (df["signal"] == 1).sum()

    if filter_mode == "block":
        # 核心逻辑：只有在我们有 LLM 数据的日期，且情感 <= 阈值时，才阻止买入
        # 没有 LLM 数据的日期（news_count == 0）：信号原样保留（"没消息就是好消息"）
        blocked = (
            (df["signal"] == 1) &
            has_llm_data &
            (df["llm_sentiment_daily"] <= sentiment_threshold)
        )
        df.loc[blocked, "signal"] = 0

        blocked_count = blocked.sum()
        print(f"[LLM-Filter] 模式=block, 阈值={sentiment_threshold}, "
              f"有LLM数据天数={has_llm_data.sum()}/{len(df)}, "
              f"原买入信号 {original_buy_count} → 阻止 {blocked_count} 个 → "
              f"剩余 {original_buy_count - blocked_count} 个")

    elif filter_mode == "scale":
        # 扩展点：按情感强度缩放仓位
        print("[LLM-Filter] 模式=scale 暂未实现，保持原始信号")
        pass

    # 重新计算 position（因为 signal 变了）
    df["position"] = np.where(df["signal"] == 1, 1,
                      np.where(df["signal"] == -1, 0, np.nan))
    df["position"] = df["position"].ffill().fillna(0).astype(int)

    return df


# ============================================================================
#  LLM 事件驱动策略 — LLM 作为主信号源
# ============================================================================

class LLMEventStrategy:
    """
    LLM 事件驱动策略：公司公告/研报 → LLM 判断 → 直接生成买卖信号。

    与 MACrossoverStrategy 的区别:
    - MACrossoverStrategy: 技术面金叉死叉 → LLM 只做二次过滤（配角）
    - LLMEventStrategy:    LLM 直接基于事件内容决策买卖（主角）

    参数
    ----
    confidence_threshold : float
        信心度阈值，只有 confidence >= threshold 的事件才执行
    default_hold_days : int
        LLM 未指定 horizon 时的默认持有天数
    allow_short : bool
        是否允许做空（默认不允许，只做多）
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        default_hold_days: int = 5,
        allow_short: bool = False,
    ):
        self.confidence_threshold = confidence_threshold
        self.default_hold_days = default_hold_days
        self.allow_short = allow_short

    def generate_signals(
        self,
        df_price: pd.DataFrame,
        df_events_scored: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        根据 LLM 评分后的事件生成交易信号。

        参数
        ----
        df_price : pd.DataFrame
            行情数据，必须包含 date, close
        df_events_scored : pd.DataFrame
            LLMFactorEngine.batch_score_events() 的输出，
            必须包含 event_date, llm_action, llm_confidence,
            llm_horizon_days

        返回
        ----
        pd.DataFrame
            df_price 加上 signal, position 列
        """
        df = df_price.copy()
        df["signal"] = 0
        df["position"] = 0

        if df_events_scored.empty:
            print("[LLMEventStrategy] 无事件数据，返回空仓信号")
            return df

        # 筛选高置信度事件
        events = df_events_scored.copy()
        events["event_date"] = pd.to_datetime(events["event_date"])
        df["date"] = pd.to_datetime(df["date"])

        # 只保留 actionable 事件
        actionable = events[
            (events["llm_confidence"] >= self.confidence_threshold) &
            (events["llm_action"].isin(["buy", "sell"]))
        ]

        if actionable.empty:
            print(f"[LLMEventStrategy] 无高置信度事件 (阈值={self.confidence_threshold})")
            return df

        buy_events = actionable[actionable["llm_action"] == "buy"]
        sell_events = actionable[actionable["llm_action"] == "sell"]

        # 将事件映射到交易日的 signal 列
        for _, event in buy_events.iterrows():
            event_dt = event["event_date"]
            horizon = int(event.get("llm_horizon_days", self.default_hold_days))

            # 找到事件日之后最近的交易日
            matching = df[df["date"] >= event_dt]
            if matching.empty:
                continue

            # 买入信号: 事件日
            entry_idx = matching.index[0]
            df.at[entry_idx, "signal"] = 1

            # 卖出信号: 持有 horizon_days 后
            exit_cutoff = event_dt + pd.DateOffset(days=horizon)
            exit_matching = df[(df["date"] >= exit_cutoff) | (df.index > entry_idx)]
            if not exit_matching.empty:
                # 取 horizon 日后的第一个交易日，或倒数第二个（避免最后一天）
                exit_idx = min(exit_matching.index[0], len(df) - 1)
                df.at[exit_idx, "signal"] = -1

        # 卖出事件：在高置信度 sell 事件日直接卖出（如果持有）
        for _, event in sell_events.iterrows():
            event_dt = event["event_date"]
            matching = df[df["date"] >= event_dt]
            if matching.empty:
                continue
            idx = matching.index[0]
            df.at[idx, "signal"] = -1

        # 计算持仓状态
        df["position"] = np.where(df["signal"] == 1, 1,
                          np.where(df["signal"] == -1, 0, np.nan))
        df["position"] = df["position"].ffill().fillna(0).astype(int)

        buy_count = (df["signal"] == 1).sum()
        sell_count = (df["signal"] == -1).sum()
        print(f"[LLMEventStrategy] 事件驱动信号: "
              f"买入 {buy_count} 次, 卖出 {sell_count} 次 "
              f"(阈值={self.confidence_threshold}, {len(buy_events)} buy事件, "
              f"{len(sell_events)} sell事件)")

        return df


# ============================================================================
#  增强版 MA 策略 — 量价确认 + 趋势过滤 + 均线排列
# ============================================================================

class EnhancedMACrossoverStrategy:
    """
    增强版双均线交叉策略（基于 investoday 七维度框架优化）。

    新增过滤条件:
    1. 量价确认: 金叉日量比 > 1.5（放量突破才有效）
    2. 趋势过滤: MA20 > MA60（只在中期上升趋势中做多）
    3. 均线排列: 检测多头/空头/粘合，仅在多头或粘合时交易

    参数
    ----
    short_window : int   短期均线 (默认5)
    long_window : int    长期均线 (默认20)
    trend_window : int   趋势均线 (默认60)
    volume_ratio_min : float  最小量比阈值 (默认1.5)
    enable_volume_filter : bool  是否启用量比过滤
    enable_trend_filter : bool   是否启用趋势过滤
    """

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
        trend_window: int = 60,
        volume_ratio_min: float = 1.5,
        enable_volume_filter: bool = True,
        enable_trend_filter: bool = False,
        enable_adx_filter: bool = True,       # ★ ADX趋势强度过滤
        adx_threshold: int = 25,               # ★ ADX>25才允许交易
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.trend_window = trend_window
        self.volume_ratio_min = volume_ratio_min
        self.enable_volume_filter = enable_volume_filter
        self.enable_trend_filter = enable_trend_filter
        self.enable_adx_filter = enable_adx_filter
        self.adx_threshold = adx_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成增强版交易信号。

        参数
        ----
        df : pd.DataFrame
            必须包含: close, volume (可选: 无volume列时跳过量化过滤)

        返回
        ----
        pd.DataFrame with signal, position, ma_short, ma_long, ma_trend
        """
        df = df.copy()

        # ---- 多周期均线 ----
        df["ma_short"] = df["close"].rolling(self.short_window).mean()
        df["ma_long"] = df["close"].rolling(self.long_window).mean()
        df["ma_trend"] = df["close"].rolling(self.trend_window).mean()

        # ---- 量比 (5日均量) ----
        has_volume = "volume" in df.columns and df["volume"].notna().any()
        if has_volume:
            df["vol_ma5"] = df["volume"].rolling(5).mean()
            df["vol_ratio"] = df["volume"] / df["vol_ma5"].replace(0, float("nan"))
            df["vol_ratio"] = df["vol_ratio"].fillna(1.0)
        else:
            df["vol_ratio"] = 1.0

        # ---- 金叉/死叉 ----
        valid = (
            df["ma_short"].notna() & df["ma_long"].notna() &
            df["ma_short"].shift(1).notna() & df["ma_long"].shift(1).notna()
        )
        golden = valid & (df["ma_short"].shift(1) <= df["ma_long"].shift(1)) & (df["ma_short"] > df["ma_long"])
        death = valid & (df["ma_short"].shift(1) >= df["ma_long"].shift(1)) & (df["ma_short"] < df["ma_long"])

        # ---- 均线排列状态 ----
        df["ma_alignment"] = "unknown"
        bullish = (df["ma_short"] > df["ma_long"]) & (df["ma_long"] > df["ma_trend"])
        bearish = (df["ma_short"] < df["ma_long"]) & (df["ma_long"] < df["ma_trend"])
        # 粘合: 三均线差距 < 2%
        sticky = (
            df["ma_short"].notna() & df["ma_long"].notna() & df["ma_trend"].notna() &
            (abs(df["ma_short"] / df["ma_long"] - 1) < 0.02) &
            (abs(df["ma_long"] / df["ma_trend"] - 1) < 0.02)
        )
        df.loc[bullish, "ma_alignment"] = "bullish"
        df.loc[bearish, "ma_alignment"] = "bearish"
        df.loc[sticky & ~bullish & ~bearish, "ma_alignment"] = "sticky"

        # ---- 趋势过滤: MA20 > MA60 ----
        trend_ok = df["ma_long"] > df["ma_trend"]

        # ---- 量价确认 ----
        volume_ok = df["vol_ratio"] >= self.volume_ratio_min

        # ---- ADX 趋势强度 ----
        adx_ok = pd.Series(True, index=df.index)
        if self.enable_adx_filter and all(c in df.columns for c in ["high", "low"]):
            from indicators import ADX
            df["adx_14"] = ADX(df["high"], df["low"], df["close"], 14)
            adx_ok = df["adx_14"] >= self.adx_threshold

        # ---- 综合过滤 ----
        block_reasons = []

        # 金叉过滤
        golden_pass = golden.copy()
        if self.enable_volume_filter and has_volume:
            no_vol = golden & ~volume_ok
            block_reasons.append(("量比不足", no_vol.sum()))
            golden_pass = golden_pass & (volume_ok | ~golden)
        if self.enable_trend_filter:
            no_trend = golden & ~trend_ok
            block_reasons.append(("趋势向下(MA20<MA60)", no_trend.sum()))
            golden_pass = golden_pass & (trend_ok | ~golden)
        if self.enable_adx_filter:
            no_adx = golden & ~adx_ok
            block_reasons.append((f"ADX<{self.adx_threshold}(趋势弱)", no_adx.sum()))
            golden_pass = golden_pass & (adx_ok | ~golden)

        # 死叉不额外过滤（让卖出信号畅通，避免"因为量小就死扛"）
        death_pass = death

        # ---- 信号生成 ----
        df["signal"] = 0
        df.loc[golden_pass, "signal"] = 1
        df.loc[death_pass, "signal"] = -1

        # 持仓状态
        df["position"] = np.where(df["signal"] == 1, 1,
                          np.where(df["signal"] == -1, 0, np.nan))
        df["position"] = df["position"].ffill().fillna(0).astype(int)

        # 统计
        buy_raw = golden.sum()
        buy_pass = golden_pass.sum()
        sell_count = (df["signal"] == -1).sum()
        print(f"[EnhancedMA] MA{self.short_window}×MA{self.long_window}×MA{self.trend_window}")
        print(f"  金叉: {buy_raw}次 → 通过{buy_pass}次, 死叉: {sell_count}次")
        for reason, count in block_reasons:
            if count > 0:
                print(f"  过滤[{reason}]: {count}次")

        return df

    def get_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """返回因子值DataFrame（供信号分析）。"""
        df = self.generate_signals(df.copy())
        return df[["date", "close", "ma_short", "ma_long", "ma_trend",
                   "vol_ratio", "ma_alignment", "signal"]]


# ============================================================================
#  RSI 均值回复策略 — 高波动股票 (参考 Backtrader kselrsi.py)
# ============================================================================

class RSIMeanReversionStrategy:
    """
    RSI 均值回复策略 — 震荡市/高波动股的替代方案。

    逻辑:
    - 买入: RSI 从超卖区(<30)回升
    - 卖出: RSI 从超买区(>70)回落 或 回到50中轴

    参数
    ----
    rsi_period : int      RSI周期 (默认14)
    rsi_oversold : float  超卖阈值 (默认30)
    rsi_overbought : float 超买阈值 (默认70)
    rsi_exit : float      平仓中轴 (默认50)
    """

    def __init__(self, rsi_period: int = 14, rsi_oversold: float = 30,
                 rsi_overbought: float = 70, rsi_exit: float = 50):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.rsi_exit = rsi_exit

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        from indicators import RSI
        df["rsi"] = RSI(df["close"], self.rsi_period)

        # 买入: RSI上穿超卖线
        buy_signal = (
            (df["rsi"].shift(1) <= self.rsi_oversold) &
            (df["rsi"] > self.rsi_oversold)
        )
        # 卖出: RSI下穿超买线 或 跌破中轴
        sell_signal = (
            (df["rsi"].shift(1) >= self.rsi_overbought) &
            (df["rsi"] < self.rsi_overbought)
        ) | (
            (df["rsi"].shift(1) >= self.rsi_exit) &
            (df["rsi"] < self.rsi_exit)
        )

        df["signal"] = 0
        df.loc[buy_signal, "signal"] = 1
        df.loc[sell_signal, "signal"] = -1

        df["position"] = np.where(df["signal"] == 1, 1,
                          np.where(df["signal"] == -1, 0, np.nan))
        df["position"] = df["position"].ffill().fillna(0).astype(int)

        buy_count = (df["signal"] == 1).sum()
        sell_count = (df["signal"] == -1).sum()
        print(f"[RSI-MeanRev] RSI{self.rsi_period} 超卖{self.rsi_oversold}/超买{self.rsi_overbought}: "
              f"买入 {buy_count} 次, 卖出 {sell_count} 次")

        return df


# ============================================================================
#  策略路由器 — 根据股票特征自动选择最优策略
# ============================================================================

class StrategyRouter:
    """
    策略路由器 — 分析股票特征，自动选择最优策略。

    规则:
    - 趋势强度高 (ADX>25): → EnhancedMACrossoverStrategy (趋势跟踪)
    - 波动率高 (ATR/Price>3%): → RSIMeanReversionStrategy (均值回复)
    - 其他: → 空仓/持有现金

    用法:
        router = StrategyRouter()
        strategy = router.select(df)
        df_signal = strategy.generate_signals(df)
    """

    def __init__(self):
        self._last_choice = None

    def analyze(self, df: pd.DataFrame) -> dict:
        """分析股票特征，返回特征字典。"""
        close = df["close"]
        returns = close.pct_change().dropna()
        n = len(returns)

        if n < 50:
            return {"type": "insufficient_data"}

        # 年化波动率
        annual_vol = returns.std() * np.sqrt(252)

        # 趋势强度 (ADX)
        adx_val = 0
        if all(c in df.columns for c in ["high", "low"]):
            from indicators import ADX
            adx_series = ADX(df["high"], df["low"], close, 14)
            adx_val = adx_series.iloc[-1] if not adx_series.isna().all() else 0

        # 收益率偏度 (正偏=趋势, 负偏=均值回复)
        skew = returns.skew() if len(returns) > 10 else 0

        # 日均振幅
        if all(c in df.columns for c in ["high", "low"]):
            daily_range = ((df["high"] - df["low"]) / close).mean()
        else:
            daily_range = 0.02

        return {
            "annual_vol": annual_vol,
            "adx": adx_val,
            "skew": skew,
            "daily_range": daily_range,
            "n_days": n,
        }

    def select(self, df: pd.DataFrame):
        """
        根据股票特征选择策略。

        返回: (strategy_instance, strategy_name)
        """
        features = self.analyze(df)

        if features.get("type") == "insufficient_data":
            return None, "insufficient_data"

        adx = features["adx"]
        vol = features["annual_vol"]
        daily_range = features["daily_range"]

        # 规则1: 强趋势 → 趋势跟踪
        if adx > 25 and vol > 0.15:
            self._last_choice = "trend"
            return EnhancedMACrossoverStrategy(
                enable_volume_filter=True, enable_adx_filter=False,
                volume_ratio_min=1.0,
            ), "trend_following"

        # 规则2: 高波动(>3%日均振幅) + 弱趋势(ADX<25) → 均值回复
        if daily_range > 0.03 and adx < 25:
            self._last_choice = "mean_reversion"
            return RSIMeanReversionStrategy(
                rsi_period=14, rsi_oversold=30, rsi_overbought=70
            ), "rsi_mean_reversion"

        # 规则3: 有趋势 → 趋势跟踪
        if adx > 20:
            self._last_choice = "trend_loose"
            return EnhancedMACrossoverStrategy(
                enable_volume_filter=True, enable_adx_filter=False,
                volume_ratio_min=1.0,
            ), "trend_loose"

        # 规则4: 默认趋势跟踪(宽松)
        self._last_choice = "trend_default"
        return EnhancedMACrossoverStrategy(
            enable_volume_filter=True, enable_adx_filter=False,
            volume_ratio_min=1.0,
        ), "trend_default"

