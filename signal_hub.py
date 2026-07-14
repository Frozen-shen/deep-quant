"""
信号聚合中心 — 多策略信号 → 统一决策

用法:
    hub = SignalHub("01810", "hk")
    hub.register("ma_cross", ma_strategy, weight=1.0)
    hub.register("llm_event", llm_strategy, weight=1.5)

    decision = hub.generate(df_price, events_df=None, daily_factors=None)
    → {action: "BUY"/"SELL"/"HOLD", confidence, reason, signals: [...]}
"""

import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import storage


@dataclass
class StrategySignal:
    """单个策略的输出信号"""
    strategy: str
    action: str        # "BUY" / "SELL" / "HOLD"
    confidence: float  # 0.0 ~ 1.0
    reason: str
    weight: float = 1.0


@dataclass
class HubDecision:
    """信号中心最终决策"""
    symbol: str
    date: str
    action: str         # "BUY" / "SELL" / "HOLD"
    confidence: float
    reason: str
    signals: List[StrategySignal] = field(default_factory=list)
    should_trade: bool = False


class SignalHub:
    """
    多策略信号聚合中心。

    新增: 大盘趋势过滤器
    - 指数 MA60 以下 → 所有买入信号转为 HOLD
    - 只有指数在中期上升趋势中才允许做多

    规则:
    1. 所有策略独立生成信号
    2. 同向信号: 叠加置信度
    3. 反向信号: 取加权置信度高者
    4. 大盘过滤: 指数<MA60 → 所有BUY→HOLD
    5. 已持仓时忽略买入信号
    """

    # 各市场对应的主要指数
    MARKET_INDEX = {
        "a": "000001",   # 上证指数
        "hk": "HSI",     # 恒生指数 (用 akshare 获取)
    }

    def __init__(self, symbol: str, market: str = "hk",
                 confidence_threshold: float = 0.5,
                 enable_market_filter: bool = False):  # 默认关闭(港股接口不稳定)
        self.symbol = symbol
        self.market = market
        self.confidence_threshold = confidence_threshold
        self.enable_market_filter = enable_market_filter
        self._strategies: Dict[str, Dict] = {}
        self._market_bullish = True  # 默认看多

    def check_market_trend(self, df_price: pd.DataFrame = None) -> bool:
        """
        检查大盘趋势: 指数是否在 MA60 上方。

        返回: True = 可以交易, False = 市场偏弱
        """
        if not self.enable_market_filter:
            return True

        try:
            from data_fetcher import DataFetcher

            index_code = self.MARKET_INDEX.get(self.market, "")
            if not index_code:
                return True

            # A股: 拉上证指数
            if self.market == "a":
                fetcher = DataFetcher()
                df_idx = fetcher.fetch(index_code, "20240101", "20260710", "qfq", market="a")
                if len(df_idx) < 60:
                    return True
                ma60 = df_idx["close"].rolling(60).mean().iloc[-1]
                last_close = df_idx["close"].iloc[-1]
                self._market_bullish = last_close > ma60
                print(f"[SignalHub] 大盘趋势: 上证 {last_close:.0f} vs MA60 {ma60:.0f} "
                      f"→ {'看多' if self._market_bullish else '看空'}")

            # 港股: 拉恒生指数
            elif self.market == "hk":
                try:
                    import akshare as ak
                    df_hsi = ak.stock_hk_index_daily_em(symbol="HSI")
                    if len(df_hsi) >= 60:
                        ma60 = df_hsi["close"].rolling(60).mean().iloc[-1]
                        last_close = df_hsi["close"].iloc[-1]
                        self._market_bullish = last_close > ma60
                        print(f"[SignalHub] 大盘趋势: 恒指 {last_close:.0f} vs MA60 {ma60:.0f} "
                              f"→ {'看多' if self._market_bullish else '看空'}")
                except Exception:
                    pass

        except Exception as e:
            print(f"[SignalHub] 大盘趋势检测失败: {e}")

        return self._market_bullish

    def register(self, name: str, signal_fn: Callable,
                 weight: float = 1.0, **fn_kwargs):
        """
        注册策略。

        参数
        ----
        name : str
            策略名称
        signal_fn : callable
            信号生成函数: fn(df_price, **kwargs) → signal_df
        weight : float
            策略权重 (默认1.0)
        fn_kwargs : dict
            传递给 signal_fn 的额外参数
        """
        self._strategies[name] = {
            "fn": signal_fn,
            "weight": weight,
            "kwargs": fn_kwargs,
        }

    def generate(self, df_price: pd.DataFrame,
                 extra_kwargs: Optional[Dict] = None) -> HubDecision:
        """
        生成当日统一信号。

        返回
        ----
        HubDecision
        """
        today = datetime.now().strftime("%Y-%m-%d")
        all_signals: List[StrategySignal] = []

        # 1. 每个策略独立生成信号
        for name, cfg in self._strategies.items():
            fn = cfg["fn"]
            weight = cfg["weight"]
            kwargs = dict(cfg["kwargs"])
            if extra_kwargs:
                kwargs.update(extra_kwargs)

            try:
                result_df = fn(df_price, **kwargs)
                # 取最新一天的信号
                last = result_df.iloc[-1]
                raw_signal = last.get("signal", 0)

                action_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
                action = action_map.get(raw_signal, "HOLD")

                # 置信度: MA策略基于信号强度, LLM策略自带confidence
                confidence = abs(raw_signal) * 0.7  # MA信号默认0.7置信度
                reason = f"{name}:{action}"

                sig = StrategySignal(
                    strategy=name,
                    action=action,
                    confidence=confidence,
                    reason=reason,
                    weight=weight,
                )
                all_signals.append(sig)

                # 记录到数据库
                storage.record_signal(
                    today, self.symbol, name, raw_signal, confidence, reason
                )

            except Exception as e:
                print(f"[SignalHub] {name} 策略失败: {e}")

        # 2. 聚合决策
        decision = self._resolve(all_signals)
        decision.symbol = self.symbol
        decision.date = today
        decision.signals = all_signals

        return decision

    def _resolve(self, signals: List[StrategySignal]) -> HubDecision:
        """冲突解决逻辑。"""
        if not signals:
            return HubDecision(
                symbol=self.symbol, date="",
                action="HOLD", confidence=0.0,
                reason="无有效策略信号",
            )

        # 加权分数
        buy_score = 0.0
        sell_score = 0.0
        buy_reasons = []
        sell_reasons = []

        for s in signals:
            w = s.weight
            if s.action == "BUY" and s.confidence >= self.confidence_threshold:
                buy_score += s.confidence * w
                buy_reasons.append(s.reason)
            elif s.action == "SELL" and s.confidence >= self.confidence_threshold:
                sell_score += s.confidence * w
                sell_reasons.append(s.reason)

        # 大盘过滤器: 市场看空 → 所有BUY→HOLD
        if not self._market_bullish and buy_score > 0:
            buy_score = 0
            buy_reasons = [f"{r}[大盘MA60以下,阻止买入]" for r in buy_reasons]

        # 检查当前持仓
        existing = storage.get_position(self.symbol)
        has_position = existing and existing["qty"] > 0

        # 决策
        if buy_score > sell_score and buy_score > 0:
            if has_position:
                return HubDecision(
                    symbol=self.symbol, date="",
                    action="HOLD", confidence=buy_score,
                    reason=f"已有持仓,跳过买入 ({'; '.join(buy_reasons)})",
                    should_trade=False,
                )
            return HubDecision(
                symbol=self.symbol, date="",
                action="BUY", confidence=buy_score,
                reason="; ".join(buy_reasons),
                should_trade=True,
            )

        elif sell_score > buy_score and sell_score > 0:
            if not has_position:
                return HubDecision(
                    symbol=self.symbol, date="",
                    action="HOLD", confidence=sell_score,
                    reason=f"无持仓,跳过卖出 ({'; '.join(sell_reasons)})",
                    should_trade=False,
                )
            return HubDecision(
                symbol=self.symbol, date="",
                action="SELL", confidence=sell_score,
                reason="; ".join(sell_reasons),
                should_trade=True,
            )

        else:
            return HubDecision(
                symbol=self.symbol, date="",
                action="HOLD", confidence=max(buy_score, sell_score),
                reason="无明确信号",
                should_trade=False,
            )


# ================================================================
#  便捷工厂函数
# ================================================================

def make_ma_signal_fn(short_window=5, long_window=20):
    """创建 MA 交叉信号函数（适配 SignalHub 接口）。"""
    from strategy import MACrossoverStrategy
    s = MACrossoverStrategy(short_window, long_window)
    return lambda df, **kw: s.generate_signals(df)


def make_llm_event_signal_fn(confidence_threshold=0.6):
    """创建 LLM 事件信号函数。"""
    from strategy import LLMEventStrategy
    s = LLMEventStrategy(confidence_threshold=confidence_threshold)
    def _fn(df, events_df=None, **kw):
        if events_df is None or (hasattr(events_df, 'empty') and events_df.empty):
            # 返回空信号
            df_out = df.copy()
            df_out["signal"] = 0
            df_out["position"] = 0
            return df_out
        return s.generate_signals(df, events_df)
    return _fn
