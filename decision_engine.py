"""
decision_engine.py — 统一决策引擎

封装完整的多因子信号生成管道:
  因子计算 → IC加权合成 → 排名决策 → 风控过滤 → 信号输出

支持四类因子:
  - price_volume: 价量技术因子 (FactorScorer)
  - fundamental:  基本面因子 (fundamental_fetcher)
  - relative:     相对因子 (relative_factors)
  - minute:       分钟频因子 (minute_factors, Baostock 15min)

用法:
  engine = DecisionEngine()
  engine.initialize()          # 加载数据 + 预计算 (~5min)
  result = engine.generate_signals("2026-07-31")
  # result: {"buy": [...], "sell": [...], "hold": [...], "scores": {...}, ...}
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(BASE_DIR, "data", "ic_validation", "p5_portfolio_report.json")
SIGNAL_LOG = os.path.join(BASE_DIR, "data", "paper_signals_v3.jsonl")

log = logging.getLogger("quant.decision")


class DecisionEngine:
    """
    多因子决策引擎 (单例模式, 供 API 调用)。

    生命周期:
      1. __init__()     — 轻量, 仅加载配置
      2. initialize()   — 重量, 加载全市场数据 + 预计算因子
      3. generate_signals() — 每次调仓时调用
    """

    def __init__(self):
        self._initialized = False
        self._factors: List[dict] = []
        self._all_data: dict = {}
        self._factor_cache = None
        self._minute_data: dict = {}
        self._fund_panel: dict = {}
        self._config: dict = {}
        self._last_scores: Dict[str, float] = {}
        self._last_result: Optional[dict] = None
        self._init_time: Optional[str] = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def factor_count(self) -> int:
        return len(self._factors)

    # ════════════════════════════════════════════════════════
    #  初始化
    # ════════════════════════════════════════════════════════

    def load_factor_config(self) -> List[dict]:
        """从 P5 报告加载锁定因子列表。"""
        if not os.path.exists(REPORT_PATH):
            log.warning("P5 报告不存在: %s", REPORT_PATH)
            return []
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            report = json.load(f)
        factors = report.get("selected_factors", [])
        log.info("加载 P5 因子: %d 个", len(factors))
        return factors

    def initialize(self, force: bool = False):
        """
        重量级初始化: 加载全市场数据, 预计算因子。

        Args:
          force: 强制重新初始化 (即使已初始化)
        """
        if self._initialized and not force:
            log.info("已初始化, 跳过")
            return

        log.info("=" * 50)
        log.info("  决策引擎初始化")
        log.info("=" * 50)

        # 1. 加载配置
        import yaml
        config_path = os.path.join(BASE_DIR, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)

        # 2. 加载因子列表
        self._factors = self.load_factor_config()
        if not self._factors:
            raise RuntimeError("无可用因子, 请先运行 P5 验证")

        # 3. 加载日线数据
        log.info("加载日线数据...")
        from data_cache import load_all, get_cached_symbols
        syms = get_cached_symbols()
        self._all_data = load_all(syms)
        self._all_data = {s: df for s, df in self._all_data.items()
                         if df is not None and len(df) >= 100}
        log.info("  有效: %d 只", len(self._all_data))

        # 4. 预计算价量因子
        log.info("预计算价量因子...")
        from factor_scorer import FactorScorer
        from factor_cache import FactorCache
        scorer = FactorScorer.from_preset("full_auto")
        pv_names = sorted(scorer.factor_weights.keys())
        self._factor_cache = FactorCache(scorer, pv_names)
        self._factor_cache.precompute(self._all_data)
        log.info("  价量因子: %d 个", len(pv_names))

        # 5. 过滤可计算因子
        computable = set(pv_names)
        # 基本面和相对因子也可计算
        computable.update(f["name"] for f in self._factors
                         if f.get("category") in ("fundamental", "relative", "minute"))
        self._factors = [f for f in self._factors if f["name"] in computable]
        log.info("  可用因子: %d 个", len(self._factors))

        # 6. 加载分钟数据
        has_minute = any(f.get("category") == "minute" for f in self._factors)
        if has_minute:
            log.info("加载分钟数据...")
            from minute_factors import load_minute_data
            self._minute_data = load_minute_data(use_cache=True)
            log.info("  分钟数据: %d 只", len(self._minute_data))

        # 7. 加载基本面数据
        has_fund = any(f.get("category") == "fundamental" for f in self._factors)
        if has_fund:
            log.info("加载基本面数据...")
            try:
                from fundamental_fetcher import load_fundamental_panel
                self._fund_panel = load_fundamental_panel()
                log.info("  基本面: %d 只", len(self._fund_panel))
            except Exception as e:
                log.warning("  基本面加载失败: %s", e)
                self._fund_panel = {}

        self._initialized = True
        self._init_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log.info("  初始化完成 (%s)", self._init_time)

    # ════════════════════════════════════════════════════════
    #  信号生成
    # ════════════════════════════════════════════════════════

    def compute_composite_scores(self, as_of_date) -> Dict[str, float]:
        """
        IC加权线性组合:
          composite_i = sum(ICIR_j × z_score(factor_j_i)) / sum(|ICIR_j|)

        支持四类因子: price_volume, fundamental, relative, minute
        """
        if not self._initialized:
            raise RuntimeError("引擎未初始化, 请先调用 initialize()")

        today = pd.Timestamp(as_of_date)
        factor_names = [f["name"] for f in self._factors]
        weights = np.array([f["icir"] * f.get("weight_multiplier", 1.0)
                           for f in self._factors])
        abs_weight_sum = np.sum(np.abs(weights))
        if abs_weight_sum < 1e-9:
            return {}

        # 收集各类因子原始值
        raw_values = {name: {} for name in factor_names}

        # ── 价量因子 (from FactorCache) ──
        pv_factors = [f for f in self._factors if f.get("category") == "price_volume"]
        if pv_factors and self._factor_cache:
            for sym in self._all_data:
                feats = self._factor_cache.get(sym, today)
                if feats is None:
                    continue
                for f in pv_factors:
                    name = f["name"]
                    if name in feats and not np.isnan(feats[name]):
                        raw_values[name][sym] = feats[name]

        # ── 基本面因子 ──
        fund_factors = [f for f in self._factors if f.get("category") == "fundamental"]
        if fund_factors and self._fund_panel:
            try:
                from fundamental_fetcher import compute_fundamental_factors
                fund_values = compute_fundamental_factors(
                    self._fund_panel, self._all_data, today)
                for sym, fvals in fund_values.items():
                    for name, val in fvals.items():
                        if name in raw_values and not np.isnan(val):
                            raw_values[name][sym] = val
            except Exception as e:
                log.warning("基本面因子计算失败: %s", e)

        # ── 相对因子 ──
        rel_factors = [f for f in self._factors if f.get("category") == "relative"]
        if rel_factors:
            try:
                from relative_factors import compute_relative_factors_batch
                rel_values = compute_relative_factors_batch(self._all_data, today)
                for sym, fvals in rel_values.items():
                    for name, val in fvals.items():
                        if name in raw_values and not np.isnan(val):
                            raw_values[name][sym] = val
            except Exception as e:
                log.warning("相对因子计算失败: %s", e)

        # ── 分钟频因子 ──
        min_factors = [f for f in self._factors if f.get("category") == "minute"]
        if min_factors and self._minute_data:
            try:
                from minute_factors import compute_minute_factors_batch
                min_values = compute_minute_factors_batch(
                    self._minute_data, today, lookback=20)
                for sym, fvals in min_values.items():
                    for name, val in fvals.items():
                        if name in raw_values and not np.isnan(val):
                            raw_values[name][sym] = val
            except Exception as e:
                log.warning("分钟因子计算失败: %s", e)

        # 覆盖率过滤: 至少 50% 因子有值
        n_factors = len(factor_names)
        sym_coverage = {}
        for name in factor_names:
            for sym in raw_values[name]:
                sym_coverage[sym] = sym_coverage.get(sym, 0) + 1
        valid_syms = sorted(
            s for s, c in sym_coverage.items() if c >= n_factors * 0.5)

        if len(valid_syms) < 10:
            log.warning("有效股票不足: %d", len(valid_syms))
            return {}

        # 截面 z-score + IC加权
        n = len(valid_syms)
        composite = np.zeros(n)
        for fi, name in enumerate(factor_names):
            vals = np.array([raw_values[name].get(s, np.nan) for s in valid_syms])
            valid_mask = ~np.isnan(vals)
            if valid_mask.sum() < 10:
                continue
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            if std < 1e-9:
                continue
            z = np.where(valid_mask, (vals - mean) / std, 0.0)
            composite += weights[fi] * z

        composite /= abs_weight_sum
        return dict(zip(valid_syms, composite.tolist()))

    def generate_signals(self, as_of_date=None,
                         execute: bool = False) -> dict:
        """
        生成交易信号。

        Args:
          as_of_date: 信号日期 (None=最新交易日)
          execute: 是否执行订单 (False=仅生成信号)

        Returns:
          {
            "signal_date": "2026-07-31",
            "buy": ["000001", "000002"],
            "sell": ["000003"],
            "hold": ["000004"],
            "top_k": ["000001", "000002", ...],
            "scores": {"000001": 1.23, ...},
            "n_factors": 63,
            "n_scored": 2500,
            "regime": "range",
            "circuit_breaker": "active",
          }
        """
        if not self._initialized:
            raise RuntimeError("引擎未初始化")

        # 确定日期
        if as_of_date is None:
            latest = None
            for sym in list(self._all_data.keys())[:5]:
                df = self._all_data[sym]
                df["date"] = pd.to_datetime(df["date"])
                last = df["date"].max()
                if latest is None or last > latest:
                    latest = last
            today = latest or pd.Timestamp.now().normalize()
        else:
            today = pd.Timestamp(as_of_date)

        log.info("生成信号: %s", today.date())

        # 1. 计算复合评分
        scores = self.compute_composite_scores(today)
        if not scores:
            return {"error": "无法计算评分", "signal_date": str(today.date())}

        self._last_scores = scores
        log.info("  评分: %d 只", len(scores))

        # 2. 交易规则过滤
        from trading_rules import TradingRules
        rules = TradingRules()
        tradeable = {}
        for sym, sc in scores.items():
            if sym in self._all_data:
                dt = self._all_data[sym][
                    self._all_data[sym]["date"] <= today].tail(2)
                if len(dt) >= 2 and not rules.is_suspended(sym, dt):
                    tradeable[sym] = sc

        # 3. 排名决策
        from portfolio_ranker import PortfolioRanker
        exec_cfg = self._config.get("execution", {})
        port_cfg = self._config.get("portfolio", {})

        top_k = exec_cfg.get("top_k", 30)
        ranker = PortfolioRanker(
            top_k=top_k,
            n_drop=port_cfg.get("n_drop", 2),
            hold_thresh=port_cfg.get("hold_thresh", 30),
            sell_rank_buffer=port_cfg.get("sell_rank_buffer", 2),
            buy_confirm_days=port_cfg.get("buy_confirm_days", 1),
            cost_threshold=port_cfg.get("cost_threshold", 0.08),
        )

        # 获取当前持仓
        holdings = []
        executor = None
        if execute:
            from execution.paper_executor import PaperExecutor
            executor = PaperExecutor(
                initial_capital=exec_cfg.get("initial_capital", 100000),
                top_k=top_k,
                lot_size=exec_cfg.get("lot_size", 100),
                slippage_bps=exec_cfg.get("slippage_bps", 30),
            )
            state = executor.load_state()
            holdings = list(state.positions.keys())

        decision = ranker.rank(tradeable, holdings)

        # 涨跌停过滤
        decision["buy"] = [s for s in decision.get("buy", [])
                          if s in self._all_data and rules.can_buy(
                              s, self._all_data[s][
                                  self._all_data[s]["date"] <= today].tail(2))]
        decision["sell"] = [s for s in decision.get("sell", [])
                           if s in self._all_data and rules.can_sell(
                               s, self._all_data[s][
                                   self._all_data[s]["date"] <= today].tail(2))]

        # 4. 熔断检查
        cb_state = "active"
        try:
            from execution.circuit_breaker import CircuitBreaker
            cb = CircuitBreaker()
            cb_state = cb.check()
            if execute:
                buy_allowed, reason = cb.allow_buy()
                if not buy_allowed:
                    log.warning("熔断: %s, 禁止买入", reason)
                    decision["buy"] = []
        except Exception:
            pass

        # 5. 构建结果
        # Top-K 得分详情
        top_k_list = decision.get("top_k", sorted(
            tradeable, key=tradeable.get, reverse=True)[:top_k])
        top_scores = {s: round(scores.get(s, 0), 4) for s in top_k_list[:top_k]}

        result = {
            "signal_date": str(today.date()),
            "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "mode": "live" if execute else "dry_run",
            "buy": decision.get("buy", []),
            "sell": decision.get("sell", []),
            "hold": decision.get("hold", []),
            "top_k": top_k_list[:top_k],
            "top_scores": top_scores,
            "n_factors": len(self._factors),
            "n_scored": len(scores),
            "n_tradeable": len(tradeable),
            "circuit_breaker": cb_state,
            "holdings": holdings,
        }

        # 6. 执行 (可选)
        if execute and executor:
            close_prices = {}
            for sym in self._all_data:
                dt = self._all_data[sym][
                    self._all_data[sym]["date"] <= today].tail(1)
                if len(dt) > 0:
                    close_prices[sym] = float(dt["close"].iloc[-1])

            report = executor.execute_orders(
                buy_list=decision.get("buy", []),
                sell_list=decision.get("sell", []),
                today=today,
                all_data=self._all_data,
                close_prices=close_prices,
            )
            executor.snapshot(str(today.date()), close_prices)
            result["execution"] = {
                "buy_filled": len(report.buy_filled),
                "buy_rejected": len(report.buy_rejected),
                "sell_filled": len(report.sell_filled),
                "sell_rejected": len(report.sell_rejected),
                "total_commission": report.total_commission,
                "equity_before": report.equity_before,
                "equity_after": report.equity_after,
            }

        # 7. 写入信号日志
        os.makedirs(os.path.dirname(SIGNAL_LOG), exist_ok=True)
        with open(SIGNAL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        self._last_result = result
        log.info("  信号: 买%d 卖%d 持有%d",
                 len(result["buy"]), len(result["sell"]), len(result["hold"]))
        return result

    # ════════════════════════════════════════════════════════
    #  查询接口
    # ════════════════════════════════════════════════════════

    def score_stock(self, symbol: str, as_of_date=None) -> Optional[dict]:
        """获取单只股票的因子分解。"""
        if not self._initialized:
            return None

        today = pd.Timestamp(as_of_date) if as_of_date else None
        if today is None:
            # 用最新日期
            if symbol in self._all_data:
                df = self._all_data[symbol]
                df["date"] = pd.to_datetime(df["date"])
                today = df["date"].max()
            else:
                return None

        result = {"symbol": symbol, "date": str(today.date()), "factors": {}}

        # 价量因子
        if self._factor_cache:
            feats = self._factor_cache.get(symbol, today)
            if feats:
                for f in self._factors:
                    if f.get("category") == "price_volume" and f["name"] in feats:
                        result["factors"][f["name"]] = {
                            "value": round(float(feats[f["name"]]), 6),
                            "icir": f["icir"],
                            "category": "price_volume",
                        }

        # 分钟因子
        if self._minute_data and symbol in self._minute_data:
            from minute_factors import compute_minute_factors
            mf = compute_minute_factors(self._minute_data[symbol], today)
            if mf:
                for f in self._factors:
                    if f.get("category") == "minute" and f["name"] in mf:
                        result["factors"][f["name"]] = {
                            "value": round(float(mf[f["name"]]), 6),
                            "icir": f["icir"],
                            "category": "minute",
                        }

        # 综合得分
        if symbol in self._last_scores:
            result["composite_score"] = round(self._last_scores[symbol], 4)

        return result

    def get_factor_summary(self) -> List[dict]:
        """获取因子配置摘要 (未初始化时从文件加载)。"""
        factors = self._factors if self._factors else self.load_factor_config()
        return [{
            "name": f["name"],
            "icir": f["icir"],
            "category": f.get("category", "unknown"),
            "weight_multiplier": f.get("weight_multiplier", 1.0),
        } for f in factors]

    def get_status(self) -> dict:
        """引擎状态。"""
        return {
            "initialized": self._initialized,
            "init_time": self._init_time,
            "n_factors": len(self._factors),
            "n_stocks": len(self._all_data),
            "n_minute_stocks": len(self._minute_data),
            "n_fund_stocks": len(self._fund_panel),
            "has_last_result": self._last_result is not None,
            "last_signal_date": self._last_result.get("signal_date")
                if self._last_result else None,
            "factor_categories": self._count_categories(),
        }

    def _count_categories(self) -> dict:
        cats = {}
        factors = self._factors if self._factors else self.load_factor_config()
        for f in factors:
            c = f.get("category", "unknown")
            cats[c] = cats.get(c, 0) + 1
        return cats


# ── 模块级单例 ──
_engine: Optional[DecisionEngine] = None


def get_engine() -> DecisionEngine:
    """获取全局决策引擎单例。"""
    global _engine
    if _engine is None:
        _engine = DecisionEngine()
    return _engine
