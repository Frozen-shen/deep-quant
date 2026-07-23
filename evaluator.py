"""
量化模型评测模块 — 三层评测体系 (单窗口 → 多窗口聚合 → 综合评级)

层级:
  Level 1: analyze_window()     — 单窗口详评 (27个指标)
  Level 2: analyze_cross_window() — 多窗口聚合 (均值/标准差/稳定性)
  Level 3: grade()              — 综合评级 (A/B/C/D + 百分制)

用法:
  from evaluator import ModelEvaluator
  eval = ModelEvaluator()
  
  # 单窗口
  report = eval.analyze_window(equity_series, daily_returns, benchmark_returns, trades_list)
  
  # 多窗口 (滚动重训练结果)
  cross = eval.analyze_cross_window(windows_data)  # list of per-window metrics
  
  # 综合评级
  grade = eval.grade(window_metrics_list)

输出:
  ReportCard = { grade, score, pass, returns, risk, risk_adjusted,
                 trading_quality, stability, statistical }
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ================================================================
#  评级阈值 — 实盘导向 (对标真实投资标准, 非学术指标)
#
#  核心理念: 赚不到真金白银 = 不合格, 不管超额多好看
#  每一档含义:
#    min  = 勉强可用的底线 (不到此线直接0分)
#    good = 值得投入真金白银的水平
#    great = 优秀的量化策略
# ================================================================

GRADE_THRESHOLDS = {
    # ★ 绝对收益 (新增) — 实盘赚钱才是硬道理
    "annual_return":     {"min": 0.05, "good": 0.12, "great": 0.20, "weight": 14},
    # 超额收益 — 保留但降权
    "annual_excess":     {"min": 0.03, "good": 0.10, "great": 0.20, "weight": 8},
    # 夏普 — 大幅提升门槛 (0.3太宽松, 实盘<1.0没人敢用)
    "sharpe_ratio":      {"min": 0.5,  "good": 1.00, "great": 1.50, "weight": 12},
    # 最大回撤 — 收紧 (实盘-35%已爆仓)
    "max_drawdown":      {"min": -0.30,"good": -0.20,"great": -0.12, "weight": 10},
    # Calmar
    "calmar_ratio":      {"min": 0.3,  "good": 0.60, "great": 1.20, "weight": 8},
    # 盈亏因子 — 1.1太低(<1.5说明模型不稳定)
    "profit_factor":     {"min": 1.3,  "good": 1.60, "great": 2.00, "weight": 10},
    # 胜率 — 35%太宽容
    "win_rate_trade":    {"min": 0.40, "good": 0.50, "great": 0.60, "weight": 8},
    # 盈亏比
    "payoff_ratio":      {"min": 1.2,  "good": 1.60, "great": 2.50, "weight": 8},
    # 期望收益
    "expectancy_pct":    {"min": 0.002,"good": 0.006,"great": 0.015,"weight": 6},
    # ★ 最差窗口惩罚 (新增) — 一个窗口崩盘=不合格
    "worst_window":      {"min": -0.25,"good": -0.10,"great": 0.00, "weight": 10},
    # 正窗口比例
    "pos_window_pct":    {"min": 0.50, "good": 0.75, "great": 0.90, "weight": 6},
    # ★ Phase 1 新增: 统计分布
    "skewness":          {"min": -0.5, "good": 0.0,  "great": 0.50, "weight": 5},
    # ★ Phase 1 新增: 捕获率
    "up_capture":        {"min": 0.5,  "good": 0.8,  "great": 1.20, "weight": 5},
    # ★ Phase 1 新增: 溃疡指数性能
    "upi":               {"min": 0.3,  "good": 0.8,  "great": 1.50, "weight": 6},
    # ★ Phase 1 新增: 系统质量
    "sqn":               {"min": 1.0,  "good": 2.0,  "great": 3.00, "weight": 5},
    # ★ Phase 1 新增: 滚动夏普稳定性
    "rolling_sharpe_min": {"min": 0.0,  "good": 0.5,  "great": 1.00, "weight": 5},
    # ★ Phase 1 新增: DSR
    "deflated_sharpe":   {"min": 0.0,  "good": 0.3,  "great": 0.70, "weight": 6},
}


def _score_metric(value: float, thresholds: dict, invert: bool = False) -> float:
    """将指标值映射为 0~1 分数。"""
    if invert:
        value = -value
    if value >= thresholds["great"]:
        return 1.0
    if value >= thresholds["good"]:
        return 0.7 + 0.3 * (value - thresholds["good"]) / (thresholds["great"] - thresholds["good"])
    if thresholds["min"] > 0 and value >= thresholds["min"]:
        return 0.3 + 0.4 * (value - thresholds["min"]) / (thresholds["good"] - thresholds["min"])
    if thresholds["min"] <= 0 and value > 0:
        return 0.3 + 0.4 * value / thresholds["good"]  # min=0 时用 good 做参考
    return max(0.0, 0.3 * value / max(thresholds["min"], 0.001))


# ================================================================
#  ModelEvaluator
# ================================================================

class ModelEvaluator:
    """量化模型评测器 — 三层评测 + 综合评级。"""

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate

    # ════════════════════════════════════
    #  Level 1: 单窗口评测
    # ════════════════════════════════════

    def analyze_window(self, equity: np.ndarray, daily_returns: np.ndarray,
                       benchmark_returns: np.ndarray = None,
                       trades: List[Dict] = None,
                       initial_capital: float = 100_000) -> Dict:
        """
        单窗口完整评测。

        Args:
          equity: 每日权益序列
          daily_returns: 每日收益率序列
          benchmark_returns: 基准每日收益率 (可选)
          trades: 交易列表 [{"date","symbol","action","price","qty","commission","pnl"}, ...]
          initial_capital: 初始资金

        Returns:
          dict: 27+ 指标
        """
        n_days = len(daily_returns)
        n_years = max(n_days / 252, 0.05)
        final_equity = equity[-1]
        total_return = final_equity / initial_capital - 1
        annual_return = (final_equity / initial_capital) ** (1 / n_years) - 1

        # ── 波动率 ──
        annual_vol = float(daily_returns.std() * np.sqrt(252))
        downside = daily_returns[daily_returns < 0]
        downside_std = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else 0.0

        # ── 统计分布 ──
        skewness = float(stats.skew(daily_returns)) if len(daily_returns) > 2 else 0.0
        kurtosis = float(stats.kurtosis(daily_returns)) if len(daily_returns) > 2 else 0.0

        # ── 超额收益 (相对无风险) ──
        excess_daily = daily_returns - self.risk_free_rate / 252
        excess_annual = float(excess_daily.mean() * 252)

        # ── Sharpe ──
        sr = float(excess_daily.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0.0

        # ── Sortino ──
        sortino = float(excess_annual / downside_std) if downside_std > 0 else 0.0

        # ── 最大回撤 ──
        cummax = np.maximum.accumulate(equity)
        drawdowns = (equity - cummax) / cummax
        max_dd = float(drawdowns.min())
        max_dd_duration = 0
        current_duration = 0
        for dd in drawdowns:
            if dd < 0:
                current_duration += 1
                max_dd_duration = max(max_dd_duration, current_duration)
            else:
                current_duration = 0

        # ── Calmar ──
        calmar = float(annual_return / abs(max_dd)) if max_dd != 0 else 0.0

        # ── Ulcer Index & UPI ──
        ulcer = float(np.sqrt(np.mean(drawdowns ** 2))) if len(drawdowns) > 0 else 0.0
        upi = float((total_return - self.risk_free_rate * n_years) / ulcer) if ulcer > 0 else 0.0

        # ── VaR / CVaR ──
        var_95 = float(np.percentile(daily_returns, 5))
        var_99 = float(np.percentile(daily_returns, 1))
        cvar_95 = float(daily_returns[daily_returns <= var_95].mean()) if (daily_returns <= var_95).sum() > 0 else var_95

        # ── Omega Ratio ──
        gains = np.maximum(daily_returns, 0).sum()
        losses = abs(np.minimum(daily_returns, 0).sum())
        omega = float(gains / losses) if losses > 0 else float("inf")

        # ── 日频胜率 ──
        win_days = (daily_returns > 0).sum()
        lose_days = (daily_returns < 0).sum()
        win_rate_daily = float(win_days / (win_days + lose_days)) if (win_days + lose_days) > 0 else 0.0

        # ── 基准对比 ──
        bench_total = 0.0
        alpha = 0.0
        beta = 0.0
        ir = 0.0
        tracking_err = 0.0

        if benchmark_returns is not None and len(benchmark_returns) == len(daily_returns):
            bench_total = float(np.prod(1 + benchmark_returns) - 1)
            benchmark_annual = (1 + bench_total) ** (1 / n_years) - 1 if n_years > 0 else 0.0
            active = daily_returns - benchmark_returns
            tracking_err = float(active.std() * np.sqrt(252))
            ir = float(active.mean() / active.std() * np.sqrt(252)) if active.std() > 0 else 0.0
            if len(daily_returns) > 10:
                cov = np.cov(daily_returns, benchmark_returns)
                if cov.shape == (2, 2) and cov[1, 1] > 0:
                    beta = float(cov[0, 1] / cov[1, 1])
                    alpha = float((daily_returns.mean() - beta * benchmark_returns.mean()) * 252)

        excess_vs_bench = total_return - bench_total
        annual_excess = annual_return - benchmark_annual

        # ── Capture Ratio (Up/Down) ──
        up_capture = 1.0
        down_capture = 1.0
        if benchmark_returns is not None and len(benchmark_returns) == len(daily_returns):
            bench_up = benchmark_returns > 0
            bench_down = benchmark_returns < 0
            if bench_up.sum() > 0:
                up_capture = float(daily_returns[bench_up].mean() / benchmark_returns[bench_up].mean())
            if bench_down.sum() > 0:
                down_capture = float(daily_returns[bench_down].mean() / benchmark_returns[bench_down].mean())

        # ── DSR (Deflated Sharpe Ratio) ──
        dsr = _deflated_sharpe(sr, n_days, skewness, kurtosis)

        # ── Rolling Sharpe ──
        rolling_sr = _rolling_sharpe_stats(daily_returns, window=126)  # 6M

        # ── 交易质量 (基于逐笔交易) ──
        tq = self._analyze_trades(trades, n_years) if trades else {}

        # ── 汇总 ──
        return {
            "n_days": n_days, "n_years": round(n_years, 2),
            "initial_capital": initial_capital, "final_equity": round(final_equity, 2),
            "total_return": total_return, "annual_return": annual_return,
            "annual_volatility": annual_vol, "downside_std": downside_std,
            "skewness": skewness, "kurtosis": kurtosis,
            "max_drawdown": max_dd, "max_dd_duration_days": max_dd_duration,
            "sharpe_ratio": sr, "sortino_ratio": sortino, "calmar_ratio": calmar,
            "ulcer_index": ulcer, "upi": upi,
            "var_95": var_95, "var_99": var_99, "cvar_95": cvar_95, "omega_ratio": omega,
            "win_rate_daily": win_rate_daily,
            "benchmark_return": bench_total, "excess_vs_benchmark": excess_vs_bench,
            "annual_excess": annual_excess,
            "up_capture": up_capture, "down_capture": down_capture,
            "tracking_error": tracking_err, "information_ratio": ir,
            "alpha": alpha, "beta": beta,
            "deflated_sharpe": dsr,
            "rolling_sharpe_min": rolling_sr["min"],
            "rolling_sharpe_mean": rolling_sr["mean"],
            "rolling_sharpe_std": rolling_sr["std"],
            **tq,
        }

    def _analyze_trades(self, trades: List[Dict], n_years: float) -> Dict:
        """分析逐笔交易质量。"""
        if not trades:
            return {}

        df = pd.DataFrame(trades)
        pnls = df["pnl"].values if "pnl" in df.columns else np.zeros(len(df))

        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        n_trades = len(pnls)
        n_wins = len(wins)
        n_losses = len(losses)
        win_rate = n_wins / n_trades if n_trades > 0 else 0.0

        avg_win = float(wins.mean()) if n_wins > 0 else 0.0
        avg_loss = float(abs(losses.mean())) if n_losses > 0 else 0.0
        payoff = avg_win / avg_loss if avg_loss > 0 else float("inf")

        total_profit = float(wins.sum()) if n_wins > 0 else 0.0
        total_loss = float(abs(losses.sum())) if n_losses > 0 else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        # Expectancy: 每笔平均收益(金额)
        expectancy = float(pnls.mean())

        # Expectancy %
        avg_trade_value = df["price"].mean() * df["qty"].mean() if "price" in df.columns and "qty" in df.columns else 10000
        expectancy_pct = expectancy / avg_trade_value if avg_trade_value > 0 else 0.0

        # 最大连续亏损
        max_consec_loss = 0
        current = 0
        for p in pnls:
            if p < 0:
                current += 1
                max_consec_loss = max(max_consec_loss, current)
            else:
                current = 0

        # 换手率
        total_volume = df["qty"].sum() if "qty" in df.columns else 0
        turnover = total_volume / n_years if n_years > 0 else 0.0

        # SQN (System Quality Number)
        sqn = float(np.sqrt(n_trades) * pnls.mean() / pnls.std()) if n_trades > 1 and pnls.std() > 0 else 0.0

        # 胜率最高的前N笔
        hold_days = df["hold_days"].values if "hold_days" in df.columns else np.zeros(n_trades)
        avg_hold = float(hold_days.mean()) if len(hold_days) > 0 else 0.0

        return {
            "n_trades": n_trades, "n_wins": n_wins, "n_losses": n_losses,
            "win_rate_trade": win_rate, "payoff_ratio": payoff if payoff != float("inf") else 999.0,
            "profit_factor": profit_factor if profit_factor != float("inf") else 999.0,
            "expectancy": expectancy, "expectancy_pct": expectancy_pct,
            "avg_win": avg_win, "avg_loss": avg_loss,
            "max_consecutive_losses": max_consec_loss,
            "avg_hold_days": avg_hold, "turnover_annual": turnover,
            "sqn": sqn,
        }

    # ════════════════════════════════════
    #  Level 2: 多窗口聚合
    # ════════════════════════════════════

    def analyze_cross_window(self, window_metrics: List[Dict]) -> Dict:
        """
        多窗口聚合评测 (滚动重训练结果)。

        Args:
          window_metrics: list of per-window metrics dict (from analyze_window)

        Returns:
          dict: 聚合统计
        """
        if not window_metrics:
            return {}

        keys = [k for k in window_metrics[0].keys()
                if isinstance(window_metrics[0][k], (int, float, np.floating))]

        agg = {}
        for k in keys:
            vals = [m[k] for m in window_metrics if k in m and np.isfinite(m[k])]
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            agg[k] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "n_valid": len(arr),
            }

        # 正窗口比例
        excess_vals = [m.get("excess_vs_benchmark", 0) for m in window_metrics]
        pos_windows = sum(1 for e in excess_vals if e > 0)
        total = len(excess_vals)

        # 窗口间 IR (超额均值和标准差之比)
        excess_arr = np.array([e for e in excess_vals if np.isfinite(e)], dtype=float)
        cross_window_ir = float(excess_arr.mean() / excess_arr.std()) if excess_arr.std() > 0 else 0.0

        return {
            "n_windows": total,
            "pos_windows": pos_windows,
            "pos_window_pct": pos_windows / total if total > 0 else 0.0,
            "cross_window_ir": cross_window_ir,
            "per_metric": agg,
        }

    # ════════════════════════════════════
    #  Level 3: 综合评级
    # ════════════════════════════════════

    def grade(self, window_metrics: List[Dict], trades: List[Dict] = None) -> Dict:
        """
        [DEPRECATED] 综合评级 — 不区分开发集/盲测集, 混在一起打分。

        请使用 grade_blind() 获取真实评分。

        Returns:
          {"grade": "C+", "score": 55.2, "pass": False, "details": {...}}
        """
        return self._compute_grade(window_metrics, trades, label="mixed")

    def grade_dev(self, dev_metrics: List[Dict]) -> Dict:
        """
        开发集参考评分 — 仅用于了解模型在调参数据上的拟合程度。
        ⚠️ 此评分不纳入最终成绩, 因为参数是根据这些数据选出来的。
        """
        result = self._compute_grade(dev_metrics, None, label="dev")
        result["_note"] = "仅供参考: 开发集数据参与了参数选择, 不代表真实表现"
        return result

    def grade_blind(self, test_metrics: List[Dict], trades: List[Dict] = None) -> Dict:
        """
        ★ 盲测集最终评分 — 这是真正的成绩单。

        盲测集数据从未在开发过程中被查看。参数完全冻结后才跑一次。
        结果直接就是最终评估, 不再做任何调整。
        """
        result = self._compute_grade(test_metrics, trades, label="blind")
        result["_note"] = "★ 最终评分: 基于未参与开发的盲测数据"
        return result

    def report(self, dev_metrics: List[Dict], blind_metrics: List[Dict],
               blind_trades: List[Dict] = None) -> Dict:
        """
        完整评估报告 — 明确分离开发集和盲测集。

        Returns:
          {
            "dev": {开发集参考评分, 不计入最终成绩},
            "blind": {★ 盲测集最终评分},
            "oos_pct": 盲测数据占比,
            "reliability": 盲测集可信度评估,
          }
        """
        # 计算 OOS 占比
        n_dev = len(dev_metrics)
        n_blind = len(blind_metrics)
        n_total = n_dev + n_blind
        oos_pct = n_blind / n_total if n_total > 0 else 0

        # 评估盲测可靠性
        blind_days = sum(m.get("n_days", 0) for m in blind_metrics)
        if blind_days >= 500:     reliability = "高"
        elif blind_days >= 250:   reliability = "中"
        elif blind_days >= 100:   reliability = "低"
        else:                     reliability = "极低"

        # 盲测窗口数
        if n_blind >= 4:          window_conf = "充分"
        elif n_blind >= 2:        window_conf = "勉强"
        else:                     window_conf = "不足"

        dev_score = self.grade_dev(dev_metrics)
        blind_score = self.grade_blind(blind_metrics, blind_trades)

        # 综合可信度
        trust = "高" if (reliability == "高" and window_conf == "充分") else \
                "中" if (reliability in ("高","中") and window_conf in ("充分","勉强")) else "低"

        return {
            "dev": {
                "grade": dev_score["grade"],
                "score": dev_score["score"],
                "windows": n_dev,
                "_note": "开发集 — 参与参数选择, 仅供参考, 不纳入最终成绩",
                "details": dev_score.get("details", {}),
            },
            "blind": {
                "grade": blind_score["grade"],
                "score": blind_score["score"],
                "windows": n_blind,
                "_note": "★ 盲测集 — 未参与开发, 这是真实水平",
                "details": blind_score.get("details", {}),
            },
            "oos_pct": round(oos_pct * 100, 1),
            "reliability": reliability,
            "blind_windows": window_conf,
            "trust": f"可信度: {trust} (盲测{blind_days}天/{n_blind}窗)"
        }

    def _compute_grade(self, window_metrics: List[Dict], trades: List[Dict] = None,
                       label: str = "mixed") -> Dict:
        """内部实现: 单组窗口的评分计算。"""
        cross = self.analyze_cross_window(window_metrics)
        agg = cross.get("per_metric", {})

        # 汇总各窗口的交易
        all_trades = []
        for m in window_metrics:
            if "all_trades" in m:
                all_trades.extend(m["all_trades"])
        if trades:
            all_trades = trades

        # ★ 最差窗口
        window_returns = [m.get("total_return", 0) for m in window_metrics]
        worst_window = min(window_returns) if window_returns else 0.0

        # ★ 近期加权: 最近窗口权重×2, 次近×1.5, 其余×1
        n = len(window_metrics)
        recent_weights = []
        for i in range(n):
            if i == n - 1:      recent_weights.append(2.0)
            elif i == n - 2:    recent_weights.append(1.5)
            else:               recent_weights.append(1.0)
        w_sum = sum(recent_weights)
        recent_weights = [w / w_sum * n for w in recent_weights] if w_sum > 0 else [1.0]*n

        # ★ 近期加权平均
        recent_weighted = {}
        for key in ["annual_return", "annual_excess", "sharpe_ratio",
                    "max_drawdown", "calmar_ratio", "up_capture", "upi",
                    "skewness", "deflated_sharpe", "rolling_sharpe_min"]:
            vals = [m.get(key, 0) for m in window_metrics]
            recent_weighted[key] = sum(v * w for v, w in zip(vals, recent_weights)) / n

        total_score = 0.0
        total_weight = 0.0
        details = {}

        for metric_name, cfg in GRADE_THRESHOLDS.items():
            invert = metric_name in ("max_drawdown",)
            weight = cfg["weight"]
            total_weight += weight

            if metric_name == "pos_window_pct":
                val = cross.get("pos_window_pct", 0)
            elif metric_name == "worst_window":
                val = worst_window
            elif metric_name in ("win_rate_trade", "payoff_ratio", "profit_factor",
                                "expectancy_pct", "sqn"):
                tq = self._analyze_trades(all_trades, 1.0)
                val = tq.get(metric_name, 0)
            elif metric_name in recent_weighted:
                val = recent_weighted[metric_name]
            elif metric_name in agg:
                val = agg[metric_name]["mean"]
            else:
                val = 0.0

            s = _score_metric(val, cfg, invert=invert)
            total_score += s * weight
            details[metric_name] = {"value": round(val, 4), "score": round(s, 2),
                                     "weight": weight, "grade": _score_to_grade(s)}

        if total_weight == 0:
            return {"grade": "N/A", "score": 0, "pass": False, "details": {}}

        final_score = round(total_score / total_weight * 100, 1)

        # 稳健性惩罚 — 只在盲测集上严格, 开发集宽松
        window_excesses = [m.get("excess_vs_benchmark", 0) for m in window_metrics]
        worst_excess = min(window_excesses) if window_excesses else 0.0
        robustness_penalty = 1.0

        if label == "blind":
            # 盲测集: 严格的稳健性要求
            if worst_excess < -0.10:
                robustness_penalty = 0.80
            elif worst_excess < -0.05:
                robustness_penalty = 0.90
            if len(window_excesses) >= 2:
                mean_ex = np.mean(window_excesses)
                std_ex = np.std(window_excesses)
                if abs(mean_ex) > 0.001:
                    cv = std_ex / abs(mean_ex)
                    if cv > 2.5:
                        robustness_penalty *= 0.85
                    elif cv > 1.5:
                        robustness_penalty *= 0.92
        else:
            # 开发集: 宽松惩罚 (本来就不会用于最终评分)
            if worst_excess < -0.15:
                robustness_penalty = 0.75

        final_score *= robustness_penalty
        if robustness_penalty < 0.99:
            details["_robustness_penalty"] = {"value": round(robustness_penalty, 3),
                                               "score": round(robustness_penalty, 2),
                                               "weight": 0, "grade": ""}

        grade = _final_grade(final_score)

        return {
            "grade": grade,
            "score": final_score,
            "pass": final_score >= 50,
            "details": details,
        }


def _score_to_grade(s: float) -> str:
    if s >= 0.85: return "A"
    if s >= 0.70: return "B"
    if s >= 0.40: return "C"
    return "D"


def _final_grade(score: float) -> str:
    if score >= 85: return "A"
    if score >= 80: return "A-"
    if score >= 75: return "B+"
    if score >= 70: return "B"
    if score >= 65: return "B-"
    if score >= 60: return "C+"
    if score >= 55: return "C"
    if score >= 50: return "C-"
    return "D"


# ════════════════════════════════════════
#  辅助函数: DSR + Rolling Sharpe
# ════════════════════════════════════════

def _deflated_sharpe(sr: float, n_days: int, skew: float, kurt: float,
                     n_trials: int = 6) -> float:
    """
    Deflated Sharpe Ratio (Harvey & Liu 2015).
    考虑多重测试后 Sharpe 的统计显著性。
    """
    if sr <= 0 or n_days < 20:
        return 0.0
    # 夏普标准误 (Lo 2002)
    se = np.sqrt((1 + 0.5 * sr**2 - skew * sr + kurt * sr**2 / 4) / n_days)
    if se <= 0:
        return 0.0
    # Expected Max SR under null
    from scipy.stats import norm
    gamma = 0.5772  # Euler-Mascheroni constant
    e_max = norm.ppf(1 - 1/n_trials) if n_trials > 0 else 0.0
    # Deflated SR
    dsr_val = (sr - e_max * se) / se
    return float(max(0.0, norm.cdf(dsr_val)))


def _rolling_sharpe_stats(daily_returns: np.ndarray, window: int = 126) -> dict:
    """计算滚动夏普的统计量。window=126 ≈ 6个月。"""
    n = len(daily_returns)
    if n < window:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    rolling_sr = []
    for i in range(window, n):
        rets = daily_returns[i-window:i]
        mu = rets.mean()
        sigma = rets.std()
        if sigma > 0:
            rolling_sr.append(float(mu / sigma * np.sqrt(252)))
    if not rolling_sr:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(rolling_sr)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


# ================================================================
#  便捷适配器: 对接 test_rolling_v3.py
# ================================================================

def evaluate_from_v3_results(csv_path: str, equity_logs: List[Dict] = None) -> Dict:
    """
    从 test_rolling_v3.py 的结果 CSV 中读取窗口数据 → 综合评测。

    Args:
      csv_path: test_results/rolling_v3_*.csv
      equity_logs: 可选, 每个窗口的权益曲线 [{equity, returns, benchmark, trades}, ...]

    Returns:
      完整的 ReportCard
    """
    df = pd.read_csv(csv_path)
    evaluator = ModelEvaluator()

    window_metrics = []
    for _, row in df.iterrows():
        m = {
            "total_return": row["strategy"] / 100,
            "benchmark_return": row["benchmark"] / 100,
            "excess_vs_benchmark": row["excess"] / 100,
            "n_trades": row.get("trades", 0),
            "sharpe_ratio": 0.0,   # 需补充
            "max_drawdown": 0.0,   # 需补充
            "calmar_ratio": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "win_rate_trade": 0.0,
            "payoff_ratio": 0.0,
            "profit_factor_trade": 0.0,
            "expectancy": 0.0,
            "expectancy_pct": 0.0,
        }
        # 如果有详细数据则合并
        if equity_logs and len(equity_logs) > len(window_metrics):
            el = equity_logs[len(window_metrics)]
            wm = evaluator.analyze_window(
                np.array(el.get("equity", [])),
                np.array(el.get("returns", [])),
                np.array(el.get("bench_returns", [])) if el.get("bench_returns") else None,
                el.get("trades", []),
            )
            m.update(wm)

        window_metrics.append(m)

    cross = evaluator.analyze_cross_window(window_metrics)
    grade = evaluator.grade(window_metrics)

    return {
        "summary": grade,
        "cross_window": cross,
        "windows": window_metrics,
    }


# ================================================================
#  Demo
# ================================================================

def demo():
    """演示评测模块。"""
    print("ModelEvaluator 演示...")
    np.random.seed(42)

    # 模拟权益曲线
    n = 252 * 2
    returns = np.random.randn(n) * 0.02 + 0.0005  # mean ~12% annual
    equity = 100000 * np.cumprod(1 + returns)
    bench_rets = np.random.randn(n) * 0.015 + 0.0003

    # 模拟交易
    trades = []
    for i in range(50):
        pnl = np.random.randn() * 500 + 200
        trades.append({
            "date": f"2024-{i//25+1:02d}-{i%25+1:02d}",
            "symbol": "600519", "action": "BUY" if i % 2 == 0 else "SELL",
            "price": 100 + i * 2, "qty": 100, "commission": 5,
            "pnl": pnl, "hold_days": np.random.randint(1, 30),
        })

    eval = ModelEvaluator()
    wm = eval.analyze_window(equity, returns, bench_rets, trades)
    cross = eval.analyze_cross_window([wm, wm, wm])  # 模拟3窗口
    grade = eval.grade([wm, wm, wm])

    print(f"\n  评级: {grade['grade']} (得分: {grade['score']})")
    print(f"  合格: {'✅' if grade['pass'] else '❌'}")
    print(f"\n  关键指标:")
    for k in ["sharpe_ratio", "max_drawdown", "calmar_ratio", "profit_factor_trade",
              "win_rate_trade", "payoff_ratio", "expectancy_pct"]:
        if k in wm:
            print(f"    {k}: {wm[k]:.4f}")

    print(f"\n  多窗口: {cross['pos_windows']}/{cross['n_windows']} 正, "
          f"IR={cross['cross_window_ir']:.2f}")

    # 从 v3 CSV 读取
    import os
    v3_files = sorted([f for f in os.listdir("test_results") if f.startswith("rolling_v3_")],
                      reverse=True) if os.path.exists("test_results") else []
    if v3_files:
        path = os.path.join("test_results", v3_files[0])
        report = evaluate_from_v3_results(path)
        print(f"\n  v3 实际结果评测: {report['summary']['grade']} "
              f"(得分: {report['summary']['score']})")
    else:
        print("\n  (无 v3 结果文件, 跳过实际数据评测)")

    print("\n✅ ModelEvaluator 正常")


if __name__ == "__main__":
    demo()
