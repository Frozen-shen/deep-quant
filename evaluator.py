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
#  评级阈值
# ================================================================

GRADE_THRESHOLDS = {
    "annual_excess":     {"min": 0.03, "good": 0.10, "great": 0.20, "weight": 15},
    "sharpe_ratio":      {"min": 0.3,  "good": 0.80, "great": 1.50, "weight": 15},
    "max_drawdown":      {"min": -0.35,"good": -0.20,"great": -0.12, "weight": 10},
    "calmar_ratio":      {"min": 0.2,  "good": 0.50, "great": 1.00, "weight": 10},
    "profit_factor":     {"min": 1.1,  "good": 1.50, "great": 2.00, "weight": 12},
    "win_rate_trade":    {"min": 0.35, "good": 0.50, "great": 0.60, "weight": 8},
    "payoff_ratio":      {"min": 1.0,  "good": 1.50, "great": 2.50, "weight": 8},
    "expectancy_pct":    {"min": 0.001,"good": 0.005,"great": 0.015,"weight": 7},
    "pos_window_pct":    {"min": 0.50, "good": 0.70, "great": 0.85, "weight": 8},
    "excess_ir":         {"min": 0.3,  "good": 0.80, "great": 1.50, "weight": 7},
}


def _score_metric(value: float, thresholds: dict, invert: bool = False) -> float:
    """将指标值映射为 0~1 分数。"""
    if invert:
        value = -value
    if value >= thresholds["great"]:
        return 1.0
    if value >= thresholds["good"]:
        return 0.7 + 0.3 * (value - thresholds["good"]) / (thresholds["great"] - thresholds["good"])
    if value >= thresholds["min"]:
        return 0.3 + 0.4 * (value - thresholds["min"]) / (thresholds["good"] - thresholds["min"])
    return max(0.0, 0.3 * value / thresholds["min"])


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

        # ── 交易质量 (基于逐笔交易) ──
        tq = self._analyze_trades(trades, n_years) if trades else {}

        # ── 汇总 ──
        return {
            "n_days": n_days, "n_years": round(n_years, 2),
            "initial_capital": initial_capital, "final_equity": round(final_equity, 2),
            "total_return": total_return, "annual_return": annual_return,
            "annual_volatility": annual_vol, "downside_std": downside_std,
            "max_drawdown": max_dd, "max_dd_duration_days": max_dd_duration,
            "sharpe_ratio": sr, "sortino_ratio": sortino, "calmar_ratio": calmar,
            "var_95": var_95, "var_99": var_99, "cvar_95": cvar_95, "omega_ratio": omega,
            "win_rate_daily": win_rate_daily,
            "benchmark_return": bench_total, "excess_vs_benchmark": excess_vs_bench,
            "annual_excess": annual_excess,
            "tracking_error": tracking_err, "information_ratio": ir,
            "alpha": alpha, "beta": beta,
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
        综合评级 — 基于所有窗口的平均指标打分。

        Returns:
          {"grade": "B+", "score": 72.5, "pass": True, "details": {...}}
        """
        cross = self.analyze_cross_window(window_metrics)
        agg = cross.get("per_metric", {})

        # 汇总各窗口的交易
        all_trades = []
        for m in window_metrics:
            if "all_trades" in m:
                all_trades.extend(m["all_trades"])
        if trades:
            all_trades = trades

        total_score = 0.0
        total_weight = 0.0
        details = {}

        for metric_name, cfg in GRADE_THRESHOLDS.items():
            invert = metric_name in ("max_drawdown",)
            weight = cfg["weight"]
            total_weight += weight

            if metric_name == "pos_window_pct":
                val = cross.get("pos_window_pct", 0)
            elif metric_name == "excess_ir":
                val = cross.get("cross_window_ir", 0)
            elif metric_name in ("win_rate_trade", "payoff_ratio", "profit_factor",
                                "expectancy_pct"):
                tq = self._analyze_trades(all_trades, 1.0)
                val = tq.get(metric_name, 0)
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
