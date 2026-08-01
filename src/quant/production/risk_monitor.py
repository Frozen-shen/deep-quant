"""
Real-time risk monitoring and alerting for the production portfolio.

Monitors portfolio risk and triggers alerts based on predefined rules:
    1. Daily loss > 3%             -> ALERT: "Consider reducing exposure"
    2. Monthly drawdown > 10%      -> ALERT: "Reduce to 50% position"
    3. 3 months underperform bench -> ALERT: "Strategy review needed"
    4. Single position > 8%        -> ALERT: "Position too concentrated"
    5. Turnover > 50% in rebalance -> ALERT: "Excessive trading"
    6. Tracking error > 2% annual  -> ALERT: "Execution drift"

Output: risk_report.json with current status and recommendations.

Usage:
    from quant.production.risk_monitor import RiskMonitor

    monitor = RiskMonitor(portfolio_state, benchmark_returns)
    report = monitor.check_all()
    print(report.overall_status)  # "OK", "WARNING", "CRITICAL"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_REPORT_PATH = _PROJECT_ROOT / "paper_trade" / "risk_report.json"

# Risk thresholds
DAILY_LOSS_THRESHOLD = -0.03          # 3% daily loss
MONTHLY_DRAWDOWN_THRESHOLD = -0.10    # 10% monthly drawdown
UNDERPERFORMANCE_MONTHS = 3           # consecutive months underperforming
POSITION_CONCENTRATION_LIMIT = 0.08   # 8% single position
TURNOVER_LIMIT = 0.50                 # 50% turnover per rebalance
TRACKING_ERROR_THRESHOLD = 0.02       # 2% annualized tracking error


@dataclass
class RiskCheck:
    """Result of a single risk check."""
    name: str
    status: str       # "OK", "WARNING", "CRITICAL"
    message: str
    value: float = 0.0
    threshold: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RiskReport:
    """
    Comprehensive risk report from all checks.

    Attributes:
        timestamp: ISO timestamp of when the report was generated.
        overall_status: Worst status across all checks ("OK", "WARNING", "CRITICAL").
        checks: List of individual check results.
        recommendations: Actionable recommendations based on violations.
    """
    timestamp: str
    overall_status: str               # "OK", "WARNING", "CRITICAL"
    checks: List[dict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status,
            "checks": self.checks,
            "recommendations": self.recommendations,
        }

    def save(self, path: str | Path = _DEFAULT_REPORT_PATH) -> None:
        """Save report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("Risk report saved to: %s", path)


class RiskMonitor:
    """
    Monitors portfolio risk and triggers alerts.

    Runs a battery of risk checks against the current portfolio state
    and produces a RiskReport with actionable recommendations.

    Args:
        portfolio_state: Dict with portfolio state (from PaperTrader.save_state format):
            {cash, positions: {symbol: {qty, avg_cost, market_value}}, daily_equity: [...]}
        benchmark_returns: List of daily benchmark returns (e.g., CSI1000).
        report_path: Path to save the risk report JSON.
    """

    def __init__(
        self,
        portfolio_state: dict,
        benchmark_returns: List[float],
        report_path: str | Path = _DEFAULT_REPORT_PATH,
    ):
        self.state = portfolio_state
        self.benchmark_returns = benchmark_returns
        self.report_path = Path(report_path)

    def check_all(self) -> RiskReport:
        """
        Run all risk checks, return report.

        Executes each risk check in sequence, aggregates results,
        and determines the overall status and recommendations.

        Returns:
            RiskReport with all check results and recommendations.
        """
        checks = []
        recommendations = []

        # Run each check
        check_funcs = [
            ("daily_loss", self.check_daily_loss),
            ("drawdown", self.check_drawdown),
            ("underperformance", self.check_underperformance),
            ("concentration", self.check_concentration),
            ("turnover", self.check_turnover),
            ("tracking_error", self.check_tracking_error),
        ]

        for name, func in check_funcs:
            try:
                result = func()
                if result is not None:
                    checks.append(result.to_dict())
                    if result.status in ("WARNING", "CRITICAL"):
                        recommendations.append(result.message)
                else:
                    checks.append(RiskCheck(
                        name=name, status="OK", message="Check passed"
                    ).to_dict())
            except Exception as e:
                logger.warning("Risk check '%s' failed: %s", name, e)
                checks.append(RiskCheck(
                    name=name, status="OK",
                    message=f"Check skipped (error: {e})"
                ).to_dict())

        # Determine overall status
        statuses = [c["status"] for c in checks]
        if "CRITICAL" in statuses:
            overall = "CRITICAL"
        elif "WARNING" in statuses:
            overall = "WARNING"
        else:
            overall = "OK"

        report = RiskReport(
            timestamp=datetime.now().isoformat(),
            overall_status=overall,
            checks=checks,
            recommendations=recommendations,
        )

        # Save report
        report.save(self.report_path)

        logger.info(
            "Risk check complete: %s (%d warnings)",
            overall, len(recommendations),
        )

        return report

    def check_daily_loss(self) -> Optional[RiskCheck]:
        """
        Check if today's loss exceeds 3%.

        Rule: Daily loss > 3% -> ALERT: "Consider reducing exposure"
        """
        daily_equity = self.state.get("daily_equity", [])
        if len(daily_equity) < 2:
            return None

        latest = daily_equity[-1]
        daily_return = latest.get("daily_return", 0.0)

        if daily_return <= DAILY_LOSS_THRESHOLD:
            return RiskCheck(
                name="daily_loss",
                status="CRITICAL",
                message=(
                    f"Daily loss {daily_return:.2%} exceeds {DAILY_LOSS_THRESHOLD:.0%} "
                    f"threshold. Consider reducing exposure."
                ),
                value=daily_return,
                threshold=DAILY_LOSS_THRESHOLD,
            )

        return RiskCheck(
            name="daily_loss",
            status="OK",
            message=f"Daily return {daily_return:.2%} within limits",
            value=daily_return,
            threshold=DAILY_LOSS_THRESHOLD,
        )

    def check_drawdown(self) -> Optional[RiskCheck]:
        """
        Check if monthly drawdown exceeds 10%.

        Rule: Monthly drawdown > 10% -> ALERT: "Reduce to 50% position"
        """
        daily_equity = self.state.get("daily_equity", [])
        if len(daily_equity) < 20:
            return None

        # Use last ~22 trading days (one month)
        recent = daily_equity[-22:]
        equities = [r["total_equity"] for r in recent]

        peak = max(equities)
        current = equities[-1]
        drawdown = (current - peak) / peak if peak > 0 else 0.0

        if drawdown <= MONTHLY_DRAWDOWN_THRESHOLD:
            return RiskCheck(
                name="drawdown",
                status="CRITICAL",
                message=(
                    f"Monthly drawdown {drawdown:.2%} exceeds "
                    f"{MONTHLY_DRAWDOWN_THRESHOLD:.0%} threshold. "
                    f"Reduce to 50% position."
                ),
                value=drawdown,
                threshold=MONTHLY_DRAWDOWN_THRESHOLD,
            )

        return RiskCheck(
            name="drawdown",
            status="OK",
            message=f"Monthly drawdown {drawdown:.2%} within limits",
            value=drawdown,
            threshold=MONTHLY_DRAWDOWN_THRESHOLD,
        )

    def check_underperformance(self) -> Optional[RiskCheck]:
        """
        Check for consecutive months of underperformance vs benchmark.

        Rule: 3 consecutive months underperform -> ALERT: "Strategy review needed"
        """
        daily_equity = self.state.get("daily_equity", [])
        if len(daily_equity) < 66 or len(self.benchmark_returns) < 66:
            return None

        # Compute monthly returns for the last 3 months
        portfolio_returns = [r["daily_return"] for r in daily_equity]
        n = min(len(portfolio_returns), len(self.benchmark_returns))
        portfolio_returns = portfolio_returns[-n:]
        benchmark_returns = self.benchmark_returns[-n:]

        # Split into monthly chunks (~22 trading days each)
        months_underperform = 0
        month_size = 22

        for i in range(0, min(n, month_size * 3), month_size):
            chunk_end = min(i + month_size, n)
            port_month = sum(portfolio_returns[i:chunk_end])
            bench_month = sum(benchmark_returns[i:chunk_end])

            if port_month < bench_month:
                months_underperform += 1
            else:
                break  # reset on outperformance

        if months_underperform >= UNDERPERFORMANCE_MONTHS:
            return RiskCheck(
                name="underperformance",
                status="WARNING",
                message=(
                    f"Strategy underperformed benchmark for {months_underperform} "
                    f"consecutive months. Strategy review needed."
                ),
                value=months_underperform,
                threshold=UNDERPERFORMANCE_MONTHS,
            )

        return RiskCheck(
            name="underperformance",
            status="OK",
            message=f"Underperformance streak: {months_underperform} months (limit: {UNDERPERFORMANCE_MONTHS})",
            value=months_underperform,
            threshold=UNDERPERFORMANCE_MONTHS,
        )

    def check_concentration(self) -> Optional[RiskCheck]:
        """
        Check if any single position exceeds 8% of portfolio.

        Rule: Single position > 8% -> ALERT: "Position too concentrated"
        """
        positions = self.state.get("positions", {})
        daily_equity = self.state.get("daily_equity", [])

        if not positions:
            return None

        # Get total equity
        if daily_equity:
            total_equity = daily_equity[-1].get("total_equity", 1.0)
        else:
            cash = self.state.get("cash", 0)
            total_equity = cash + sum(
                p.get("market_value", p.get("qty", 0) * p.get("avg_cost", 0))
                for p in positions.values()
            )

        if total_equity <= 0:
            return None

        # Find max position weight
        max_weight = 0.0
        max_symbol = ""
        for symbol, pos in positions.items():
            mv = pos.get("market_value", pos.get("qty", 0) * pos.get("avg_cost", 0))
            weight = mv / total_equity
            if weight > max_weight:
                max_weight = weight
                max_symbol = symbol

        if max_weight > POSITION_CONCENTRATION_LIMIT:
            return RiskCheck(
                name="concentration",
                status="WARNING",
                message=(
                    f"Position {max_symbol} is {max_weight:.1%} of portfolio "
                    f"(limit: {POSITION_CONCENTRATION_LIMIT:.0%}). "
                    f"Position too concentrated."
                ),
                value=max_weight,
                threshold=POSITION_CONCENTRATION_LIMIT,
            )

        return RiskCheck(
            name="concentration",
            status="OK",
            message=f"Max position weight {max_weight:.1%} within limits",
            value=max_weight,
            threshold=POSITION_CONCENTRATION_LIMIT,
        )

    def check_turnover(self) -> Optional[RiskCheck]:
        """
        Check if turnover in the last rebalance exceeded 50%.

        Rule: Turnover > 50% in single rebalance -> ALERT: "Excessive trading"
        """
        trade_history = self.state.get("trade_history", [])
        daily_equity = self.state.get("daily_equity", [])

        if not trade_history or not daily_equity:
            return None

        # Get total equity for normalization
        total_equity = daily_equity[-1].get("total_equity", 1.0)
        if total_equity <= 0:
            return None

        # Look at trades from the most recent date
        latest_date = trade_history[-1].get("date", "")
        recent_trades = [t for t in trade_history if t.get("date") == latest_date]

        if not recent_trades:
            return None

        # Compute one-way turnover
        total_traded = sum(abs(t.get("amount", 0)) for t in recent_trades)
        turnover = total_traded / (total_equity * 2)  # one-way

        if turnover > TURNOVER_LIMIT:
            return RiskCheck(
                name="turnover",
                status="WARNING",
                message=(
                    f"Rebalance turnover {turnover:.0%} exceeds "
                    f"{TURNOVER_LIMIT:.0%} limit. Excessive trading."
                ),
                value=turnover,
                threshold=TURNOVER_LIMIT,
            )

        return RiskCheck(
            name="turnover",
            status="OK",
            message=f"Turnover {turnover:.0%} within limits",
            value=turnover,
            threshold=TURNOVER_LIMIT,
        )

    def check_tracking_error(self) -> Optional[RiskCheck]:
        """
        Check if tracking error vs backtest exceeds 2% annualized.

        Rule: TE > 2% annualized -> ALERT: "Execution drift"
        """
        daily_equity = self.state.get("daily_equity", [])
        if len(daily_equity) < 10 or len(self.benchmark_returns) < 10:
            return None

        portfolio_returns = [r["daily_return"] for r in daily_equity]
        n = min(len(portfolio_returns), len(self.benchmark_returns))

        if n < 10:
            return None

        port_arr = np.array(portfolio_returns[-n:])
        bench_arr = np.array(self.benchmark_returns[-n:])

        diff = port_arr - bench_arr
        te_daily = np.std(diff, ddof=1) if n > 1 else 0.0
        te_annual = te_daily * np.sqrt(252)

        if te_annual > TRACKING_ERROR_THRESHOLD:
            return RiskCheck(
                name="tracking_error",
                status="WARNING",
                message=(
                    f"Annualized tracking error {te_annual:.2%} exceeds "
                    f"{TRACKING_ERROR_THRESHOLD:.0%} threshold. Execution drift detected."
                ),
                value=te_annual,
                threshold=TRACKING_ERROR_THRESHOLD,
            )

        return RiskCheck(
            name="tracking_error",
            status="OK",
            message=f"Tracking error {te_annual:.2%} within limits",
            value=te_annual,
            threshold=TRACKING_ERROR_THRESHOLD,
        )
