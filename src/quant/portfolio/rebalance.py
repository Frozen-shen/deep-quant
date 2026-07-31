"""
Rebalancing logic with turnover control.

Controls WHEN and HOW to rebalance:
    - Calendar-based: fixed schedule (quarterly/monthly)
    - Threshold-based: rebalance when drift exceeds threshold
    - Hybrid: calendar + threshold override

Turnover control is critical for A-share strategies:
    - Old system: >1000% annual turnover destroyed all alpha
    - New target: max 300% annual turnover (~75% per quarterly rebalance)
    - With 10.5bp round-trip cost, 300% turnover costs ~31.5bp/year
"""

from datetime import date, timedelta
from typing import Dict, Optional

import numpy as np


class RebalanceController:
    """
    Controls WHEN and HOW to rebalance.

    Strategies:
        - calendar: rebalance on fixed schedule (quarterly/monthly)
        - threshold: rebalance when drift exceeds threshold
        - hybrid: calendar + threshold override

    Turnover control:
        - max_turnover_per_rebalance: cap on single rebalance
        - If desired turnover > cap, partially rebalance (move proportionally)
        - Buffer zone: don't trade if score change is small

    Args:
        freq: Rebalance frequency. One of "quarterly", "monthly", "weekly".
        max_turnover: Maximum one-way turnover per rebalance event.
            0.50 means at most 50% of portfolio value changes hands.
        drift_threshold: If using threshold/hybrid mode, rebalance when
            portfolio drift (from target) exceeds this fraction.
        buffer_days: Minimum days between rebalances (prevents over-trading
            near calendar boundaries).
    """

    VALID_FREQUENCIES = ("quarterly", "monthly", "weekly")

    def __init__(
        self,
        freq: str = "quarterly",
        max_turnover: float = 0.50,
        drift_threshold: float = 0.10,
        buffer_days: int = 5,
    ):
        if freq not in self.VALID_FREQUENCIES:
            raise ValueError(
                f"freq must be one of {self.VALID_FREQUENCIES}, got '{freq}'"
            )
        if not 0 < max_turnover <= 2.0:
            raise ValueError("max_turnover must be in (0, 2]")
        if not 0 < drift_threshold <= 1.0:
            raise ValueError("drift_threshold must be in (0, 1]")
        if buffer_days < 0:
            raise ValueError("buffer_days must be >= 0")

        self.freq = freq
        self.max_turnover = max_turnover
        self.drift_threshold = drift_threshold
        self.buffer_days = buffer_days

    def should_rebalance(
        self,
        current_date: date,
        last_rebalance_date: Optional[date],
    ) -> bool:
        """
        Determine if a rebalance should occur on current_date.

        For calendar-based rebalancing, checks if we've crossed into a new
        period since the last rebalance, respecting the buffer zone.

        Args:
            current_date: The date to check.
            last_rebalance_date: Date of the most recent rebalance, or None
                if no rebalance has occurred yet.

        Returns:
            True if a rebalance should be triggered.
        """
        # First rebalance always happens
        if last_rebalance_date is None:
            return True

        # Respect buffer zone
        days_since = (current_date - last_rebalance_date).days
        if days_since < self.buffer_days:
            return False

        if self.freq == "quarterly":
            return self._crossed_quarter(last_rebalance_date, current_date)
        elif self.freq == "monthly":
            return self._crossed_month(last_rebalance_date, current_date)
        elif self.freq == "weekly":
            return days_since >= 7

        return False

    def should_rebalance_threshold(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> bool:
        """
        Check if portfolio drift exceeds the threshold.

        Drift is measured as one-way turnover between current and target.

        Args:
            current_weights: Current portfolio weights.
            target_weights: Desired target weights from model.

        Returns:
            True if drift exceeds the threshold.
        """
        drift = self._compute_drift(current_weights, target_weights)
        return drift > self.drift_threshold

    def compute_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute the trades needed to move from current to target weights.

        Returns {symbol: trade_weight} where positive = buy, negative = sell.
        Trades are subject to the turnover cap.

        Args:
            current_weights: Current portfolio weights.
            target_weights: Desired target weights.

        Returns:
            Trade weights (positive = buy, negative = sell).
        """
        # Compute raw trades
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        raw_trades: Dict[str, float] = {}
        for s in all_symbols:
            current_w = current_weights.get(s, 0.0)
            target_w = target_weights.get(s, 0.0)
            trade = target_w - current_w
            if abs(trade) > 1e-8:  # Ignore dust trades
                raw_trades[s] = trade

        # Apply turnover cap
        capped_trades = self.apply_turnover_cap(raw_trades)

        return capped_trades

    def apply_turnover_cap(self, trades: Dict[str, float]) -> Dict[str, float]:
        """
        Cap trades to respect maximum turnover per rebalance.

        If total turnover exceeds the cap, all trades are scaled down
        proportionally. This means we partially rebalance toward the target
        rather than fully rebalancing.

        The scaling preserves the relative priority of trades: large
        conviction trades get proportionally more execution than small ones.

        Args:
            trades: {symbol: trade_weight} uncapped trades.

        Returns:
            Capped trades where total one-way turnover <= max_turnover.
        """
        if not trades:
            return {}

        # Compute total one-way turnover
        total_turnover = sum(abs(t) for t in trades.values()) / 2.0

        if total_turnover <= self.max_turnover + 1e-12:
            return trades

        # Scale down proportionally
        scale_factor = self.max_turnover / total_turnover
        capped = {s: t * scale_factor for s, t in trades.items()}

        # Remove dust after scaling
        capped = {s: t for s, t in capped.items() if abs(t) > 1e-8}

        return capped

    def compute_buffer_zone_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        score_changes: Optional[Dict[str, float]] = None,
        min_score_change: float = 0.05,
    ) -> Dict[str, float]:
        """
        Compute trades with a buffer zone on score changes.

        Positions where the score hasn't changed much are not traded,
        reducing unnecessary turnover from noise in model outputs.

        Args:
            current_weights: Current portfolio weights.
            target_weights: Target weights from model.
            score_changes: {symbol: abs_score_change} if available.
            min_score_change: Minimum score change to trigger a trade.

        Returns:
            Filtered and capped trades.
        """
        trades = self.compute_trades(current_weights, target_weights)

        if score_changes is not None:
            # Only trade positions where score changed significantly
            filtered = {}
            for s, t in trades.items():
                score_change = score_changes.get(s, float("inf"))
                if score_change >= min_score_change:
                    filtered[s] = t
                # If score barely changed, skip the trade (keep current weight)
            trades = filtered

        # Re-apply turnover cap after filtering
        return self.apply_turnover_cap(trades)

    def estimate_annual_turnover(self) -> float:
        """
        Estimate maximum annual turnover given the rebalance frequency
        and per-rebalance cap.

        Returns:
            Maximum annual one-way turnover.
        """
        rebalances_per_year = {
            "quarterly": 4,
            "monthly": 12,
            "weekly": 52,
        }
        return self.max_turnover * rebalances_per_year[self.freq]

    def _crossed_quarter(self, last_date: date, current_date: date) -> bool:
        """Check if we've crossed a quarter boundary."""
        last_quarter = (last_date.year, (last_date.month - 1) // 3)
        current_quarter = (current_date.year, (current_date.month - 1) // 3)
        return current_quarter != last_quarter

    def _crossed_month(self, last_date: date, current_date: date) -> bool:
        """Check if we've crossed a month boundary."""
        last_month = (last_date.year, last_date.month)
        current_month = (current_date.year, current_date.month)
        return current_month != last_month

    def _compute_drift(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> float:
        """Compute one-way drift between current and target."""
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        if not all_symbols:
            return 0.0

        total_change = sum(
            abs(target_weights.get(s, 0.0) - current_weights.get(s, 0.0))
            for s in all_symbols
        )
        return total_change / 2.0
