"""
Risk management constraints for portfolio construction.

Provides pre-trade and post-trade risk checks including:
    - Single position limits
    - Sector concentration limits
    - Total exposure bounds
    - Turnover limits per rebalance
    - Volatility-based position scaling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class RiskReport:
    """
    Summary of risk checks on a set of portfolio weights.

    Attributes:
        max_position_weight: Largest single position weight.
        max_sector_weight: Largest sector aggregate weight.
        total_exposure: Sum of all position weights (1.0 = fully invested).
        n_positions: Number of non-zero positions.
        violations: List of human-readable violation descriptions.
        is_valid: True if no violations were found.
    """

    max_position_weight: float
    max_sector_weight: float
    total_exposure: float
    n_positions: int
    violations: List[str] = field(default_factory=list)
    is_valid: bool = True


class RiskManager:
    """
    Pre-trade and post-trade risk checks.

    Constraints:
        - Single position: max 5% of portfolio
        - Sector exposure: max 30% per sector
        - Total exposure: 80-100% (allow small cash buffer)
        - Turnover limit: max N% of portfolio value traded per rebalance
        - Volatility cap: reduce position if realized vol > threshold

    Args:
        max_position: Maximum weight for a single position.
        max_sector: Maximum aggregate weight for a single sector.
        max_turnover_per_rebalance: Maximum fraction of portfolio that can
            be traded in a single rebalance event.
        min_exposure: Minimum total portfolio exposure (allows cash buffer).
        max_exposure: Maximum total portfolio exposure.
    """

    def __init__(
        self,
        max_position: float = 0.05,
        max_sector: float = 0.30,
        max_turnover_per_rebalance: float = 0.50,
        min_exposure: float = 0.80,
        max_exposure: float = 1.0,
    ):
        if not 0 < max_position <= 1.0:
            raise ValueError("max_position must be in (0, 1]")
        if not 0 < max_sector <= 1.0:
            raise ValueError("max_sector must be in (0, 1]")
        if not 0 < max_turnover_per_rebalance <= 2.0:
            raise ValueError("max_turnover_per_rebalance must be in (0, 2]")
        if not 0 <= min_exposure <= max_exposure <= 1.0:
            raise ValueError("Must have 0 <= min_exposure <= max_exposure <= 1.0")

        self.max_position = max_position
        self.max_sector = max_sector
        self.max_turnover_per_rebalance = max_turnover_per_rebalance
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure

    def check_weights(
        self,
        weights: Dict[str, float],
        sector_map: Optional[Dict[str, str]] = None,
    ) -> RiskReport:
        """
        Run all risk checks on a set of weights and produce a report.

        Args:
            weights: {symbol: weight} portfolio weights.
            sector_map: {symbol: sector_name} for sector concentration checks.

        Returns:
            RiskReport with all computed metrics and any violations found.
        """
        if not weights:
            return RiskReport(
                max_position_weight=0.0,
                max_sector_weight=0.0,
                total_exposure=0.0,
                n_positions=0,
                violations=["Empty portfolio"],
                is_valid=False,
            )

        violations: List[str] = []

        # Filter to non-zero weights
        active_weights = {s: w for s, w in weights.items() if abs(w) > 1e-10}
        n_positions = len(active_weights)

        # Check single position limit
        max_pos_weight = max(active_weights.values()) if active_weights else 0.0
        if max_pos_weight > self.max_position + 1e-8:
            over_positions = [
                s for s, w in active_weights.items()
                if w > self.max_position + 1e-8
            ]
            violations.append(
                f"Position limit violated: {len(over_positions)} position(s) exceed "
                f"{self.max_position:.1%} (max={max_pos_weight:.2%})"
            )

        # Check sector concentration
        max_sec_weight = 0.0
        if sector_map is not None:
            sector_weights: Dict[str, float] = {}
            for s, w in active_weights.items():
                sector = sector_map.get(s, "unknown")
                sector_weights[sector] = sector_weights.get(sector, 0.0) + w

            if sector_weights:
                max_sec_weight = max(sector_weights.values())
                if max_sec_weight > self.max_sector + 1e-8:
                    over_sectors = [
                        (sec, sw)
                        for sec, sw in sector_weights.items()
                        if sw > self.max_sector + 1e-10
                    ]
                    sector_str = ", ".join(
                        f"{sec}={sw:.1%}" for sec, sw in over_sectors
                    )
                    violations.append(
                        f"Sector limit violated: {sector_str} "
                        f"(limit={self.max_sector:.1%})"
                    )

        # Check total exposure
        total_exposure = sum(active_weights.values())
        if total_exposure < self.min_exposure - 1e-8:
            violations.append(
                f"Underexposed: total={total_exposure:.2%} < min={self.min_exposure:.2%}"
            )
        if total_exposure > self.max_exposure + 1e-8:
            violations.append(
                f"Overexposed: total={total_exposure:.2%} > max={self.max_exposure:.2%}"
            )

        # Check for negative weights (long-only constraint)
        negative_positions = [s for s, w in active_weights.items() if w < -1e-8]
        if negative_positions:
            violations.append(
                f"Negative weights found: {len(negative_positions)} position(s)"
            )

        return RiskReport(
            max_position_weight=max_pos_weight,
            max_sector_weight=max_sec_weight,
            total_exposure=total_exposure,
            n_positions=n_positions,
            violations=violations,
            is_valid=len(violations) == 0,
        )

    def clip_weights(
        self,
        weights: Dict[str, float],
        sector_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """
        Clip weights to satisfy all constraints.

        Applies constraints iteratively:
            1. Clip individual positions to max_position
            2. Clip sector aggregates to max_sector
            3. Ensure total exposure within [min_exposure, max_exposure]
            4. Normalize

        Args:
            weights: Raw weights that may violate constraints.
            sector_map: Sector mapping for concentration limits.

        Returns:
            Constrained weights summing to at most max_exposure.
        """
        if not weights:
            return {}

        clipped = dict(weights)

        # Step 1: Remove negative weights (long-only)
        clipped = {s: max(w, 0.0) for s, w in clipped.items()}

        # Step 2: Iteratively clip positions and sectors
        for _ in range(50):
            changed = False

            # Clip individual positions
            for s in list(clipped.keys()):
                if clipped[s] > self.max_position + 1e-12:
                    clipped[s] = self.max_position
                    changed = True

            # Clip sectors
            if sector_map is not None:
                sector_weights: Dict[str, float] = {}
                sector_members: Dict[str, List[str]] = {}
                for s, w in clipped.items():
                    sector = sector_map.get(s, "unknown")
                    sector_weights[sector] = sector_weights.get(sector, 0.0) + w
                    sector_members.setdefault(sector, []).append(s)

                for sector, sw in sector_weights.items():
                    if sw > self.max_sector + 1e-12:
                        scale = self.max_sector / sw
                        for s in sector_members[sector]:
                            clipped[s] *= scale
                        changed = True

            if not changed:
                break

        # Step 3: Enforce total exposure bounds
        total = sum(clipped.values())
        if total > self.max_exposure + 1e-12:
            scale = self.max_exposure / total
            clipped = {s: w * scale for s, w in clipped.items()}
        elif total < self.min_exposure - 1e-12 and total > 0:
            # Scale up to min_exposure (but respect position caps)
            scale = self.min_exposure / total
            clipped = {s: min(w * scale, self.max_position) for s, w in clipped.items()}

        # Step 4: Remove dust
        clipped = {s: w for s, w in clipped.items() if w > 1e-6}

        return clipped

    def compute_turnover(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float],
    ) -> float:
        """
        Compute one-way turnover between two weight vectors.

        Turnover = 0.5 * sum(|w_new - w_old|)

        A turnover of 1.0 means the entire portfolio was replaced.
        Annual turnover of 3.0 (300%) is the target maximum.

        Args:
            old_weights: Previous portfolio weights.
            new_weights: Target portfolio weights.

        Returns:
            One-way turnover as a fraction of portfolio value.
        """
        all_symbols = set(old_weights.keys()) | set(new_weights.keys())
        if not all_symbols:
            return 0.0

        total_change = 0.0
        for s in all_symbols:
            old_w = old_weights.get(s, 0.0)
            new_w = new_weights.get(s, 0.0)
            total_change += abs(new_w - old_w)

        # One-way turnover is half the total absolute change
        return total_change / 2.0

    def check_turnover(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float],
    ) -> bool:
        """
        Check if turnover between two weight vectors is within limits.

        Args:
            old_weights: Current portfolio weights.
            new_weights: Desired portfolio weights.

        Returns:
            True if turnover is within the per-rebalance limit.
        """
        turnover = self.compute_turnover(old_weights, new_weights)
        return turnover <= self.max_turnover_per_rebalance + 1e-10

    def scale_volatility_positions(
        self,
        weights: Dict[str, float],
        volatility: Dict[str, float],
        vol_threshold: float = 0.60,
    ) -> Dict[str, float]:
        """
        Reduce positions for stocks with realized volatility above threshold.

        Positions with vol > vol_threshold are scaled down by
        (vol_threshold / vol) to control tail risk.

        Args:
            weights: Current weights.
            volatility: {symbol: annualized_realized_vol}.
            vol_threshold: Maximum acceptable annualized volatility.

        Returns:
            Adjusted weights (may sum to less than original).
        """
        if not weights or not volatility:
            return weights

        adjusted = {}
        for s, w in weights.items():
            vol = volatility.get(s)
            if vol is not None and vol > vol_threshold and vol > 0:
                scale = vol_threshold / vol
                adjusted[s] = w * scale
            else:
                adjusted[s] = w

        return adjusted
