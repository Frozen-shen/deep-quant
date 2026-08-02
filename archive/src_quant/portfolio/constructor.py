"""
Portfolio construction from model scores.

Pipeline:
    scores -> rank -> top-K selection -> weight assignment -> risk constraints -> final weights

Weighting schemes:
    - equal: 1/K per position (robust baseline, IR=0.377 in old system)
    - score_weighted: proportional to score (normalized)
    - risk_parity: inverse volatility weighting
"""

from typing import Dict, List, Optional

import numpy as np


class PortfolioConstructor:
    """
    Converts model scores into portfolio weights.

    The constructor applies a multi-step pipeline to transform raw model
    outputs into tradeable portfolio weights that respect position limits,
    sector constraints, and risk budgets.

    Args:
        top_k: Number of positions to hold. Default 30 provides good
            diversification while concentrating on highest-conviction names.
        weighting: Weight assignment scheme. One of "equal", "score_weighted",
            or "risk_parity".
        max_single_weight: Maximum weight for any single position (default 5%).
        max_sector_weight: Maximum total weight for any single sector (default 30%).
    """

    VALID_WEIGHTING_SCHEMES = ("equal", "score_weighted", "risk_parity")

    def __init__(
        self,
        top_k: int = 30,
        weighting: str = "equal",
        max_single_weight: float = 0.05,
        max_sector_weight: float = 0.30,
    ):
        if weighting not in self.VALID_WEIGHTING_SCHEMES:
            raise ValueError(
                f"weighting must be one of {self.VALID_WEIGHTING_SCHEMES}, got '{weighting}'"
            )
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 0 < max_single_weight <= 1.0:
            raise ValueError("max_single_weight must be in (0, 1]")
        if not 0 < max_sector_weight <= 1.0:
            raise ValueError("max_sector_weight must be in (0, 1]")

        self.top_k = top_k
        self.weighting = weighting
        self.max_single_weight = max_single_weight
        self.max_sector_weight = max_sector_weight

    def construct(
        self,
        scores: Dict[str, float],
        sector_map: Optional[Dict[str, str]] = None,
        volatility: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Convert model scores into portfolio weights.

        Args:
            scores: {symbol: score} from model. Higher is better.
            sector_map: {symbol: sector_name} for sector constraints.
                If None, sector constraints are skipped.
            volatility: {symbol: annualized_vol} for risk parity weighting.
                Required if weighting="risk_parity".

        Returns:
            weights: {symbol: weight} summing to 1.0 (or less if constraints
                force cash allocation).
        """
        if not scores:
            return {}

        # Step 1: Select top-K by score
        selected = self._select_top_k(scores)

        # Step 2: Assign initial weights
        weights = self._assign_weights(selected, scores, volatility)

        # Step 3: Apply single-position cap
        weights = self._clip_positions(weights)

        # Step 4: Apply sector constraints
        if sector_map is not None:
            weights = self._apply_constraints(weights, sector_map)

        # Step 5: Final normalization (ensure sum <= 1.0)
        weights = self._normalize(weights)

        return weights

    def _select_top_k(self, scores: Dict[str, float]) -> List[str]:
        """
        Select the top-K symbols by score.

        Uses numpy argsort for efficiency. Ties are broken arbitrarily
        (by dict insertion order via stable sort).
        """
        symbols = list(scores.keys())
        score_values = np.array([scores[s] for s in symbols], dtype=np.float64)

        # argsort ascending, take last K (highest scores)
        n_select = min(self.top_k, len(symbols))
        # Use argpartition for O(n) selection, then sort the selected
        if n_select < len(symbols):
            partition_idx = np.argpartition(score_values, -n_select)[-n_select:]
            # Sort selected by score descending for deterministic ordering
            sorted_order = partition_idx[np.argsort(score_values[partition_idx])[::-1]]
        else:
            sorted_order = np.argsort(score_values)[::-1]

        return [symbols[i] for i in sorted_order]

    def _assign_weights(
        self,
        selected: List[str],
        scores: Dict[str, float],
        volatility: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Assign weights according to the chosen scheme.

        Args:
            selected: List of selected symbols.
            scores: Full score dictionary.
            volatility: Volatility dictionary (for risk_parity).

        Returns:
            Unnormalized weights dict.
        """
        n = len(selected)
        if n == 0:
            return {}

        if self.weighting == "equal":
            w = 1.0 / n
            return {s: w for s in selected}

        elif self.weighting == "score_weighted":
            # Shift scores to be positive, then normalize
            score_arr = np.array([scores[s] for s in selected], dtype=np.float64)
            # Shift so minimum is at least a small positive value
            min_score = score_arr.min()
            if min_score <= 0:
                score_arr = score_arr - min_score + 1e-8
            total = score_arr.sum()
            if total <= 0:
                # Fallback to equal weight if all scores are zero/negative
                w = 1.0 / n
                return {s: w for s in selected}
            normalized = score_arr / total
            return {s: float(normalized[i]) for i, s in enumerate(selected)}

        elif self.weighting == "risk_parity":
            if volatility is None:
                raise ValueError(
                    "volatility dict required for risk_parity weighting"
                )
            # Inverse volatility weighting
            inv_vols = []
            valid_symbols = []
            for s in selected:
                vol = volatility.get(s)
                if vol is not None and vol > 0:
                    inv_vols.append(1.0 / vol)
                    valid_symbols.append(s)
                else:
                    # If no vol data, use median inverse vol as fallback
                    inv_vols.append(None)
                    valid_symbols.append(s)

            # Fill None values with median of known inverse vols
            known_inv_vols = [v for v in inv_vols if v is not None]
            if not known_inv_vols:
                # No volatility data at all, fallback to equal weight
                w = 1.0 / n
                return {s: w for s in selected}

            median_inv_vol = float(np.median(known_inv_vols))
            inv_vols = [v if v is not None else median_inv_vol for v in inv_vols]

            inv_vol_arr = np.array(inv_vols, dtype=np.float64)
            total = inv_vol_arr.sum()
            normalized = inv_vol_arr / total
            return {s: float(normalized[i]) for i, s in enumerate(selected)}

        # Should never reach here due to __init__ validation
        raise ValueError(f"Unknown weighting scheme: {self.weighting}")

    def _clip_positions(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Iteratively clip positions exceeding max_single_weight.

        Uses the water-filling algorithm: clip oversized positions and
        redistribute excess weight proportionally to uncapped positions.
        If no uncapped positions exist, the excess becomes cash (portfolio
        sums to less than 1.0).
        """
        if not weights:
            return weights

        clipped = dict(weights)
        max_iterations = 100  # Safety bound

        for _ in range(max_iterations):
            # Find positions exceeding the cap
            over_cap = {
                s: w for s, w in clipped.items() if w > self.max_single_weight + 1e-12
            }
            if not over_cap:
                break

            # Clip and compute excess
            excess = 0.0
            for s in over_cap:
                excess += clipped[s] - self.max_single_weight
                clipped[s] = self.max_single_weight

            # Redistribute excess to uncapped positions proportionally
            uncapped = {s: w for s, w in clipped.items() if w < self.max_single_weight - 1e-12}
            if not uncapped:
                # All positions at cap, no room to redistribute.
                # Excess becomes cash (portfolio sums to < 1.0).
                break
            else:
                total_uncapped = sum(uncapped.values())
                if total_uncapped > 0:
                    for s in uncapped:
                        clipped[s] += excess * (uncapped[s] / total_uncapped)
                else:
                    per_position = excess / len(uncapped)
                    for s in uncapped:
                        clipped[s] += per_position

        return clipped

    def _apply_constraints(
        self, weights: Dict[str, float], sector_map: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Apply sector concentration constraints.

        Iteratively reduces overweight sectors by scaling down positions
        within the sector proportionally, then redistributes to underweight
        sectors.
        """
        if not weights:
            return weights

        constrained = dict(weights)
        max_iterations = 50

        for _ in range(max_iterations):
            # Compute sector weights
            sector_weights: Dict[str, float] = {}
            sector_members: Dict[str, List[str]] = {}
            for s, w in constrained.items():
                sector = sector_map.get(s, "unknown")
                sector_weights[sector] = sector_weights.get(sector, 0.0) + w
                sector_members.setdefault(sector, []).append(s)

            # Find sectors exceeding the cap
            over_sectors = {
                sec: sw
                for sec, sw in sector_weights.items()
                if sw > self.max_sector_weight + 1e-12
            }
            if not over_sectors:
                break

            # Scale down overweight sectors
            for sector, sector_weight in over_sectors.items():
                scale_factor = self.max_sector_weight / sector_weight
                for s in sector_members[sector]:
                    constrained[s] *= scale_factor

            # Renormalize to sum to 1.0
            total = sum(constrained.values())
            if total > 0:
                for s in constrained:
                    constrained[s] /= total

        # Re-apply position cap after sector constraint (may have been violated)
        constrained = self._clip_positions(constrained)

        return constrained

    def _normalize(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Final normalization: ensure weights sum to at most 1.0 without
        violating the single-position cap.

        If scaling up to 1.0 would push positions above max_single_weight,
        the portfolio is left partially in cash (sum < 1.0). This is correct
        behavior: it's better to hold cash than violate risk limits.

        Removes any positions with negligible weight (< 1bp) to avoid
        dust trades.
        """
        if not weights:
            return {}

        # Remove dust positions
        min_weight = 1e-4  # 1bp
        filtered = {s: w for s, w in weights.items() if w >= min_weight}

        if not filtered:
            return {}

        total = sum(filtered.values())
        if total <= 0:
            return {}

        if total > 1.0:
            # Scale down to 1.0 (this never violates position caps)
            normalized = {s: w / total for s, w in filtered.items()}
        elif total < 1.0:
            # Try to scale up, but check if it would violate position cap
            scale = 1.0 / total
            max_after_scale = max(filtered.values()) * scale
            if max_after_scale <= self.max_single_weight + 1e-10:
                # Safe to scale up
                normalized = {s: w * scale for s, w in filtered.items()}
            else:
                # Scaling up would violate caps; leave partially in cash
                normalized = dict(filtered)
        else:
            normalized = dict(filtered)

        return normalized
