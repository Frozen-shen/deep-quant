"""Event-driven factors: buyback announcements with time decay."""
import os
import numpy as np
import pandas as pd
from typing import Dict


class BuybackFactor:
    """Buyback announcement factor — small/medium buybacks have CAR +1.4~2.9% (p<0.01)."""

    def __init__(self, cache_path: str = None, decay_days: int = 20, half_life: int = 10):
        if cache_path is None:
            cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      'data', 'event_cache', 'buyback_events.parquet')
        self.decay_days = decay_days
        self.half_life = half_life
        self.events = None
        if os.path.exists(cache_path):
            self.events = pd.read_parquet(cache_path)
            self.events['event_date'] = pd.to_datetime(self.events['event_date'])

    def compute_scores(self, as_of_date) -> Dict[str, float]:
        """Return {symbol: score} for stocks with recent buyback events."""
        if self.events is None or len(self.events) == 0:
            return {}
        as_of = pd.Timestamp(as_of_date)
        mask = ((self.events['event_date'] <= as_of) &
                (self.events['event_date'] >= as_of - pd.Timedelta(days=self.decay_days)))
        recent = self.events[mask]
        scores = {}
        for _, row in recent.iterrows():
            days_ago = (as_of - row['event_date']).days
            decay = np.exp(-days_ago / self.half_life)
            base = float(row.get('size_score', 2.0))
            sym = str(row['symbol'])
            # Keep highest score if multiple events
            if sym not in scores or base * decay > scores[sym]:
                scores[sym] = base * decay
        return scores

    def enhance_scores(self, base_scores: Dict[str, float],
                       buyback_scores: Dict[str, float],
                       weight: float = 0.15) -> Dict[str, float]:
        """Add buyback signal as overlay on base model scores."""
        if not buyback_scores:
            return base_scores
        vals = list(buyback_scores.values())
        if len(vals) < 2:
            return base_scores
        mu, sigma = np.mean(vals), np.std(vals)
        if sigma < 1e-8:
            return base_scores
        enhanced = dict(base_scores)
        for sym, bb_val in buyback_scores.items():
            if sym in enhanced:
                z = (bb_val - mu) / sigma
                enhanced[sym] += weight * z
        return enhanced
