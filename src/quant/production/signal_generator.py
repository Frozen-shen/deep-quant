"""
Daily signal generation for the IC-weighted linear model.

Workflow:
    1. Load latest market data (from data_cache/ parquet files)
    2. Compute factors for all stocks using the FactorEngine DSL
    3. Load pre-computed IC weights (from model_weights/ic_weights.json)
    4. Score all stocks cross-sectionally (IC-weighted linear combination)
    5. Generate target portfolio (Top-30 with buffer zone Top-80)
    6. Compare with current holdings -> generate buy/sell orders
    7. Save signal to signals/YYYY-MM-DD.json

The signal file is designed to be human-readable so a trader can manually
execute the orders in their brokerage account.

Usage:
    from quant.production.signal_generator import SignalGenerator

    gen = SignalGenerator()
    signal = gen.generate()  # generates for latest available date
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Project root: three levels up from src/quant/production/signal_generator.py
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Default paths (relative to project root)
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "conservative.yaml"
_DEFAULT_WEIGHTS_PATH = _PROJECT_ROOT / "model_weights" / "ic_weights.json"
_DEFAULT_DATA_CACHE = _PROJECT_ROOT / "data_cache"
_DEFAULT_SIGNALS_DIR = _PROJECT_ROOT / "signals"
_DEFAULT_PAPER_TRADE_DIR = _PROJECT_ROOT / "paper_trade"

# Active factors for the production model (from IC analysis)
ACTIVE_FACTORS = [
    "illiquidity_trend",
    "amihud",
    "ret_20d",
    "ret_10d",
    "overnight_return",
]

# Factor expressions (from the factor library DSL)
FACTOR_EXPRESSIONS = {
    "illiquidity_trend": (
        "Mean(Abs($close/Ref($close,1)-1)/($amount+1), 5) / "
        "(Mean(Abs($close/Ref($close,1)-1)/($amount+1), 20) + 0.0001)"
    ),
    "amihud": "Mean(Abs($close/Ref($close,1)-1) / ($amount+1), 20)",
    "ret_20d": "Ref($close, 20) / $close - 1",
    "ret_10d": "Ref($close, 10) / $close - 1",
    "overnight_return": "Mean($open/Ref($close,1)-1, 20)",
}

# Factor directions: 1 = higher is better, -1 = lower is better
# These are used to flip factor values so that higher always = better
FACTOR_DIRECTIONS = {
    "illiquidity_trend": -1,  # lower illiquidity trend is better
    "amihud": -1,             # lower illiquidity is better
    "ret_20d": -1,            # reversed sign convention (see library.py)
    "ret_10d": -1,            # reversed sign convention
    "overnight_return": -1,   # lower overnight return is better
}

# Portfolio construction parameters
TOP_K = 30                # target portfolio size
BUFFER_ZONE = 80          # don't sell until rank drops below this
MAX_CHANGES_PER_REBALANCE = 10  # turnover cap per rebalance


class SignalGenerator:
    """
    Generates daily trading signals based on the IC-Linear model.

    The generator loads market data from the local parquet cache, computes
    factor values using the DSL engine, applies IC-weighted scoring, and
    produces a target portfolio with buy/sell orders.

    Args:
        config_path: Path to the YAML configuration file.
        weights_path: Path to the IC weights JSON file.
        data_cache_dir: Directory containing per-symbol parquet files.
        signals_dir: Directory to write signal JSON files.
    """

    def __init__(
        self,
        config_path: str | Path = _DEFAULT_CONFIG_PATH,
        weights_path: str | Path = _DEFAULT_WEIGHTS_PATH,
        data_cache_dir: str | Path = _DEFAULT_DATA_CACHE,
        signals_dir: str | Path = _DEFAULT_SIGNALS_DIR,
    ):
        self.config_path = Path(config_path)
        self.weights_path = Path(weights_path)
        self.data_cache_dir = Path(data_cache_dir)
        self.signals_dir = Path(signals_dir)

        # Load configuration
        self._config = self._load_config()

        # Load IC weights
        self._ic_weights = self._load_weights()

        # Initialize factor engine (lazy import to avoid circular deps)
        self._engine = None

        # Ensure output directory exists
        self.signals_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, as_of_date: Optional[str] = None) -> dict:
        """
        Generate signal for today (or specified date).

        Args:
            as_of_date: Date string (YYYY-MM-DD). If None, uses the latest
                date available in the data cache.

        Returns:
            Signal dictionary (also saved to signals/ directory).
        """
        logger.info("Starting signal generation...")

        # Step 1: Load data
        data = self._load_data(as_of_date)
        if not data:
            raise RuntimeError("No data available for signal generation")

        # Determine the actual as-of date from loaded data
        actual_date = self._get_latest_date(data)
        if as_of_date is None:
            as_of_date = actual_date
        logger.info("Generating signal for date: %s", as_of_date)

        # Step 2: Compute factor scores
        scores = self._compute_scores(data, as_of_date)
        if not scores:
            raise RuntimeError("Failed to compute scores - no valid factor data")

        # Step 3: Load current holdings (from paper trade state if available)
        current_holdings = self._load_current_holdings()

        # Step 4: Construct target portfolio
        target = self._construct_portfolio(scores, current_holdings)

        # Step 5: Generate orders
        orders = self._generate_orders(target, current_holdings)

        # Step 6: Risk check
        risk_check = self._run_risk_check(target, orders)

        # Step 7: Determine if rebalance is due
        rebalance_due = self._is_rebalance_due(as_of_date)

        # Build signal document
        signal = {
            "date": as_of_date,
            "model_version": "ic_linear_v4",
            "rebalance_due": rebalance_due,
            "target_portfolio": target,
            "orders": orders,
            "risk_check": risk_check,
            "metadata": {
                "n_stocks_scored": len(scores),
                "n_factors_active": len(self._ic_weights),
                "top_factor": self._get_top_factor(),
                "avg_turnover_20d": self._compute_avg_turnover(data, as_of_date),
                "generated_at": datetime.now().isoformat(),
                "data_cache_dir": str(self.data_cache_dir),
            },
        }

        # Step 8: Save signal
        signal_path = self._save_signal(signal, as_of_date)
        logger.info("Signal saved to: %s", signal_path)

        return signal

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self, as_of_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load latest data for all stocks from the parquet cache.

        Returns:
            Dict mapping symbol -> DataFrame with OHLCV data.
        """
        data = {}
        cache_dir = self.data_cache_dir

        if not cache_dir.exists():
            logger.error("Data cache directory not found: %s", cache_dir)
            return data

        parquet_files = sorted(cache_dir.glob("*.parquet"))
        logger.info("Found %d parquet files in cache", len(parquet_files))

        for fpath in parquet_files:
            symbol = fpath.stem
            try:
                df = pd.read_parquet(fpath)
                if df.empty:
                    continue

                # Ensure date column is datetime
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date").reset_index(drop=True)

                # Filter to as_of_date if specified
                if as_of_date is not None:
                    cutoff = pd.Timestamp(as_of_date)
                    df = df[df["date"] <= cutoff]

                # Need at least 60 rows for factor computation (lookback)
                if len(df) >= 60:
                    data[symbol] = df

            except Exception as e:
                logger.warning("Failed to load %s: %s", symbol, e)
                continue

        logger.info("Loaded data for %d stocks", len(data))
        return data

    def _get_latest_date(self, data: Dict[str, pd.DataFrame]) -> str:
        """Get the latest common date across all loaded data."""
        max_dates = []
        for df in data.values():
            if "date" in df.columns and not df.empty:
                max_dates.append(df["date"].max())
        if not max_dates:
            return date.today().isoformat()
        # Use the median date to avoid outliers
        latest = max(max_dates)
        return latest.strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Factor computation and scoring
    # ------------------------------------------------------------------

    def _get_engine(self):
        """Lazy-initialize the factor engine."""
        if self._engine is None:
            from quant.factors.engine import FactorEngine
            self._engine = FactorEngine()
        return self._engine

    def _compute_scores(
        self, data: Dict[str, pd.DataFrame], as_of_date: str
    ) -> Dict[str, float]:
        """
        Compute cross-sectional scores using IC weights.

        For each stock, computes active factor values on the as_of_date,
        standardizes them cross-sectionally (z-score), then applies
        IC-weighted linear combination.

        Args:
            data: Symbol -> DataFrame mapping.
            as_of_date: The date to compute scores for.

        Returns:
            Dict mapping symbol -> composite score (higher is better).
        """
        engine = self._get_engine()
        target_date = pd.Timestamp(as_of_date)

        # Compute raw factor values for each stock
        raw_factors: Dict[str, Dict[str, float]] = {}  # symbol -> {factor: value}

        for symbol, df in data.items():
            # Get data up to as_of_date
            mask = df["date"] <= target_date
            work_df = df[mask].copy()

            if len(work_df) < 25:  # minimum for 20-day lookback factors
                continue

            factor_values = {}
            for factor_name, expression in FACTOR_EXPRESSIONS.items():
                try:
                    result = engine.compute(expression, work_df)
                    if result is not None and len(result) > 0:
                        last_val = result.iloc[-1]
                        if pd.notna(last_val) and np.isfinite(last_val):
                            factor_values[factor_name] = float(last_val)
                except Exception as e:
                    logger.debug(
                        "Factor %s failed for %s: %s", factor_name, symbol, e
                    )
                    continue

            if factor_values:
                raw_factors[symbol] = factor_values

        if not raw_factors:
            logger.error("No valid factor values computed")
            return {}

        logger.info(
            "Computed factors for %d stocks (%d factors each)",
            len(raw_factors),
            len(ACTIVE_FACTORS),
        )

        # Cross-sectional standardization (z-score per factor)
        symbols = list(raw_factors.keys())
        factor_names = ACTIVE_FACTORS

        # Build factor matrix (n_stocks x n_factors)
        n_stocks = len(symbols)
        n_factors = len(factor_names)
        factor_matrix = np.full((n_stocks, n_factors), np.nan, dtype=float)

        for i, sym in enumerate(symbols):
            for j, fname in enumerate(factor_names):
                val = raw_factors[sym].get(fname)
                if val is not None:
                    factor_matrix[i, j] = val

        # Z-score standardization per column (cross-sectional)
        standardized = np.zeros_like(factor_matrix)
        for j in range(n_factors):
            col = factor_matrix[:, j]
            valid_mask = ~np.isnan(col)
            if valid_mask.sum() < 5:
                continue
            valid_vals = col[valid_mask]
            mean = np.mean(valid_vals)
            std = np.std(valid_vals)
            if std > 1e-10:
                standardized[valid_mask, j] = (col[valid_mask] - mean) / std
            # NaN stays as 0 (neutral)

        # Apply factor directions (flip so higher = better)
        for j, fname in enumerate(factor_names):
            direction = FACTOR_DIRECTIONS.get(fname, 1)
            if direction == -1:
                standardized[:, j] = -standardized[:, j]

        # Apply IC weights to compute composite score
        weights = np.array(
            [self._ic_weights.get(fname, 0.0) for fname in factor_names],
            dtype=float,
        )
        weight_sum = weights.sum()

        if weight_sum <= 0:
            # Fallback: equal weight all active factors
            weights = np.ones(n_factors) / n_factors
            weight_sum = 1.0
            logger.warning("No IC weights loaded, using equal weights")

        scores_array = (standardized * weights).sum(axis=1) / weight_sum

        # Build score dictionary
        scores = {}
        for i, sym in enumerate(symbols):
            score_val = float(scores_array[i])
            if np.isfinite(score_val):
                scores[sym] = score_val

        logger.info("Scored %d stocks, score range: [%.3f, %.3f]",
                    len(scores),
                    min(scores.values()) if scores else 0,
                    max(scores.values()) if scores else 0)
        return scores

    # ------------------------------------------------------------------
    # Portfolio construction
    # ------------------------------------------------------------------

    def _construct_portfolio(
        self, scores: Dict[str, float], current_holdings: List[str]
    ) -> List[dict]:
        """
        Apply buffer zone, turnover cap, and liquidity filter.

        Rules:
            - Select Top-30 by score as the target portfolio
            - Buffer zone: existing holdings are kept if they rank within Top-80
            - Max 10 changes (buys + sells) per rebalance
            - Equal weight (1/30) per position

        Args:
            scores: Symbol -> composite score.
            current_holdings: List of currently held symbols.

        Returns:
            List of dicts: [{"symbol", "weight", "score", "rank"}]
        """
        if not scores:
            return []

        # Rank all stocks by score (descending)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rank_map = {sym: rank + 1 for rank, (sym, _) in enumerate(ranked)}

        # Determine target portfolio with buffer zone logic
        current_set = set(current_holdings)
        top_k_symbols = [sym for sym, _ in ranked[:TOP_K]]
        top_k_set = set(top_k_symbols)

        # Buffer zone: keep existing holdings if they rank within BUFFER_ZONE
        keep_from_current = []
        for sym in current_holdings:
            rank = rank_map.get(sym, len(ranked) + 1)
            if rank <= BUFFER_ZONE:
                keep_from_current.append(sym)

        # New entries: top-K stocks not currently held
        new_entries = [sym for sym in top_k_symbols if sym not in current_set]

        # Build target: start with kept holdings, add new entries up to TOP_K
        target_symbols = list(keep_from_current)

        # Add new entries (respecting max changes constraint)
        n_sells = len(current_set) - len(set(keep_from_current))
        max_buys = MAX_CHANGES_PER_REBALANCE - n_sells
        max_buys = max(0, min(max_buys, MAX_CHANGES_PER_REBALANCE))

        for sym in new_entries:
            if len(target_symbols) >= TOP_K:
                break
            if len(new_entries[:new_entries.index(sym) + 1]) <= max_buys:
                target_symbols.append(sym)

        # If we still have room, fill from top-K
        if len(target_symbols) < TOP_K:
            for sym in top_k_symbols:
                if sym not in set(target_symbols):
                    target_symbols.append(sym)
                if len(target_symbols) >= TOP_K:
                    break

        # Truncate to TOP_K
        target_symbols = target_symbols[:TOP_K]

        # Assign equal weights
        weight = 1.0 / len(target_symbols) if target_symbols else 0.0

        target_portfolio = []
        for sym in target_symbols:
            target_portfolio.append({
                "symbol": sym,
                "weight": round(weight, 4),
                "score": round(scores.get(sym, 0.0), 4),
                "rank": rank_map.get(sym, 0),
            })

        # Sort by rank for readability
        target_portfolio.sort(key=lambda x: x["rank"])

        logger.info(
            "Constructed portfolio: %d positions, %d from current, %d new",
            len(target_portfolio),
            len(keep_from_current),
            len(target_portfolio) - len(keep_from_current),
        )

        return target_portfolio

    # ------------------------------------------------------------------
    # Order generation
    # ------------------------------------------------------------------

    def _generate_orders(
        self, target: List[dict], current_holdings: List[str]
    ) -> List[dict]:
        """
        Diff target vs current holdings to generate buy/sell orders.

        Args:
            target: Target portfolio list from _construct_portfolio.
            current_holdings: Currently held symbols.

        Returns:
            List of order dicts: [{"action", "symbol", "weight", "reason"}]
        """
        target_symbols = {item["symbol"] for item in target}
        target_weights = {item["symbol"]: item["weight"] for item in target}
        current_set = set(current_holdings)

        orders = []

        # SELL: in current but not in target
        for sym in current_holdings:
            if sym not in target_symbols:
                orders.append({
                    "action": "SELL",
                    "symbol": sym,
                    "weight": 0.0,
                    "reason": "dropped below rank 80",
                })

        # BUY: in target but not in current
        for sym in target_symbols:
            if sym not in current_set:
                orders.append({
                    "action": "BUY",
                    "symbol": sym,
                    "weight": target_weights.get(sym, 0.0),
                    "reason": "new entry",
                })

        # Sort: sells first (free up cash), then buys
        orders.sort(key=lambda x: (0 if x["action"] == "SELL" else 1, x["symbol"]))

        logger.info(
            "Generated %d orders: %d BUY, %d SELL",
            len(orders),
            sum(1 for o in orders if o["action"] == "BUY"),
            sum(1 for o in orders if o["action"] == "SELL"),
        )

        return orders

    # ------------------------------------------------------------------
    # Risk checks
    # ------------------------------------------------------------------

    def _run_risk_check(self, target: List[dict], orders: List[dict]) -> dict:
        """
        Pre-trade risk checks on the signal.

        Checks:
            - Position concentration (no single position > 8%)
            - Excessive turnover (> 50% of portfolio changing)
            - Minimum diversification (at least 10 positions)

        Returns:
            {"passed": bool, "violations": [str]}
        """
        violations = []

        # Check position concentration
        for item in target:
            if item["weight"] > 0.08:
                violations.append(
                    f"Position {item['symbol']} weight {item['weight']:.1%} > 8%"
                )

        # Check turnover
        n_changes = len(orders)
        n_positions = max(len(target), 1)
        turnover_ratio = n_changes / (n_positions * 2)  # approximate
        if turnover_ratio > 0.50:
            violations.append(
                f"Turnover {turnover_ratio:.0%} exceeds 50% threshold"
            )

        # Check minimum diversification
        if len(target) < 10:
            violations.append(
                f"Only {len(target)} positions (minimum 10 required)"
            )

        return {
            "passed": len(violations) == 0,
            "violations": violations,
        }

    # ------------------------------------------------------------------
    # Rebalance scheduling
    # ------------------------------------------------------------------

    def _is_rebalance_due(self, as_of_date: str) -> bool:
        """
        Check if a rebalance is due based on quarterly schedule.

        Rebalance months: January, April, July, October (first trading day).
        For simplicity, we check if the month is a rebalance month.
        """
        dt = pd.Timestamp(as_of_date)
        rebalance_months = {1, 4, 7, 10}
        return dt.month in rebalance_months

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        """Load configuration from YAML."""
        if self.config_path.exists():
            try:
                import yaml
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning("Failed to load config: %s", e)
        return {}

    def _load_weights(self) -> Dict[str, float]:
        """
        Load IC weights from JSON file.

        Falls back to equal weights if file not found.
        """
        if self.weights_path.exists():
            try:
                with open(self.weights_path, "r", encoding="utf-8") as f:
                    weights = json.load(f)
                logger.info("Loaded IC weights: %s", weights)
                return weights
            except Exception as e:
                logger.warning("Failed to load weights: %s", e)

        # Fallback: equal weights for active factors
        logger.warning(
            "Weights file not found at %s, using equal weights", self.weights_path
        )
        n = len(ACTIVE_FACTORS)
        return {f: 1.0 / n for f in ACTIVE_FACTORS}

    def _load_current_holdings(self) -> List[str]:
        """
        Load current holdings from paper trade state.

        Returns empty list if no state file exists (first run).
        """
        state_file = _DEFAULT_PAPER_TRADE_DIR / "portfolio.json"
        if state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                positions = state.get("positions", {})
                return [sym for sym, pos in positions.items() if pos.get("qty", 0) > 0]
            except Exception as e:
                logger.warning("Failed to load holdings: %s", e)
        return []

    def _get_top_factor(self) -> str:
        """Get the factor with the highest IC weight."""
        if not self._ic_weights:
            return "none"
        return max(self._ic_weights, key=self._ic_weights.get)

    def _compute_avg_turnover(
        self, data: Dict[str, pd.DataFrame], as_of_date: str
    ) -> float:
        """Compute average 20-day turnover (amount) across all stocks."""
        target_date = pd.Timestamp(as_of_date)
        turnovers = []

        for df in data.values():
            if "amount" not in df.columns:
                continue
            mask = df["date"] <= target_date
            recent = df[mask].tail(20)
            if len(recent) >= 10:
                avg_amount = recent["amount"].mean()
                if pd.notna(avg_amount):
                    turnovers.append(float(avg_amount))

        if not turnovers:
            return 0.0
        return float(np.mean(turnovers))

    def _save_signal(self, signal: dict, as_of_date: str) -> Path:
        """Save signal to JSON file."""
        filename = f"{as_of_date}.json"
        path = self.signals_dir / filename

        with open(path, "w", encoding="utf-8") as f:
            json.dump(signal, f, ensure_ascii=False, indent=2)

        return path
