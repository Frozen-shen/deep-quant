"""
Unified configuration system for quant-starter.

Uses dataclasses for type-safe configuration with YAML file loading
and nested merge support. Configuration is immutable after creation
(frozen dataclasses) to prevent accidental mutation during backtests.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class DataConfig:
    """Data acquisition and storage settings."""

    cache_dir: str = "data_store"
    universe: str = "csi1000"  # csi300, csi500, csi1000, all_a
    start_date: str = "2018-01-01"
    end_date: str = "2025-12-31"
    frequency: str = "daily"  # daily, weekly, monthly
    adjust_type: str = "qfq"  # qfq (forward), hfq (backward), none
    data_source: str = "akshare"


@dataclass(frozen=True)
class FactorConfig:
    """Factor selection and filtering settings."""

    preset: str = "auto"  # auto = use all factors with IC > threshold
    ic_threshold: float = 0.02  # minimum IC to include a factor
    use_fundamental: bool = True
    use_technical: bool = True
    use_events: bool = True
    max_factors: int = 50  # cap to prevent overfitting
    decay_halflife: int = 60  # days for IC decay weighting
    neutralize: bool = True  # industry + market-cap neutralization
    winsorize_sigma: float = 3.0  # MAD winsorization threshold


@dataclass(frozen=True)
class ModelConfig:
    """Model training and prediction settings."""

    type: str = "ic_weighted_linear"  # PRIMARY model
    # ML fallback (only if linear underperforms)
    ml_type: str = "lightgbm"
    ml_n_estimators: int = 50  # EXTREMELY conservative
    ml_max_depth: int = 2
    ml_min_data_in_leaf: int = 500
    ml_learning_rate: float = 0.01
    ml_lambda_l1: float = 1.0
    ml_lambda_l2: float = 1.0
    ml_feature_fraction: float = 0.8
    ml_bagging_fraction: float = 0.8
    ml_bagging_freq: int = 5
    # Walk-forward optimization settings
    train_window_days: int = 504  # 2 years
    step_days: int = 63  # quarterly retrain
    val_ratio: float = 0.15
    # Ensemble
    ensemble_method: str = "ic_weighted"  # ic_weighted, equal, rank_average


@dataclass(frozen=True)
class PortfolioConfig:
    """Portfolio construction and risk management settings."""

    top_k: int = 30
    max_positions: int = 30
    rebalance_freq: str = "quarterly"  # monthly, quarterly
    max_turnover_annual: float = 3.0  # 300% max annual turnover
    max_single_weight: float = 0.05  # 5% per stock
    max_sector_weight: float = 0.30  # 30% per sector
    min_holding_days: int = 20
    sell_rank_buffer: int = 5  # don't sell until rank drops below top_k + buffer
    buy_confirm_days: int = 3  # confirm buy signal for N consecutive days
    weight_method: str = "equal"  # equal, score_weighted, risk_parity
    allow_short: bool = False


@dataclass(frozen=True)
class BacktestConfig:
    """Backtesting simulation settings."""

    commission_rate: float = 0.00025  # 万2.5
    slippage_rate: float = 0.0005  # 万5
    impact_rate: float = 0.0003  # 万3
    initial_capital: float = 1_000_000
    benchmark: str = "000852"  # CSI1000
    price_field: str = "close"  # close, vwap
    use_limit_check: bool = True  # skip if limit-up/down
    suspend_handling: str = "hold"  # hold, sell_at_open

    @property
    def total_cost_bps(self) -> float:
        """Total one-way transaction cost in basis points."""
        return (self.commission_rate + self.slippage_rate + self.impact_rate) * 10000


@dataclass(frozen=True)
class QuantConfig:
    """Root configuration combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    factors: FactorConfig = field(default_factory=FactorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QuantConfig":
        """
        Load configuration from a YAML file.

        Supports nested partial overrides: only specified fields are
        overridden, all others retain their defaults.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A fully-resolved QuantConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If the YAML contains invalid field names.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        return cls._from_dict(raw)

    @classmethod
    def from_yaml_with_base(
        cls, base_path: str | Path, override_path: str | Path
    ) -> "QuantConfig":
        """
        Load a base config then deep-merge an override file on top.

        Args:
            base_path: Path to the base YAML (e.g., configs/base.yaml).
            override_path: Path to the override YAML (e.g., configs/aggressive.yaml).

        Returns:
            A merged QuantConfig instance.
        """
        base_path = Path(base_path)
        override_path = Path(override_path)

        with open(base_path, "r", encoding="utf-8") as f:
            base_raw = yaml.safe_load(f) or {}

        with open(override_path, "r", encoding="utf-8") as f:
            override_raw = yaml.safe_load(f) or {}

        merged = _deep_merge(base_raw, override_raw)
        return cls._from_dict(merged)

    @classmethod
    def _from_dict(cls, raw: Dict[str, Any]) -> "QuantConfig":
        """Construct QuantConfig from a raw dictionary."""
        return cls(
            data=_build_dataclass(DataConfig, raw.get("data", {})),
            factors=_build_dataclass(FactorConfig, raw.get("factors", {})),
            model=_build_dataclass(ModelConfig, raw.get("model", {})),
            portfolio=_build_dataclass(PortfolioConfig, raw.get("portfolio", {})),
            backtest=_build_dataclass(BacktestConfig, raw.get("backtest", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire config to a plain dictionary."""
        result = asdict(self)
        # Add computed property
        result["backtest"]["total_cost_bps"] = self.backtest.total_cost_bps
        return result

    def to_yaml(self, path: str | Path) -> None:
        """Write the configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.to_dict(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    def summary(self) -> str:
        """Return a human-readable summary of key settings."""
        lines = [
            "=== QuantConfig Summary ===",
            f"  Universe:       {self.data.universe}",
            f"  Date Range:     {self.data.start_date} ~ {self.data.end_date}",
            f"  Model:          {self.model.type}",
            f"  Rebalance:      {self.portfolio.rebalance_freq}",
            f"  Top-K:          {self.portfolio.top_k}",
            f"  Max Turnover:   {self.portfolio.max_turnover_annual:.0%}/yr",
            f"  Cost (1-way):   {self.backtest.total_cost_bps:.1f} bp",
            f"  Benchmark:      {self.backtest.benchmark}",
            "===========================",
        ]
        return "\n".join(lines)


def _build_dataclass(cls: Type[T], raw: Dict[str, Any]) -> T:
    """
    Build a dataclass instance from a dict, ignoring unknown keys
    and filtering out properties (like total_cost_bps).
    """
    valid_fields = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in valid_fields}
    return cls(**filtered)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override dict into base dict.
    Override values take precedence. Nested dicts are merged recursively.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
