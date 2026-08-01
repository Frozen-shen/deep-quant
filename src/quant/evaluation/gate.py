"""
验证门 (Validation Gate) — 回测结果必须通过此门才能被报告。

Usage:
    from quant.evaluation.gate import validate_result, GateResult
    
    result = {...}  # backtest output
    gate = validate_result(result)
    if not gate.passed:
        print(f"BLOCKED: {gate.failures}")
    else:
        print("PASSED: result is reportable")

Rules (from DEVELOPMENT_DISCIPLINE.md):
  1. Turnover must be >5% and <95% per rebalance
  2. Sharpe must be <2.5 (abnormally high = likely bug)
  3. IR must be <1.5 (abnormally high = likely bias)
  4. Must have >=3 years of data
  5. Cost rate must be >=15bp round-trip
  6. Must have >0 positions
  7. Partition must not be "blind"
"""
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class GateResult:
    """Result of validation gate check."""
    passed: bool
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checks_run: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __str__(self):
        status = "PASS" if self.passed else "BLOCKED"
        msg = f"[{status}] {self.checks_run} checks"
        if self.failures:
            msg += f", {len(self.failures)} failures: {self.failures}"
        if self.warnings:
            msg += f", {len(self.warnings)} warnings: {self.warnings}"
        return msg


# Fixed data partitions (NEVER change these)
DATE_RANGES = {
    "train": ("2018-01-01", "2023-12-31"),
    "val": ("2024-01-01", "2025-06-30"),
    "test": ("2025-07-01", "2025-12-31"),
    "blind": ("2026-01-01", "2026-07-10"),
}

# Thresholds
MIN_TURNOVER = 0.05       # 5% per rebalance (strategy must trade)
MAX_TURNOVER = 0.95       # 95% per rebalance (not random)
MAX_SHARPE = 2.5          # Above this = likely bug or bias
MAX_IR = 1.5              # Above this = likely look-ahead or survivorship
MIN_YEARS = 3             # Need at least 3 years
MIN_COST_BP = 15.0        # Minimum 15bp round-trip cost
MIN_POSITIONS = 1         # Must hold something


def validate_result(result: dict, partition: str = "val") -> GateResult:
    """
    Validate a backtest result against the discipline rules.
    
    Args:
        result: dict with keys like 'sharpe', 'ir', 'avg_turnover', 
                'yearly', 'cost_rate_bp', 'n_positions'
        partition: which data partition was used
    
    Returns:
        GateResult with passed/failed status and details
    """
    failures = []
    warnings = []
    checks = 0

    # Rule 0: Partition check
    checks += 1
    if partition == "blind":
        failures.append("BLIND partition used for backtest (FORBIDDEN)")
    elif partition not in DATE_RANGES:
        failures.append(f"Unknown partition '{partition}'")

    # Rule 1: Turnover bounds
    checks += 1
    avg_to = result.get("avg_turnover", result.get("avg_to", 0))
    if avg_to < MIN_TURNOVER:
        failures.append(
            f"Turnover too low: {avg_to*100:.1f}%/rb < {MIN_TURNOVER*100:.0f}% "
            f"(portfolio may be frozen)"
        )
    elif avg_to > MAX_TURNOVER:
        failures.append(
            f"Turnover too high: {avg_to*100:.1f}%/rb > {MAX_TURNOVER*100:.0f}% "
            f"(essentially random trading)"
        )

    # Rule 2: Sharpe sanity
    checks += 1
    sharpe = result.get("sharpe", 0)
    if abs(sharpe) > MAX_SHARPE:
        failures.append(
            f"Sharpe abnormally high: {sharpe:.2f} > {MAX_SHARPE} "
            f"(likely bug, bias, or overfitting)"
        )

    # Rule 3: IR sanity
    checks += 1
    ir = result.get("ir", result.get("information_ratio", 0))
    if ir is not None and abs(ir) > MAX_IR:
        failures.append(
            f"IR abnormally high: {ir:.3f} > {MAX_IR} "
            f"(likely look-ahead bias or survivorship)"
        )

    # Rule 4: Sufficient data
    checks += 1
    yearly = result.get("yearly", {})
    n_years = len(yearly) if isinstance(yearly, dict) else 0
    if n_years < MIN_YEARS:
        failures.append(
            f"Insufficient data: {n_years} years < {MIN_YEARS} minimum"
        )

    # Rule 5: Cost model
    checks += 1
    cost_bp = result.get("cost_rate_bp", result.get("cost_bp", 0))
    if cost_bp < MIN_COST_BP:
        failures.append(
            f"Cost too low: {cost_bp:.1f}bp < {MIN_COST_BP:.0f}bp minimum "
            f"(unrealistic cost model)"
        )

    # Rule 6: Positions
    checks += 1
    n_pos = result.get("n_positions", result.get("top_k", 0))
    if n_pos < MIN_POSITIONS:
        failures.append(f"No positions held (n_positions={n_pos})")

    # Warnings (don't block, but flag)
    yearly_excess = result.get("yearly_excess", {})
    if yearly_excess:
        wins = sum(1 for v in yearly_excess.values() if v > 0)
        win_rate = wins / len(yearly_excess)
        if win_rate < 0.67:
            warnings.append(
                f"Year win rate {win_rate*100:.0f}% < 67% "
                f"({wins}/{len(yearly_excess)} years positive excess)"
            )

    if sharpe is not None and 1.5 < abs(sharpe) <= MAX_SHARPE:
        warnings.append(f"Sharpe {sharpe:.2f} is high — verify methodology")

    return GateResult(
        passed=len(failures) == 0,
        failures=failures,
        warnings=warnings,
        checks_run=checks,
    )


def save_experiment(result: dict, gate: GateResult, params: dict,
                    partition: str, notes: str = "",
                    exp_dir: str = "experiments") -> str:
    """
    Save experiment record to JSON file.
    
    Returns the file path.
    """
    exp_path = Path(exp_dir)
    exp_path.mkdir(parents=True, exist_ok=True)

    # Generate ID
    existing = list(exp_path.glob("*.json"))
    exp_id = f"exp_{datetime.now().strftime('%Y%m%d')}_{len(existing)+1:03d}"

    record = {
        "experiment_id": exp_id,
        "timestamp": datetime.now().isoformat(),
        "partition": partition,
        "parameters": params,
        "results": result,
        "validation": {
            "passed": gate.passed,
            "failures": gate.failures,
            "warnings": gate.warnings,
            "checks_run": gate.checks_run,
        },
        "notes": notes,
    }

    filepath = exp_path / f"{exp_id}.json"
    filepath.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str))
    return str(filepath)


def assert_partition(start_date: str, end_date: str, partition: str):
    """
    Assert that the given date range matches the declared partition.
    Call at the top of every backtest script.
    
    Usage:
        assert_partition("2024-01-01", "2025-06-30", "val")
    """
    assert partition != "blind", (
        "FORBIDDEN: blind partition (2026+) must never be used for backtesting. "
        "See DEVELOPMENT_DISCIPLINE.md Rule #1."
    )
    expected_start, expected_end = DATE_RANGES[partition]
    assert start_date >= expected_start, (
        f"Start {start_date} is before {partition} partition start {expected_start}"
    )
    assert end_date <= expected_end, (
        f"End {end_date} exceeds {partition} partition end {expected_end}. "
        f"See DEVELOPMENT_DISCIPLINE.md Rule #1."
    )
