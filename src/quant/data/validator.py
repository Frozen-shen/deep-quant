"""
Data quality validation module.

Provides comprehensive data quality checks for market data:
  - Gap detection (missing trading days)
  - Outlier detection (abnormal price movements)
  - Limit-up/down flagging
  - Survivorship bias analysis
  - Full panel validation with structured reporting

Usage:
    from quant.data.validator import DataValidator, ValidationReport
    from quant.data.calendar import TradingCalendar

    cal = TradingCalendar()
    validator = DataValidator(calendar=cal)

    report = validator.validate_panel(panel)
    print(report.summary())

    gaps = validator.check_gaps(df, cal)
    outliers = validator.check_outliers(df, threshold=5.0)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SymbolValidation:
    """Validation results for a single symbol."""

    symbol: str
    row_count: int = 0
    date_range: str = ""
    gaps: List[str] = field(default_factory=list)
    outliers: List[str] = field(default_factory=list)
    limit_flags: int = 0
    ohlc_issues: List[str] = field(default_factory=list)
    duplicates: int = 0
    issues: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """No critical issues found."""
        return len(self.issues) == 0 and self.duplicates == 0

    @property
    def total_issues(self) -> int:
        return (
            len(self.gaps)
            + len(self.outliers)
            + len(self.ohlc_issues)
            + len(self.issues)
            + self.duplicates
        )


class ValidationReport:
    """
    Aggregated validation report for a DataPanel or batch of symbols.

    Collects per-symbol validation results and provides summary statistics.
    """

    def __init__(self):
        self.timestamp: str = datetime.now().isoformat()
        self.total_symbols: int = 0
        self.checked_symbols: int = 0
        self.symbol_results: Dict[str, SymbolValidation] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []

    @property
    def passed(self) -> bool:
        """True if no critical errors were found."""
        return len(self.errors) == 0

    @property
    def symbols_with_issues(self) -> int:
        return sum(1 for r in self.symbol_results.values() if not r.passed)

    def add_result(self, result: SymbolValidation) -> None:
        """Add a per-symbol validation result."""
        self.symbol_results[result.symbol] = result
        self.checked_symbols += 1

        # Classify issues as errors or warnings
        for issue in result.issues:
            if any(kw in issue for kw in ("empty", "missing", "corrupt", "high<low")):
                self.errors.append(f"{result.symbol}: {issue}")
            else:
                self.warnings.append(f"{result.symbol}: {issue}")

        for gap in result.gaps[:5]:  # Cap to avoid flooding
            self.warnings.append(f"{result.symbol}: gap: {gap}")

        for outlier in result.outliers[:5]:
            self.warnings.append(f"{result.symbol}: outlier: {outlier}")

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            "=" * 60,
            "Data Quality Validation Report",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Symbols checked: {self.checked_symbols}/{self.total_symbols}",
            f"Status: {'PASS' if self.passed else 'FAIL'}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
            f"Symbols with issues: {self.symbols_with_issues}",
            "-" * 60,
        ]

        if self.errors:
            lines.append("ERRORS:")
            for e in self.errors[:20]:
                lines.append(f"  [E] {e}")
            if len(self.errors) > 20:
                lines.append(f"  ... and {len(self.errors) - 20} more")

        if self.warnings:
            lines.append("WARNINGS (first 10):")
            for w in self.warnings[:10]:
                lines.append(f"  [W] {w}")
            if len(self.warnings) > 10:
                lines.append(f"  ... and {len(self.warnings) - 10} more")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize report to a dictionary."""
        return {
            "timestamp": self.timestamp,
            "total_symbols": self.total_symbols,
            "checked_symbols": self.checked_symbols,
            "passed": self.passed,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "symbols_with_issues": self.symbols_with_issues,
            "errors": self.errors[:100],
            "warnings": self.warnings[:100],
        }


class DataValidator:
    """
    Data quality validator for market data.

    Parameters
    ----------
    calendar : TradingCalendar, optional
        Trading calendar instance for gap detection. If None, gap
        detection will use a simpler heuristic (max 5-day gaps).
    outlier_threshold : float
        Z-score threshold for outlier detection. Default 5.0.
    max_suspension_days : int
        Maximum consecutive zero-volume days before flagging. Default 60.
    """

    def __init__(
        self,
        calendar=None,
        outlier_threshold: float = 5.0,
        max_suspension_days: int = 60,
    ):
        self._calendar = calendar
        self._outlier_threshold = outlier_threshold
        self._max_suspension_days = max_suspension_days

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_gaps(
        self,
        df: pd.DataFrame,
        calendar=None,
    ) -> List[str]:
        """
        Check for missing trading days in the data.

        Parameters
        ----------
        df : pd.DataFrame
            Must have a 'date' column with datetime values.
        calendar : TradingCalendar, optional
            If provided, uses the actual trading calendar for precise
            gap detection. Otherwise falls back to a heuristic.

        Returns
        -------
        List[str]
            List of human-readable gap descriptions.
        """
        cal = calendar or self._calendar
        if df is None or df.empty or len(df) < 2:
            return []

        dates = pd.to_datetime(df["date"]).sort_values()
        issues: List[str] = []

        if cal is not None:
            # Precise: compare against actual trading calendar
            start = dates.iloc[0]
            end = dates.iloc[-1]
            expected_days = set(cal.get_trading_days(start, end))
            actual_days = set(dates.tolist())
            missing = sorted(expected_days - actual_days)

            if missing:
                # Group consecutive missing days
                groups = self._group_consecutive(missing)
                for group in groups:
                    if len(group) == 1:
                        issues.append(
                            f"Missing trading day: {group[0].strftime('%Y-%m-%d')}"
                        )
                    else:
                        issues.append(
                            f"Missing {len(group)} trading days: "
                            f"{group[0].strftime('%Y-%m-%d')} ~ "
                            f"{group[-1].strftime('%Y-%m-%d')}"
                        )
        else:
            # Heuristic: flag gaps > 5 calendar days
            diffs = dates.diff().dropna()
            for i, gap in enumerate(diffs):
                if gap.days > 5:
                    date_str = dates.iloc[i + 1].strftime("%Y-%m-%d")
                    issues.append(
                        f"Date gap of {gap.days} days before {date_str}"
                    )

        return issues

    def check_outliers(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> List[str]:
        """
        Detect price outliers using rolling z-score of daily returns.

        Parameters
        ----------
        df : pd.DataFrame
            Must have 'date' and 'close' columns.
        threshold : float, optional
            Z-score threshold. Defaults to self._outlier_threshold.

        Returns
        -------
        List[str]
            Descriptions of suspicious rows.
        """
        threshold = threshold or self._outlier_threshold
        if df is None or df.empty or len(df) < 30:
            return []

        df_sorted = df.sort_values("date").reset_index(drop=True)
        returns = df_sorted["close"].pct_change()

        # Rolling z-score (60-day window)
        window = 60
        rolling_mean = returns.rolling(window, min_periods=20).mean()
        rolling_std = returns.rolling(window, min_periods=20).std()

        issues: List[str] = []
        for i in range(window, len(df_sorted)):
            std = rolling_std.iloc[i]
            if pd.isna(std) or std == 0:
                continue
            z_score = (returns.iloc[i] - rolling_mean.iloc[i]) / std
            if abs(z_score) > threshold:
                date_str = df_sorted["date"].iloc[i].strftime("%Y-%m-%d")
                ret_val = returns.iloc[i]
                issues.append(
                    f"{date_str}: return {ret_val:+.2%}, "
                    f"z-score {z_score:+.1f} (threshold: {threshold:.1f})"
                )

        return issues

    def check_limit_up_down(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag limit-up and limit-down days.

        Adds columns:
          - 'pct_return': daily percentage return
          - 'is_limit_up': True if at upper limit
          - 'is_limit_down': True if at lower limit
          - 'limit_type': 'up', 'down', or None

        Parameters
        ----------
        df : pd.DataFrame
            Must have 'date', 'close', and optionally 'open', 'high', 'low'.

        Returns
        -------
        pd.DataFrame
            Copy of input with limit flag columns added.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        result = df.copy().sort_values("date").reset_index(drop=True)
        result["pct_return"] = result["close"].pct_change() * 100

        # Determine limit threshold per row (simplified: use first digit)
        # In practice, you'd look up the symbol's board type
        limit_pct = 10.0  # Default for main board

        result["is_limit_up"] = result["pct_return"] >= (limit_pct - 0.2)
        result["is_limit_down"] = result["pct_return"] <= -(limit_pct - 0.2)

        # One-word board detection (cannot trade)
        if all(c in result.columns for c in ["open", "high", "low", "close"]):
            one_word = (
                (result["open"] == result["high"])
                & (result["high"] == result["low"])
                & (result["low"] == result["close"])
            )
            result["is_limit_up"] = result["is_limit_up"] | (
                one_word & (result["pct_return"] > 0)
            )
            result["is_limit_down"] = result["is_limit_down"] | (
                one_word & (result["pct_return"] < 0)
            )

        # Combine into limit_type column
        conditions = [result["is_limit_up"], result["is_limit_down"]]
        choices = ["up", "down"]
        result["limit_type"] = np.select(conditions, choices, default=None)

        return result

    def check_survivorship(
        self,
        universe_history: Dict[str, List[str]],
    ) -> dict:
        """
        Analyze survivorship bias in universe history.

        Parameters
        ----------
        universe_history : Dict[str, List[str]]
            Mapping of period (e.g., "2023-01") to symbol lists.

        Returns
        -------
        dict
            Bias report containing:
              - added: symbols that appeared over time
              - removed: symbols that disappeared (potential delistings)
              - survivorship_ratio: fraction of original still present
              - is_biased: whether significant bias is detected
        """
        if not universe_history:
            return {"is_biased": False, "message": "No history provided"}

        sorted_keys = sorted(universe_history.keys())
        if len(sorted_keys) < 2:
            return {"is_biased": False, "message": "Insufficient history"}

        first_set = set(universe_history[sorted_keys[0]])
        last_set = set(universe_history[sorted_keys[-1]])

        removed = first_set - last_set
        added = last_set - first_set

        survivorship_ratio = len(first_set & last_set) / max(len(first_set), 1)

        # Significant bias if >10% of original universe was removed
        is_biased = len(removed) / max(len(first_set), 1) > 0.10

        report = {
            "period_start": sorted_keys[0],
            "period_end": sorted_keys[-1],
            "original_count": len(first_set),
            "final_count": len(last_set),
            "removed_count": len(removed),
            "added_count": len(added),
            "removed_symbols": sorted(removed)[:50],
            "added_symbols": sorted(added)[:50],
            "survivorship_ratio": round(survivorship_ratio, 4),
            "is_biased": is_biased,
        }

        if is_biased:
            report["recommendation"] = (
                "Significant survivorship bias detected. "
                "Use point-in-time universe snapshots for backtesting."
            )

        return report

    def validate_panel(self, panel) -> ValidationReport:
        """
        Run all validation checks on a DataPanel.

        Parameters
        ----------
        panel : DataPanel
            The data panel to validate.

        Returns
        -------
        ValidationReport
            Aggregated report with per-symbol results.
        """
        report = ValidationReport()
        symbols = panel.symbols
        report.total_symbols = len(symbols)

        logger.info("Validating %d symbols...", len(symbols))

        for symbol in symbols:
            df = panel.get(symbol)
            result = self._validate_single(symbol, df)
            report.add_result(result)

        logger.info(
            "Validation complete: %d checked, %d errors, %d warnings",
            report.checked_symbols, len(report.errors), len(report.warnings),
        )
        return report

    def validate_symbol(self, symbol: str, df: pd.DataFrame) -> SymbolValidation:
        """
        Validate a single symbol's data.

        Parameters
        ----------
        symbol : str
            Stock code.
        df : pd.DataFrame
            The symbol's data.

        Returns
        -------
        SymbolValidation
        """
        return self._validate_single(symbol, df)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_single(self, symbol: str, df: pd.DataFrame) -> SymbolValidation:
        """Run all checks on a single symbol."""
        result = SymbolValidation(symbol=symbol)

        # Empty check
        if df is None or df.empty:
            result.issues.append("empty_data: no rows")
            return result

        result.row_count = len(df)
        dates = pd.to_datetime(df["date"])
        result.date_range = f"{dates.min().date()} ~ {dates.max().date()}"

        # Minimum row count
        if result.row_count < 100:
            result.issues.append(
                f"low_row_count: only {result.row_count} rows"
            )

        # Duplicate dates
        dup_count = dates.duplicated().sum()
        if dup_count > 0:
            result.duplicates = int(dup_count)
            result.issues.append(f"duplicate_dates: {dup_count} duplicates")

        # OHLC consistency
        ohlc_issues = self._check_ohlc(df)
        result.ohlc_issues = ohlc_issues
        if ohlc_issues:
            result.issues.extend(ohlc_issues)

        # Gaps
        gaps = self.check_gaps(df)
        result.gaps = gaps

        # Outliers
        outliers = self.check_outliers(df)
        result.outliers = outliers

        # Limit flags count
        if len(df) > 1:
            limit_df = self.check_limit_up_down(df)
            result.limit_flags = int(
                limit_df["is_limit_up"].sum() + limit_df["is_limit_down"].sum()
            )

        # Suspension detection
        suspension_issues = self._check_suspension(df)
        if suspension_issues:
            result.issues.extend(suspension_issues)

        return result

    def _check_ohlc(self, df: pd.DataFrame) -> List[str]:
        """Validate OHLC data consistency."""
        issues: List[str] = []

        if not all(c in df.columns for c in ["open", "high", "low", "close"]):
            issues.append("missing_ohlc: not all OHLC columns present")
            return issues

        # high >= low
        bad_hl = df[df["high"] < df["low"]]
        if len(bad_hl) > 0:
            issues.append(f"high<low: {len(bad_hl)} rows")

        # high >= max(open, close)
        bad_h = df[df["high"] < df[["open", "close"]].max(axis=1)]
        if len(bad_h) > 0:
            issues.append(f"high<max(open,close): {len(bad_h)} rows")

        # low <= min(open, close)
        bad_l = df[df["low"] > df[["open", "close"]].min(axis=1)]
        if len(bad_l) > 0:
            issues.append(f"low>min(open,close): {len(bad_l)} rows")

        # Zero or negative prices
        for col in ["open", "high", "low", "close"]:
            bad = df[df[col] <= 0]
            if len(bad) > 0:
                issues.append(f"{col}<=0: {len(bad)} rows")

        return issues

    def _check_suspension(self, df: pd.DataFrame) -> List[str]:
        """Detect abnormally long suspension periods."""
        issues: List[str] = []
        if "volume" not in df.columns or len(df) < 2:
            return issues

        df_sorted = df.sort_values("date").reset_index(drop=True)
        is_zero = (df_sorted["volume"] == 0).astype(int)

        # Find consecutive zero-volume segments
        groups = (is_zero.diff() != 0).cumsum()
        zero_segments = df_sorted[is_zero == 1].groupby(groups[is_zero == 1])

        for _, segment in zero_segments:
            if len(segment) >= self._max_suspension_days:
                start = segment["date"].iloc[0].strftime("%Y-%m-%d")
                end = segment["date"].iloc[-1].strftime("%Y-%m-%d")
                issues.append(
                    f"long_suspension: {start} ~ {end} "
                    f"({len(segment)} consecutive zero-volume days)"
                )

        return issues

    @staticmethod
    def _group_consecutive(dates: List[pd.Timestamp]) -> List[List[pd.Timestamp]]:
        """Group consecutive dates (within 4 calendar days) into segments."""
        if not dates:
            return []

        groups: List[List[pd.Timestamp]] = []
        current_group = [dates[0]]

        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i - 1]).days
            if gap <= 4:  # Allow weekends within a gap segment
                current_group.append(dates[i])
            else:
                groups.append(current_group)
                current_group = [dates[i]]

        groups.append(current_group)
        return groups
