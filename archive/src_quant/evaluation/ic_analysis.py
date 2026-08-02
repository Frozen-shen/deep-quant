"""
Information Coefficient (IC) analysis toolkit.

Computes:
    - Rank IC (Spearman) per factor per period
    - IC mean, std, IR (= mean/std)
    - IC decay curve (how fast does predictive power fade?)
    - IC by market cap group
    - IC by sector
    - Factor ranking by predictive power

The IC is the single most important metric for factor research.
A factor with IC_IR > 0.5 is considered strong for A-shares.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


class ICAnalyzer:
    """
    Information Coefficient analysis toolkit.

    The IC measures the cross-sectional correlation between factor values
    and subsequent returns. Rank IC (Spearman) is preferred over Pearson
    because it's robust to outliers and non-linear relationships.

    Key thresholds for A-share factors:
        - |IC mean| > 0.03: potentially useful
        - |IC IR| > 0.5: strong factor
        - |IC IR| > 1.0: exceptional (rare, likely overfit)
    """

    def compute_ic_series(
        self,
        factor_panel: pd.DataFrame,
        returns_panel: pd.DataFrame,
        forward_period: int = 5,
    ) -> pd.Series:
        """
        Compute time series of cross-sectional Rank IC.

        For each date, computes the Spearman rank correlation between
        factor values and forward returns over the specified period.

        Args:
            factor_panel: DataFrame (date x symbol) of factor values.
            returns_panel: DataFrame (date x symbol) of daily returns.
            forward_period: Number of days to look forward for returns.
                Default 5 = weekly forward return.

        Returns:
            Series indexed by date with Rank IC values.
        """
        # Compute forward returns
        forward_returns = self._compute_forward_returns(
            returns_panel, forward_period
        )

        # Align factor and forward returns
        common_dates = factor_panel.index.intersection(forward_returns.index)
        common_symbols = factor_panel.columns.intersection(forward_returns.columns)

        factor_aligned = factor_panel.loc[common_dates, common_symbols]
        returns_aligned = forward_returns.loc[common_dates, common_symbols]

        # Compute cross-sectional rank IC for each date
        ic_values = {}
        for dt in common_dates:
            f = factor_aligned.loc[dt]
            r = returns_aligned.loc[dt]

            # Remove NaN pairs
            valid_mask = f.notna() & r.notna()
            if valid_mask.sum() < 10:  # Need minimum cross-section
                continue

            f_valid = f[valid_mask]
            r_valid = r[valid_mask]

            # Skip if either array is constant (correlation undefined)
            if f_valid.std() < 1e-12 or r_valid.std() < 1e-12:
                continue

            # Spearman rank correlation
            corr, _ = stats.spearmanr(f_valid.values, r_valid.values)
            if not np.isnan(corr):
                ic_values[dt] = corr

        return pd.Series(ic_values, name="rank_ic")

    def compute_ic_summary(self, ic_series: pd.Series) -> Dict:
        """
        Compute summary statistics for an IC time series.

        Args:
            ic_series: Time series of IC values from compute_ic_series().

        Returns:
            Dictionary with:
                - ic_mean: Average IC
                - ic_std: Standard deviation of IC
                - ic_ir: Information Ratio of IC (mean/std)
                - ic_positive_ratio: Fraction of periods with positive IC
                - ic_abs_mean: Mean of |IC| (measures consistency regardless of direction)
                - t_stat: t-statistic for IC mean != 0
                - n_periods: Number of observations
        """
        if len(ic_series) == 0:
            return {
                "ic_mean": 0.0,
                "ic_std": 0.0,
                "ic_ir": 0.0,
                "ic_positive_ratio": 0.0,
                "ic_abs_mean": 0.0,
                "t_stat": 0.0,
                "n_periods": 0,
            }

        ic_mean = ic_series.mean()
        ic_std = ic_series.std(ddof=1)
        n = len(ic_series)

        # IC IR (analogous to Sharpe ratio for IC)
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0

        # t-statistic
        t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else 0.0

        # Positive ratio
        positive_ratio = (ic_series > 0).mean()

        return {
            "ic_mean": float(ic_mean),
            "ic_std": float(ic_std),
            "ic_ir": float(ic_ir),
            "ic_positive_ratio": float(positive_ratio),
            "ic_abs_mean": float(ic_series.abs().mean()),
            "t_stat": float(t_stat),
            "n_periods": int(n),
        }

    def compute_ic_decay(
        self,
        factor_panel: pd.DataFrame,
        returns_panel: pd.DataFrame,
        max_horizon: int = 20,
    ) -> pd.Series:
        """
        Compute IC decay curve: how predictive power fades over time.

        For each horizon h in [1, max_horizon], compute the average IC
        using h-day forward returns. A fast decay suggests the factor
        captures short-term alpha that requires frequent rebalancing.

        Args:
            factor_panel: DataFrame (date x symbol) of factor values.
            returns_panel: DataFrame (date x symbol) of daily returns.
            max_horizon: Maximum forward period to test.

        Returns:
            Series indexed by horizon (1 to max_horizon) with mean IC values.
        """
        decay = {}
        for h in range(1, max_horizon + 1):
            ic_series = self.compute_ic_series(factor_panel, returns_panel, h)
            if len(ic_series) > 0:
                decay[h] = ic_series.mean()
            else:
                decay[h] = 0.0

        return pd.Series(decay, name="ic_decay")

    def compute_ic_by_group(
        self,
        factor_panel: pd.DataFrame,
        returns_panel: pd.DataFrame,
        group_panel: pd.DataFrame,
        forward_period: int = 5,
    ) -> Dict[str, pd.Series]:
        """
        Compute IC separately for each group (e.g., sector, market cap).

        Args:
            factor_panel: DataFrame (date x symbol) of factor values.
            returns_panel: DataFrame (date x symbol) of daily returns.
            group_panel: DataFrame (date x symbol) of group labels
                (e.g., sector names or cap group labels).
            forward_period: Forward return period.

        Returns:
            {group_name: IC_series} dictionary.
        """
        forward_returns = self._compute_forward_returns(
            returns_panel, forward_period
        )

        common_dates = factor_panel.index.intersection(forward_returns.index)
        common_symbols = factor_panel.columns.intersection(forward_returns.columns)

        factor_aligned = factor_panel.loc[common_dates, common_symbols]
        returns_aligned = forward_returns.loc[common_dates, common_symbols]
        group_aligned = group_panel.loc[common_dates, common_symbols]

        # Find all unique groups
        all_groups = set()
        for dt in common_dates:
            groups = group_aligned.loc[dt].dropna().unique()
            all_groups.update(groups)

        # Compute IC per group per date
        group_ic: Dict[str, Dict] = {g: {} for g in all_groups}

        for dt in common_dates:
            f = factor_aligned.loc[dt]
            r = returns_aligned.loc[dt]
            g = group_aligned.loc[dt]

            valid_mask = f.notna() & r.notna() & g.notna()
            if valid_mask.sum() < 10:
                continue

            f_valid = f[valid_mask]
            r_valid = r[valid_mask]
            g_valid = g[valid_mask]

            for group_name in all_groups:
                group_mask = g_valid == group_name
                if group_mask.sum() < 5:  # Minimum group size
                    continue

                f_group = f_valid[group_mask]
                r_group = r_valid[group_mask]

                # Skip if either array is constant
                if f_group.std() < 1e-12 or r_group.std() < 1e-12:
                    continue

                corr, _ = stats.spearmanr(f_group.values, r_group.values)
                if not np.isnan(corr):
                    group_ic[group_name][dt] = corr

        # Convert to Series
        result = {}
        for group_name, ic_dict in group_ic.items():
            if ic_dict:
                result[group_name] = pd.Series(ic_dict, name=f"ic_{group_name}")

        return result

    def rank_factors(
        self,
        factor_panels: Dict[str, pd.DataFrame],
        returns_panel: pd.DataFrame,
        forward_period: int = 5,
    ) -> pd.DataFrame:
        """
        Rank multiple factors by predictive power.

        Computes IC summary for each factor and returns a ranked table.

        Args:
            factor_panels: {factor_name: DataFrame (date x symbol)} of factor values.
            returns_panel: DataFrame (date x symbol) of daily returns.
            forward_period: Forward return period for IC computation.

        Returns:
            DataFrame with factors as rows and IC statistics as columns,
            sorted by |IC IR| descending.
        """
        records = []
        for factor_name, factor_panel in factor_panels.items():
            ic_series = self.compute_ic_series(
                factor_panel, returns_panel, forward_period
            )
            summary = self.compute_ic_summary(ic_series)
            summary["factor"] = factor_name
            records.append(summary)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.set_index("factor")

        # Sort by absolute IC IR descending
        df["abs_ic_ir"] = df["ic_ir"].abs()
        df = df.sort_values("abs_ic_ir", ascending=False)
        df = df.drop(columns=["abs_ic_ir"])

        return df

    def _compute_forward_returns(
        self,
        returns_panel: pd.DataFrame,
        period: int,
    ) -> pd.DataFrame:
        """
        Compute forward cumulative returns over a given period.

        forward_return[t] = product(1 + r[t+1], ..., 1 + r[t+period]) - 1

        Args:
            returns_panel: DataFrame (date x symbol) of daily returns.
            period: Number of days forward.

        Returns:
            DataFrame of forward returns (shifted so index aligns with
            the signal date).
        """
        # Cumulative product of (1 + r) over the forward window
        # Use rolling product then shift back
        cum_returns = (1 + returns_panel).rolling(
            window=period, min_periods=period
        ).apply(np.prod, raw=True) - 1.0

        # Shift back so that index date t has the return from t to t+period
        forward = cum_returns.shift(-period)

        return forward
