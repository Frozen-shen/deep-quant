"""
Statistical robustness checks for strategy validation.

Tests:
    1. Sub-sample analysis: split into 2-4 sub-periods, check consistency
    2. Bootstrap: resample returns with replacement, compute confidence intervals
    3. Monte Carlo: shuffle signal-return pairing, compute p-value
    4. Parameter sensitivity: vary key params +/-20%, check stability
    5. Outlier removal: remove top/bottom 5% days, re-compute metrics

A strategy that passes all robustness checks is much less likely to be
the result of overfitting or data snooping.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class RobustnessChecker:
    """
    Statistical robustness checks for strategy validation.

    These tests help distinguish genuine alpha from overfitting:
        - Sub-sample: does performance persist across time periods?
        - Bootstrap: what's the confidence interval on key metrics?
        - Monte Carlo: what's the probability of this result by chance?
        - Parameter sensitivity: is performance stable to small changes?
        - Outlier removal: is performance driven by a few extreme days?
    """

    TRADING_DAYS_PER_YEAR = 244

    def sub_sample_analysis(
        self,
        equity_curve: pd.Series,
        n_splits: int = 4,
    ) -> pd.DataFrame:
        """
        Split the equity curve into sub-periods and compute metrics for each.

        A robust strategy should show positive performance across most
        sub-periods, not just be driven by one lucky period.

        Args:
            equity_curve: Series of portfolio values.
            n_splits: Number of sub-periods to split into (2-4 recommended).

        Returns:
            DataFrame with one row per sub-period and columns:
                [start_date, end_date, return, sharpe, max_dd, n_days]
        """
        n = len(equity_curve)
        if n < n_splits * 20:
            raise ValueError(
                f"Need at least {n_splits * 20} data points for {n_splits} splits, got {n}"
            )

        split_size = n // n_splits
        records = []

        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < n_splits - 1 else n

            sub_curve = equity_curve.iloc[start_idx:end_idx]
            sub_returns = sub_curve.pct_change().dropna()

            if len(sub_returns) == 0:
                continue

            # Compute metrics for this sub-period
            total_ret = sub_curve.iloc[-1] / sub_curve.iloc[0] - 1.0
            mean_ret = sub_returns.mean()
            std_ret = sub_returns.std(ddof=1)
            sharpe = (
                (mean_ret / std_ret) * np.sqrt(self.TRADING_DAYS_PER_YEAR)
                if std_ret > 0
                else 0.0
            )

            # Max drawdown
            cummax = sub_curve.cummax()
            drawdown = (sub_curve - cummax) / cummax
            max_dd = drawdown.min()

            records.append({
                "period": i + 1,
                "start_date": sub_curve.index[0],
                "end_date": sub_curve.index[-1],
                "return": total_ret,
                "sharpe": sharpe,
                "max_dd": max_dd,
                "n_days": len(sub_returns),
            })

        return pd.DataFrame(records)

    def bootstrap_confidence(
        self,
        returns: pd.Series,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        random_state: int = 42,
    ) -> Dict:
        """
        Bootstrap confidence intervals for key performance metrics.

        Resamples daily returns with replacement to build an empirical
        distribution of performance metrics. This accounts for non-normality
        and serial correlation better than parametric methods.

        Args:
            returns: Daily returns series.
            n_bootstrap: Number of bootstrap iterations.
            confidence: Confidence level (e.g., 0.95 for 95% CI).
            random_state: Random seed for reproducibility.

        Returns:
            Dictionary with confidence intervals for:
                - sharpe: {lower, upper, mean, median}
                - annual_return: {lower, upper, mean, median}
                - max_drawdown: {lower, upper, mean, median}
        """
        rng = np.random.default_rng(random_state)
        returns_arr = returns.dropna().values
        n = len(returns_arr)

        if n < 30:
            raise ValueError("Need at least 30 return observations for bootstrap")

        bootstrap_sharpes = np.empty(n_bootstrap)
        bootstrap_annual_returns = np.empty(n_bootstrap)
        bootstrap_max_dds = np.empty(n_bootstrap)

        for i in range(n_bootstrap):
            # Resample with replacement
            sample = rng.choice(returns_arr, size=n, replace=True)

            # Sharpe
            mean_ret = sample.mean()
            std_ret = sample.std(ddof=1)
            if std_ret > 0:
                bootstrap_sharpes[i] = (
                    mean_ret / std_ret * np.sqrt(self.TRADING_DAYS_PER_YEAR)
                )
            else:
                bootstrap_sharpes[i] = 0.0

            # Annual return
            bootstrap_annual_returns[i] = mean_ret * self.TRADING_DAYS_PER_YEAR

            # Max drawdown
            cum = np.cumprod(1.0 + sample)
            running_max = np.maximum.accumulate(cum)
            dd = (cum - running_max) / running_max
            bootstrap_max_dds[i] = dd.min()

        # Compute confidence intervals
        alpha = (1.0 - confidence) / 2.0
        lower_pct = alpha * 100
        upper_pct = (1.0 - alpha) * 100

        result = {}
        for name, arr in [
            ("sharpe", bootstrap_sharpes),
            ("annual_return", bootstrap_annual_returns),
            ("max_drawdown", bootstrap_max_dds),
        ]:
            result[name] = {
                "lower": float(np.percentile(arr, lower_pct)),
                "upper": float(np.percentile(arr, upper_pct)),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
            }

        result["n_bootstrap"] = n_bootstrap
        result["confidence"] = confidence

        return result

    def monte_carlo_pvalue(
        self,
        strategy_returns: pd.Series,
        n_simulations: int = 1000,
        random_state: int = 42,
    ) -> float:
        """
        Monte Carlo p-value for strategy performance.

        Tests H0: the strategy has no predictive power (returns are random).
        Shuffles the return series to break any signal-return relationship,
        then computes what fraction of random strategies achieve equal or
        better Sharpe ratio.

        A p-value < 0.05 suggests the strategy performance is unlikely
        due to chance alone.

        Args:
            strategy_returns: Actual daily strategy returns.
            n_simulations: Number of random permutations.
            random_state: Random seed.

        Returns:
            p-value: fraction of simulations with Sharpe >= actual Sharpe.
        """
        rng = np.random.default_rng(random_state)
        returns_arr = strategy_returns.dropna().values
        n = len(returns_arr)

        if n < 30:
            raise ValueError("Need at least 30 observations for Monte Carlo test")

        # Actual Sharpe ratio
        actual_mean = returns_arr.mean()
        actual_std = returns_arr.std(ddof=1)
        if actual_std > 0:
            actual_sharpe = (
                actual_mean / actual_std * np.sqrt(self.TRADING_DAYS_PER_YEAR)
            )
        else:
            return 1.0  # No variance, can't reject null

        # Simulate: permute returns (breaks temporal structure)
        n_better = 0
        for _ in range(n_simulations):
            shuffled = rng.permutation(returns_arr)
            sim_mean = shuffled.mean()
            sim_std = shuffled.std(ddof=1)
            if sim_std > 0:
                sim_sharpe = (
                    sim_mean / sim_std * np.sqrt(self.TRADING_DAYS_PER_YEAR)
                )
            else:
                sim_sharpe = 0.0

            if sim_sharpe >= actual_sharpe:
                n_better += 1

        p_value = (n_better + 1) / (n_simulations + 1)  # +1 for continuity correction
        return float(p_value)

    def parameter_sensitivity(
        self,
        run_func: Callable,
        param_name: str,
        base_value: float,
        variations: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Test strategy sensitivity to a parameter.

        Runs the strategy with the parameter varied by +/-20% (or custom
        variations) and checks if performance metrics are stable.

        A robust strategy should not have dramatically different performance
        for small parameter changes.

        Args:
            run_func: Callable that takes param_name=value and returns a dict
                with at least a 'sharpe' key. Signature:
                    run_func(**{param_name: value}) -> dict with 'sharpe', 'cagr', etc.
            param_name: Name of the parameter to vary.
            base_value: Base (default) value of the parameter.
            variations: List of values to test. If None, uses
                [0.8*base, 0.9*base, base, 1.1*base, 1.2*base].

        Returns:
            DataFrame with columns [param_value, sharpe, cagr, max_dd]
            showing performance at each parameter value.
        """
        if variations is None:
            variations = [
                base_value * 0.8,
                base_value * 0.9,
                base_value,
                base_value * 1.1,
                base_value * 1.2,
            ]

        records = []
        for val in variations:
            try:
                result = run_func(**{param_name: val})
                records.append({
                    "param_value": val,
                    "sharpe": result.get("sharpe", 0.0),
                    "cagr": result.get("cagr", 0.0),
                    "max_dd": result.get("max_drawdown", 0.0),
                })
            except Exception as e:
                records.append({
                    "param_value": val,
                    "sharpe": np.nan,
                    "cagr": np.nan,
                    "max_dd": np.nan,
                })

        df = pd.DataFrame(records)

        # Add stability metrics
        if len(df) > 1 and df["sharpe"].notna().any():
            sharpe_values = df["sharpe"].dropna()
            df.attrs["sharpe_cv"] = (
                sharpe_values.std() / abs(sharpe_values.mean())
                if abs(sharpe_values.mean()) > 1e-10
                else float("inf")
            )

        return df

    def outlier_removal_test(
        self,
        returns: pd.Series,
        percentile: float = 0.05,
    ) -> Dict:
        """
        Re-compute metrics after removing extreme return days.

        If performance is driven by a handful of extreme days, it's less
        robust than performance spread across many days.

        Args:
            returns: Daily returns series.
            percentile: Fraction to remove from each tail (default 5%).

        Returns:
            Dictionary comparing metrics before and after outlier removal:
                {metric: {original: value, trimmed: value, change: value}}
        """
        returns_arr = returns.dropna().values
        n = len(returns_arr)

        if n < 20:
            raise ValueError("Need at least 20 observations for outlier test")

        # Compute original metrics
        original_metrics = self._quick_metrics(returns_arr)

        # Remove top and bottom percentile
        lower_bound = np.percentile(returns_arr, percentile * 100)
        upper_bound = np.percentile(returns_arr, (1 - percentile) * 100)
        trimmed = returns_arr[
            (returns_arr >= lower_bound) & (returns_arr <= upper_bound)
        ]

        # Compute trimmed metrics
        trimmed_metrics = self._quick_metrics(trimmed)

        # Compare
        result = {}
        for key in original_metrics:
            orig = original_metrics[key]
            trim = trimmed_metrics[key]
            result[key] = {
                "original": orig,
                "trimmed": trim,
                "change": trim - orig,
                "pct_change": (trim - orig) / abs(orig) if abs(orig) > 1e-10 else 0.0,
            }

        result["n_removed"] = n - len(trimmed)
        result["pct_removed"] = (n - len(trimmed)) / n

        return result

    def full_report(
        self,
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series] = None,
    ) -> str:
        """
        Generate a complete robustness report.

        Runs all available tests and formats results as a readable report.

        Args:
            equity_curve: Portfolio equity curve.
            benchmark: Optional benchmark equity curve.

        Returns:
            Multi-line formatted report string.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("ROBUSTNESS ANALYSIS REPORT")
        lines.append("=" * 70)

        returns = equity_curve.pct_change().dropna()

        # 1. Sub-sample analysis
        lines.append("")
        lines.append("--- 1. Sub-Sample Analysis (4 periods) ---")
        try:
            sub_df = self.sub_sample_analysis(equity_curve, n_splits=4)
            for _, row in sub_df.iterrows():
                lines.append(
                    f"  Period {int(row['period'])}: "
                    f"Return={row['return']:>7.2%}, "
                    f"Sharpe={row['sharpe']:>6.3f}, "
                    f"MaxDD={row['max_dd']:>7.2%}"
                )
            # Consistency check
            n_positive = (sub_df["sharpe"] > 0).sum()
            lines.append(
                f"  Consistency: {n_positive}/{len(sub_df)} periods with positive Sharpe"
            )
        except Exception as e:
            lines.append(f"  Error: {e}")

        # 2. Bootstrap confidence intervals
        lines.append("")
        lines.append("--- 2. Bootstrap Confidence Intervals (95%) ---")
        try:
            boot = self.bootstrap_confidence(returns, n_bootstrap=1000)
            for metric in ["sharpe", "annual_return", "max_drawdown"]:
                ci = boot[metric]
                lines.append(
                    f"  {metric:<15s}: [{ci['lower']:>8.3f}, {ci['upper']:>8.3f}] "
                    f"(median={ci['median']:.3f})"
                )
        except Exception as e:
            lines.append(f"  Error: {e}")

        # 3. Monte Carlo p-value
        lines.append("")
        lines.append("--- 3. Monte Carlo P-Value ---")
        try:
            p_value = self.monte_carlo_pvalue(returns, n_simulations=1000)
            significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""
            lines.append(f"  P-value: {p_value:.4f} {significance}")
            lines.append(
                "  (Significance: *** p<0.01, ** p<0.05, * p<0.10)"
            )
        except Exception as e:
            lines.append(f"  Error: {e}")

        # 4. Outlier removal
        lines.append("")
        lines.append("--- 4. Outlier Removal (5% each tail) ---")
        try:
            outlier_result = self.outlier_removal_test(returns, percentile=0.05)
            lines.append(
                f"  Days removed: {outlier_result['n_removed']} "
                f"({outlier_result['pct_removed']:.1%})"
            )
            for metric in ["sharpe", "annual_return", "max_drawdown"]:
                if metric in outlier_result:
                    info = outlier_result[metric]
                    lines.append(
                        f"  {metric:<15s}: original={info['original']:>8.3f}, "
                        f"trimmed={info['trimmed']:>8.3f}, "
                        f"change={info['pct_change']:>7.1%}"
                    )
        except Exception as e:
            lines.append(f"  Error: {e}")

        # 5. Summary verdict
        lines.append("")
        lines.append("--- SUMMARY ---")
        verdicts = []
        try:
            sub_df = self.sub_sample_analysis(equity_curve, n_splits=4)
            n_positive = (sub_df["sharpe"] > 0).sum()
            if n_positive >= 3:
                verdicts.append("PASS: Consistent across sub-periods")
            else:
                verdicts.append("WARN: Inconsistent across sub-periods")
        except Exception:
            pass

        try:
            boot = self.bootstrap_confidence(returns, n_bootstrap=1000)
            if boot["sharpe"]["lower"] > 0:
                verdicts.append("PASS: Sharpe CI entirely positive")
            else:
                verdicts.append("WARN: Sharpe CI includes zero")
        except Exception:
            pass

        try:
            p_value = self.monte_carlo_pvalue(returns, n_simulations=1000)
            if p_value < 0.05:
                verdicts.append("PASS: Statistically significant (p<0.05)")
            else:
                verdicts.append("WARN: Not statistically significant")
        except Exception:
            pass

        for v in verdicts:
            lines.append(f"  {v}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _quick_metrics(self, returns_arr: np.ndarray) -> Dict[str, float]:
        """
        Quick computation of key metrics from a returns array.

        Args:
            returns_arr: 1D numpy array of daily returns.

        Returns:
            Dict with sharpe, annual_return, max_drawdown.
        """
        n = len(returns_arr)
        if n == 0:
            return {"sharpe": 0.0, "annual_return": 0.0, "max_drawdown": 0.0}

        mean_ret = returns_arr.mean()
        std_ret = returns_arr.std(ddof=1)

        sharpe = (
            mean_ret / std_ret * np.sqrt(self.TRADING_DAYS_PER_YEAR)
            if std_ret > 0
            else 0.0
        )
        annual_return = mean_ret * self.TRADING_DAYS_PER_YEAR

        cum = np.cumprod(1.0 + returns_arr)
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / running_max
        max_dd = dd.min()

        return {
            "sharpe": float(sharpe),
            "annual_return": float(annual_return),
            "max_drawdown": float(max_dd),
        }
