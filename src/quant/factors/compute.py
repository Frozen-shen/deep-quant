"""
Batch factor computation with caching and IC-based filtering.

:class:`FactorComputer` turns a raw OHLCV *panel* (a DataFrame indexed by
``(date, symbol)``, as produced by ``data.store.load_panel``) into a set of
*factor panels* — one ``date x symbol`` DataFrame per factor. It supports
incremental recomputation (only newly arrived dates are computed, while a
lookback warm-up window keeps rolling operators correct) and information-
coefficient (IC) screening to drop factors that carry no predictive signal.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from quant.factors.engine import FactorEngine
from quant.factors.library import FactorDef

__all__ = ["FactorComputer", "DataPanel"]

# A DataPanel is a "long" panel: MultiIndex (date, symbol) with OHLCV columns
# (open, high, low, close, volume, amount, turnover). This is exactly what
# data.store.load_panel() returns, so no conversion is required by callers.
DataPanel = pd.DataFrame


class FactorComputer:
    """
    Batch factor computation with caching.

    - Computes all factors for all symbols in a DataPanel.
    - Returns factor panels: ``Dict[factor_name, DataFrame(date x symbol)]``.
    - Supports incremental computation (only compute new dates).
    - IC-based filtering: compute rank-IC for each factor and drop those whose
      absolute mean IC falls below a threshold.
    """

    def __init__(self, engine: FactorEngine, factor_defs: Sequence[FactorDef]):
        self.engine = engine
        self.factor_defs: List[FactorDef] = list(factor_defs)
        self._factor_names: List[str] = [d.name for d in self.factor_defs]
        self._expr_map: Dict[str, str] = {d.name: d.expression for d in self.factor_defs}
        self._max_lookback: int = max(
            (d.lookback for d in self.factor_defs), default=1
        )

        # factor_name -> DataFrame(date x symbol)
        self._panels: Dict[str, pd.DataFrame] = {}
        # symbol -> last computed date (drives incremental updates)
        self._last_computed: Dict[str, pd.Timestamp] = {}

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------
    @property
    def factor_names(self) -> List[str]:
        """Ordered list of factor names produced by this computer."""
        return list(self._factor_names)

    @property
    def panels(self) -> Dict[str, pd.DataFrame]:
        """The currently cached factor panels (date x symbol each)."""
        return self._panels

    # ------------------------------------------------------------------
    #  Core computation
    # ------------------------------------------------------------------
    def compute_single(self, expression: str, df: pd.DataFrame) -> pd.Series:
        """Compute one arbitrary expression on a per-symbol DataFrame."""
        return self.engine.compute(expression, df)

    def compute_all(
        self,
        panel: DataPanel,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute every factor for every symbol in ``panel``.

        Parameters
        ----------
        panel : DataPanel
            Long panel with MultiIndex ``(date, symbol)`` and OHLCV columns.
        symbols : list of str, optional
            Restrict to these symbols. Defaults to all symbols in ``panel``.

        Returns
        -------
        dict
            ``{factor_name: DataFrame}`` where each DataFrame is indexed by date
            with one column per symbol. Results are also cached internally so
            that subsequent calls are incremental.
        """
        if not isinstance(panel.index, pd.MultiIndex):
            raise ValueError(
                "panel must have a MultiIndex (date, symbol); "
                "see data.store.load_panel()."
            )

        if symbols is None:
            symbols = list(panel.index.get_level_values("symbol").unique())

        # Calendar-day buffer that comfortably covers the largest rolling
        # lookback (in trading days) so rolling values stay correct when we
        # only recompute the tail of each symbol's history.
        warmup_days = self._max_lookback * 2 + 10

        # symbol -> DataFrame(date x factor) holding newly computed rows only
        fresh: Dict[str, pd.DataFrame] = {}

        for sym in symbols:
            try:
                sdf = panel.xs(sym, level="symbol")
            except KeyError:
                continue
            if sdf is None or len(sdf) == 0:
                continue

            sdf = sdf.sort_index()
            last = self._last_computed.get(sym)

            if last is not None:
                cutoff = last - pd.Timedelta(days=warmup_days)
                work = sdf[sdf.index >= cutoff]
            else:
                work = sdf

            if len(work) == 0:
                continue

            values = self.engine.compute_batch(self._expr_map, work)
            values.index = work.index

            if last is not None:
                values = values[values.index > last]

            if len(values) == 0:
                continue

            fresh[sym] = values
            self._last_computed[sym] = values.index.max()

        self._merge_fresh(fresh)
        return self._panels

    def _merge_fresh(self, fresh: Dict[str, pd.DataFrame]) -> None:
        """Merge newly computed per-symbol rows into the cached factor panels."""
        if not fresh:
            return

        for name in self._factor_names:
            pieces = []
            for sym, df in fresh.items():
                if name not in df.columns:
                    continue
                col = df[[name]].copy()
                col.columns = [sym]
                pieces.append(col)

            if not pieces:
                continue

            new_block = pd.concat(pieces, axis=1).sort_index()
            existing = self._panels.get(name)
            if existing is None:
                self._panels[name] = new_block
            else:
                # Overwrite overlapping dates, then append newer ones.
                combined = pd.concat([existing, new_block])
                combined = combined[~combined.index.duplicated(keep="last")]
                self._panels[name] = combined.sort_index()

    # ------------------------------------------------------------------
    #  IC-based filtering
    # ------------------------------------------------------------------
    def compute_ic(
        self,
        factor_panels: Dict[str, pd.DataFrame],
        returns_panel: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Mean cross-sectional rank-IC (Spearman) for each factor.

        Parameters
        ----------
        factor_panels : dict
            ``{factor_name: DataFrame(date x symbol)}`` (from :meth:`compute_all`).
        returns_panel : DataFrame
            ``date x symbol`` matrix of *forward* returns aligned to the factor
            dates (typically the next-period return).

        Returns
        -------
        dict
            ``{factor_name: mean_ic}``. Factors with too few overlapping
            cross-sections get ``NaN``.
        """
        mean_ic: Dict[str, float] = {}
        for name, fp in factor_panels.items():
            common_dates = fp.index.intersection(returns_panel.index)
            common_syms = fp.columns.intersection(returns_panel.columns)
            if len(common_dates) == 0 or len(common_syms) < 5:
                mean_ic[name] = np.nan
                continue

            f = fp.loc[common_dates, common_syms]
            r = returns_panel.loc[common_dates, common_syms]

            ics = []
            for date in common_dates:
                fv = f.loc[date]
                rv = r.loc[date]
                mask = fv.notna() & rv.notna()
                if mask.sum() < 5:
                    continue
                # Spearman = Pearson on ranks. Rank first, then skip degenerate
                # cross-sections where a ranked array is constant (all ties);
                # scipy otherwise emits a ConstantInputWarning.
                fr = fv[mask].rank()
                rr = rv[mask].rank()
                if fr.std() == 0 or rr.std() == 0:
                    continue
                ic = fr.corr(rr)
                if ic is not None and not np.isnan(ic):
                    ics.append(ic)

            mean_ic[name] = float(np.mean(ics)) if ics else np.nan
        return mean_ic

    def filter_by_ic(
        self,
        factor_panels: Dict[str, pd.DataFrame],
        returns_panel: pd.DataFrame,
        threshold: float = 0.02,
    ) -> List[str]:
        """
        Return the names of factors whose ``|mean IC| >= threshold``.

        Factors with ``NaN`` IC (insufficient data) are dropped.
        """
        ic = self.compute_ic(factor_panels, returns_panel)
        kept = [
            name for name, value in ic.items()
            if value is not None and not np.isnan(value) and abs(value) >= threshold
        ]
        # Preserve library ordering for determinism.
        return [name for name in self._factor_names if name in kept]

    # ------------------------------------------------------------------
    #  Matrix access
    # ------------------------------------------------------------------
    def get_factor_matrix(
        self,
        date,
        symbols: List[str],
    ) -> np.ndarray:
        """
        Build a ``(n_symbols, n_factors)`` matrix for a single date.

        Rows follow ``symbols`` order; columns follow :attr:`factor_names`
        order. Missing values are ``NaN``. Call :meth:`compute_all` first to
        populate the cache.
        """
        ts = pd.Timestamp(date)
        n_sym, n_fac = len(symbols), len(self._factor_names)
        matrix = np.full((n_sym, n_fac), np.nan, dtype=float)

        for j, name in enumerate(self._factor_names):
            panel = self._panels.get(name)
            if panel is None or ts not in panel.index:
                continue
            row = panel.loc[ts]
            for i, sym in enumerate(symbols):
                if sym in row.index:
                    val = row[sym]
                    if pd.notna(val):
                        matrix[i, j] = float(val)
        return matrix
