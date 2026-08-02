"""
Factor expression DSL engine.

Parses Qlib-style factor expressions such as ``"Mean($close, 20) / $close - 1"``
into an evaluatable AST and computes them on OHLCV DataFrames.

Adapted from the legacy ``factor_engine.py`` (which this replaces). The core
recursive-descent parser and rolling-operator semantics are preserved; the
public surface is now the :class:`FactorEngine` class, with added support for
the power operator (``**``) and the ``Delta`` / ``Sign`` / ``Clip`` functions.

Example
-------
>>> import pandas as pd
>>> engine = FactorEngine()
>>> result = engine.compute("Mean($close, 20) / $close - 1", df)

All computations propagate ``NaN`` gracefully (they never raise on missing or
degenerate data); division by zero yields ``NaN`` rather than ``inf``.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

__all__ = ["FactorEngine", "parse_factor", "Factor"]


# ============================================================================
#  AST nodes
# ============================================================================

class Factor(ABC):
    """Abstract base for every node in a factor expression tree."""

    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """Evaluate this node against ``df`` and return an aligned Series."""

    @abstractmethod
    def __repr__(self) -> str:  # pragma: no cover - trivial
        ...


class FieldFactor(Factor):
    """Reference to an OHLCV column, written ``$close``, ``$volume``, etc."""

    VALID_FIELDS = {
        "open", "high", "low", "close",
        "volume", "amount", "turnover",
        "vwap", "outstanding_share",
    }
    _ALIASES = {"vwap": "amount"}

    def __init__(self, field: str):
        field = field.strip().lower()
        if field.startswith("$"):
            field = field[1:]
        if field not in self.VALID_FIELDS:
            raise ValueError(
                f"Unknown field: {field}. Valid fields: {sorted(self.VALID_FIELDS)}"
            )
        self.field = field

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        col = self._ALIASES.get(self.field, self.field)
        if col not in df.columns:
            # Missing column -> all-NaN series rather than a crash.
            return pd.Series(np.nan, index=df.index, dtype=float)
        return df[col].astype(float)

    def __repr__(self) -> str:
        return f"${self.field}"


class ConstFactor(Factor):
    """A numeric literal."""

    def __init__(self, value: float):
        self.value = float(value)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=df.index, dtype=float)

    def __repr__(self) -> str:
        return str(self.value)


class RollingFactor(Factor):
    """
    Rolling-window operator applied to a child expression.

    Covers: Ref, Mean, Std, Max, Min, Sum, Median, Skew, Kurt, EMA, Rank, Delta.
    ``Delta(x, n)`` is defined as ``x - Ref(x, n)``.
    """

    _ROLLING_OPS = {
        "Ref", "Mean", "Std", "Max", "Min", "Sum",
        "Median", "Skew", "Kurt", "EMA", "Rank", "Delta",
    }

    def __init__(self, child: Factor, window: int, op_name: str):
        if op_name not in self._ROLLING_OPS:
            raise ValueError(f"Unknown rolling operator: {op_name}")
        self.child = child
        self.window = int(window)
        self.op_name = op_name

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        series = self.child.evaluate(df)
        if self.op_name == "Delta":
            return series - series.shift(self.window)
        return self._rolling(series)

    def _rolling(self, series: pd.Series) -> pd.Series:
        n = self.window
        op = self.op_name
        if op == "Ref":
            return series.shift(n)
        if op == "Mean":
            return series.rolling(n, min_periods=max(1, n // 2)).mean()
        if op == "Std":
            return series.rolling(n, min_periods=max(2, n // 2)).std()
        if op == "Max":
            return series.rolling(n, min_periods=1).max()
        if op == "Min":
            return series.rolling(n, min_periods=1).min()
        if op == "Sum":
            return series.rolling(n, min_periods=1).sum()
        if op == "Median":
            return series.rolling(n, min_periods=1).median()
        if op == "Skew":
            return series.rolling(n, min_periods=3).skew()
        if op == "Kurt":
            return series.rolling(n, min_periods=4).kurt()
        if op == "EMA":
            return series.ewm(span=n, adjust=False).mean()
        if op == "Rank":
            return series.rolling(n, min_periods=1).rank(pct=True)
        raise ValueError(f"Unknown rolling operator: {op}")  # pragma: no cover

    def __repr__(self) -> str:
        return f"{self.op_name}({self.child}, {self.window})"


class ArithFactor(Factor):
    """Binary arithmetic: +, -, *, /, **."""

    def __init__(self, left: Factor, right: Factor, op: str):
        self.left = left
        self.right = right
        self.op = op

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        lv = self.left.evaluate(df)
        rv = self.right.evaluate(df)
        if self.op == "+":
            return lv + rv
        if self.op == "-":
            return lv - rv
        if self.op == "*":
            return lv * rv
        if self.op == "/":
            # Safe division: divide-by-zero -> NaN (never inf).
            return lv / rv.replace(0, np.nan)
        if self.op == "**":
            return lv ** rv
        raise ValueError(f"Unknown operator: {self.op}")

    def __repr__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


class CmpFactor(Factor):
    """Comparison operators producing 0/1 floats: >, <, >=, <=, ==, !=."""

    _OPS: Dict[str, Callable[[pd.Series, pd.Series], pd.Series]] = {
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }

    def __init__(self, left: Factor, right: Factor, op: str):
        if op not in self._OPS:
            raise ValueError(f"Unknown comparison: {op}")
        self.left = left
        self.right = right
        self.op = op

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        lv = self.left.evaluate(df)
        rv = self.right.evaluate(df)
        return self._OPS[self.op](lv, rv).astype(float)

    def __repr__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


class UnaryFunc(Factor):
    """Element-wise unary function: Abs, Log, Sign."""

    def __init__(self, child: Factor, func: Callable[[pd.Series], pd.Series], name: str):
        self.child = child
        self.func = func
        self.name = name

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        return self.func(self.child.evaluate(df))

    def __repr__(self) -> str:
        return f"{self.name}({self.child})"


class ClipFactor(Factor):
    """Clip(child, lo, hi) — bound values into [lo, hi]."""

    def __init__(self, child: Factor, lo: float, hi: float):
        self.child = child
        self.lo = float(lo)
        self.hi = float(hi)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        return self.child.evaluate(df).clip(lower=self.lo, upper=self.hi)

    def __repr__(self) -> str:
        return f"Clip({self.child}, {self.lo}, {self.hi})"


class CorrFactor(Factor):
    """Rolling Pearson correlation between two child expressions."""

    def __init__(self, child_a: Factor, child_b: Factor, window: int):
        self.child_a = child_a
        self.child_b = child_b
        self.window = int(window)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        a = self.child_a.evaluate(df)
        b = self.child_b.evaluate(df)
        return a.rolling(self.window, min_periods=max(5, self.window // 2)).corr(b)

    def __repr__(self) -> str:
        return f"Corr({self.child_a}, {self.child_b}, {self.window})"


class IfFactor(Factor):
    """If(cond, true_val, false_val) — selects by ``cond > 0``."""

    def __init__(self, cond: Factor, true_val: Factor, false_val: Factor):
        self.cond = cond
        self.true_val = true_val
        self.false_val = false_val

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        c = self.cond.evaluate(df)
        tv = self.true_val.evaluate(df)
        fv = self.false_val.evaluate(df)
        return pd.Series(np.where(c > 0, tv, fv), index=df.index, dtype=float)

    def __repr__(self) -> str:
        return f"If({self.cond}, {self.true_val}, {self.false_val})"


class CrossFactor(Factor):
    """Cross(a, b): +1 when ``a`` crosses above ``b``, -1 on cross below, else 0."""

    def __init__(self, a: Factor, b: Factor):
        self.a = a
        self.b = b

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        av = self.a.evaluate(df)
        bv = self.b.evaluate(df)
        av_1, bv_1 = av.shift(1), bv.shift(1)
        golden = (av_1 <= bv_1) & (av > bv)
        death = (av_1 >= bv_1) & (av < bv)
        result = pd.Series(0.0, index=df.index)
        result[golden] = 1.0
        result[death] = -1.0
        return result

    def __repr__(self) -> str:
        return f"Cross({self.a}, {self.b})"


class RSqrFactor(Factor):
    """Rolling coefficient of determination (R^2) of a linear trend fit."""

    def __init__(self, child: Factor, window: int):
        self.child = child
        self.window = int(window)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        s = self.child.evaluate(df).astype(float)
        n = self.window
        result = pd.Series(np.nan, index=s.index, dtype=float)
        values = s.to_numpy()
        x = np.arange(n, dtype=float)
        mx = x.mean()
        ss_xx = np.sum((x - mx) ** 2)
        for i in range(n - 1, len(values)):
            y = values[i - n + 1: i + 1]
            if np.any(np.isnan(y)):
                continue
            if np.std(y) == 0:
                result.iloc[i] = 0.0
                continue
            my = y.mean()
            ss_xy = np.sum((x - mx) * (y - my))
            ss_yy = np.sum((y - my) ** 2)
            if ss_xx > 0 and ss_yy > 0:
                slope = ss_xy / ss_xx
                y_pred = my + slope * (x - mx)
                ss_res = np.sum((y - y_pred) ** 2)
                result.iloc[i] = max(0.0, 1.0 - ss_res / ss_yy)
            else:
                result.iloc[i] = 0.0
        return result

    def __repr__(self) -> str:
        return f"RSqr({self.child}, {self.window})"


# ============================================================================
#  Tokenizer
# ============================================================================

# Order matters: POW must be tried before OP so that "**" is not split into
# two "*" tokens. FUNC must precede FIELD/NUMBER (names are alphabetic).
TOKEN_PATTERNS: List[Tuple[str, str]] = [
    ("NUMBER", r"\d+\.?\d*"),
    ("FIELD", r"\$[a-zA-Z_]+"),
    ("FUNC", r"[A-Za-z_][A-Za-z0-9_]*\s*\("),
    ("POW", r"\*\*"),
    ("OP", r"[+\-*/]"),
    ("CMP", r">=?|<=?|==|!="),
    ("COMMA", r","),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("WS", r"\s+"),
]
_COMPILED = [(name, re.compile(pat)) for name, pat in TOKEN_PATTERNS]


def _tokenize(expr: str) -> List[Tuple[str, str]]:
    """Split an expression string into (type, value) tokens."""
    tokens: List[Tuple[str, str]] = []
    pos = 0
    n = len(expr)
    while pos < n:
        matched = False
        for tok_type, regex in _COMPILED:
            m = regex.match(expr, pos)
            if not m:
                continue
            if tok_type != "WS":
                val = m.group()
                if tok_type == "FUNC":
                    # "Mean(" -> FUNC("Mean") + LPAREN
                    tokens.append(("FUNC", val[:-1].strip()))
                    tokens.append(("LPAREN", "("))
                else:
                    tokens.append((tok_type, val))
            pos = m.end()
            matched = True
            break
        if not matched:
            raise SyntaxError(
                f"Cannot tokenize at position {pos}: '{expr[pos:pos + 12]}'"
            )
    return tokens


# ============================================================================
#  Recursive-descent parser
# ============================================================================

def _const_value(node: Factor, default: int) -> int:
    """Extract an integer window/argument from a constant node."""
    if isinstance(node, ConstFactor):
        return int(node.value)
    return default


def _make_func(name: str, args: List[Factor]) -> Factor:
    """Build the AST node for a function call."""
    rolling_ops = {
        "Ref", "Mean", "Std", "Max", "Min", "Sum",
        "Median", "Skew", "Kurt", "EMA", "Rank", "Delta",
    }
    if name in rolling_ops:
        if len(args) != 2:
            raise SyntaxError(f"{name} requires 2 args: ({name}(factor, window))")
        return RollingFactor(args[0], _const_value(args[1], 5), name)

    if name == "Corr":
        if len(args) != 3:
            raise SyntaxError("Corr requires 3 args: Corr(x, y, window)")
        return CorrFactor(args[0], args[1], _const_value(args[2], 10))

    if name == "RSqr":
        if len(args) != 2:
            raise SyntaxError("RSqr requires 2 args: RSqr(field, window)")
        return RSqrFactor(args[0], _const_value(args[1], 20))

    if name == "If":
        if len(args) != 3:
            raise SyntaxError("If requires 3 args: If(cond, true, false)")
        return IfFactor(args[0], args[1], args[2])

    if name == "Cross":
        if len(args) != 2:
            raise SyntaxError("Cross requires 2 args: Cross(a, b)")
        return CrossFactor(args[0], args[1])

    if name == "Clip":
        if len(args) != 3:
            raise SyntaxError("Clip requires 3 args: Clip(x, lo, hi)")
        return ClipFactor(args[0], _const_value(args[1], 0), _const_value(args[2], 1))

    if name == "Abs":
        if len(args) != 1:
            raise SyntaxError("Abs requires 1 arg: Abs(x)")
        return UnaryFunc(args[0], lambda s: s.abs(), "Abs")

    if name == "Log":
        if len(args) != 1:
            raise SyntaxError("Log requires 1 arg: Log(x)")
        return UnaryFunc(args[0], lambda s: np.log(np.maximum(s, 1e-12)), "Log")

    if name == "Sign":
        if len(args) != 1:
            raise SyntaxError("Sign requires 1 arg: Sign(x)")
        return UnaryFunc(args[0], lambda s: np.sign(s), "Sign")

    raise ValueError(f"Unknown function: {name}")


def _parse_expr(tokens: List[Tuple[str, str]], pos: int = 0) -> Tuple[Factor, int]:
    """
    Grammar (lowest to highest precedence):

        expr     -> add_sub (CMP add_sub)*
        add_sub  -> mul_div (('+' | '-') mul_div)*
        mul_div  -> power (('*' | '/') power)*
        power    -> unary ('**' power)?          # right-associative
        unary    -> ('+' | '-')? atom
        atom     -> NUMBER | FIELD | FUNC(args) | '(' expr ')'
    """

    def parse_add_sub(pos: int) -> Tuple[Factor, int]:
        left, pos = parse_mul_div(pos)
        while pos < len(tokens) and tokens[pos][0] == "OP" and tokens[pos][1] in ("+", "-"):
            op = tokens[pos][1]
            pos += 1
            right, pos = parse_mul_div(pos)
            left = ArithFactor(left, right, op)
        return left, pos

    def parse_mul_div(pos: int) -> Tuple[Factor, int]:
        left, pos = parse_power(pos)
        while pos < len(tokens) and tokens[pos][0] == "OP" and tokens[pos][1] in ("*", "/"):
            op = tokens[pos][1]
            pos += 1
            right, pos = parse_power(pos)
            left = ArithFactor(left, right, op)
        return left, pos

    def parse_power(pos: int) -> Tuple[Factor, int]:
        base, pos = parse_unary(pos)
        if pos < len(tokens) and tokens[pos][0] == "POW":
            pos += 1
            exponent, pos = parse_power(pos)  # right-associative
            base = ArithFactor(base, exponent, "**")
        return base, pos

    def parse_unary(pos: int) -> Tuple[Factor, int]:
        unary_op = None
        if pos < len(tokens) and tokens[pos][0] == "OP" and tokens[pos][1] in ("+", "-"):
            unary_op = tokens[pos][1]
            pos += 1
        node, pos = parse_atom(pos)
        if unary_op == "-":
            node = ArithFactor(ConstFactor(0), node, "-")
        return node, pos

    def parse_atom(pos: int) -> Tuple[Factor, int]:
        if pos >= len(tokens):
            raise SyntaxError("Incomplete expression")

        tok_type, val = tokens[pos]
        pos += 1

        if tok_type == "NUMBER":
            return ConstFactor(float(val)), pos

        if tok_type == "FIELD":
            return FieldFactor(val), pos

        if tok_type == "FUNC":
            func_name = val
            if pos < len(tokens) and tokens[pos][0] == "LPAREN":
                pos += 1
            args: List[Factor] = []
            if pos < len(tokens) and tokens[pos][0] != "RPAREN":
                arg, pos = _parse_expr(tokens, pos)
                args.append(arg)
                while pos < len(tokens) and tokens[pos][0] == "COMMA":
                    pos += 1
                    arg, pos = _parse_expr(tokens, pos)
                    args.append(arg)
            if pos >= len(tokens) or tokens[pos][0] != "RPAREN":
                raise SyntaxError(f"Function {func_name} missing closing ')'")
            pos += 1  # consume RPAREN
            return _make_func(func_name, args), pos

        if tok_type == "LPAREN":
            node, pos = _parse_expr(tokens, pos)
            if pos >= len(tokens) or tokens[pos][0] != "RPAREN":
                raise SyntaxError("Missing closing ')'")
            pos += 1
            return node, pos

        raise SyntaxError(f"Unexpected token: {tok_type}:{val}")

    left, pos = parse_add_sub(pos)
    while pos < len(tokens) and tokens[pos][0] == "CMP":
        op = tokens[pos][1]
        pos += 1
        right, pos = parse_add_sub(pos)
        left = CmpFactor(left, right, op)
    return left, pos


def parse_factor(expr: str) -> Factor:
    """Parse a factor expression string into an evaluatable AST."""
    tokens = _tokenize(expr)
    factor, pos = _parse_expr(tokens, 0)
    if pos < len(tokens):
        raise SyntaxError(f"Trailing tokens after expression: {tokens[pos:]}")
    return factor


# ============================================================================
#  Public engine
# ============================================================================

class FactorEngine:
    """
    Expression-based factor computation engine.

    Parses factor expressions and computes them on OHLCV DataFrames.

    Usage
    -----
    >>> engine = FactorEngine()
    >>> result = engine.compute("Mean($close, 20) / $close - 1", df)
    # Returns a Series with the computed factor values

    Supported syntax
    ----------------
    - Variables: $open, $high, $low, $close, $volume, $amount, $turnover
    - Functions: Mean(x, n), Std(x, n), Max(x, n), Min(x, n), Sum(x, n),
                 Corr(x, y, n), Rank(x, n), Skew(x, n), Kurt(x, n), EMA(x, n),
                 Delta(x, n), Ref(x, n), Abs(x), Log(x), Sign(x),
                 Clip(x, lo, hi), Median(x, n), If(c, a, b), Cross(a, b),
                 RSqr(x, n)
    - Operators: +, -, *, /, **
    - Comparisons: >, <, >=, <=, ==, !=  (yield 0/1)
    - Constants: numeric literals

    Computed expressions are cached so that repeated calls with the same
    expression string reuse the parsed AST.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Factor] = {}

    def _get_ast(self, expression: str) -> Factor:
        ast = self._cache.get(expression)
        if ast is None:
            ast = parse_factor(expression)
            self._cache[expression] = ast
        return ast

    def compute(self, expression: str, df: pd.DataFrame) -> pd.Series:
        """
        Compute a single factor expression on ``df``.

        Returns a Series aligned to ``df.index``. NaN values in the inputs
        propagate through the computation rather than raising.
        """
        ast = self._get_ast(expression)
        result = ast.evaluate(df)
        if not isinstance(result, pd.Series):
            result = pd.Series(result, index=df.index, dtype=float)
        return result.astype(float)

    def compute_batch(self, expressions: Dict[str, str], df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute many factors at once.

        Parameters
        ----------
        expressions : dict
            ``{factor_name: expression_string}``
        df : DataFrame
            OHLCV data with the standard columns.

        Returns
        -------
        DataFrame with one column per factor (named by the dict keys), aligned
        to ``df.index``. A factor that fails to compute is filled with NaN and
        does not abort the batch.
        """
        out = pd.DataFrame(index=df.index)
        for name, expr in expressions.items():
            try:
                out[name] = self.compute(expr, df)
            except Exception as exc:  # noqa: BLE001 - batch must be resilient
                print(f"[FactorEngine] '{name}' = '{expr}' failed: {exc}")
                out[name] = np.nan
        return out

    def validate(self, expression: str) -> bool:
        """Return True if ``expression`` parses without error, else False."""
        try:
            self._get_ast(expression)
            return True
        except Exception:  # noqa: BLE001 - validation is boolean by contract
            return False
