"""
因子表达式引擎 — Qlib 风格的字符串因子 DSL

将 "Mean($close, 5) / Mean($close, 20) - 1" 自动解析为可计算因子树。

用法:
    from factor_engine import parse_factor, FactorLibrary

    expr = parse_factor("Ref($close, 5) / $close - 1")   # 5日动量
    result = expr.evaluate(df)                              # → pd.Series

    lib = FactorLibrary.from_config({
        "momentum_5": "Ref($close, 5) / $close - 1",
        "volume_ratio": "$volume / Mean($volume, 5)",
        "ma_cross": "Mean($close, 5) > Mean($close, 20)",
    })
    factors_df = lib.evaluate_all(df)
"""

import re
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union


# ============================================================================
#  Factor AST 节点
# ============================================================================

class Factor(ABC):
    """因子抽象基类。"""
    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """在 DataFrame 上计算该因子。"""
        ...

    @abstractmethod
    def __repr__(self) -> str:
        ...


class FieldFactor(Factor):
    """字段引用: $close, $open, $high, $low, $volume, $amount"""

    VALID_FIELDS = {"close", "open", "high", "low", "volume", "amount", "vwap", "turnover", "outstanding_share"}

    def __init__(self, field: str):
        field = field.strip().lower()
        if field.startswith("$"):
            field = field[1:]
        if field not in self.VALID_FIELDS:
            raise ValueError(f"不支持的字段: {field}，可选: {self.VALID_FIELDS}")
        self.field = field

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        if self.field not in df.columns:
            # 常见别名
            aliases = {"vwap": "amount"}
            col = aliases.get(self.field, self.field)
            if col not in df.columns:
                return pd.Series(np.nan, index=df.index)
        else:
            col = self.field
        return df[col].astype(float)

    def __repr__(self):
        return f"${self.field}"


class ConstFactor(Factor):
    """常数因子。"""
    def __init__(self, value: float):
        self.value = value

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=df.index)

    def __repr__(self):
        return str(self.value)


class RollingFactor(Factor):
    """滚动算子基类: Ref, Mean, Std, Max, Min, Sum"""
    def __init__(self, child: Factor, window: int, op_name: str):
        self.child = child
        self.window = window
        self.op_name = op_name

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        series = self.child.evaluate(df)
        return self._rolling(series)

    def _rolling(self, series: pd.Series) -> pd.Series:
        if self.op_name == "Ref":
            return series.shift(self.window)
        elif self.op_name == "Mean":
            return series.rolling(self.window, min_periods=max(1, self.window // 2)).mean()
        elif self.op_name == "Std":
            return series.rolling(self.window, min_periods=max(2, self.window // 2)).std()
        elif self.op_name == "Max":
            return series.rolling(self.window, min_periods=1).max()
        elif self.op_name == "Min":
            return series.rolling(self.window, min_periods=1).min()
        elif self.op_name == "Sum":
            return series.rolling(self.window, min_periods=1).sum()
        elif self.op_name == "Median":
            return series.rolling(self.window, min_periods=1).median()
        elif self.op_name == "Skew":
            return series.rolling(self.window, min_periods=3).skew()
        elif self.op_name == "Kurt":
            return series.rolling(self.window, min_periods=4).kurt()
        elif self.op_name == "EMA":
            return series.ewm(span=self.window, adjust=False).mean()
        elif self.op_name == "Rank":
            return series.rolling(self.window, min_periods=1).rank(pct=True)
        else:
            raise ValueError(f"未知滚动算子: {self.op_name}")

    def __repr__(self):
        return f"{self.op_name}({self.child}, {self.window})"


class ArithFactor(Factor):
    """算术运算: +, -, *, /"""
    OPS = {"+": "add", "-": "sub", "*": "mul", "/": "div"}

    def __init__(self, left: Factor, right: Factor, op: str):
        self.left = left
        self.right = right
        self.op = op

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        lv = self.left.evaluate(df)
        rv = self.right.evaluate(df)
        if self.op == "+":
            return lv + rv
        elif self.op == "-":
            return lv - rv
        elif self.op == "*":
            return lv * rv
        elif self.op == "/":
            # 安全除法
            return lv / rv.replace(0, np.nan)
        raise ValueError(f"未知运算: {self.op}")

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"


class CmpFactor(Factor):
    """比较运算: >, <, >=, <=, ==, != → 返回 0/1"""
    def __init__(self, left: Factor, right: Factor, op: str):
        self.left = left
        self.right = right
        self.op = op

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        lv = self.left.evaluate(df)
        rv = self.right.evaluate(df)
        if self.op == ">":
            return (lv > rv).astype(float)
        elif self.op == "<":
            return (lv < rv).astype(float)
        elif self.op == ">=":
            return (lv >= rv).astype(float)
        elif self.op == "<=":
            return (lv <= rv).astype(float)
        elif self.op == "==":
            return (lv == rv).astype(float)
        elif self.op == "!=":
            return (lv != rv).astype(float)
        raise ValueError(f"未知比较: {self.op}")

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"


class IfFactor(Factor):
    """条件因子: If(cond, true_expr, false_expr)"""
    def __init__(self, cond: Factor, true_val: Factor, false_val: Factor):
        self.cond = cond
        self.true_val = true_val
        self.false_val = false_val

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        c = self.cond.evaluate(df)
        tv = self.true_val.evaluate(df)
        fv = self.false_val.evaluate(df)
        return pd.Series(np.where(c > 0, tv, fv), index=df.index)

    def __repr__(self):
        return f"If({self.cond}, {self.true_val}, {self.false_val})"


class CrossFactor(Factor):
    """交叉检测: Cross(a, b) → a上穿b=1, a下穿b=-1"""
    def __init__(self, a: Factor, b: Factor):
        self.a = a
        self.b = b

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        av = self.a.evaluate(df)
        bv = self.b.evaluate(df)
        av_1 = av.shift(1)
        bv_1 = bv.shift(1)
        golden = (av_1 <= bv_1) & (av > bv)
        death = (av_1 >= bv_1) & (av < bv)
        result = pd.Series(0, index=df.index, dtype=float)
        result[golden] = 1
        result[death] = -1
        return result

    def __repr__(self):
        return f"Cross({self.a}, {self.b})"


class RSVFactor(Factor):
    """RSV (Raw Stochastic Value): ($close - Min($low,N)) / (Max($high,N) - Min($low,N) + eps)"""
    def __init__(self, window: int = 9):
        self.window = window

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        c, h, l = df["close"], df["high"], df["low"]
        low_n = l.rolling(self.window, min_periods=1).min()
        high_n = h.rolling(self.window, min_periods=1).max()
        return (c - low_n) / (high_n - low_n + 1e-12)

    def __repr__(self):
        return f"RSV({self.window})"


class _UnaryFunc(Factor):
    """一元函数: Abs, Log, Sign 等"""
    def __init__(self, child: Factor, func, name: str):
        self.child = child
        self.func = func
        self.name = name

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        return self.func(self.child.evaluate(df))

    def __repr__(self):
        return f"{self.name}({self.child})"


# ============================================================================
#  Parser — 字符串 → Factor 树
# ============================================================================

# Token 类型
TOKEN_PATTERNS = [
    ("NUMBER",  r"\d+\.?\d*"),
    ("FIELD",   r"\$[a-zA-Z_]+"),
    ("FUNC",    r"[A-Za-z_][A-Za-z0-9_]*\s*\("),
    ("OP",      r"[+\-*/]"),
    ("CMP",     r">=?|<=?|==|!="),
    ("COMMA",   r","),
    ("LPAREN",  r"\("),
    ("RPAREN",  r"\)"),
    ("WS",      r"\s+"),
]


def _tokenize(expr: str) -> List[tuple]:
    """将表达式字符串分词。"""
    tokens = []
    pos = 0
    while pos < len(expr):
        match = None
        for tok_type, pattern in TOKEN_PATTERNS:
            regex = re.compile(pattern)
            m = regex.match(expr, pos)
            if m:
                if tok_type != "WS":
                    val = m.group()
                    if tok_type == "FUNC":
                        val = val[:-1]  # Remove trailing (
                        tokens.append(("FUNC", val))
                        tokens.append(("LPAREN", "("))
                    else:
                        tokens.append((tok_type, val))
                pos = m.end()
                match = True
                break
        if not match:
            raise SyntaxError(f"无法解析位置 {pos}: '{expr[pos:pos+10]}...'")
    return tokens


def _parse_expr(tokens: List[tuple], pos: int = 0) -> (Factor, int):
    """递归下降解析器: expr → term (CMP term)*"""

    def parse_term(tokens, pos):
        """term → factor (OP factor)*"""
        left, pos = parse_factor(tokens, pos)
        while pos < len(tokens) and tokens[pos][0] == "OP":
            op = tokens[pos][1]
            pos += 1
            right, pos = parse_factor(tokens, pos)
            left = ArithFactor(left, right, op)
        return left, pos

    def parse_factor(tokens, pos):
        """factor → NUMBER | FIELD | FUNC(args) | LPAREN expr RPAREN"""
        if pos >= len(tokens):
            raise SyntaxError("表达式不完整")

        tok_type, val = tokens[pos]
        pos += 1

        if tok_type == "NUMBER":
            return ConstFactor(float(val)), pos
        elif tok_type == "FIELD":
            return FieldFactor(val), pos
        elif tok_type == "FUNC":
            # 函数调用: FUNC LPAREN args RPAREN
            func_name = val
            # 跳过 LPAREN
            if pos < len(tokens) and tokens[pos][0] == "LPAREN":
                pos += 1
            args = []
            if pos < len(tokens) and tokens[pos][0] != "RPAREN":
                # 解析第一个参数
                arg, pos = _parse_expr(tokens, pos)
                args.append(arg)
                while pos < len(tokens) and tokens[pos][0] == "COMMA":
                    pos += 1
                    arg, pos = _parse_expr(tokens, pos)
                    args.append(arg)
            if pos >= len(tokens) or tokens[pos][0] != "RPAREN":
                raise SyntaxError(f"函数 {func_name} 缺少右括号")
            pos += 1  # skip RPAREN

            return _make_func(func_name, args), pos
        elif tok_type == "LPAREN":
            expr, pos = _parse_expr(tokens, pos)
            if pos >= len(tokens) or tokens[pos][0] != "RPAREN":
                raise SyntaxError("缺少右括号")
            pos += 1
            return expr, pos
        else:
            raise SyntaxError(f"意外的 token: {tok_type}:{val}")

    # expr → term (CMP term)*
    left, pos = parse_term(tokens, pos)
    while pos < len(tokens) and tokens[pos][0] == "CMP":
        op = tokens[pos][1]
        pos += 1
        right, pos = parse_term(tokens, pos)
        left = CmpFactor(left, right, op)
    return left, pos


def _make_func(name: str, args: List[Factor]) -> Factor:
    """根据函数名和参数创建因子节点。"""
    # 滚动算子: Mean(child, window)
    rolling_ops = {"Ref", "Mean", "Std", "Max", "Min", "Sum", "Median", "Skew", "Kurt", "EMA", "Rank"}
    if name in rolling_ops:
        if len(args) != 2:
            raise SyntaxError(f"{name} 需要2个参数: (factor, window)")
        window = int(float(args[1].__repr__())) if isinstance(args[1], ConstFactor) else 5
        return RollingFactor(args[0], window, name)

    # 条件: If(cond, true, false)
    if name == "If":
        if len(args) != 3:
            raise SyntaxError("If 需要3个参数: (cond, true, false)")
        return IfFactor(args[0], args[1], args[2])

    # 交叉: Cross(a, b)
    if name == "Cross":
        if len(args) != 2:
            raise SyntaxError("Cross 需要2个参数: (a, b)")
        return CrossFactor(args[0], args[1])

    # RSV: RSV(window)
    if name == "RSV":
        if len(args) != 1:
            raise SyntaxError("RSV 需要1个参数: (window)")
        window = int(float(args[0].__repr__())) if isinstance(args[0], ConstFactor) else 9
        return RSVFactor(window)

    # Abs / Log / Sign
    if name == "Abs" and len(args) == 1:
        return _UnaryFunc(args[0], np.abs, "Abs")
    if name == "Log" and len(args) == 1:
        return _UnaryFunc(args[0], lambda x: np.log(np.maximum(x, 1e-12)), "Log")

    raise ValueError(f"未知函数: {name}")


def parse_factor(expr: str) -> Factor:
    """将字符串表达式解析为因子树。"""
    tokens = _tokenize(expr)
    factor, pos = _parse_expr(tokens, 0)
    if pos < len(tokens):
        raise SyntaxError(f"表达式末尾有多余 token: {tokens[pos:]}")
    return factor


# ============================================================================
#  FactorLibrary — 批量因子管理
# ============================================================================

class FactorLibrary:
    """
    因子库 — 从配置加载和管理多个因子。

    用法:
        lib = FactorLibrary.from_config({
            "momentum_5":  "Ref($close, 5) / $close - 1",
            "ma_spread":   "Mean($close, 5) / Mean($close, 20) - 1",
            "vol_ratio":   "$volume / Mean($volume, 5)",
        }, default_fields={"volume": 0})
        factors_df = lib.evaluate_all(df)
    """

    def __init__(self, factors: Dict[str, Factor]):
        self.factors = factors
        self._cache: Dict[str, Factor] = {}

    @classmethod
    def from_config(cls, config: Dict[str, str],
                    default_fields: Optional[Dict[str, float]] = None) -> "FactorLibrary":
        """
        从字符串表达式配置创建因子库。

        参数
        ----
        config : dict
            {因子名: 表达式字符串}
        default_fields : dict
            默认字段值(如 {"volume": 0}，当数据无 volume 列时填充)
        """
        factors = {}
        for name, expr_str in config.items():
            try:
                factors[name] = parse_factor(expr_str)
            except Exception as e:
                print(f"[FactorLib] 解析 '{name}' = '{expr_str}' 失败: {e}")
        return cls(factors)

    def evaluate_all(self, df: pd.DataFrame, cache_to_db: bool = False,
                      symbol: str = "") -> pd.DataFrame:
        """计算所有因子，返回因子 DataFrame。可选择缓存到数据库。"""
        result = pd.DataFrame(index=df.index)
        result["date"] = df["date"].values if "date" in df.columns else df.index
        for name, factor in self.factors.items():
            try:
                result[name] = factor.evaluate(df)
            except Exception as e:
                print(f"[FactorLib] 计算 '{name}' 失败: {e}")
                result[name] = np.nan

        # 缓存到数据库
        if cache_to_db and symbol:
            try:
                import storage
                storage.save_factor_snapshots_batch(symbol, result)
            except Exception as e:
                print(f"[FactorLib] 缓存失败: {e}")

        return result

    def get(self, name: str) -> Factor:
        return self.factors[name]

    def __len__(self):
        return len(self.factors)

    def __repr__(self):
        return f"FactorLibrary({len(self.factors)} factors: {list(self.factors.keys())[:5]}...)"
