"""
因子注册表 — 元数据管理 + 生命周期

每个因子有:
  - name: 唯一标识
  - expr: DSL 表达式 (价量因子) 或 source 标识 (基本面/相对因子)
  - category: momentum/value/volatility/liquidity/quality/growth/relative/northbound
  - freq: daily / quarterly / snapshot
  - status: active / deprecated / experimental
  - version: 语义版本号
  - ic_history: 最近N期IC记录 (用于衰减检测)
  - meta: 经济逻辑描述、数据来源、创建时间

持久化: data/factor_registry.json
"""

import os
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(BASE_DIR, "data", "factor_registry.json")


@dataclass
class FactorMeta:
    """因子元数据。"""
    name: str
    expr: str = ""                          # DSL表达式 (价量因子)
    source: str = "price_volume"            # price_volume / fundamental / relative / northbound / flow
    category: str = "momentum"              # 因子大类
    freq: str = "daily"                     # daily / quarterly / snapshot
    status: str = "active"                  # active / deprecated / experimental
    version: str = "1.0.0"
    description: str = ""                   # 经济逻辑
    created_at: str = ""
    updated_at: str = ""
    ic_latest: Optional[float] = None       # 最近一期IC
    icir_latest: Optional[float] = None     # 最近ICIR
    ic_half_life: Optional[float] = None    # IC半衰期 (天)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FactorMeta":
        # 过滤未知字段
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


class FactorRegistry:
    """
    因子注册表 — 管理所有因子的元数据和生命周期。

    用法:
        reg = FactorRegistry.load()
        reg.register("mom_20d", expr="Ref($close, 20) / $close - 1",
                     category="momentum", description="20日动量")
        reg.save()

        # 查询
        active = reg.query(status="active", category="momentum")
        f = reg.get("mom_20d")
    """

    def __init__(self, factors: Dict[str, FactorMeta] = None):
        self.factors: Dict[str, FactorMeta] = factors or {}

    @classmethod
    def load(cls, path: str = REGISTRY_PATH) -> "FactorRegistry":
        """从 JSON 加载注册表。"""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            factors = {name: FactorMeta.from_dict(d) for name, d in data.items()}
            return cls(factors)
        return cls()

    def save(self, path: str = REGISTRY_PATH):
        """持久化到 JSON。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {name: meta.to_dict() for name, meta in self.factors.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def register(self, name: str, expr: str = "", source: str = "price_volume",
                 category: str = "momentum", freq: str = "daily",
                 status: str = "active", description: str = "",
                 tags: List[str] = None, **kwargs) -> FactorMeta:
        """注册或更新一个因子。"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        if name in self.factors:
            meta = self.factors[name]
            meta.expr = expr or meta.expr
            meta.source = source
            meta.category = category
            meta.freq = freq
            meta.status = status
            meta.description = description or meta.description
            meta.updated_at = now
            if tags:
                meta.tags = tags
        else:
            meta = FactorMeta(
                name=name, expr=expr, source=source, category=category,
                freq=freq, status=status, description=description,
                created_at=now, updated_at=now, tags=tags or [],
                **{k: v for k, v in kwargs.items()
                   if k in FactorMeta.__dataclass_fields__}
            )
            self.factors[name] = meta
        return meta

    def deprecate(self, name: str, reason: str = ""):
        """废弃因子。"""
        if name in self.factors:
            self.factors[name].status = "deprecated"
            self.factors[name].updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
            if reason:
                self.factors[name].description += f" [DEPRECATED: {reason}]"

    def get(self, name: str) -> Optional[FactorMeta]:
        return self.factors.get(name)

    def query(self, status: str = None, category: str = None,
              source: str = None, freq: str = None) -> List[FactorMeta]:
        """按条件查询因子。"""
        results = list(self.factors.values())
        if status:
            results = [f for f in results if f.status == status]
        if category:
            results = [f for f in results if f.category == category]
        if source:
            results = [f for f in results if f.source == source]
        if freq:
            results = [f for f in results if f.freq == freq]
        return results

    def update_ic(self, name: str, ic: float, icir: float,
                  half_life: float = None):
        """更新因子的最新IC指标。"""
        if name in self.factors:
            meta = self.factors[name]
            meta.ic_latest = ic
            meta.icir_latest = icir
            if half_life is not None:
                meta.ic_half_life = half_life
            meta.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    def sync_from_library(self):
        """从 factor_library.py 同步所有已定义因子到注册表。"""
        import sys
        sys.path.insert(0, BASE_DIR)
        import factor_library as fl
        # 收集所有 *_FACTORS 字典 (包含 DSL 表达式字符串)
        all_raw = {}
        for attr in dir(fl):
            obj = getattr(fl, attr)
            if isinstance(obj, dict) and attr.endswith("_FACTORS"):
                for name, expr_str in obj.items():
                    if isinstance(expr_str, str) and "$" in expr_str:
                        all_raw[name] = expr_str
        for name, expr_str in all_raw.items():
            if name not in self.factors:
                self.register(name, expr=expr_str, source="price_volume",
                              category=self._infer_category(name),
                              status="active")

    @staticmethod
    def _infer_category(name: str) -> str:
        """从因子名推断类别。"""
        name_lower = name.lower()
        if any(k in name_lower for k in ["return", "mom", "ma", "cross", "streak", "position"]):
            return "momentum"
        if any(k in name_lower for k in ["vol", "std", "skew", "kurt", "downside", "sortino"]):
            return "volatility"
        if any(k in name_lower for k in ["volume", "turnover", "amount", "flow"]):
            return "liquidity"
        if any(k in name_lower for k in ["corr", "cord", "sync", "beta"]):
            return "correlation"
        if any(k in name_lower for k in ["fund_", "ep", "bp", "sp", "ocf", "accrual"]):
            return "value"
        if any(k in name_lower for k in ["nb_", "north"]):
            return "northbound"
        if any(k in name_lower for k in ["rel_", "idio", "excess"]):
            return "relative"
        return "other"

    def __len__(self):
        return len(self.factors)

    def __repr__(self):
        n_active = sum(1 for f in self.factors.values() if f.status == "active")
        return f"FactorRegistry({len(self.factors)} total, {n_active} active)"
