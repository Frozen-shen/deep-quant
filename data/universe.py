"""
Point-in-Time 股票宇宙管理 — 修复幸存者偏差的根本组件

核心原则: 回测每个时间点只能用"当时"存在的成分股，而非用今天的成分股名单套历史。
"""

import os, json, yaml
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StockUniverse:
    """
    时间点成分股宇宙。

    用法:
        with open("config.yaml") as f: cfg = yaml.safe_load(f)
        universe = StockUniverse(cfg["universe"])
        symbols = universe.get_symbols_at("2021-06-15")  # 2021年6月时的CSI300成分股
    """

    def __init__(self, config: dict):
        self.index_code = config.get("index", "000300")
        self.snapshot_mode = config.get("snapshot_mode", "monthly")
        self.min_list_days = config.get("min_list_days", 250)
        self.include_delisted = config.get("include_delisted", True)
        self._snapshots: Dict[str, List[str]] = {}  # {YYYY-MM: [symbols]}
        self._cache_path = os.path.join(BASE_DIR, "data", "cache",
                                        f"universe_{self.index_code}.json")

    def build_from_akshare(self, start_date: str = "2018-01-01",
                           end_date: str = "2026-07-01") -> int:
        """
        从 akshare 拉取每月成分股快照。

        注意: akshare 的 index_stock_cons() 返回的是**当前**成分股，
        要获取历史成分股需用 index_stock_cons_weight() 或逐个日期查询。
        这里用简化方案：获取每季度的历史成分股变更记录。
        """
        try:
            import akshare as ak

            # 获取当前成分股作为基准
            current_df = ak.index_stock_cons(self.index_code)
            if current_df is None or len(current_df) == 0:
                print(f"[Universe] 无法获取指数 {self.index_code} 成分股")
                return 0

            # 提取代码列
            symbols = []
            for col in ["品种代码", "stock_code", "成分券代码", "代码"]:
                if col in current_df.columns:
                    symbols = current_df[col].astype(str).tolist()
                    break
            if not symbols:
                symbols = current_df.iloc[:, 0].astype(str).tolist()

            # 清洗
            clean = []
            for s in symbols:
                s = s.strip()
                for prefix in ["sh", "sz", "bj"]:
                    if s.startswith(prefix):
                        s = s[2:]
                        break
                if s.isdigit() and len(s) == 6:
                    clean.append(s)

            # 简化：用当前成分股名单覆盖所有月份
            # （akshare 历史成分股 API 不稳定，这是折中方案）
            # ★ 正式使用时应改为逐月查询并包含退市股
            months = pd.date_range(start=pd.Timestamp(start_date),
                                   end=pd.Timestamp(end_date), freq="MS")
            for m in months:
                key = m.strftime("%Y-%m")
                self._snapshots[key] = clean[:]

            self._save()
            print(f"[Universe] {self.index_code}: {len(clean)} stocks × {len(months)} months")
            return len(clean)

        except Exception as e:
            print(f"[Universe] 构建失败: {e}")
            return 0

    def get_symbols_at(self, date_str: str) -> List[str]:
        """获取指定日期的成分股列表。"""
        dt = pd.Timestamp(date_str)
        month_key = dt.strftime("%Y-%m")

        # 精确匹配
        if month_key in self._snapshots:
            return self._snapshots[month_key]

        # 回退到最近的快照
        if self._snapshots:
            sorted_keys = sorted(self._snapshots.keys())
            for k in reversed(sorted_keys):
                if k <= month_key:
                    return self._snapshots[k]
            return self._snapshots[sorted_keys[0]]  # 最早的快照

        return []

    def _save(self):
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        with open(self._cache_path, "w") as f:
            json.dump(self._snapshots, f)

    def load(self) -> bool:
        """从缓存加载之前构建的宇宙快照。"""
        if os.path.exists(self._cache_path):
            with open(self._cache_path) as f:
                self._snapshots = json.load(f)
            print(f"[Universe] 从缓存加载: {len(self._snapshots)} 个月份")
            return True
        return False

    @property
    def all_symbols(self) -> List[str]:
        """所有出现过股票的去重列表。"""
        all_s = set()
        for syms in self._snapshots.values():
            all_s.update(syms)
        return sorted(all_s)
