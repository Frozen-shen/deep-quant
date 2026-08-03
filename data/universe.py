"""
Point-in-Time 股票宇宙管理 — 消除幸存者偏差的根本组件

核心原则: 回测每个时间点只能用"当时"存在的成分股，而非用今天的成分股名单套历史。

数据源:
  - CSI300 (000300): baostock query_hs300_stocks(date) — 真实历史成分股
  - CSI1000 (000852): baostock query_zz500_stocks(date) 作为部分历史代理
    (ZZ500 覆盖 CSI1000 中排名靠前的约500只股票)
    + akshare index_stock_cons_csindex('000852') 补充最新成分股

PIT_QUALITY 标记:
  - "HIGH": 真实历史成分股 (CSI300 via baostock)
  - "MEDIUM": 部分历史代理 (CSI1000 via ZZ500 proxy, 覆盖约50%)
  - "LOW": 使用当前成分股回填 (无历史数据时的 fallback)
"""

import os, json, time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# baostock 查询间隔 (秒)，避免被限流
_QUERY_INTERVAL = 0.3


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
        self._pit_quality: Dict[str, str] = {}  # {YYYY-MM: quality_flag}
        self._cache_path = os.path.join(BASE_DIR, "data", "cache",
                                        f"universe_{self.index_code}.json")

    def build_from_baostock(self, start_date: str = "2018-01-01",
                            end_date: str = "2026-07-01",
                            sample_day: int = 15) -> int:
        """
        从 baostock 拉取每月真实历史成分股快照。

        CSI300: 使用 query_hs300_stocks(date) — 真实历史数据
        CSI1000: 使用 query_zz500_stocks(date) 作为部分代理 + 最新成分股补充

        Args:
            start_date: 起始日期
            end_date: 结束日期
            sample_day: 每月采样日 (默认15日，避开月初月末调仓日)

        Returns:
            成功获取的月份数量
        """
        import baostock as bs

        # 确定使用哪个 baostock 查询函数
        if self.index_code == "000300":
            query_func = bs.query_hs300_stocks
            quality = "HIGH"
            print(f"[Universe] CSI300: 使用 baostock 真实历史成分股")
        elif self.index_code in ("000852", "000905"):
            # CSI1000 没有免费历史API，用 ZZ500 作为部分代理
            query_func = bs.query_zz500_stocks
            quality = "MEDIUM"
            print(f"[Universe] CSI1000: 使用 ZZ500 作为部分历史代理 (覆盖约50%)")
        elif self.index_code == "000016":
            query_func = bs.query_sz50_stocks
            quality = "HIGH"
            print(f"[Universe] SZ50: 使用 baostock 真实历史成分股")
        else:
            print(f"[Universe] 指数 {self.index_code} 无 baostock 历史数据，"
                  f"使用当前成分股回填 (LOW quality)")
            return self._build_fallback(start_date, end_date)

        # 登录 baostock
        lg = bs.login()
        if lg.error_code != '0':
            print(f"[Universe] baostock 登录失败: {lg.error_msg}")
            return 0

        try:
            months = pd.date_range(start=pd.Timestamp(start_date),
                                   end=pd.Timestamp(end_date), freq="MS")
            success_count = 0
            last_query_time = 0
            consecutive_failures = 0

            for i, m in enumerate(months):
                # 构造查询日期 (每月 sample_day 日)
                query_date = m.replace(day=min(sample_day, 28)).strftime("%Y-%m-%d")
                month_key = m.strftime("%Y-%m")

                # 限流
                elapsed = time.time() - last_query_time
                if elapsed < _QUERY_INTERVAL:
                    time.sleep(_QUERY_INTERVAL - elapsed)

                # 查询 (带重连逻辑)
                last_query_time = time.time()
                rs = query_func(date=query_date)

                if rs.error_code != '0':
                    consecutive_failures += 1
                    # 连续失败时尝试重新登录
                    if consecutive_failures >= 2:
                        print(f"[Universe] 连续失败 {consecutive_failures} 次，尝试重连...")
                        try:
                            bs.logout()
                        except Exception:
                            pass
                        time.sleep(1)
                        lg = bs.login()
                        if lg.error_code == '0':
                            print(f"[Universe] 重连成功")
                            consecutive_failures = 0
                            # 重试当前查询
                            time.sleep(_QUERY_INTERVAL)
                            rs = query_func(date=query_date)
                            if rs.error_code != '0':
                                print(f"[Universe] {month_key} 重试仍失败: {rs.error_msg}")
                                continue
                        else:
                            print(f"[Universe] 重连失败: {lg.error_msg}")
                            continue
                    else:
                        print(f"[Universe] {month_key} 查询失败: {rs.error_msg}")
                        continue
                else:
                    consecutive_failures = 0

                # 提取成分股代码
                symbols = []
                while rs.next():
                    row = rs.get_row_data()
                    # row format: [updateDate, code, code_name]
                    # code format: "sh.600000" or "sz.000001"
                    code = row[1] if len(row) > 1 else ""
                    # 去掉 "sh." / "sz." 前缀
                    if "." in code:
                        code = code.split(".")[1]
                    if code.isdigit() and len(code) == 6:
                        symbols.append(code)

                if symbols:
                    self._snapshots[month_key] = sorted(symbols)
                    self._pit_quality[month_key] = quality
                    success_count += 1

                # 进度报告
                if (i + 1) % 12 == 0 or (i + 1) == len(months):
                    print(f"[Universe] 进度: {i+1}/{len(months)} 月, "
                          f"成功 {success_count} 个")

        finally:
            try:
                bs.logout()
            except Exception:
                pass

        # 对 CSI1000，补充最新成分股到最近月份
        if self.index_code == "000852":
            self._supplement_csi1000_latest(end_date)

        self._save()
        print(f"[Universe] {self.index_code}: 完成 {success_count} 个月份快照")
        return success_count

    def _supplement_csi1000_latest(self, end_date: str):
        """用 akshare 获取最新 CSI1000 成分股，补充到最近月份。"""
        try:
            import akshare as ak
            df = ak.index_stock_cons_csindex(symbol="000852")
            if df is not None and len(df) > 0:
                # 提取代码列
                symbols = []
                for col in ["成分券代码", "品种代码", "stock_code", "代码"]:
                    if col in df.columns:
                        symbols = df[col].astype(str).str.zfill(6).tolist()
                        break
                if not symbols:
                    symbols = df.iloc[:, 0].astype(str).tolist()

                # 清洗
                clean = []
                for s in symbols:
                    s = s.strip()
                    if s.isdigit() and len(s) == 6:
                        clean.append(s)

                if clean:
                    # 写入最新月份
                    latest_key = pd.Timestamp(end_date).strftime("%Y-%m")
                    self._snapshots[latest_key] = sorted(clean)
                    self._pit_quality[latest_key] = "HIGH"
                    print(f"[Universe] CSI1000 最新成分股 ({len(clean)} 只) "
                          f"已补充到 {latest_key}")
        except Exception as e:
            print(f"[Universe] CSI1000 最新成分股补充失败: {e}")

    def _build_fallback(self, start_date: str, end_date: str) -> int:
        """
        Fallback: 使用当前成分股回填所有月份 (有幸存者偏差)。
        标记为 LOW quality。
        """
        try:
            import akshare as ak
            if self.index_code in ("000300", "000852", "000016", "000905"):
                df = ak.index_stock_cons_csindex(symbol=self.index_code)
            else:
                df = ak.index_stock_cons(self.index_code)

            if df is None or len(df) == 0:
                print(f"[Universe] 无法获取指数 {self.index_code} 成分股")
                return 0

            symbols = []
            for col in ["成分券代码", "品种代码", "stock_code", "代码"]:
                if col in df.columns:
                    symbols = df[col].astype(str).tolist()
                    break
            if not symbols:
                symbols = df.iloc[:, 0].astype(str).tolist()

            clean = []
            for s in symbols:
                s = s.strip()
                for prefix in ["sh", "sz", "bj"]:
                    if s.startswith(prefix):
                        s = s[2:]
                        break
                if s.isdigit() and len(s) == 6:
                    clean.append(s)

            months = pd.date_range(start=pd.Timestamp(start_date),
                                   end=pd.Timestamp(end_date), freq="MS")
            for m in months:
                key = m.strftime("%Y-%m")
                self._snapshots[key] = clean[:]
                self._pit_quality[key] = "LOW"

            self._save()
            print(f"[Universe] {self.index_code}: {len(clean)} stocks × "
                  f"{len(months)} months (LOW quality - 当前成分股回填)")
            return len(months)

        except Exception as e:
            print(f"[Universe] Fallback 构建失败: {e}")
            return 0

    def build_from_akshare(self, start_date: str = "2018-01-01",
                           end_date: str = "2026-07-01") -> int:
        """
        [已废弃] 旧接口，保留兼容性。内部转发到 build_from_baostock。
        """
        print("[Universe] WARNING: build_from_akshare() 已废弃，"
              "请使用 build_from_baostock() 获取真实历史成分股")
        return self.build_from_baostock(start_date, end_date)

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

    def get_pit_quality_at(self, date_str: str) -> str:
        """获取指定日期的 PIT 数据质量标记。"""
        dt = pd.Timestamp(date_str)
        month_key = dt.strftime("%Y-%m")

        if month_key in self._pit_quality:
            return self._pit_quality[month_key]

        # 回退
        if self._pit_quality:
            sorted_keys = sorted(self._pit_quality.keys())
            for k in reversed(sorted_keys):
                if k <= month_key:
                    return self._pit_quality[k]
            return self._pit_quality[sorted_keys[0]]

        return "UNKNOWN"

    def _save(self):
        """保存快照到 JSON 文件。格式: {YYYY-MM: [stock_codes]}"""
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        with open(self._cache_path, "w") as f:
            json.dump(self._snapshots, f)

        # 保存质量元数据
        quality_path = self._cache_path.replace(".json", "_quality.json")
        with open(quality_path, "w") as f:
            json.dump(self._pit_quality, f)

    def load(self) -> bool:
        """从缓存加载之前构建的宇宙快照。"""
        if os.path.exists(self._cache_path):
            with open(self._cache_path) as f:
                self._snapshots = json.load(f)

            # 加载质量元数据
            quality_path = self._cache_path.replace(".json", "_quality.json")
            if os.path.exists(quality_path):
                with open(quality_path) as f:
                    self._pit_quality = json.load(f)

            print(f"[Universe] 从缓存加载: {len(self._snapshots)} 个月份")
            if self._pit_quality:
                qualities = set(self._pit_quality.values())
                print(f"[Universe] PIT质量: {qualities}")
            return True
        return False

    @property
    def all_symbols(self) -> List[str]:
        """所有出现过股票的去重列表。"""
        all_s = set()
        for syms in self._snapshots.values():
            all_s.update(syms)
        return sorted(all_s)

    @property
    def quality_summary(self) -> Dict[str, int]:
        """各质量等级的月份数量统计。"""
        summary = {}
        for q in self._pit_quality.values():
            summary[q] = summary.get(q, 0) + 1
        return summary


# =============================================================================
# 独立构建脚本入口
# =============================================================================

def build_all_universes(start_date: str = "2018-01-01",
                        end_date: str = "2026-07-01"):
    """构建所有指数的 PIT 宇宙并保存到缓存。"""
    indices = [
        {"index": "000300", "name": "CSI300"},
        {"index": "000852", "name": "CSI1000"},
    ]

    for idx_cfg in indices:
        print(f"\n{'='*60}")
        print(f"构建 {idx_cfg['name']} ({idx_cfg['index']}) PIT 宇宙")
        print(f"{'='*60}")

        universe = StockUniverse(idx_cfg)
        count = universe.build_from_baostock(start_date, end_date)
        if count > 0:
            print(f"[Universe] {idx_cfg['name']}: 成功构建 {count} 个月份")
            print(f"[Universe] 质量分布: {universe.quality_summary}")
            # 验证不同月份的成分股确实不同
            keys = sorted(universe._snapshots.keys())
            if len(keys) >= 2:
                s1 = set(universe._snapshots[keys[0]])
                s2 = set(universe._snapshots[keys[-1]])
                overlap = len(s1 & s2)
                print(f"[Universe] 验证: {keys[0]} vs {keys[-1]} "
                      f"重叠 {overlap}/{len(s1)} "
                      f"({'真实PIT' if overlap < len(s1) else '警告: 成分股相同!'})")
        else:
            print(f"[Universe] {idx_cfg['name']}: 构建失败!")


if __name__ == "__main__":
    build_all_universes()
