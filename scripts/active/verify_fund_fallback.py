"""快速验证: fetch_financials 旧缓存回退 + _fund_report_factors 消费"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "active"))

import pandas as pd
from data.fundamental import fetch_financials
from run_walkforward_backtest import _fund_report_factors

# 1) 000001: 在 fundamental_cache (akshare 中文列名) 中 → 主路径
df1 = fetch_financials("000001")
assert df1 is not None and "日期" in df1.columns, "000001 主缓存读取失败"
print(f"[1] 000001 主缓存: {len(df1)} 行, 列含 日期/净资产收益率(%): "
      f"{'净资产收益率(%)' in df1.columns}")

# 2) 000002: 仅在旧缓存 (英文列名) → 回退 + 列名转换 + 回写
df2 = fetch_financials("000002")
assert df2 is not None, "000002 旧缓存回退失败"
assert "日期" in df2.columns and "净资产收益率(%)" in df2.columns, "000002 列名未转换"
assert not any(c in df2.columns for c in
               ("report_date", "roe", "profit_growth_deducted", "revenue", "net_profit")), \
    "英文列/语义不符列残留"
assert pd.api.types.is_datetime64_any_dtype(df2["日期"]), "日期列应为 datetime"
print(f"[2] 000002 旧缓存回退: {len(df2)} 行 ({df2['日期'].min().date()} ~ "
      f"{df2['日期'].max().date()}), 列: {df2.columns.tolist()}")

# 3) _fund_report_factors 能正常消费转换后的数据
factors = _fund_report_factors(df2)
assert factors is not None and len(factors) > 0, "000002 因子计算失败"
assert "roe" in factors.columns and "eps_ttm" in factors.columns, "核心因子缺失"
# 未映射字段 (扣非净利润绝对额) → profit_growth_ded 不应出现 (宁缺勿错)
print(f"[3] _fund_report_factors(000002): {len(factors)} 期, 因子列: "
      f"{factors.columns.tolist()}")
assert "profit_growth_ded" not in factors.columns, "profit_growth_ded 不应由旧缓存产生"

# 4) 000001 走 _fund_report_factors (akshare 全字段, 含扣非绝对额)
f1 = _fund_report_factors(df1)
assert f1 is not None and "profit_growth_ded" in f1.columns, "akshare 扣非因子缺失"
print(f"[4] _fund_report_factors(000001): profit_growth_ded 正常 = "
      f"{f1['profit_growth_ded'].notna().sum()} 期非空")

# 5) 覆盖统计 (相对 data_cache 股票列表)
from data_cache import get_cached_symbols
syms = get_cached_symbols()
covered = [s for s in syms if fetch_financials(s) is not None]
print(f"[5] 覆盖: {len(covered)}/{len(syms)} 只 (原 424, 新增 {len(covered)-424})")
assert len(covered) > 2000, "覆盖数异常偏低"
print("\nALL CHECKS PASSED")
