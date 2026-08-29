"""验证 fills 时间修复 (datetime 列)"""
import sys
sys.path.insert(0, r"C:\Users\Frozen\ZCodeProject\quant-starter")
from data.minute_fetcher import MinuteFetcher

mf = MinuteFetcher()
res = mf.get_pov_fills("600519", "2026-08-07", 300)
print("n_fills:", res["n_fills"])
print("前3:", res["fills"][:3])
print("末2:", res["fills"][-2:])
print("price:", round(res["price"], 2))
