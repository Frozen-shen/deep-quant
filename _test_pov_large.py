"""验证: 大订单走 POV 拆单 (多时段), 小订单走 VWAP"""
import sys
sys.path.insert(0, r"C:\Users\Frozen\ZCodeProject\quant-starter")
from data.minute_fetcher import MinuteFetcher

mf = MinuteFetcher()

# 大订单 (茅台日成交约 300 万股, 50 万股占比 16% → 拆单)
r = mf.get_pov_fills("600519", "2026-08-07", 500000)
print("大订单 500000股:")
print("  n_fills:", r["n_fills"])
print("  price:", round(r["price"], 2))
print("  前3:", r["fills"][:3])
print("  末2:", r["fills"][-2:])

# 中订单 (1 万股占比 0.3% → 拆单)
r2 = mf.get_pov_fills("600519", "2026-08-07", 10000)
print("\n中订单 10000股:")
print("  n_fills:", r2["n_fills"], "| price:", round(r2["price"], 2))
