"""单测: POV fills 记录到交易"""
import sys
sys.path.insert(0, r"C:\Users\Frozen\ZCodeProject\quant-starter")
import pandas as pd
from model.engine import SimpleBacktest
from data.minute_fetcher import MinuteFetcher

# 1. get_pov_fills 返回时段明细
mf = MinuteFetcher()
res = mf.get_pov_fills("600519", "2026-08-07", 300)
print("600519 qty=300 fills:")
print("  price:", res["price"] if res else None)
print("  n_fills:", res["n_fills"] if res else None)
if res:
    print("  前3 fills:", res["fills"][:3])
    print("  末3 fills:", res["fills"][-3:])

# 2. engine 买入记录 fill_times
df = pd.DataFrame({
    "date": pd.to_datetime(["2026-08-06", "2026-08-07"]),
    "open": [1300.0, 1306.0], "high": [1310.0, 1315.0],
    "low": [1295.0, 1295.0], "close": [1305.0, 1310.0],
    "volume": [100000, 100000]})
all_data = {"600519": df}
bt = SimpleBacktest(initial_capital=1000000, top_k=5, lot_size=100,
                    slippage_bps=10, execution_price="pov")
bt.vwap_panel = {}
decision = {"buy": ["600519"], "sell": [], "weights": None, "cash_scale": 1.0}
buys, sells, trades = bt.execute(decision, pd.Timestamp("2026-08-07"), all_data, None)
for t in trades:
    print("\n交易:", t["action"], t["symbol"], "qty", t["qty"], "px", round(t["price"], 2))
    print("  fill_times:", t.get("fill_times"))
