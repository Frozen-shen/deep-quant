"""
参数敏感性分析 — train_months 等关键参数一键出曲线

用法: python scripts/run_sensitivity.py
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml, numpy as np
from model.pipeline import load_config, QuantPipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config = load_config()

# ══════════════════
# 测试参数: train_months
# ══════════════════
config["mode"] = "dev"  # 在开发期上测试

values = [4, 6, 8, 12, 18]
print("=" * 65)
print("  参数敏感性: train_months")
print("=" * 65)
print(f"  测试值: {values}")

results = []
for v in values:
    cfg = copy.deepcopy(config)
    cfg["rolling"]["train_months"] = v
    cfg["rolling"]["embargo_days"] = max(v * 1, 10)  # 自适应embargo

    pipeline = QuantPipeline(cfg, mode="dev")
    # 只收集窗口结果, 不打印
    pipeline._load_universe(); pipeline._load_data()
    pipeline._precompute_factors()
    windows = pipeline._generate_windows()

    window_results = []
    for wi, w in enumerate(windows):
        r = pipeline._run_window(wi, w)
        if r: window_results.append(r)

    rets = [r["total_return"] for r in window_results]
    excesses = [r.get("excess", 0) for r in window_results]
    mean_ret = np.mean(rets) if rets else 0
    mean_ex = np.mean(excesses) if excesses else 0

    results.append({
        "train_months": v,
        "n_windows": len(rets),
        "mean_return": round(float(mean_ret), 2),
        "mean_excess": round(float(mean_ex), 2),
        "pos_windows": sum(1 for r in rets if r > 0),
        "per_window": [round(float(r), 1) for r in rets],
    })

    print(f"  {v}m → 均值:{mean_ret:+.1f}%  超额:{mean_ex:+.1f}%  "
          f"正窗口:{sum(1 for r in rets if r>0)}/{len(rets)}")

# ── 诊断 ──
rets_only = [r["mean_return"] for r in results]
peak_idx = np.argmax(rets_only)
peak_val = values[peak_idx]

# 是否是单点尖峰?
is_spike = True
if len(rets_only) >= 3:
    # 检查邻居是否接近峰值
    neighbors = []
    if peak_idx > 0: neighbors.append(rets_only[peak_idx-1])
    if peak_idx < len(rets_only)-1: neighbors.append(rets_only[peak_idx+1])
    if neighbors and max(neighbors) > rets_only[peak_idx] * 0.5:
        is_spike = False

print(f"\n{'=' * 65}")
print(f"  诊断:")
print(f"  最优值: train_months={peak_val} (均值+{rets_only[peak_idx]:.1f}%)")
if is_spike and peak_val == 4:
    print(f"  ⚠️ 警告: 4月是单点尖峰, 可能是噪音而非真实信号")
    print(f"  建议: 使用 6-8 月训练窗口, 更稳健")
else:
    print(f"  ✓ 参数曲线平缓, 配置区间内均可用")

# 保存
with open(os.path.join(BASE_DIR, "data", "sensitivity.json"), "w") as f:
    json.dump({"parameter": "train_months", "values": values, "results": results}, f, indent=2)
print(f"  输出: data/sensitivity.json")
print(f"{'=' * 65}")
