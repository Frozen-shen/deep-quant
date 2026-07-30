"""
T+3~T+15 全曲线 horizon 扫描 — 高原 vs 尖峰检测

目的: 判断 T+7 是真正的预测窗口甜点(平滑高原)还是噪音尖峰。
高原定义: 最优值 ±1 范围内的相邻值至少 2 个以上。
尖峰定义: 最优值明显孤立, 相邻值急剧下降。
"""
import sys, os, json, copy
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from model.pipeline import load_config, QuantPipeline

HORIZONS = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]

def run_sweep():
    config = load_config()

    results = []
    print("=" * 70)
    print("  Horizon 全曲线扫描: T+{} ~ T+{}".format(HORIZONS[0], HORIZONS[-1]))
    print("=" * 70)

    for h in HORIZONS:
        cfg = copy.deepcopy(config)
        cfg["label"]["horizon_days"] = h
        # embargo = horizon * 2, 最少 10 天
        cfg["rolling"]["embargo_days"] = max(h * 2, 10)

        print(f"\n── T+{h}d (embargo={cfg['rolling']['embargo_days']}d) ──")

        pipeline = QuantPipeline(cfg, mode="dev")
        pipeline._load_universe()
        pipeline._load_data()
        pipeline._precompute_factors()
        windows = pipeline._generate_windows()

        per_window = []
        for wi, w in enumerate(windows):
            r = pipeline._run_window(wi, w)
            if r:
                per_window.append({
                    "window": r["window"],
                    "test_start": r.get("test_start", ""),
                    "test_end": r.get("test_end", ""),
                    "total_return": r["total_return"],
                    "excess": r["excess"],
                    "trades": r["trades"],
                })

        if per_window:
            mean_ret = float(np.mean([pw["total_return"] for pw in per_window]))
            mean_excess = float(np.mean([pw["excess"] for pw in per_window]))
            pos = sum(1 for pw in per_window if pw["excess"] > 0)
        else:
            mean_ret = mean_excess = pos = 0

        print(f"  均值 策略:{mean_ret:+.1f}%  超额:{mean_excess:+.1f}%  正超额:{pos}/{len(per_window)}")

        results.append({
            "horizon": h,
            "embargo": cfg["rolling"]["embargo_days"],
            "mean_return": round(mean_ret, 2),
            "mean_excess": round(mean_excess, 2),
            "n_windows": len(per_window),
            "pos_excess": pos,
            "per_window": per_window,
        })

    # ── 高原检测 ──
    excesses = [r["mean_excess"] for r in results]
    best_idx = int(np.argmax(excesses))
    best_h = results[best_idx]["horizon"]
    best_excess = results[best_idx]["mean_excess"]

    # 最优值 ±1 范围内的相邻值
    plateau_count = 0
    plateau_horizons = []
    for r in results:
        if abs(r["mean_excess"] - best_excess) <= 1.0:
            plateau_count += 1
            plateau_horizons.append(r["horizon"])

    is_plateau = plateau_count >= 3  # 至少3个值在高原上

    print("\n" + "=" * 70)
    print("  高原检测结果")
    print("=" * 70)
    print(f"  最优: T+{best_h}d  超额:{best_excess:+.2f}%")
    print(f"  高原区间: {plateau_horizons}")
    print(f"  高原点数: {plateau_count}/{len(results)}")
    print(f"  判断: {'✅ 高原 — T+{}/{} 在平滑区间内'.format(best_h, '/'.join(str(h) for h in plateau_horizons)) if is_plateau else '❌ 尖峰 — T+{} 是孤立最优值，相邻急剧下降'.format(best_h)}")

    # ── 输出表格 ──
    print(f"\n{'Horizon':>8s} {'Embargo':>8s} {'策略均值':>10s} {'超额均值':>10s} {'正超额':>8s}")
    print("-" * 52)
    for r in results:
        marker = " ←" if r["horizon"] == best_h else ""
        print(f"  T+{r['horizon']:>3d}  {r['embargo']:>5d}d  {r['mean_return']:>+8.2f}%  {r['mean_excess']:>+8.2f}%  "
              f"{r['pos_excess']}/{r['n_windows']}{marker}")

    # ── 保存 ──
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "horizons_tested": HORIZONS,
        "best": {"horizon": best_h, "mean_excess": best_excess},
        "plateau_detected": is_plateau,
        "plateau_horizons": plateau_horizons,
        "plateau_count": plateau_count,
        "results": results,
    }
    out_path = os.path.join(BASE_DIR, "data", "horizon_sweep.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")

    return output


if __name__ == "__main__":
    run_sweep()
