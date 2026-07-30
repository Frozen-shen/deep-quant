"""
模型基线对比 — L0(等权) vs L1(单因子) vs L2(线性) vs L3(LightGBM)

每个模型在 dev 期跑一次回测, 输出对比报告。
"""
import sys, os, json, copy
from datetime import datetime
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from model.pipeline import load_config, QuantPipeline

MODELS = ["l0", "l1", "l2", "lgb"]
MODEL_NAMES = {"l0": "L0 等权", "l1": "L1 单因子", "l2": "L2 Ridge线性", "lgb": "L3 LightGBM"}


def run_comparison():
    config = load_config()
    results = []

    print("=" * 70)
    print("  模型基线对比: L0 → L1 → L2 → L3")
    print("=" * 70)

    for mtype in MODELS:
        cfg = copy.deepcopy(config)
        cfg["model"]["type"] = mtype

        print(f"\n── {MODEL_NAMES[mtype]} ──")

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
            rets = [pw["total_return"] for pw in per_window]
            excesses = [pw["excess"] for pw in per_window]
            mean_ret = float(np.mean(rets))
            mean_excess = float(np.mean(excesses))
            std_excess = float(np.std(excesses, ddof=1)) if len(excesses) > 1 else 0.0
            ir = mean_excess / std_excess if std_excess > 0 else 0.0
            pos = sum(1 for e in excesses if e > 0)
        else:
            mean_ret = mean_excess = std_excess = ir = pos = 0

        results.append({
            "model": mtype,
            "name": MODEL_NAMES[mtype],
            "mean_return": round(mean_ret, 2),
            "mean_excess": round(mean_excess, 2),
            "excess_std": round(std_excess, 2),
            "ir": round(ir, 3),
            "n_windows": len(per_window),
            "pos_excess": pos,
            "per_window": per_window,
        })

        print(f"  策略:{mean_ret:+.1f}%  超额:{mean_excess:+.1f}%  IR:{ir:.3f}  正超额:{pos}/{len(per_window)}")

    # ── 输出对比表 ──
    print("\n" + "=" * 70)
    print("  对比汇总")
    print("=" * 70)
    print(f"  {'模型':<16s} {'策略均值':>10s} {'超额均值':>10s} {'超额std':>10s} {'IR':>8s} {'正超额':>8s}")
    print("  " + "-" * 68)
    lgb = next((r for r in results if r["model"] == "lgb"), None)
    for r in results:
        vs_lgb = ""
        if lgb and r["model"] != "lgb":
            delta = r["mean_excess"] - lgb["mean_excess"]
            vs_lgb = f"  vs L3 {delta:+.1f}%"
        print(f"  {r['name']:<16s} {r['mean_return']:>+8.2f}%  {r['mean_excess']:>+8.2f}%  "
              f"{r['excess_std']:>8.2f}%  {r['ir']:>+8.3f}  {r['pos_excess']}/{r['n_windows']}{vs_lgb}")

    # ── 保存 ──
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    out_path = os.path.join(BASE_DIR, "data", "baseline_comparison.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")
    return output


if __name__ == "__main__":
    run_comparison()
