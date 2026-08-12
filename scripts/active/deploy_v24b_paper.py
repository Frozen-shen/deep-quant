"""
scripts/active/deploy_v24b_paper.py — 部署 v24b 最优权重到模拟盘

将 walkforward_results_v24b_vwap.json 的 stable_factors + 中位数ICIR
落成 p5_portfolio_report.json 格式 (run_paper_signal.py 读取), 替换旧的
2026-08-03 P5 报告 (39 因子, SKIP)。

分类规则 (与 run_paper_signal.py 的分组一致):
  - price_volume: 价量类 (默认)
  - fundamental:  fund_ 前缀
  - relative:     relative 前缀
  - minute:       min_ 前缀
  - aux:          aux_ 前缀 → price_volume 分组 (回测中按截面z-score处理)

用法:
  py scripts/active/deploy_v24b_paper.py            # 生成并替换
  py scripts/active/deploy_v24b_paper.py --dry-run  # 只打印不写入
"""
import argparse
import json
import os
import shutil
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IC_DIR = os.path.join(BASE_DIR, "data", "ic_validation")
SRC = os.path.join(IC_DIR, "walkforward_results_v24b_vwap.json")
DST = os.path.join(IC_DIR, "p5_portfolio_report.json")


def classify(name: str) -> str:
    if name.startswith("fund_"):
        return "fundamental"
    if name.startswith("relative_"):
        return "relative"
    if name.startswith("min_"):
        return "minute"
    return "price_volume"  # aux_* 与价量同组


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(SRC):
        print(f"❌ 源文件不存在: {SRC}")
        return 1
    v24 = json.load(open(SRC, encoding="utf-8"))
    meta = v24.get("meta", {})
    stable = meta.get("stable_factors", [])
    icir = meta.get("stable_factor_icir_median", {})
    if not stable:
        print("❌ v24b meta 无 stable_factors")
        return 1

    factors = []
    for name in stable:
        factors.append({
            "name": name,
            "icir": icir.get(name, 0.0),
            "category": classify(name),
            "weight_multiplier": 1.0,
        })

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "walkforward_results_v24b_vwap.json",
        "verdict": "v24b-prod",
        "note": ("v24b (VWAP执行层) 生产权重: fold均值+10.10%, EXTEND+2.1%, "
                 "Sharpe 1.26; 替换 2026-08-03 p5 (SKIP)"),
        "selected_factors": factors,
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "v24b 最优配置部署 (VWAP执行+10bps残差, pool_filter+vol_target, 15m分钟因子)",
            "bt_config": meta.get("bt_config", {}),
            "thresholds": {"fold_min_hits": 3, "fold_t_stat_min": 1.645},
        },
    }

    # 分类统计
    cats = {}
    for f in factors:
        cats[f["category"]] = cats.get(f["category"], 0) + 1
    print(f"因子数: {len(factors)} | 分类: {cats}")
    print(f"ICIR 范围: {min(f['icir'] for f in factors):+.3f} ~ "
          f"{max(f['icir'] for f in factors):+.3f}")
    print(f"前5: {[(f['name'], round(f['icir'],3)) for f in factors[:5]]}")

    if args.dry_run:
        print("\n[dry-run] 未写入")
        return 0

    # 备份旧报告
    if os.path.exists(DST):
        bak = DST.replace(".json", f"_bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        shutil.copy2(DST, bak)
        print(f"已备份旧报告: {bak}")

    with open(DST, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"✅ 已写入: {DST}")
    print("模拟盘下次运行将使用 v24b 权重 (run_paper_signal.py 读取此文件)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
