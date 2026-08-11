"""
scripts/active/check_param_passthrough.py — 参数透传完整性检查 (防再犯)

背景: 2026-08-09~11 期间, weight_mode / vol_target_cfg / trend_timing_cfg
三个实验参数在 main → run_fold_analysis → run_backtest 链路中出现 3 次透传遗漏
(中间层签名/调用漏改, 参数被静默吞掉退化为默认值, 实验结论失真)。

本脚本用 AST 静态分析 run_walkforward_backtest.py:
  1. 关键参数必须在相关函数签名中声明 (带默认值允许, 但调用点必须显式传入)
  2. 对每个"签名含关键参数"的函数, 检查所有调用点是否传入了全部关键参数
  3. 发现缺口时打印调用点行号并退出码非 0 (CI/冒烟门禁用)

用法:
  py scripts/active/check_param_passthrough.py [--file 目标文件] [--params p1,p2]
  py scripts/active/check_param_passthrough.py --silent   # 仅退出码

冒烟门控: DEVELOPMENT_RUNBOOK.md 第3步 (每日回测前运行, 退出码 0 才继续)。
"""

import argparse
import ast
import sys

DEFAULT_TARGET = "scripts/active/run_walkforward_backtest.py"

# 关键链路参数 (新加实验参数时必须登记到这里)
DEFAULT_KEY_PARAMS = [
    "weight_mode",        # 组合加权模式 (equal/inv_vol/risk_parity)
    "pool_filter_cfg",    # 股票池分域
    "vol_target_cfg",     # 波动率目标仓位
    "trend_timing_cfg",   # 趋势择时
]

# 必须完整透传的函数 (签名含关键参数即视为"关键函数")
# 函数名可带默认值 (None/等), 但调用点不允许静默省略 → 必须显式传 config 值
TRACKED_FUNCS = [
    "run_backtest",
    "run_fold_analysis",
    "run_fold_test",
    "main",
]

# main 是 CLI 入口: 从 config.yaml 读取参数并向下传, 自身不接收外部传入,
# 因此豁免"签名必须声明关键参数"检查 (但它的调用点仍会被检查)。
SIG_CHECK_EXEMPT = {"main"}


def analyze(path: str, key_params: list) -> tuple:
    """
    返回 (缺口列表, 检查函数数, 检查调用点数)。
    缺口: (调用点行号, 被调函数, 缺失参数列表)
    """
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)

    # 收集函数签名: {name: set(param_names)}
    sigs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            sigs[node.name] = {a.arg for a in node.args.args}

    # 收集所有调用点: [(行号, 被调函数名, 传入的关键参数set)]
    call_sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = None
        if isinstance(node.func, ast.Name):
            fname = node.func.id
        elif isinstance(node.func, ast.Attribute):
            fname = node.func.attr
        if fname is None:
            continue
        kwargs = {kw.arg for kw in node.keywords if kw.arg}
        call_sites.append((node.lineno, fname, kwargs))

    gaps = []
    checked_calls = 0
    for lineno, fname, kwargs in call_sites:
        if fname not in TRACKED_FUNCS:
            continue
        sig = sigs.get(fname, set())
        # 该函数签名中有哪些关键参数
        has = sig & set(key_params)
        if not has:
            continue  # 函数不涉及关键参数, 无需检查
        checked_calls += 1
        missing = has - kwargs
        if missing:
            gaps.append((lineno, fname, sorted(missing)))

    # 额外检查: 关键函数签名本身是否声明了所有关键参数
    sig_gaps = []
    for fn in TRACKED_FUNCS:
        if fn in SIG_CHECK_EXEMPT:
            continue
        sig = sigs.get(fn, set())
        miss_in_sig = set(key_params) - sig
        if miss_in_sig:
            sig_gaps.append((fn, sorted(miss_in_sig)))

    return gaps, sig_gaps, checked_calls


def main():
    parser = argparse.ArgumentParser(description="参数透传完整性检查")
    parser.add_argument("--file", default=DEFAULT_TARGET, help="目标脚本路径")
    parser.add_argument("--params", default=",".join(DEFAULT_KEY_PARAMS),
                        help="关键参数列表 (逗号分隔)")
    parser.add_argument("--silent", action="store_true", help="仅退出码, 不输出")
    args = parser.parse_args()

    key_params = [p.strip() for p in args.params.split(",") if p.strip()]
    gaps, sig_gaps, checked = analyze(args.file, key_params)

    if args.silent:
        sys.exit(1 if (gaps or sig_gaps) else 0)

    print(f"═══ 参数透传完整性检查 ═══")
    print(f"  目标: {args.file}")
    print(f"  关键参数: {', '.join(key_params)}")
    print(f"  检查调用点: {checked} 处")
    print()

    if sig_gaps:
        print("❌ 签名缺口 (函数签名未声明关键参数):")
        for fn, miss in sig_gaps:
            print(f"    {fn} 缺: {', '.join(miss)}")
    if gaps:
        print("❌ 调用点缺口 (签名有但调用未传, 会静默退化为默认值):")
        for lineno, fname, missing in gaps:
            print(f"    行 {lineno}: {fname} 未传: {', '.join(missing)}")
    if not gaps and not sig_gaps:
        print("✅ 全链路透传完整, 无缺口")
        print("   (main → run_fold_analysis/run_fold_test → run_backtest)")
        print("   (main → run_backtest 非fold路径)")
    print()
    sys.exit(1 if (gaps or sig_gaps) else 0)


if __name__ == "__main__":
    main()
