"""
因子工厂 CLI — 命令行入口

用法:
    py -m factor_factory list [--status active] [--category momentum]
    py -m factor_factory validate mom_20d [--period 2018-2022] [--sample 500]
    py -m factor_factory validate --all [--sample 500]
    py -m factor_factory sync   # 从 factor_library.py 同步因子到注册表
    py -m factor_factory info mom_20d
"""

import os
import sys
import json
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


def cmd_list(args):
    """列出注册表中的因子。"""
    from factor_factory.registry import FactorRegistry
    reg = FactorRegistry.load()

    factors = reg.query(
        status=args.status,
        category=args.category,
        source=args.source,
    )

    if not factors:
        print("(空)")
        return

    # 表格输出
    print(f"{'Name':<30} {'Category':<14} {'Source':<14} {'Status':<12} {'ICIR':<8} {'HL(d)':<7}")
    print("-" * 90)
    for f in sorted(factors, key=lambda x: -(abs(x.icir_latest) if x.icir_latest else 0)):
        icir = f"{f.icir_latest:+.3f}" if f.icir_latest else "-"
        hl = f"{f.ic_half_life:.0f}" if f.ic_half_life else "-"
        print(f"{f.name:<30} {f.category:<14} {f.source:<14} {f.status:<12} {icir:<8} {hl:<7}")
    print(f"\n共 {len(factors)} 个因子")


def cmd_validate(args):
    """验证因子。"""
    from factor_factory.validation import validate_factor, validate_batch
    from factor_factory.registry import FactorRegistry

    reg = FactorRegistry.load()

    if args.all:
        # 验证所有 active 价量因子
        active = reg.query(status="active", source="price_volume")
        names = [f.name for f in active]
        exprs = {f.name: f.expr for f in active if f.expr}
        if not names:
            # fallback: 从 factor_library 获取
            from factor_library import get_all_factors
            all_f = get_all_factors()
            names = list(all_f.keys())
            exprs = all_f
        print(f"批量验证 {len(names)} 个因子...")
        period = tuple(args.period.split("-")) if args.period else None
        results = validate_batch(names, exprs=exprs, period=period,
                                 sample=args.sample)
        # 汇总
        n_pass = sum(1 for r in results if r.get("verdict", {}).get("verdict") == "PASS")
        n_marg = sum(1 for r in results if r.get("verdict", {}).get("verdict") == "MARGINAL")
        n_fail = sum(1 for r in results if r.get("verdict", {}).get("verdict") == "FAIL")
        print(f"\n汇总: PASS={n_pass}, MARGINAL={n_marg}, FAIL={n_fail}")
    else:
        # 验证单个因子
        name = args.name
        meta = reg.get(name)
        expr = meta.expr if meta and meta.expr else None

        period = None
        if args.period:
            parts = args.period.split("-")
            if len(parts) == 2:
                period = (parts[0], parts[1])
            elif len(parts) == 4:
                period = (f"{parts[0]}-{parts[1]}", f"{parts[2]}-{parts[3]}")

        print(f"验证因子: {name}")
        report = validate_factor(name, expr=expr, period=period,
                                 sample=args.sample)

        # 打印摘要
        print(f"\n{'='*60}")
        print(f"  因子: {name}")
        print(f"  覆盖率: {report.get('coverage', '?'):.1%}")
        print(f"  验证天数: {report.get('n_dates', '?')}")
        print(f"  IC半衰期: {report.get('half_life_days', '?')} 天")
        print(f"  滚动IC趋势: {report.get('rolling_ic_trend', '?')}")
        print(f"\n  Horizon IC:")
        for h, ic_data in report.get("horizon_ic", {}).items():
            print(f"    {h:>3}d: IC={ic_data.get('ic_mean', '?'):+.5f}  "
                  f"ICIR={ic_data.get('icir', '?'):+.4f}  "
                  f"正比例={ic_data.get('pos_ratio', '?'):.1%}")
        v = report.get("verdict", {})
        print(f"\n  判定: {v.get('verdict', '?')} ({v.get('n_pass', 0)}/4 通过)")
        for check, passed in v.get("checks", {}).items():
            mark = "✓" if passed else "✗"
            print(f"    {mark} {check}")
        print(f"{'='*60}")

        # 更新注册表
        if meta:
            main_ic = report.get("horizon_ic", {}).get(20, {})
            reg.update_ic(name,
                          ic=main_ic.get("ic_mean"),
                          icir=main_ic.get("icir"),
                          half_life=report.get("half_life_days"))
            reg.save()


def cmd_sync(args):
    """从 factor_library.py 同步因子到注册表。"""
    from factor_factory.registry import FactorRegistry
    reg = FactorRegistry.load()
    before = len(reg)
    reg.sync_from_library()
    reg.save()
    print(f"同步完成: {before} → {len(reg)} 个因子")


def cmd_info(args):
    """显示因子详情。"""
    from factor_factory.registry import FactorRegistry
    reg = FactorRegistry.load()
    meta = reg.get(args.name)
    if not meta:
        print(f"因子 '{args.name}' 未注册")
        return
    print(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        prog="factor_factory",
        description="因子研究工厂 — 注册/验证/中性化/报告"
    )
    sub = parser.add_subparsers(dest="command")

    # list
    p_list = sub.add_parser("list", help="列出因子")
    p_list.add_argument("--status", default="active")
    p_list.add_argument("--category", default=None)
    p_list.add_argument("--source", default=None)

    # validate
    p_val = sub.add_parser("validate", help="验证因子")
    p_val.add_argument("name", nargs="?", default=None)
    p_val.add_argument("--all", action="store_true", help="验证所有active因子")
    p_val.add_argument("--period", default=None, help="验证期间 (如 2018-01-01-2022-12-31)")
    p_val.add_argument("--sample", type=int, default=None, help="抽样股票数")

    # sync
    sub.add_parser("sync", help="从factor_library同步")

    # info
    p_info = sub.add_parser("info", help="因子详情")
    p_info.add_argument("name")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "sync":
        cmd_sync(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
