"""
_morning_resume.py — 明早自动续跑组合任务 (2026-08-09 深夜诊断结论后设计)

背景: 深夜(23:00+)东财/cheapproxy 网关对长进程批量请求风控异常(空返回/重试卡死),
      手动短进程稳定成功。策略: 白天窗口 + 分批小进程。

执行顺序 (全部断点续传, 失败不中断):
  1. 日线增量 (代理, 2域名) → 07-31 补到 08-07
  2. 未复权 (代理, 2域名) → 补全 5076 只
  3. 15m 增量 (代理, 2域名) → 旧文件 07-31 补到 08-07 (分钟叠加层生产依赖)
  4. 股东户数续跑 (代理) → 已有 ~1462 只, 续拉剩余
  5. 北向续跑 (代理) → 已有 ~2042 只, 续拉剩余

用法: py _morning_resume.py          # 全部执行
      py _morning_resume.py --step 1  # 只跑第 1 步
"""
import os
import sys
import time
import subprocess
import argparse

BASE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("日线增量", "py", ["_daily_proxy_update.py"]),
    ("未复权", "py", ["_unadj_proxy.py"]),
    ("15m增量", "py", ["_min15_fetch.py"]),
    ("股东户数续跑", "py", ["_gdhs_fetch.py"]),
    ("北向续跑", "py", ["_northbound_fetch.py"]),
]


def run_step(name: str, cmd: list) -> bool:
    print(f"\n{'='*60}\n▶ {name}: {' '.join(cmd)}\n{'='*60}", flush=True)
    try:
        r = subprocess.run(cmd, cwd=BASE, timeout=4 * 3600)
        ok = r.returncode == 0
        print(f"▶ {name} 结束: {'✅' if ok else '❌'} exit={r.returncode}", flush=True)
        return ok
    except subprocess.TimeoutExpired:
        print(f"▶ {name} 超时", flush=True)
        return False
    except Exception as e:
        print(f"▶ {name} 启动失败: {e}", flush=True)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=None, help="只跑指定步骤(1-5)")
    args = parser.parse_args()

    if args.step:
        name, _, cmd = STEPS[args.step - 1]
        run_step(name, ["py"] + cmd)
    else:
        for i, (name, _, cmd) in enumerate(STEPS, 1):
            print(f"\n[{i}/{len(STEPS)}] {name}", flush=True)
            run_step(name, ["py"] + cmd)
            time.sleep(10)  # 步骤间冷却, 避免网关误判


if __name__ == "__main__":
    main()
