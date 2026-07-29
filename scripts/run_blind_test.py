"""
不可作弊盲测 — 运行后立即锁定, 结果不可覆盖

用法: python scripts/run_blind_test.py

机制:
  1. 检查 config.yaml 的 locked 状态
  2. 若未锁定: 运行盲测, 输出结果到 blind_results/trial_NNN.json
  3. 结果文件带时间戳, 不可覆盖
  4. 自动写入 git tag, 更新 config trial_count
  5. 若已锁定: 拒绝运行
"""
import sys, os, json, yaml
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# ── 检查锁定状态 ──
config_path = os.path.join(BASE_DIR, "config.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

bt = cfg.get("blind_test", {})
if bt.get("locked", False):
    print("=" * 60)
    print("  ❌ 盲测已锁定.")
    print(f"  Trial #{bt.get('trial_count', '?')}: {bt.get('last_run', 'unknown')}")
    print(f"  如有疑问, 查看 blind_results/ 目录或 git log --tags")
    print("=" * 60)
    sys.exit(1)

# ── 运行盲测 ──
from model.pipeline import QuantPipeline

trial_num = bt.get("trial_count", 0) + 1
print("=" * 60)
print(f"  ★ 盲测 Trial #{trial_num}")
print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  ★ 参数冻结. 结果即为最终答案. 不可再调参.")
print("=" * 60)

pipeline = QuantPipeline(cfg, mode="blind")
pipeline.run()

# ── 保存结果 ──
os.makedirs(os.path.join(BASE_DIR, "blind_results"), exist_ok=True)
result_path = os.path.join(BASE_DIR, "blind_results", f"trial_{trial_num:03d}.json")

result_data = {
    "trial": trial_num,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "config": {k: v for k, v in cfg.items() if k != "_comment"},
    "git_commit": os.popen("cd " + BASE_DIR + " && git rev-parse HEAD").read().strip(),
    "status": "locked",
}

# 写入结果 (如果文件已存在则拒绝覆盖)
if os.path.exists(result_path):
    print(f"\n  ❌ 结果文件已存在: {result_path}")
    print(f"  盲测结果不可覆盖. 请检查文件内容.")
    sys.exit(1)

with open(result_path, "w") as f:
    json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)

# ── 锁定 config ──
bt["trial_count"] = trial_num
bt["last_run"] = datetime.now().strftime("%Y-%m-%d")
bt["locked"] = True
cfg["blind_test"] = bt

with open(config_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

# ── Git tag ──
tag = f"blind-trial-{trial_num:03d}"
os.system(f"cd {BASE_DIR} && git tag -a {tag} -m 'Blind test Trial #{trial_num}' 2>/dev/null")

print(f"\n{'=' * 60}")
print(f"  ★ 盲测完成. 参数已永久锁定.")
print(f"  结果: {result_path}")
print(f"  Git tag: {tag}")
print(f"  Config: locked=true, trial_count={trial_num}")
print(f"{'=' * 60}")
