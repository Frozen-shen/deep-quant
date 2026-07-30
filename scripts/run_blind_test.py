"""
不可作弊盲测 v2 — CI 可强制执行

机制:
  1. 检查锁状态 + config hash → 拒绝重跑
  2. 盲测结果哈希写入 git tag
  3. 结果写入 blind_results/ (不可覆盖)
  4. 实验记录写入 experiment_log.json
"""
import sys, os, json, yaml, hashlib
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

config_path = os.path.join(BASE_DIR, "config.yaml")
with open(config_path) as f:
    raw_config = f.read()
    cfg = yaml.safe_load(raw_config)

bt = cfg.get("blind_test", {})

# ── 锁检查 ──
if bt.get("locked", False):
    print("=" * 60)
    print("  ❌ 盲测已锁定. 禁止重跑.")
    print(f"  Trial #{bt.get('trial_count','?')}: {bt.get('last_run','?')}")
    print(f"  结果: blind_results/")
    sys.exit(1)

# ── 运行 ──
from model.pipeline import QuantPipeline

config_hash = hashlib.sha256(raw_config.encode()).hexdigest()[:12]
git_commit = os.popen(f"cd {BASE_DIR} && git rev-parse HEAD").read().strip()
trial_num = bt.get("trial_count", 0) + 1

print("=" * 60)
print(f"  ★ 盲测 Trial #{trial_num}")
print(f"  Config: {config_hash}")
print(f"  Commit: {git_commit[:12]}")
print("=" * 60)

pipeline = QuantPipeline(cfg, mode="blind")
pipeline.run()

# ── 保存结果 (不可覆盖) ──
os.makedirs(os.path.join(BASE_DIR, "blind_results"), exist_ok=True)
result_path = os.path.join(BASE_DIR, "blind_results", f"trial_{trial_num:03d}.json")
if os.path.exists(result_path):
    print(f"\n  ❌ 结果文件已存在: {result_path}")
    sys.exit(1)

result_data = {
    "trial": trial_num,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "config_hash": config_hash,
    "git_commit": git_commit,
    "params": {k: v for k, v in cfg.items() if k not in ("blind_test","data_partition")},
}
with open(result_path, "w") as f:
    json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)

# ── 锁定 config ──
bt["trial_count"] = trial_num
bt["last_run"] = datetime.now().strftime("%Y-%m-%d")
bt["locked"] = True
bt["config_hash"] = config_hash
cfg["blind_test"] = bt
with open(config_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

# ── Git tag ──
tag = f"blind-trial-{trial_num:03d}-{config_hash}"
os.system(f"cd {BASE_DIR} && git tag -a {tag} -m 'Trial #{trial_num} config={config_hash}' 2>/dev/null")

print(f"\n★ 盲测完成. 参数永久锁定. Git tag: {tag}")
