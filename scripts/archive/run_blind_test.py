"""
不可作弊盲测 v3 — 独立锁 + 物理递增编号 + 数字白卷终结

机制:
  1. blind_lock.yaml 独立锁 (修改config.yaml无法绕过)
  2. trial编号 = 扫描blind_results/取max+1 (物理上不可重置)
  3. 结果JSON写入实际收益数字 (不再白卷)
  4. ruamel.yaml 写 config 保注释
"""
import sys, os, json, hashlib, glob, re
from datetime import datetime
from ruamel.yaml import YAML

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

yaml_rt = YAML()
yaml_rt.preserve_quotes = True
yaml_rt.width = 120

LOCK_PATH = os.path.join(BASE_DIR, "blind_lock.yaml")
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
RESULTS_DIR = os.path.join(BASE_DIR, "blind_results")

# ═══════════════════════════════════════════════════════
# 1. 独立锁检查 — blind_lock.yaml (config.yaml 改了也没用)
# ═══════════════════════════════════════════════════════
if os.path.exists(LOCK_PATH):
    with open(LOCK_PATH) as f:
        lock = yaml_rt.load(f) or {}
    if lock.get("locked", False):
        print("=" * 60)
        print("  ❌ 盲测已锁定 (blind_lock.yaml). 禁止重跑.")
        print(f"  Trial #{lock.get('trial_count','?')}: {lock.get('last_run','?')}")
        print(f"  {lock.get('message','')}")
        sys.exit(1)

# ═══════════════════════════════════════════════════════
# 2. Trial编号: 扫描已有文件取 max+1 (物理不可重置)
# ═══════════════════════════════════════════════════════
os.makedirs(RESULTS_DIR, exist_ok=True)
existing = glob.glob(os.path.join(RESULTS_DIR, "trial_*.json"))
max_n = 0
for f in existing:
    m = re.search(r'trial_(\d+)\.json', os.path.basename(f))
    if m:
        max_n = max(max_n, int(m.group(1)))
trial_num = max_n + 1

# ═══════════════════════════════════════════════════════
# 3. 读取 config 并运行
# ═══════════════════════════════════════════════════════
with open(CONFIG_PATH) as f:
    raw_config = f.read()
cfg = yaml_rt.load(raw_config)

config_hash = hashlib.sha256(raw_config.encode()).hexdigest()[:12]
git_commit = os.popen(f"cd {BASE_DIR} && git rev-parse HEAD").read().strip()

print("=" * 60)
print(f"  ★ 盲测 Trial #{trial_num}")
print(f"  Config: {config_hash}")
print(f"  Commit: {git_commit[:12]}")
print("=" * 60)

from model.pipeline import QuantPipeline
pipeline = QuantPipeline(cfg, mode="blind")
output = pipeline.run()

# ═══════════════════════════════════════════════════════
# 4. 写入结果 — 带实际收益数字 (终结白卷)
# ═══════════════════════════════════════════════════════
result_path = os.path.join(RESULTS_DIR, f"trial_{trial_num:03d}.json")
if os.path.exists(result_path):
    print(f"\n  ❌ 结果文件已存在: {result_path}")
    print(f"  这不应该发生 (编号取自max+1). 手动检查.")
    sys.exit(1)

summary = output.get("summary", {})
per_window = summary.get("per_window", [])

# 精简 equity_curve (避免JSON过大): 只保留首尾和每月1号
for pw in per_window:
    if "equity_curve" in pw:
        curve = pw["equity_curve"]
        sampled = []
        for i, pt in enumerate(curve):
            if i == 0 or i == len(curve) - 1 or pt.get("date","").endswith("-01"):
                sampled.append(pt)
        pw["equity_curve"] = sampled

result_data = {
    "trial": trial_num,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "config_hash": config_hash,
    "git_commit": git_commit,
    "summary": {
        "strategy_mean_pct": summary.get("mean_return"),
        "excess_mean_pct": summary.get("mean_excess"),
        "n_windows": summary.get("n_windows"),
        "pos_windows": summary.get("pos_windows"),
        "pos_excess": summary.get("pos_excess"),
    },
    "per_window": [
        {
            "window": pw.get("window"),
            "test_start": pw.get("test_start"),
            "test_end": pw.get("test_end"),
            "strategy_return_pct": pw.get("total_return"),
            "benchmark_return_pct": pw.get("benchmark_return"),
            "excess_pct": pw.get("excess"),
            "trades": pw.get("trades"),
            "equity_curve": pw.get("equity_curve"),
        }
        for pw in per_window
    ],
    "params": {k: v for k, v in cfg.items() if k not in ("blind_test", "data_partition")},
}

with open(result_path, "w") as f:
    json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)

# ═══════════════════════════════════════════════════════
# 5. 更新锁和配置 (ruamel.yaml 保注释)
# ═══════════════════════════════════════════════════════
lock_data = {
    "locked": True,
    "trial_count": trial_num,
    "last_trial": trial_num,
    "last_run": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "config_hash": config_hash,
    "message": f"Trial#{trial_num}: 策略{summary.get('mean_return',0):+.1f}% 超额{summary.get('mean_excess',0):+.1f}%",
}
with open(LOCK_PATH, "w") as f:
    yaml_rt.dump(lock_data, f)

# 更新 config.yaml (保持注释)
with open(CONFIG_PATH) as f:
    cfg_doc = yaml_rt.load(f)
if "blind_test" not in cfg_doc:
    cfg_doc["blind_test"] = {}
cfg_doc["blind_test"]["trial_count"] = trial_num
cfg_doc["blind_test"]["last_run"] = datetime.now().strftime("%Y-%m-%d")
cfg_doc["blind_test"]["locked"] = True
cfg_doc["blind_test"]["config_hash"] = config_hash
with open(CONFIG_PATH, "w") as f:
    yaml_rt.dump(cfg_doc, f)

# Git tag
tag = f"blind-trial-{trial_num:03d}-{config_hash}"
os.system(f"cd {BASE_DIR} && git tag -a {tag} -m 'Trial #{trial_num} config={config_hash}' 2>/dev/null")

print(f"\n★ 盲测完成. 参数永久锁定 (blind_lock.yaml + config.yaml).")
print(f"  Git tag: {tag}")
print(f"  结果文件: {result_path}")
