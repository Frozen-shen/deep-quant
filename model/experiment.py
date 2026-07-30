"""
实验记账 — 每次 pipeline run 都落盘

用法 (pipeline 内部自动调用):
  from model.experiment import log_experiment
  log_experiment("dev", config, results, metrics)
"""
import os, json, hashlib, yaml
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(BASE_DIR, "experiment_log.json")


def log_experiment(mode: str, config: dict, results: list,
                   window_metrics: list = None):
    """
    记录一次实验。

    Args:
      mode: "dev" | "blind"
      config: 完整 config dict
      results: pipeline 返回的窗口结果列表
      window_metrics: 每窗口详细指标 (可选)
    """
    # config hash
    config_str = json.dumps(config, sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]

    # git commit
    git_commit = ""
    try:
        git_commit = os.popen(f"cd {BASE_DIR} && git rev-parse HEAD").read().strip()[:12]
    except: pass

    # 汇总指标
    rets = [r["total_return"] for r in results] if results else []
    excesses = [r.get("excess", 0) for r in results] if results else []

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "config_hash": config_hash,
        "git_commit": git_commit,
        "n_windows": len(rets),
        "mean_return": round(sum(rets) / len(rets), 2) if rets else 0,
        "mean_excess": round(sum(excesses) / len(excesses), 2) if excesses else 0,
        "pos_windows": sum(1 for r in rets if r > 0) if rets else 0,
        "pos_excess": sum(1 for e in excesses if e > 0) if excesses else 0,
        "per_window": results,
    }

    # 追加到日志
    log = []
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH) as f:
                log = json.load(f)
        except: pass

    log.append(entry)

    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  [实验记账] {mode} | config={config_hash} | "
          f"ret={entry['mean_return']:+.1f}% | excess={entry['mean_excess']:+.1f}% | "
          f"#{len(log)} logged to {os.path.basename(LOG_PATH)}")
