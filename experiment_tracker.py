"""
experiment_tracker.py — 自动实验追踪

每次回测/实验运行后调用 log_experiment() 自动记录。
记录存储到 experiments/ 目录，每个实验一个 JSON 文件。
"""

import hashlib
import json
import itertools
from datetime import datetime
from pathlib import Path

_counter = itertools.count()


def log_experiment(
    script_name: str,
    partition: str,
    config: dict,
    results: dict,
    notes: str = "",
    experiments_dir: str | None = None,
) -> str:
    """记录一次实验到 experiments/ 目录。

    Args:
        script_name: 脚本名称
        partition: 数据分区 (train/test/val)
        config: 实验参数字典
        results: 回测结果字典
        notes: 备注
        experiments_dir: 实验目录路径，默认 "experiments"

    Returns:
        exp_id: 实验唯一标识
    """
    dir_path = Path(experiments_dir) if experiments_dir else Path("experiments")
    dir_path.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    seq = next(_counter)
    exp_id = f"exp_{now:%Y%m%d_%H%M%S}_{(id(config) + seq) % 10000:04d}"

    config_json = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_json.encode()).hexdigest()[:12]

    record = {
        "experiment_id": exp_id,
        "timestamp": now.isoformat(),
        "script": script_name,
        "partition": partition,
        "config_hash": config_hash,
        "parameters": config,
        "results": results,
        "notes": notes,
    }

    filepath = dir_path / f"{exp_id}.json"
    filepath.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    return exp_id


def list_experiments(experiments_dir: str | None = None) -> list:
    """列出所有实验记录。

    Args:
        experiments_dir: 实验目录路径，默认 "experiments"

    Returns:
        按文件名排序的实验记录列表
    """
    dir_path = Path(experiments_dir) if experiments_dir else Path("experiments")
    if not dir_path.exists():
        return []

    records = []
    for f in sorted(dir_path.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        records.append(data)

    return records
