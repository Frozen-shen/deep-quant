"""
gate.py — 量化开发硬门禁

所有回测/实验脚本必须在开头调用:
    from gate import check, load_config
    config = load_config()
    check(partition="research", script_name=__file__, config=config)

不通过 → 抛出 GateViolation，脚本终止。
"""

import re
from collections import Counter
from pathlib import Path

import yaml


class GateViolation(Exception):
    """Raised when a development discipline gate check fails."""
    pass


def load_config(config_path="config.yaml"):
    """Load YAML config file. Raises GateViolation if file is missing."""
    path = Path(config_path)
    if not path.exists():
        raise GateViolation(f"配置文件不存在: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config_integrity(config, raw_yaml=None):
    """Detect duplicate YAML keys by parsing raw text.

    Returns a list of error strings (empty if no issues found).
    """
    errors = []
    if raw_yaml is None:
        return errors

    # Track keys per indentation level to detect duplicates within same block
    # Simple approach: find all key occurrences and flag duplicates
    # within the same parent block.
    lines = raw_yaml.split("\n")
    # Group keys by their indentation context
    # We track (indent, parent_context) -> list of keys seen
    seen_keys = {}  # (indent_level, parent_key) -> Counter of child keys

    current_parent = {}  # indent_level -> key at that level

    for line in lines:
        if not line.strip() or line.strip().startswith("#"):
            continue

        # Calculate indentation
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Extract key if this is a key line
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):", stripped)
        if not match:
            continue

        key = match.group(1)

        # Determine parent: the nearest key at a lower indent level
        # Update current parent tracking
        current_parent[indent] = key
        # Remove deeper levels
        for k in list(current_parent.keys()):
            if k > indent:
                del current_parent[k]

        # Find parent key (nearest lower indent)
        parent_indent = None
        for k in sorted(current_parent.keys(), reverse=True):
            if k < indent:
                parent_indent = k
                break

        parent_key = current_parent.get(parent_indent, "__root__") if parent_indent is not None else "__root__"
        context_key = (indent, parent_key)

        if context_key not in seen_keys:
            seen_keys[context_key] = Counter()

        seen_keys[context_key][key] += 1

    # Report duplicates
    for (indent, parent), counter in seen_keys.items():
        for key, count in counter.items():
            if count > 1:
                errors.append(
                    f"重复键 '{key}' 出现 {count} 次 (在 '{parent}' 下)"
                )

    return errors


def check(partition, script_name, config, raw_yaml=None):
    """Main enforcement function. Checks partition, model params, and cost floor.

    Raises GateViolation with formatted message listing all violations.
    """
    violations = []

    # 1. Partition legality
    valid_partitions = ["research", "development", "test"]
    if partition == "blind":
        violations.append("盲测集不允许直接运行脚本，请使用 blind_test.py 流程")
    elif partition not in valid_partitions:
        violations.append(f"非法分区: '{partition}'，合法分区为 {valid_partitions}")

    # 2. Model params
    model_cfg = config.get("model", {})
    n_estimators = model_cfg.get("n_estimators", 0)
    max_depth = model_cfg.get("max_depth", 0)

    if n_estimators > 100:
        violations.append(
            f"n_estimators={n_estimators} 超过上限 100，请降低模型复杂度"
        )
    if max_depth > 3:
        violations.append(
            f"max_depth={max_depth} 超过上限 3，请降低模型复杂度"
        )

    # 3. Cost floor
    exec_cfg = config.get("execution", {})
    slippage_bps = exec_cfg.get("slippage_bps", 0)
    commission_buy = exec_cfg.get("commission_buy", 0)
    commission_sell = exec_cfg.get("commission_sell", 0)

    total_cost = slippage_bps + (commission_buy + commission_sell) * 10000
    if total_cost < 15:
        violations.append(
            f"总成本={total_cost:.1f}bps 低于下限 15bps，"
            f"请确保成本假设合理（成本不能为零）"
        )

    # 4. Config integrity
    if raw_yaml is not None:
        integrity_errors = validate_config_integrity(config, raw_yaml=raw_yaml)
        violations.extend(integrity_errors)

    if violations:
        msg_lines = [f"🚫 门禁检查未通过 ({script_name}):"]
        for i, v in enumerate(violations, 1):
            msg_lines.append(f"  {i}. {v}")
        raise GateViolation("\n".join(msg_lines))
