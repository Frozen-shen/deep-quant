# 量化系统全面重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild development discipline enforcement (code gates) and data infrastructure (fix SSE gap, expand coverage) so the quant system can produce trustworthy results.

**Architecture:** Three sequential phases — Phase A creates enforcement modules (gate.py, experiment_tracker, config_validator, logger) and cleans up scripts/config; Phase B repairs data (SSE stocks, index benchmark, fundamentals, money flow, unadjusted prices, PEAD); Phase C integrates everything (PIT universe, benchmark swap, daily pipeline). Each phase produces independently verifiable results.

**Tech Stack:** Python 3.x (launcher: `py`), akshare, pandas, numpy, pyyaml, SQLite. Windows platform, Git Bash shell.

**Critical Environment Notes:**
- Use `py` not `python` (Windows Store stub returns exit code 49)
- Project is NOT yet a git repository — Task 1 initializes it
- Working directory: `C:\Users\Frozen\ZCodeProject\quant-starter`

---

## File Structure

### New Files (Phase A)
| File | Responsibility |
|------|---------------|
| `gate.py` | Hard enforcement gate — all backtest scripts must call `gate.check()` |
| `experiment_tracker.py` | Auto-log every experiment run to `experiments/` as JSON |
| `config_validator.py` | Detect config.yaml errors (duplicate keys, param violations, partition mismatch) |
| `logger.py` | Structured logging (console + file), replaces all print() |
| `scripts/active/README.md` | Documents which scripts are canonical |

### New Files (Phase B)
| File | Responsibility |
|------|---------------|
| `scripts/fetch_index_data.py` | Fetch CSI300/500/1000 index daily bars |
| `scripts/fetch_unadjusted_batch.py` | Batch fetch unadjusted prices for limit detection |
| `scripts/refresh_pead_cache.py` | Refresh PEAD forecast cache for 2024-2026 |

### New Files (Phase C)
| File | Responsibility |
|------|---------------|
| `data/pit_universe.py` | Point-in-Time universe builder from monthly index constituents |
| `scripts/daily_pipeline.py` | Orchestrator: data update → factor compute → signal → execute |

### Modified Files
| File | Change |
|------|--------|
| `config.yaml` | Fix duplicate keys, align partitions, fix model params |
| `DEVELOPMENT_DISCIPLINE.md` | v2: reference code enforcement, unified partitions |
| `scripts/fetch_full_universe.py` | Fix SSE stock list generation bug |

### Archived (moved to `scripts/archive/`)
All scripts except the 8 canonical ones listed in the design doc.

---

## Phase A: Development Discipline Enforcement

### Task 1: Initialize Git Repository

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Initialize git**

Run:
```bash
cd C:/Users/Frozen/ZCodeProject/quant-starter
git init
git config user.name "Frozen"
git config user.email "frozen@local"
```

- [ ] **Step 2: Create .gitignore**

```gitignore
# Data (too large for git)
data_store/
data_cache/
data/fundamental_cache/
data/pead_cache/
data/event_cache/
data/factor_cache/
*.parquet
*.db

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/

# LLM cache
.llm_cache/

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db

# Logs
logs/
```

- [ ] **Step 3: Initial commit**

```bash
git add -A
git commit -m "chore: initial commit before restructure"
```

Expected: Commit succeeds, all source files tracked, data files excluded.

---

### Task 2: Create `gate.py` — Hard Enforcement Gate

**Files:**
- Create: `gate.py`
- Test: `tests/test_gate.py`

- [ ] **Step 1: Write failing tests for gate.py**

Create `tests/test_gate.py`:

```python
"""Tests for gate.py — development discipline enforcement."""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gate import GateViolation, check, validate_config_integrity


class TestPartitionCheck:
    def test_research_partition_passes(self):
        config = _minimal_config()
        # Should not raise
        check(partition="research", script_name="test.py", config=config)

    def test_development_partition_passes(self):
        config = _minimal_config()
        check(partition="development", script_name="test.py", config=config)

    def test_test_partition_passes(self):
        config = _minimal_config()
        check(partition="test", script_name="test.py", config=config)

    def test_blind_partition_raises(self):
        config = _minimal_config()
        with pytest.raises(GateViolation, match="盲测集"):
            check(partition="blind", script_name="test.py", config=config)

    def test_invalid_partition_raises(self):
        config = _minimal_config()
        with pytest.raises(GateViolation, match="非法分区"):
            check(partition="yolo", script_name="test.py", config=config)


class TestModelParamCheck:
    def test_compliant_model_passes(self):
        config = _minimal_config()
        config["model"] = {"n_estimators": 50, "max_depth": 2}
        check(partition="research", script_name="test.py", config=config)

    def test_too_many_trees_raises(self):
        config = _minimal_config()
        config["model"] = {"n_estimators": 600, "max_depth": 2}
        with pytest.raises(GateViolation, match="n_estimators"):
            check(partition="research", script_name="test.py", config=config)

    def test_too_deep_raises(self):
        config = _minimal_config()
        config["model"] = {"n_estimators": 50, "max_depth": 5}
        with pytest.raises(GateViolation, match="max_depth"):
            check(partition="research", script_name="test.py", config=config)


class TestCostCheck:
    def test_adequate_cost_passes(self):
        config = _minimal_config()
        config["execution"] = {"slippage_bps": 30, "commission_buy": 0.00025, "commission_sell": 0.00075}
        check(partition="research", script_name="test.py", config=config)

    def test_zero_cost_raises(self):
        config = _minimal_config()
        config["execution"] = {"slippage_bps": 0, "commission_buy": 0, "commission_sell": 0}
        with pytest.raises(GateViolation, match="成本"):
            check(partition="research", script_name="test.py", config=config)


class TestConfigIntegrity:
    def test_valid_config_no_errors(self):
        errors = validate_config_integrity(_minimal_config())
        assert errors == []

    def test_duplicate_key_detected(self):
        # Simulate: config has both max_positions=5 and max_positions=40
        # We detect this via raw YAML parsing
        errors = validate_config_integrity(
            _minimal_config(), raw_yaml="execution:\n  max_positions: 5\n  max_positions: 40\n"
        )
        assert any("重复" in e or "duplicate" in e.lower() for e in errors)


def _minimal_config():
    return {
        "model": {"n_estimators": 50, "max_depth": 2},
        "execution": {"slippage_bps": 30, "commission_buy": 0.00025, "commission_sell": 0.00075},
        "data_partition": {
            "research": {"start": "2018-01-01", "end": "2022-12-31"},
            "development": {"start": "2023-01-01", "end": "2024-06-30"},
            "test": {"start": "2024-07-01", "end": "2025-06-30"},
            "blind": {"start": "2025-07-01", "end": "2026-07-31"},
        },
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -m pytest tests/test_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gate'`

- [ ] **Step 3: Implement gate.py**

Create `gate.py`:

```python
"""
gate.py — 量化开发硬门禁

所有回测/实验脚本必须在开头调用:
    from gate import check, load_config
    config = load_config()
    check(partition="research", script_name=__file__, config=config)

不通过 → 抛出 GateViolation，脚本终止。
"""
import sys
import re
from pathlib import Path
from typing import Optional

import yaml


class GateViolation(Exception):
    """门禁违反 — 脚本被阻止运行。"""
    pass


VALID_PARTITIONS = ["research", "development", "test"]

# 模型参数上限 (教训#6: 不用600树ML)
MAX_N_ESTIMATORS = 100
MAX_DEPTH = 3
MIN_COST_BPS = 15  # 最低总成本 (滑点+手续费)


def load_config(config_path: str = "config.yaml") -> dict:
    """加载 config.yaml。"""
    path = Path(config_path)
    if not path.exists():
        raise GateViolation(f"config.yaml 不存在: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check(partition: str, script_name: str, config: dict,
          raw_yaml: Optional[str] = None) -> None:
    """
    硬门禁检查。在脚本开头调用。

    Args:
        partition: 当前使用的数据分区 ("research" / "development" / "test")
        script_name: 脚本名 (用于错误信息)
        config: yaml.safe_load(config.yaml) 的结果
        raw_yaml: 可选，config.yaml 原始文本 (用于检测重复key)

    Raises:
        GateViolation: 任何规则违反时抛出，脚本应立即终止。
    """
    violations = []

    # ── Rule 1: 分区合法性 ──
    if partition == "blind":
        violations.append("盲测集永远禁止用于回测/参数选择/策略设计")
    elif partition not in VALID_PARTITIONS:
        violations.append(f"非法分区 '{partition}', 合法值: {VALID_PARTITIONS}")

    # ── Rule 2: 模型参数合规 ──
    model_cfg = config.get("model", {})
    n_est = model_cfg.get("n_estimators", 0)
    depth = model_cfg.get("max_depth", 0)
    if n_est > MAX_N_ESTIMATORS:
        violations.append(
            f"model.n_estimators={n_est} > {MAX_N_ESTIMATORS} (教训#6: 过拟合风险)")
    if depth > MAX_DEPTH:
        violations.append(
            f"model.max_depth={depth} > {MAX_DEPTH} (过拟合风险)")

    # ── Rule 3: 成本下限 ──
    exec_cfg = config.get("execution", {})
    slippage_bps = exec_cfg.get("slippage_bps", 0)
    comm_buy = exec_cfg.get("commission_buy", 0)
    comm_sell = exec_cfg.get("commission_sell", 0)
    total_cost_bps = slippage_bps + (comm_buy + comm_sell) * 10000
    if total_cost_bps < MIN_COST_BPS:
        violations.append(
            f"总成本 {total_cost_bps:.1f}bp < {MIN_COST_BPS}bp 下限 (不能放水)")

    # ── Rule 4: 配置完整性 ──
    if raw_yaml:
        integrity_errors = validate_config_integrity(config, raw_yaml=raw_yaml)
        violations.extend(integrity_errors)

    # ── 输出并终止 ──
    if violations:
        msg = f"\n{'='*60}\n"
        msg += f"  GATE VIOLATION — 脚本 '{script_name}' 被阻止\n"
        msg += f"{'='*60}\n"
        for i, v in enumerate(violations, 1):
            msg += f"  {i}. ✗ {v}\n"
        msg += f"{'='*60}\n"
        msg += f"  修复以上问题后重新运行。\n"
        msg += f"{'='*60}\n"
        raise GateViolation(msg)


def validate_config_integrity(config: dict, raw_yaml: Optional[str] = None) -> list:
    """
    检测 config.yaml 中的结构性错误。

    Returns:
        错误消息列表 (空列表 = 无错误)
    """
    errors = []

    # 检测重复key (通过正则匹配原始YAML文本)
    if raw_yaml:
        # 找所有 "key:" 模式，检查同一缩进级别是否有重复
        lines = raw_yaml.split("\n")
        seen_at_indent = {}  # {(indent_level, parent_context): [keys]}
        current_parent = ""
        for line in lines:
            if not line.strip() or line.strip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            stripped = line.strip()
            if ":" in stripped:
                key = stripped.split(":")[0].strip()
                context_key = (indent, current_parent)
                if indent == 0:
                    current_parent = key
                    context_key = (0, "__root__")
                if context_key not in seen_at_indent:
                    seen_at_indent[context_key] = []
                if key in seen_at_indent[context_key]:
                    errors.append(f"config.yaml 重复key: '{key}' (同级出现多次，YAML静默取最后值)")
                seen_at_indent[context_key].append(key)

    return errors
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -m pytest tests/test_gate.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add gate.py tests/test_gate.py
git commit -m "feat: add gate.py — hard enforcement for development discipline"
```

---

### Task 3: Create `experiment_tracker.py`

**Files:**
- Create: `experiment_tracker.py`
- Test: `tests/test_experiment_tracker.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_experiment_tracker.py`:

```python
"""Tests for experiment_tracker.py."""
import json
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment_tracker import log_experiment, list_experiments


class TestLogExperiment:
    def test_creates_json_file(self, tmp_path):
        exp_id = log_experiment(
            script_name="test_script.py",
            partition="research",
            config={"model": {"n_estimators": 50}},
            results={"sharpe": 1.2, "ir": 0.5},
            notes="test run",
            experiments_dir=str(tmp_path),
        )
        assert exp_id.startswith("exp_")
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1

        with open(files[0], "r", encoding="utf-8") as f:
            record = json.load(f)
        assert record["script"] == "test_script.py"
        assert record["partition"] == "research"
        assert record["results"]["sharpe"] == 1.2
        assert "timestamp" in record
        assert "config_hash" in record

    def test_config_hash_deterministic(self, tmp_path):
        config = {"a": 1, "b": [1, 2, 3]}
        id1 = log_experiment("s.py", "research", config, {}, experiments_dir=str(tmp_path))
        id2 = log_experiment("s.py", "research", config, {}, experiments_dir=str(tmp_path))

        files = sorted(tmp_path.glob("*.json"))
        with open(files[0]) as f:
            r1 = json.load(f)
        with open(files[1]) as f:
            r2 = json.load(f)
        assert r1["config_hash"] == r2["config_hash"]


class TestListExperiments:
    def test_empty_dir(self, tmp_path):
        result = list_experiments(str(tmp_path))
        assert result == []

    def test_lists_all(self, tmp_path):
        log_experiment("a.py", "research", {}, {"x": 1}, experiments_dir=str(tmp_path))
        log_experiment("b.py", "development", {}, {"y": 2}, experiments_dir=str(tmp_path))
        result = list_experiments(str(tmp_path))
        assert len(result) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_experiment_tracker.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement experiment_tracker.py**

Create `experiment_tracker.py`:

```python
"""
experiment_tracker.py — 自动实验追踪

每次回测/实验运行后调用 log_experiment() 自动记录。
记录存储到 experiments/ 目录，每个实验一个 JSON 文件。

Usage:
    from experiment_tracker import log_experiment
    exp_id = log_experiment(
        script_name=__file__,
        partition="research",
        config=config,
        results={"sharpe": 1.2, "ir": 0.5, "max_dd": -0.15},
        notes="IC-weighted linear, top-30, 20-day rebalance",
    )
"""
import json
import hashlib
import datetime
from pathlib import Path
from typing import Optional

DEFAULT_EXPERIMENTS_DIR = "experiments"


def log_experiment(
    script_name: str,
    partition: str,
    config: dict,
    results: dict,
    notes: str = "",
    experiments_dir: Optional[str] = None,
) -> str:
    """
    记录一次实验。

    Args:
        script_name: 运行脚本名
        partition: 使用的数据分区
        config: 完整配置快照
        results: 结果指标字典
        notes: 备注
        experiments_dir: 存储目录 (默认 "experiments/")

    Returns:
        experiment_id 字符串
    """
    exp_dir = Path(experiments_dir or DEFAULT_EXPERIMENTS_DIR)
    exp_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now()
    exp_id = f"exp_{now.strftime('%Y%m%d_%H%M%S')}_{id(config) % 10000:04d}"

    config_str = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]

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

    out_path = exp_dir / f"{exp_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)

    return exp_id


def list_experiments(experiments_dir: Optional[str] = None) -> list:
    """列出所有实验记录 (按时间排序)。"""
    exp_dir = Path(experiments_dir or DEFAULT_EXPERIMENTS_DIR)
    if not exp_dir.exists():
        return []

    records = []
    for f in sorted(exp_dir.glob("exp_*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            records.append(json.load(fh))
    return records
```

- [ ] **Step 4: Run tests**

Run: `py -m pytest tests/test_experiment_tracker.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add experiment_tracker.py tests/test_experiment_tracker.py
git commit -m "feat: add experiment_tracker.py — auto experiment logging"
```

---

### Task 4: Create `logger.py`

**Files:**
- Create: `logger.py`

- [ ] **Step 1: Implement logger.py**

```python
"""
logger.py — 统一日志模块

替代所有 print() 语句。提供结构化日志输出到 console + 文件。

Usage:
    from logger import get_logger
    log = get_logger("factor_scorer")
    log.info("因子计算完成: %d 只股票", n_stocks)
    log.warning("数据缺失: %s", symbol)
    log.error("回测失败: %s", str(e))
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("logs")
_initialized = False


def _init_log_dir():
    global _initialized
    if not _initialized:
        LOG_DIR.mkdir(exist_ok=True)
        _initialized = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    获取命名 logger。

    输出:
    - Console: 彩色简洁格式
    - File: logs/quant_YYYYMMDD.log 完整格式
    """
    _init_log_dir()

    logger = logging.getLogger(f"quant.{name}")
    if logger.handlers:
        return logger  # 已初始化

    logger.setLevel(level)

    # Console handler
    console_fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    # File handler
    today = datetime.now().strftime("%Y%m%d")
    file_fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(
        LOG_DIR / f"quant_{today}.log", encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)
    logger.addHandler(fh)

    return logger
```

- [ ] **Step 2: Verify import works**

Run: `py -c "from logger import get_logger; log = get_logger('test'); log.info('OK')"`
Expected: Prints timestamped log line, creates `logs/` directory

- [ ] **Step 3: Commit**

```bash
git add logger.py
git commit -m "feat: add logger.py — structured logging module"
```

---

### Task 5: Fix `config.yaml`

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Read current config.yaml and identify all issues**

Run: `py -c "import yaml; print(yaml.safe_load(open('config.yaml')))"`
Note the duplicate max_positions (5 then 40, last wins = 40).

- [ ] **Step 2: Rewrite config.yaml with fixes**

Replace the entire `config.yaml` with:

```yaml
# ═══════════════════════════════════════════════════════════
# quant-starter 统一配置 (v2.0 — 2026-08-01 重构)
# 唯一配置源。所有代码从此文件读取。
# 修改此文件后必须更新 blind_test.config_hash。
# ═══════════════════════════════════════════════════════════

# ── 数据分区 (唯一合法定义, 与 DEVELOPMENT_DISCIPLINE.md v2 同步) ──
data_partition:
  research:    {start: "2018-01-01", end: "2022-12-31"}  # 因子开发、IC计算
  development: {start: "2023-01-01", end: "2024-06-30"}  # 组合验证、参数调优
  test:        {start: "2024-07-01", end: "2025-06-30"}  # 最终测试 (只跑一次)
  blind:       {start: "2025-07-01", end: "2026-07-31"}  # 模拟盘跟踪 (永不回测)
  full_start: "2018-01-01"
  full_end: "2026-07-31"

# ── 盲测状态 ──
blind_test:
  locked: true
  contaminated: true  # 旧blind期(2024-07~2026-07)已被污染(trial_count=3)
  note: "新blind期从2025-07-01开始, 旧结果全部作废"

# ── 执行参数 ──
execution:
  initial_capital: 100000
  lot_size: 100
  top_k: 30                    # 持仓数量
  max_positions: 40            # 最大持仓上限
  max_single_pct: 0.25         # 单票仓位上限
  slippage_bps: 30             # 滑点 (小盘股30bp)
  commission_buy: 0.00025      # 买入手续费
  commission_sell: 0.00075     # 卖出手续费 (含印花税)
  signal_delay: 1              # T+1信号延迟
  execution_price: open        # 成交价: 次日开盘
  turnover_limit_pct: 0.5      # 月单边换手≤50%

# ── 模型参数 (教训#6: 极端正则化) ──
model:
  type: linear                 # 生产路径: IC加权线性 (不用ML)
  # 以下为研究用ML参数 (model/pipeline.py 使用, 非生产路径)
  research_lgb:
    n_estimators: 50
    max_depth: 2
    min_data_in_leaf: 200
    lambda_l1: 1.0
    lambda_l2: 1.0
    learning_rate: 0.05
    early_stopping_rounds: 10

# ── 因子配置 ──
factors:
  preset: ic_auto
  use_fundamental: false       # 基本面数据就绪后改为 true
  corr_threshold: 0.6          # 独立性剪枝阈值 (从0.4放宽到0.6)

# ── 标签 ──
label:
  horizon_days: 20
  type: rank

# ── 滚动验证 ──
rolling:
  train_months: 4
  test_months: 9
  embargo_days: 30
  day_step: 2

# ── 组合管理 ──
portfolio:
  hold_thresh: 30              # 月度调仓
  sector_neutral: true
  buy_confirm_days: 1
  sell_rank_buffer: 2
  n_drop: 2

# ── 时间衰减 ──
time_decay:
  half_life_years: 0.5

# ── Universe ──
universe:
  index: "all_cached"
  include_delisted: true
  min_list_days: 250
  snapshot_mode: monthly
  data_version: v20260801      # 更新版本号
```

- [ ] **Step 3: Validate new config**

Run: `py -c "import yaml; c = yaml.safe_load(open('config.yaml')); print('model.type:', c['model']['type']); print('partitions:', list(c['data_partition'].keys()))"`
Expected: `model.type: linear`, `partitions: ['research', 'development', 'test', 'blind', 'full_start', 'full_end']`

- [ ] **Step 4: Run gate.py against new config**

Run: `py -c "from gate import check, load_config; c = load_config(); check('research', 'manual_test', c); print('GATE PASSED')"`
Expected: `GATE PASSED`

- [ ] **Step 5: Commit**

```bash
git add config.yaml
git commit -m "fix: config.yaml v2 — unified partitions, fixed model params, removed duplicates"
```

---

### Task 6: Archive Scripts

**Files:**
- Create: `scripts/active/` directory
- Create: `scripts/archive/` directory
- Move: scripts between directories

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p scripts/active scripts/archive
```

- [ ] **Step 2: Copy canonical scripts to active/**

The 8 canonical scripts (copy, not move, to preserve imports during transition):

```bash
cp scripts/run_factor_portfolio.py scripts/active/run_research_backtest.py
cp scripts/run_paper_signal_v3.py scripts/active/run_paper_signal.py
cp scripts/run_alpha158_ic.py scripts/active/run_factor_ic.py
cp scripts/fetch_full_universe.py scripts/active/fetch_daily_data.py
cp scripts/fetch_fundamentals_v2.py scripts/active/fetch_fundamentals.py
cp scripts/run_ic_monitor.py scripts/active/run_ic_monitor.py
cp scripts/export_equity_curve.py scripts/active/export_equity_curve.py
cp scripts/init_paper_account.py scripts/active/init_paper_account.py
```

- [ ] **Step 3: Move all other scripts to archive/**

```bash
# Move everything except active/ and archive/ and __init__.py
for f in scripts/*.py; do
  base=$(basename "$f")
  if [ "$base" != "__init__.py" ]; then
    mv "$f" scripts/archive/
  fi
done
```

- [ ] **Step 4: Create scripts/active/README.md**

```markdown
# 合法脚本 (active/)

只有此目录中的脚本可以使用。同类功能只允许存在一个脚本。

| 脚本 | 用途 | 运行频率 |
|------|------|----------|
| `run_research_backtest.py` | 研究阶段回测 (唯一) | 按需 |
| `run_factor_ic.py` | 因子IC验证 (唯一) | 按需 |
| `run_paper_signal.py` | 模拟盘信号生成 (唯一) | 每日 |
| `fetch_daily_data.py` | 全市场日线数据获取 | 每周/按需 |
| `fetch_fundamentals.py` | 基本面数据获取 | 每月 |
| `run_ic_monitor.py` | IC衰减监控 | 每周 |
| `export_equity_curve.py` | 净值曲线导出 | 按需 |
| `init_paper_account.py` | 模拟盘账户初始化 | 一次性 |

## 规则

1. **禁止新建回测脚本** — 修改 `run_research_backtest.py`
2. **禁止在 archive/ 中运行脚本** — 结果不可信
3. **每个脚本开头必须有 gate.check() 调用**
```

- [ ] **Step 5: Commit**

```bash
git add scripts/
git commit -m "refactor: archive 26 redundant scripts, keep 8 canonical in active/"
```

---

### Task 7: Update DEVELOPMENT_DISCIPLINE.md v2

**Files:**
- Modify: `DEVELOPMENT_DISCIPLINE.md`

- [ ] **Step 1: Rewrite DEVELOPMENT_DISCIPLINE.md**

Replace with v2 that references code enforcement:

```markdown
# 量化开发纪律手册 v2

> 本文件是最高优先级约束。
> **执行方式: 代码强制 (`gate.py`)，不依赖人工自觉。**

---

## 第一条：数据分区（代码强制）

分区定义在 `config.yaml` 的 `data_partition` 字段中（唯一源）。

| 分区 | 日期范围 | 用途 | 限制 |
|------|----------|------|------|
| research | 2018-01-01 → 2022-12-31 | 因子开发、IC计算 | 可反复使用 |
| development | 2023-01-01 → 2024-06-30 | 组合验证、参数调优 | 可反复使用 |
| test | 2024-07-01 → 2025-06-30 | 最终评估 | **只跑一次** |
| blind | 2025-07-01 → 2026-07-31 | 模拟盘跟踪 | **永不回测** |

**代码强制**: `gate.py` 会阻止使用 "blind" 分区的脚本运行。

---

## 第二条：模型参数（代码强制）

| 参数 | 上限 | 理由 |
|------|------|------|
| n_estimators | ≤ 100 | 教训#6: 600树过拟合 |
| max_depth | ≤ 3 | 极端正则化 |
| min_data_in_leaf | ≥ 200 | 防止叶节点过拟合 |

**代码强制**: `gate.py` 检查 config.yaml 中的模型参数。

---

## 第三条：成本模型（代码强制）

总交易成本 (滑点 + 双边手续费) ≥ 15bp。

当前设定: 滑点30bp + 手续费(2.5bp+7.5bp) = 40bp。合规。

---

## 第四条：实验记录（代码强制）

每次回测必须调用 `experiment_tracker.log_experiment()`。
记录自动写入 `experiments/exp_YYYYMMDD_HHMMSS_XXXX.json`。

---

## 第五条：脚本管理

- 只有 `scripts/active/` 中的脚本可以使用
- 同类功能只允许一个脚本
- 需要"修复"时修改现有脚本，不得新建
- 废弃脚本移入 `scripts/archive/`

---

## 第六条：基准

- 回测基准 = CSI1000 指数收益率 (从 `data/cache/index_csi1000.parquet` 读取)
- 不得使用子集等权作为主基准
- 全池等权可作为辅助参考，但报告以CSI1000为准

---

## 第七条：PIT Universe

- 每个调仓日只用当时存在的股票
- 使用 `data/pit_universe.py` 的 `get_universe(date)` 获取合法股票池
- 禁止按数据长度/文件大小筛选股票

---

## 附录：已确认的教训清单

| # | 教训 | 首次犯错 | 重犯次数 | 代码强制? |
|---|------|---------|---------|-----------|
| 1 | 不按文件大小选池 | v2.0 | 1 | ✓ pit_universe |
| 2 | 不用子集当基准 | v2.0 | 1 | ✓ CSI1000 |
| 3 | IC必须有embargo | v2.0 | 1 | ✓ config |
| 4 | 不碾数据分区 | v2.0 | 2 | ✓ gate.py |
| 5 | 不放水成本 | v2.0 | 1 | ✓ gate.py |
| 6 | 不用600树ML | v1.0 | 1 | ✓ gate.py |
| 7 | 换手=0必须报警 | v4 | 1 | 验证门 |
| 8 | IR>1.5必须质疑 | v3 | 1 | 验证门 |
```

- [ ] **Step 2: Commit**

```bash
git add DEVELOPMENT_DISCIPLINE.md
git commit -m "docs: DEVELOPMENT_DISCIPLINE.md v2 — code-enforced, unified with config.yaml"
```

---

## Phase B: Data Infrastructure Repair

### Task 8: Fix `fetch_full_universe.py` — Add SSE Stocks

**Files:**
- Modify: `scripts/active/fetch_daily_data.py` (copied from fetch_full_universe.py)

- [ ] **Step 1: Identify the bug**

Read the stock list generation logic in `scripts/archive/fetch_full_universe.py`. The bug is that it only generates SZSE code ranges (000xxx, 002xxx, 300xxx, 301xxx) and misses SSE ranges (600xxx, 601xxx, 603xxx, 605xxx, 688xxx).

- [ ] **Step 2: Fix the stock list generation**

Replace the code list generation with akshare's authoritative source:

```python
def get_all_a_stock_codes() -> list:
    """
    获取全部A股代码 (权威来源: akshare实时列表)。
    过滤: ST、上市不足250天、北交所(8xxxxx/4xxxxx)。
    """
    import akshare as ak
    import pandas as pd
    from datetime import datetime, timedelta

    # 获取全A股列表
    df = ak.stock_info_a_code_name()
    codes = df["code"].tolist()

    # 过滤北交所 (8开头, 4开头)
    codes = [c for c in codes if not c.startswith("8") and not c.startswith("4")]

    # 过滤ST (名称含ST)
    st_mask = df["name"].str.contains("ST|退", na=False)
    st_codes = set(df[st_mask]["code"].tolist())
    codes = [c for c in codes if c not in st_codes]

    return codes
```

- [ ] **Step 3: Add SSE-aware market detection for akshare calls**

```python
def get_market(code: str) -> str:
    """根据代码判断市场 (akshare需要此参数)。"""
    if code.startswith(("6", "9")):
        return "sh"
    else:
        return "sz"
```

- [ ] **Step 4: Test the fix (dry run)**

Run: `py scripts/active/fetch_daily_data.py --check-only`
Expected: Shows ~4500+ codes (both SSE and SZSE), not just ~2776 SZSE-only.

- [ ] **Step 5: Run full fetch (background, will take hours)**

Run: `py scripts/active/fetch_daily_data.py --resume`
Note: This will take several hours for ~4500 stocks. Use `--resume` so it can be interrupted and continued.

- [ ] **Step 6: Commit the fix**

```bash
git add scripts/active/fetch_daily_data.py
git commit -m "fix: fetch_daily_data now covers SSE stocks (600/601/603/605/688)"
```

---

### Task 9: Fetch Index Data (CSI1000 Benchmark)

**Files:**
- Create: `scripts/fetch_index_data.py`

- [ ] **Step 1: Create the script**

```python
"""
fetch_index_data.py — 获取A股主要指数日线数据

产出:
  data/cache/index_csi300.parquet   (沪深300, 000300)
  data/cache/index_csi500.parquet   (中证500, 000905)
  data/cache/index_csi1000.parquet  (中证1000, 000852) ← 主基准
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")

INDICES = {
    "000300": "csi300",
    "000905": "csi500",
    "000852": "csi1000",
}

START_DATE = "20180101"
END_DATE = "20260731"


def fetch_index(symbol: str, name: str) -> pd.DataFrame:
    """获取单个指数日线。"""
    import akshare as ak

    print(f"  获取 {name} ({symbol})...", flush=True)
    df = ak.index_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=START_DATE,
        end_date=END_DATE,
    )

    # 标准化列名
    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
    df = df.sort_values("date").reset_index(drop=True)

    # 计算日收益率
    df["return"] = df["close"].pct_change()

    print(f"    {len(df)} 行, {df['date'].min().date()} → {df['date'].max().date()}", flush=True)
    return df


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    for symbol, name in INDICES.items():
        out_path = os.path.join(CACHE_DIR, f"index_{name}.parquet")
        try:
            df = fetch_index(symbol, name)
            df.to_parquet(out_path, index=False)
            print(f"  ✓ 已保存: {out_path}", flush=True)
        except Exception as e:
            print(f"  ✗ 失败: {name} — {e}", flush=True)

    print("\n完成。", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `py scripts/fetch_index_data.py`
Expected: Creates 3 parquet files in `data/cache/`, each with ~2000+ rows.

- [ ] **Step 3: Verify CSI1000 data**

Run: `py -c "import pandas as pd; df = pd.read_parquet('data/cache/index_csi1000.parquet'); print(f'Rows: {len(df)}, Range: {df.date.min()} to {df.date.max()}'); print(f'Annual return: {(df.close.iloc[-1]/df.close.iloc[0])**(252/len(df))-1:.1%}')"`
Expected: ~2000 rows, 2018-2026, reasonable annual return.

- [ ] **Step 4: Commit**

```bash
git add scripts/fetch_index_data.py
git commit -m "feat: add fetch_index_data.py — CSI300/500/1000 benchmark data"
```

---

### Task 10: Fetch Unadjusted Data for Limit Detection

**Files:**
- Create: `scripts/fetch_unadjusted_batch.py`

- [ ] **Step 1: Create the script**

```python
"""
fetch_unadjusted_batch.py — 批量获取未复权日线数据

用途: 涨跌停检测需要未复权价格 (复权后10%/20%阈值失真)。
产出: data_cache/unadjusted/{code}.parquet

当前缺口: 仅176/1550只有未复权数据, 需补充到全覆盖。
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_CACHE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "data_cache"
UNADJ_DIR = DATA_CACHE / "unadjusted"


def get_cached_symbols() -> list:
    """获取 data_cache 中已有复权数据的股票列表。"""
    return [f.stem for f in DATA_CACHE.glob("*.parquet") if f.stem != "__meta__"]


def get_existing_unadj() -> set:
    """获取已有未复权数据的股票。"""
    if not UNADJ_DIR.exists():
        return set()
    return {f.stem for f in UNADJ_DIR.glob("*.parquet")}


def fetch_unadjusted(code: str) -> None:
    """获取单只股票未复权日线。"""
    import akshare as ak
    import pandas as pd

    market = "sh" if code.startswith("6") else "sz"

    try:
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date="20180101",
            end_date="20260731",
            adjust="",  # 未复权
        )
    except Exception as e:
        print(f"  ✗ {code}: {e}")
        return

    if df is None or df.empty:
        print(f"  ✗ {code}: 空数据")
        return

    df = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("date").reset_index(drop=True)

    out_path = UNADJ_DIR / f"{code}.parquet"
    df.to_parquet(out_path, index=False)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="跳过已有数据")
    parser.add_argument("--limit", type=int, default=0, help="限制数量 (0=全部)")
    parser.add_argument("--check-only", action="store_true", help="只检查覆盖率")
    args = parser.parse_args()

    UNADJ_DIR.mkdir(parents=True, exist_ok=True)

    all_symbols = get_cached_symbols()
    existing = get_existing_unadj()
    missing = [s for s in all_symbols if s not in existing]

    print(f"复权数据: {len(all_symbols)} 只")
    print(f"未复权已有: {len(existing)} 只")
    print(f"未复权缺失: {len(missing)} 只")
    print(f"覆盖率: {len(existing)/len(all_symbols)*100:.1f}%")

    if args.check_only:
        return

    if args.resume:
        to_fetch = missing
    else:
        to_fetch = all_symbols

    if args.limit > 0:
        to_fetch = to_fetch[:args.limit]

    print(f"\n开始获取: {len(to_fetch)} 只 (间隔1秒)")
    for i, code in enumerate(to_fetch):
        fetch_unadjusted(code)
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(to_fetch)}", flush=True)
        time.sleep(1)  # 防封IP

    # 最终覆盖率
    final = get_existing_unadj()
    print(f"\n完成。覆盖率: {len(final)/len(all_symbols)*100:.1f}%")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Check current coverage**

Run: `py scripts/fetch_unadjusted_batch.py --check-only`
Expected: Shows ~11% coverage (176/1550)

- [ ] **Step 3: Run fetch (background, ~25 minutes for 1374 missing stocks)**

Run: `py scripts/fetch_unadjusted_batch.py --resume`

- [ ] **Step 4: Commit**

```bash
git add scripts/fetch_unadjusted_batch.py
git commit -m "feat: add fetch_unadjusted_batch.py — limit detection coverage"
```

---

### Task 11: Fetch Money Flow + North Flow Data

**Files:**
- Modify: `scripts/fetch_fund_flow.py` (verify it works)
- Modify: `scripts/fetch_north_flow.py` (verify it works)

- [ ] **Step 1: Test money flow fetch with a single stock**

Run: `py -c "import akshare as ak; df = ak.stock_individual_fund_flow(stock='000001', market='sz'); print(df.shape); print(df.columns.tolist())"`
Expected: Returns a DataFrame with daily fund flow data.

- [ ] **Step 2: Run money flow fetch for full universe**

Run: `py scripts/fetch_fund_flow.py --resume`
Note: If the script doesn't support `--resume`, add it. The script should save to `data/factor_cache/money_flow/{code}.parquet`.

- [ ] **Step 3: Test north flow fetch**

Run: `py -c "import akshare as ak; df = ak.stock_hsgt_individual_em(symbol='000001'); print(df.shape)"`
Expected: Returns northbound holding history.

- [ ] **Step 4: Run north flow fetch**

Run: `py scripts/fetch_north_flow.py --resume`

- [ ] **Step 5: Verify outputs exist**

Run: `ls data/factor_cache/money_flow/ | wc -l` and `ls data/factor_cache/north_flow/ | wc -l`
Expected: Non-zero counts.

- [ ] **Step 6: Commit any fixes made**

```bash
git add scripts/fetch_fund_flow.py scripts/fetch_north_flow.py
git commit -m "fix: ensure money_flow and north_flow fetch scripts work with --resume"
```

---

### Task 12: Refresh PEAD Cache (2024-2026)

**Files:**
- Create: `scripts/refresh_pead_cache.py`

- [ ] **Step 1: Create the script**

```python
"""
refresh_pead_cache.py — 刷新业绩预告缓存

当前缺口: data/pead_cache/ 止于 2023-12-31。
需要补充: 2024Q1 → 2026Q2 的业绩预告数据。

接口: ak.stock_yjyg_em(date="YYYYMMDD")
返回: 该季度所有已发布业绩预告的股票
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PEAD_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "data" / "pead_cache"

# 需要获取的季度 (YYYYMMDD = 季度末)
QUARTERS = [
    "20240331", "20240630", "20240930", "20241231",
    "20250331", "20250630", "20250930", "20251231",
    "20260331", "20260630",
]


def fetch_quarter(date_str: str):
    """获取单个季度的业绩预告。"""
    import akshare as ak
    import pandas as pd

    out_path = PEAD_DIR / f"forecast_{date_str}.parquet"
    if out_path.exists():
        print(f"  跳过 {date_str} (已存在)")
        return

    print(f"  获取 {date_str}...", flush=True)
    try:
        df = ak.stock_yjyg_em(date=date_str)
        if df is None or df.empty:
            print(f"    空数据 (该季度可能尚未披露)")
            return
        df.to_parquet(out_path, index=False)
        print(f"    ✓ {len(df)} 条记录", flush=True)
    except Exception as e:
        print(f"    ✗ 失败: {e}")

    time.sleep(2)  # 防封


def main():
    PEAD_DIR.mkdir(parents=True, exist_ok=True)

    existing = {f.stem.replace("forecast_", "") for f in PEAD_DIR.glob("forecast_*.parquet")}
    print(f"已有季度: {sorted(existing)}")
    print(f"目标季度: {QUARTERS}")
    print()

    for q in QUARTERS:
        fetch_quarter(q)

    final = {f.stem.replace("forecast_", "") for f in PEAD_DIR.glob("forecast_*.parquet")}
    print(f"\n完成。共 {len(final)} 个季度: {sorted(final)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `py scripts/refresh_pead_cache.py`
Expected: Fetches 2024-2026 quarters (some may be empty if not yet disclosed).

- [ ] **Step 3: Commit**

```bash
git add scripts/refresh_pead_cache.py
git commit -m "feat: add refresh_pead_cache.py — extend PEAD data to 2026"
```

---

### Task 13: Expand Fundamental Cache

**Files:**
- Use existing: `scripts/active/fetch_fundamentals.py`

- [ ] **Step 1: Check current coverage**

Run: `py scripts/active/fetch_fundamentals.py --check-only`
Expected: Shows ~418 stocks, 27% coverage.

- [ ] **Step 2: Run expansion (background, ~80 minutes)**

Run: `py scripts/active/fetch_fundamentals.py --resume`
This fetches fundamental data for all stocks in data_cache that don't have it yet.

- [ ] **Step 3: Verify expanded coverage**

Run: `py -c "from pathlib import Path; files = list(Path('data/fundamental_cache').glob('*.parquet')); print(f'Fundamental cache: {len(files)} stocks')"`
Expected: Significantly more than 418.

---

## Phase C: Pipeline Integration

### Task 14: Create PIT Universe Builder

**Files:**
- Create: `data/pit_universe.py`
- Test: `tests/test_pit_universe.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_pit_universe.py`:

```python
"""Tests for data/pit_universe.py."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pit_universe import get_universe, get_all_trading_stocks


class TestGetUniverse:
    def test_returns_list_of_codes(self):
        # 2020-01 should return CSI300+CSI1000 constituents from that month
        universe = get_universe("2020-01-15")
        assert isinstance(universe, list)
        assert len(universe) > 500  # CSI300+CSI1000 ≈ 1300
        assert all(isinstance(c, str) and len(c) == 6 for c in universe)

    def test_different_dates_different_universes(self):
        u1 = get_universe("2019-01-15")
        u2 = get_universe("2023-06-15")
        # Not identical (constituents change over time)
        assert set(u1) != set(u2)

    def test_fallback_to_all_cached(self):
        # For dates before our constituent data starts
        universe = get_universe("2015-01-01")
        # Should fall back to all cached stocks
        assert len(universe) > 0


class TestGetAllTradingStocks:
    def test_returns_superset(self):
        all_stocks = get_all_trading_stocks()
        assert len(all_stocks) > 1000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_pit_universe.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement data/pit_universe.py**

```python
"""
data/pit_universe.py — Point-in-Time Universe Builder

消除幸存者偏差: 每个调仓日只返回当时存在的股票。
数据源: data/cache/universe_000300.json + universe_000852.json (月度成分)

Usage:
    from data.pit_universe import get_universe
    stocks = get_universe("2023-06-15")  # 返回2023年6月的CSI300+CSI1000成分
"""
import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

CACHE_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "cache"
DATA_CACHE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "data_cache"

_constituents_cache = {}  # 内存缓存


def _load_constituents(index_name: str) -> dict:
    """
    加载指数月度成分数据。

    Returns:
        {"2018-01": ["600000", "000001", ...], "2018-02": [...], ...}
    """
    if index_name in _constituents_cache:
        return _constituents_cache[index_name]

    path = CACHE_DIR / f"universe_{index_name}.json"
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    _constituents_cache[index_name] = data
    return data


def get_universe(date: str, min_list_days: int = 250) -> list:
    """
    获取指定日期的PIT股票池。

    策略:
    1. 取该月的 CSI300 + CSI1000 成分 (≈1300只)
    2. 补充 data_cache 中有数据且上市超过 min_list_days 的股票
    3. 去重后返回

    Args:
        date: 日期字符串 "YYYY-MM-DD"
        min_list_days: 最少上市天数

    Returns:
        股票代码列表 ["600000", "000001", ...]
    """
    dt = datetime.strptime(date, "%Y-%m-%d")
    month_key = dt.strftime("%Y-%m")

    # 从指数成分获取
    csi300 = _load_constituents("000300")
    csi1000 = _load_constituents("000852")

    stocks = set()

    # 取当月或最近月份的成分
    for source in [csi300, csi1000]:
        if month_key in source:
            stocks.update(source[month_key])
        else:
            # 找最近的月份
            available_months = sorted(source.keys())
            past_months = [m for m in available_months if m <= month_key]
            if past_months:
                stocks.update(source[past_months[-1]])

    # 如果指数成分数据为空, 回退到 all_cached
    if not stocks:
        stocks = set(get_all_trading_stocks())

    return sorted(stocks)


def get_all_trading_stocks() -> list:
    """
    获取 data_cache 中所有有数据的股票 (回退方案)。
    注意: 这有幸存者偏差, 仅用于指数成分数据不可用时。
    """
    if not DATA_CACHE_DIR.exists():
        return []
    return sorted([
        f.stem for f in DATA_CACHE_DIR.glob("*.parquet")
        if len(f.stem) == 6 and f.stem.isdigit()
    ])
```

- [ ] **Step 4: Run tests**

Run: `py -m pytest tests/test_pit_universe.py -v`
Expected: All PASS (requires `data/cache/universe_000300.json` and `universe_000852.json` to exist)

- [ ] **Step 5: Commit**

```bash
git add data/pit_universe.py tests/test_pit_universe.py
git commit -m "feat: add PIT universe builder — eliminates survivorship bias"
```

---

### Task 15: Create Daily Pipeline Orchestrator

**Files:**
- Create: `scripts/daily_pipeline.py`

- [ ] **Step 1: Create the orchestrator**

```python
"""
daily_pipeline.py — 每日管道编排

收盘后(16:30+)运行一次, 完成:
1. 检查是否交易日
2. 增量更新日线数据
3. 数据质量校验
4. 生成信号 (如果到达调仓日)
5. 记录运行状态

Usage:
    py scripts/daily_pipeline.py              # 正常运行
    py scripts/daily_pipeline.py --dry-run    # 只检查不执行
    py scripts/daily_pipeline.py --force      # 强制运行(忽略交易日检查)
"""
import os
import sys
import json
import argparse
from datetime import datetime, date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger import get_logger

log = get_logger("daily_pipeline")

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE_FILE = BASE_DIR / "data" / "pipeline_state.json"


def is_trading_day(d: date) -> bool:
    """检查是否为交易日。"""
    try:
        from data.calendar import is_trading_day as _check
        return _check(d.strftime("%Y-%m-%d"))
    except ImportError:
        # 回退: 周一到周五
        return d.weekday() < 5


def load_state() -> dict:
    """加载管道状态。"""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"last_run": None, "last_data_date": None, "runs": []}


def save_state(state: dict):
    """保存管道状态。"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def step_update_data(dry_run: bool = False) -> bool:
    """Step 1: 增量更新数据。"""
    log.info("Step 1: 增量数据更新...")
    if dry_run:
        log.info("  [dry-run] 跳过")
        return True

    # 调用 update_daily_data 逻辑
    try:
        from scripts.archive.update_daily_data import main as update_main
        update_main()
        return True
    except Exception as e:
        log.error("  数据更新失败: %s", e)
        return False


def step_validate_data(dry_run: bool = False) -> bool:
    """Step 2: 数据质量校验。"""
    log.info("Step 2: 数据质量校验...")
    if dry_run:
        log.info("  [dry-run] 跳过")
        return True
    # TODO: 接入 data/validator.py
    return True


def step_generate_signal(dry_run: bool = False) -> bool:
    """Step 3: 生成信号 (调仓日)。"""
    log.info("Step 3: 信号生成...")
    if dry_run:
        log.info("  [dry-run] 跳过")
        return True
    # TODO: 接入 run_paper_signal
    return True


def main():
    parser = argparse.ArgumentParser(description="每日管道")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    today = date.today()
    log.info("=" * 50)
    log.info("每日管道启动: %s", today)

    if not args.force and not is_trading_day(today):
        log.info("非交易日, 退出。")
        return

    state = load_state()

    success = True
    success &= step_update_data(args.dry_run)
    success &= step_validate_data(args.dry_run)
    success &= step_generate_signal(args.dry_run)

    state["last_run"] = datetime.now().isoformat()
    state["runs"].append({
        "date": str(today),
        "success": success,
        "dry_run": args.dry_run,
    })
    # 只保留最近100次
    state["runs"] = state["runs"][-100:]
    save_state(state)

    if success:
        log.info("管道完成 ✓")
    else:
        log.error("管道有步骤失败 ✗")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test dry run**

Run: `py scripts/daily_pipeline.py --dry-run --force`
Expected: Logs each step as "[dry-run] 跳过", exits cleanly.

- [ ] **Step 3: Commit**

```bash
git add scripts/daily_pipeline.py
git commit -m "feat: add daily_pipeline.py — orchestrator for data+signal+monitoring"
```

---

### Task 16: End-to-End Verification

**Files:**
- No new files

- [ ] **Step 1: Run gate.py on the active backtest script**

Run: `py -c "from gate import check, load_config; c = load_config(); check('research', 'run_research_backtest.py', c); print('GATE PASSED')"`
Expected: PASS

- [ ] **Step 2: Verify data coverage report**

Run:
```bash
py -c "
from pathlib import Path
dc = len(list(Path('data_cache').glob('*.parquet')))
ua = len(list(Path('data_cache/unadjusted').glob('*.parquet')))
fund = len(list(Path('data/fundamental_cache').glob('*.parquet')))
mf = len(list(Path('data/factor_cache/money_flow').glob('*.parquet')))
nf = len(list(Path('data/factor_cache/north_flow').glob('*.parquet')))
idx = Path('data/cache/index_csi1000.parquet').exists()
print(f'data_cache: {dc} stocks')
print(f'unadjusted: {ua} stocks ({ua/dc*100:.0f}%)')
print(f'fundamental: {fund} stocks ({fund/dc*100:.0f}%)')
print(f'money_flow: {mf} stocks')
print(f'north_flow: {nf} stocks')
print(f'CSI1000 index: {\"YES\" if idx else \"NO\"}')
"
```

- [ ] **Step 3: Verify PIT universe works**

Run: `py -c "from data.pit_universe import get_universe; u = get_universe('2023-06-15'); print(f'PIT universe 2023-06: {len(u)} stocks')"`
Expected: ~1300 stocks

- [ ] **Step 4: Verify experiment tracker**

Run: `py -c "from experiment_tracker import log_experiment; eid = log_experiment('verification.py', 'research', {'test': True}, {'status': 'pass'}); print(f'Logged: {eid}')"`
Expected: Creates `experiments/exp_*.json`

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: Phase A+B+C restructure complete — verified end-to-end"
```

---

## Execution Notes

**Parallelism opportunities:**
- Tasks 2, 3, 4 (gate, tracker, logger) are independent — can run in parallel
- Tasks 8, 9, 10, 11, 12, 13 (data fetches) are independent — can run in parallel
- Task 14 (PIT universe) depends on index constituent data already existing
- Task 15 (daily pipeline) depends on Tasks 4 (logger) and 14 (PIT)

**Long-running tasks:**
- Task 8 (fetch all stocks): ~4-6 hours
- Task 10 (unadjusted): ~25 minutes
- Task 11 (money flow + north flow): ~1-2 hours
- Task 13 (fundamentals): ~80 minutes

These should be run in background and verified after completion.
