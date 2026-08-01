# 量化系统全面重构设计文档

> **状态**: DRAFT — 待用户审阅
> **日期**: 2026-08-01
> **范围**: 开发纪律重构 + 数据基础设施重建

---

## 一、问题诊断总结

### 1.1 当前系统致命缺陷

| # | 缺陷 | 严重度 | 根因 |
|---|------|--------|------|
| 1 | data_store 2776只股票中上交所仅5只 | **CRITICAL** | fetch_full_universe.py 的股票列表生成逻辑有bug，只遍历了深交所代码段 |
| 2 | config.yaml 与 DEVELOPMENT_DISCIPLINE.md 分区定义矛盾 | **CRITICAL** | 多次迭代后未同步更新，两套分区并存 |
| 3 | config.yaml 模型参数(600树/depth5)违反纪律教训#6 | **HIGH** | 参数在某次实验中被修改后未回滚 |
| 4 | money_flow / north_flow 缓存完全为空 | **HIGH** | 脚本存在但从未成功运行完毕（akshare接口不稳定+无重试机制） |
| 5 | 无代码强制门禁(gate.py从未创建) | **HIGH** | 纪律文档写了"执行方式"但从未实现 |
| 6 | 34个脚本严重冗余(5个回测脚本并存) | **MEDIUM** | 每次"修复"都新建脚本而非修改原有脚本 |
| 7 | 实验追踪系统不存在 | **MEDIUM** | experiments/ 目录从未创建 |
| 8 | 分析师数据仅1天快照 | **MEDIUM** | 需要每日积累，但没有调度机制 |
| 9 | fundamental_cache 仅418只且偏深交所 | **MEDIUM** | fetch脚本中断后未恢复 |
| 10 | 无PIT universe(幸存者偏差) | **MEDIUM** | 有CSI300/1000月度成分数据但未接入回测 |
| 11 | 无合理基准(CSI1000指数未获取) | **LOW** | 用全池等权代替，小盘池年化+19.2%不合理 |
| 12 | config.yaml max_positions 重复定义 | **LOW** | YAML静默取最后一个值，无校验 |

### 1.2 开发纪律违反历史

已确认的纪律违反（来自 DEVELOPMENT_DISCIPLINE.md 教训清单 + 本session审计）：

1. **盲测污染**: trial_count=3，Trial#2和#3偷看了盲测结果后修改参数
2. **31次实验无Bonferroni校正**: 多次尝试后只报告最好的结果
3. **Horizon sweep是data snooping**: 在dev集上扫描T+3到T+15，选了最好的horizon
4. **参数在dev集上迭代**: 反复调参直到dev集结果好看
5. **600树/depth5**: 教训#6明确禁止，但config中仍然存在

---

## 二、设计方案：开发纪律重构

### 2.1 设计原则

> **"如果纪律不能被代码强制执行，那它就不是纪律，只是建议。"**

- **硬门禁**: 所有回测/实验脚本必须通过 gate 检查才能运行
- **单一入口**: 定义唯一合法的工作流路径，禁止绕过
- **自动追踪**: 每次运行自动记录实验，无需人工记忆
- **配置唯一源**: config.yaml 是唯一配置源，纪律文档引用它而非定义自己的值

### 2.2 模块设计

#### 2.2.1 `gate.py` — 硬门禁模块

```python
"""
gate.py — 量化开发硬门禁
所有回测/实验脚本必须在开头调用 gate.check() 才能继续运行。
不通过 → sys.exit(1)，物理上无法绕过。
"""

class GateViolation(Exception):
    """门禁违反异常"""
    pass

def check(partition: str, script_name: str, config: dict) -> None:
    """
    在脚本开头调用。检查:
    1. partition 合法性 (不能是 'blind')
    2. 日期范围与 config.yaml 一致
    3. 基准设定合法 (不能手动指定子集)
    4. 成本模型 >= 15bp
    5. 模型参数合规 (n_estimators <= 100, max_depth <= 3)
    6. config.yaml 无重复key (yaml.safe_load会静默覆盖)
    """
    violations = []
    
    # Rule 1: 分区合法性
    VALID_PARTITIONS = ["research", "development", "test"]
    if partition not in VALID_PARTITIONS:
        violations.append(f"非法分区 '{partition}', 合法值: {VALID_PARTITIONS}")
    if partition == "blind":
        violations.append("盲测集永远禁止用于回测/参数选择")
    
    # Rule 2: 日期范围一致性
    # 从 config.yaml 读取 data_partition, assert 脚本使用的日期与之匹配
    
    # Rule 3: 模型参数合规
    model_cfg = config.get("model", {})
    if model_cfg.get("n_estimators", 0) > 100:
        violations.append(f"n_estimators={model_cfg['n_estimators']} > 100, 违反教训#6")
    if model_cfg.get("max_depth", 0) > 3:
        violations.append(f"max_depth={model_cfg['max_depth']} > 3, 过拟合风险")
    
    # Rule 4: 成本下限
    exec_cfg = config.get("execution", {})
    slippage = exec_cfg.get("slippage_bps", 0)
    commission = exec_cfg.get("commission_buy", 0) + exec_cfg.get("commission_sell", 0)
    total_cost_bps = slippage + commission * 10000
    if total_cost_bps < 15:
        violations.append(f"总成本 {total_cost_bps:.1f}bp < 15bp 下限")
    
    if violations:
        msg = f"\n{'='*60}\n  GATE VIOLATION — 脚本 '{script_name}' 被阻止\n{'='*60}\n"
        for v in violations:
            msg += f"  ✗ {v}\n"
        msg += f"{'='*60}\n  修复以上问题后重新运行。\n{'='*60}\n"
        raise GateViolation(msg)
```

#### 2.2.2 `experiment_tracker.py` — 自动实验追踪

```python
"""
experiment_tracker.py — 自动实验记录
每次回测运行自动生成 JSON 记录到 experiments/ 目录。
"""

import json, hashlib, datetime
from pathlib import Path

EXPERIMENTS_DIR = Path("experiments")

def log_experiment(
    script_name: str,
    partition: str,
    config: dict,
    results: dict,
    notes: str = "",
) -> str:
    """
    自动记录实验。返回 experiment_id。
    
    记录内容:
    - config hash (可复现)
    - 完整参数快照
    - 结果指标
    - 时间戳
    - git commit hash (如果是git仓库)
    """
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    
    exp_id = f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    record = {
        "experiment_id": exp_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "script": script_name,
        "partition": partition,
        "config_hash": hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12],
        "parameters": config,
        "results": results,
        "notes": notes,
    }
    
    out_path = EXPERIMENTS_DIR / f"{exp_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    
    return exp_id
```

#### 2.2.3 `config_validator.py` — 配置一致性校验

```python
"""
config_validator.py — 检测 config.yaml 中的矛盾和错误
在 gate.check() 中调用，也可独立运行: py config_validator.py
"""

import yaml

def validate_config(config_path: str = "config.yaml") -> list:
    """
    检查:
    1. 无重复key (用yaml.compose检测)
    2. data_partition 与 DEVELOPMENT_DISCIPLINE.md 一致
    3. model参数在合规范围内
    4. execution 参数完整且合理
    5. blind_test.locked == true 时 trial_count 不再增加
    """
    errors = []
    
    # 检测重复key
    with open(config_path, "r") as f:
        content = f.read()
    # 简单检测: 同一缩进级别出现两次相同key
    # (yaml.safe_load 会静默覆盖，需要手动检测)
    
    config = yaml.safe_load(content)
    
    # 模型参数合规
    model = config.get("model", {})
    if model.get("n_estimators", 0) > 100:
        errors.append(f"model.n_estimators={model['n_estimators']} > 100")
    if model.get("max_depth", 0) > 3:
        errors.append(f"model.max_depth={model['max_depth']} > 3")
    
    # 执行参数完整性
    exec_cfg = config.get("execution", {})
    required_keys = ["initial_capital", "lot_size", "slippage_bps", "top_k"]
    for k in required_keys:
        if k not in exec_cfg:
            errors.append(f"execution.{k} 缺失")
    
    return errors
```

#### 2.2.4 脚本生命周期管理

**当前问题**: 34个脚本，5个回测脚本并存，不知道哪个是"正版"。

**解决方案**: 三层分类

```
scripts/
├── active/          # 当前合法脚本 (≤8个)
│   ├── run_research_backtest.py    # 研究阶段回测 (唯一)
│   ├── run_factor_ic.py            # 因子IC验证 (唯一)
│   ├── run_paper_signal.py         # 模拟盘信号 (唯一)
│   ├── fetch_daily_data.py         # 数据获取 (唯一)
│   ├── fetch_alternative_data.py   # 另类数据获取 (唯一)
│   ├── run_ic_monitor.py           # IC衰减监控
│   ├── export_equity_curve.py      # 净值导出
│   └── init_paper_account.py       # 账户初始化
├── archive/         # 历史脚本 (保留但标记为废弃)
│   ├── run_backtest.py
│   ├── run_corrected_backtest.py
│   ├── run_new_backtest.py
│   ├── run_optimized_backtest.py
│   ├── run_strategy_v4.py
│   └── ... (其余所有)
└── README.md        # 说明: 只有 active/ 中的脚本可以使用
```

**规则**: 
- 新建脚本必须放入 `active/`，且同类功能只允许存在一个
- 需要"修复"时，修改现有脚本，不得新建
- 废弃脚本移入 `archive/` 并在文件头加 `# DEPRECATED: use scripts/active/xxx.py`

#### 2.2.5 合法工作流定义

```
┌─────────────────────────────────────────────────────────────┐
│                    唯一合法开发流程                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 假设阶段 (纸上)                                          │
│     └→ 写下假设: "因子X在Y条件下有Z方向alpha"                 │
│     └→ 记录到 experiments/ 作为 "hypothesis"                 │
│                                                             │
│  2. 因子开发 (research分区)                                   │
│     └→ 在 research 分区 (2018-2022) 计算IC/ICIR              │
│     └→ 通过阈值 → 进入候选池                                 │
│     └→ 未通过 → 放弃，记录原因                                │
│                                                             │
│  3. 组合验证 (development分区)                                │
│     └→ 在 development 分区 (2023-2024H1) 做walk-forward      │
│     └→ 计算超额收益、IR、MaxDD、换手率                        │
│     └→ 通过验证门 → 进入test                                 │
│     └→ 未通过 → 回到步骤2或放弃                               │
│                                                             │
│  4. 最终测试 (test分区, 只跑一次)                             │
│     └→ 在 test 分区 (2024H2-2025H1) 跑一次                   │
│     └→ 结果锁定，不得修改参数后重跑                            │
│     └→ 通过 → 部署到模拟盘                                   │
│     └→ 未通过 → 整个策略废弃                                  │
│                                                             │
│  5. 模拟盘跟踪 (blind分区, 永远不回测)                        │
│     └→ 每日运行 run_paper_signal.py                          │
│     └→ 积累实际交易记录                                      │
│     └→ 3个月后评估是否毕业                                    │
│                                                             │
│  ⚠️ 禁止事项:                                               │
│     - 禁止在blind分区做任何回测                                │
│     - 禁止在test分区迭代参数                                  │
│     - 禁止新建回测脚本(修改现有的)                             │
│     - 禁止不记录实验就跑回测                                   │
│     - 禁止修改config.yaml后不更新config_hash                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 数据分区统一定义

**唯一合法分区** (写入 config.yaml，所有代码从此读取):

```yaml
data_partition:
  research:     {start: "2018-01-01", end: "2022-12-31"}  # 因子开发、IC计算
  development:  {start: "2023-01-01", end: "2024-06-30"}  # 组合验证、参数调优
  test:         {start: "2024-07-01", end: "2025-06-30"}  # 最终测试(只跑一次)
  blind:        {start: "2025-07-01", end: "2026-07-31"}  # 模拟盘跟踪(永不回测)
  full_start: "2018-01-01"
  full_end: "2026-07-31"
```

**注意**: 这是**新的统一定义**，取代当前 config.yaml 的3分区(research/development/blind_test)和 DEVELOPMENT_DISCIPLINE.md 的4分区(train/val/test/blind)。重构完成后，旧定义全部作废，所有代码只从此处读取。选择4分区而非3分区的原因：test集必须独立于development，否则参数调优会污染最终评估。

### 2.4 config.yaml 修正清单

| 问题 | 修正 |
|------|------|
| `max_positions` 重复定义 (5 和 40) | 删除第一个，保留 `max_positions: 40` |
| `n_estimators: 600` | 改为 `n_estimators: 50` |
| `max_depth: 5` | 改为 `max_depth: 2` |
| `min_data_in_leaf: 40` | 改为 `min_data_in_leaf: 200` |
| data_partition 与纪律文档不一致 | 统一为 2.3 节的定义 |
| `blind_test.trial_count: 3` | 保留但标记为"已污染，不再使用此blind期" |
| `use_fundamental: false` | 数据就绪后改为 `true` |

---

## 三、设计方案：数据基础设施重建

### 3.1 设计原则

- **纯免费**: 只使用 akshare 免费接口
- **可恢复**: 所有fetch脚本支持 `--resume`，中断后可续传
- **可验证**: 每次fetch后自动输出覆盖率报告
- **增量式**: 首次全量，后续只拉增量

### 3.2 数据获取优先级

```
优先级 P0 (阻塞一切后续工作):
├── 修复 data_store 上交所缺失 ← 当前最严重问题
└── 获取 CSI1000 指数日线 ← 回测基准

优先级 P1 (因子验证需要):
├── fundamental_cache 扩展到全市场
├── money_flow 历史数据拉取
└── north_flow 历史数据拉取

优先级 P2 (日常积累):
├── 分析师快照每日积累 (cron)
└── 数据日更管道 (每日收盘后)

优先级 P3 (质量提升):
├── PIT universe 构建 (用CSI300+1000月度成分)
└── 退市股票数据补充
```

### 3.3 数据目录统一策略

**当前问题**: 存在两套日线数据:
- `data_cache/` — 1548只，双交易所，数据较旧(约到2026-07-10)
- `data_store/` — 2776只，仅深交所，数据较新(到2026-07-31)

**决策**: 修复后统一使用 `data_store/` 作为唯一数据源:
1. 修复 fetch 脚本，补充上交所股票到 data_store
2. 将 data_cache 中有但 data_store 中没有的股票迁移过去 (主要是688/601/603)
3. 所有代码统一从 data_store 读取
4. data_cache 降级为 `data_cache_archive/`，不再使用

### 3.4 各数据源获取方案

#### 3.3.1 修复 data_store 上交所缺失 (P0)

**问题根因**: `fetch_full_universe.py` 生成股票列表时，代码段遍历逻辑有bug。

**修复方案**:
```python
# 正确的全A股代码生成:
# 上交所: 600000-601999, 603000-603999, 605000-605999, 688000-689999
# 深交所: 000001-000999, 001001-001999, 002001-002999, 003001-003999, 
#         300001-301999

# 更可靠的方案: 直接用 akshare 获取实时股票列表
import akshare as ak
stock_list = ak.stock_info_a_code_name()  # 返回所有A股代码+名称
# 然后过滤ST、上市不足250天等
```

**预期产出**: ~5000只A股 (含ST约4500只可交易)

#### 3.3.2 CSI1000 指数 (P0)

```python
# akshare 获取指数日线:
import akshare as ak
df = ak.index_zh_a_hist(symbol="000852", period="daily", 
                         start_date="20180101", end_date="20260731")
# 存入 data/cache/index_csi1000.parquet
```

#### 3.3.3 fundamental_cache 扩展 (P1)

**当前**: 418只，偏深交所
**目标**: 覆盖 data_store 中所有股票 (~5000只)
**接口**: `ak.stock_financial_analysis_indicator(symbol=code)`
**限制**: 每次调用间隔需 ≥ 1秒 (防封IP)
**预计耗时**: 5000只 × 1秒 ≈ 83分钟
**支持**: `--resume` 跳过已存在的parquet

#### 3.3.4 money_flow 历史 (P1)

**接口**: `ak.stock_individual_fund_flow(stock=code, market="sh/sz")`
**返回**: 每日主力/超大/大/中/小单净流入
**历史深度**: akshare 提供近~1年数据 (不是完整历史)
**限制**: 这是关键约束 — 资金流数据只有约1年历史
**方案**: 
- 拉取所有可交易股票近1年数据
- 从今天起每日积累
- 6个月后可用于IC验证

**⚠️ 硬约束**: akshare 资金流接口只提供近约1年历史。这意味着:
- 2026-08 之前无法对资金流因子做有效的IC验证(样本太短)
- 在积累到至少6个月(≈120个交易日)之前，money_flow 因子在 run_factor_portfolio.py 中继续 `skip: true`
- 这是一个**免费数据源的结构性限制**，无法通过代码解决

#### 3.3.5 north_flow 历史 (P1)

**接口**: `ak.stock_hsgt_individual_em(symbol=code)`
**返回**: 北向资金持股数量/市值历史
**历史深度**: 约2-3年 (2023年起)
**预计耗时**: 与fundamental类似

#### 3.3.6 分析师快照积累 (P2)

**接口**: `ak.stock_profit_forecast_em(symbol="")` (批量，返回全市场)
**频率**: 每日收盘后运行一次
**积累方式**: `data/factor_cache/analyst/snapshot_YYYYMMDD.parquet`
**可用时机**: 积累30天后可计算 revision_30d 因子

#### 3.3.7 PIT Universe (P3)

**已有数据**: `data/cache/universe_000300.json` + `universe_000852.json` (月度成分)
**方案**: 
- 每个调仓日，取当月CSI300+CSI1000成分作为可交易池
- 这消除了幸存者偏差 (退市股在退市前仍在成分中)
- 缺点: 只覆盖1300只，不是全市场
- 替代方案: 用 data_store 中各股票的首次出现日期作为"上市日"近似

### 3.4 数据质量保障

每次数据获取后自动运行校验:

```python
def validate_data_quality(symbol: str, df: pd.DataFrame) -> dict:
    """
    检查:
    1. 空数据 / 行数异常 (< 100行 → 警告)
    2. 日期连续性 (跳空 > 5天 → 标记)
    3. 价格跳变 (单日 > 20% 且非ST → 除权嫌疑)
    4. 成交量为0连续天数 (停牌检测)
    5. 重复日期
    """
```

### 3.5 数据覆盖率目标

| 数据类型 | 当前 | 目标 | 时间线 |
|----------|------|------|--------|
| 日线行情 | 2776只(仅SZ) + 1548只(双交易所) | 5000只(全A) | 1天 |
| 基本面 | 418只 | 3000只(可交易) | 2天 |
| 资金流 | 0 | 3000只(近1年) | 1天 |
| 北向资金 | 0 | 3000只(近2年) | 1天 |
| 分析师 | 1天 | 每日积累 | 30天后生效 |
| 指数基准 | 无 | CSI300+CSI500+CSI1000 | 1小时 |
| PIT universe | 无 | CSI300+1000月度 | 已有数据,需接入 |

---

## 四、实施顺序

```
Phase A: 纪律重建 (Day 1-2)
├── A1: 创建 gate.py + experiment_tracker.py + config_validator.py
├── A2: 修正 config.yaml (统一分区、修正参数、删除重复key)
├── A3: 脚本归档 (active/ vs archive/)
├── A4: 更新 DEVELOPMENT_DISCIPLINE.md v2 (引用代码而非自定义值)
└── A5: 验证 — 用gate.py检查现有所有脚本，输出合规报告

Phase B: 数据修复 (Day 2-4)
├── B1: 修复 fetch_full_universe.py 上交所bug + 重新拉取
├── B2: 获取 CSI1000/CSI300 指数日线
├── B3: 扩展 fundamental_cache 到全市场
├── B4: 拉取 money_flow 历史
├── B5: 拉取 north_flow 历史
└── B6: 数据质量校验 + 覆盖率报告

Phase C: 管道整合 (Day 4-5)
├── C1: 接入 PIT universe (CSI300+1000月度成分)
├── C2: 替换基准为 CSI1000 指数
├── C3: 设置每日数据更新调度
├── C4: 设置分析师快照每日积累
└── C5: 端到端验证 — 从数据到信号跑通一次

Phase D: 回测修复 (Day 5+, 用户确认后)
├── D1: 在 research 分区重新计算因子IC
├── D2: 在 development 分区做walk-forward
├── D3: 通过验证门后，在 test 分区跑一次
└── D4: 结果锁定，决定是否部署模拟盘
```

---

## 五、成功标准

| 指标 | 标准 |
|------|------|
| 数据覆盖 | ≥ 4000只A股日线 + 基本面 + 资金流 |
| 纪律执行 | gate.py 阻止所有违规脚本运行 |
| 实验追踪 | 每次回测自动生成 experiment JSON |
| 脚本整洁 | active/ ≤ 8个脚本，无冗余 |
| 端到端 | `py scripts/active/run_paper_signal.py` 一条命令跑通 |
| 可复现 | 任意 experiment JSON 可以精确复现结果 |

---

## 六、代码审计新增发现（补充）

### 6.1 训练-生产断裂 (CRITICAL)

**现状**: 
- `model/pipeline.py` 训练 LightGBM ensemble → 回测 → 丢弃模型（无save）
- `run_paper_signal_v3.py` 从 `p5_portfolio_report.json` 读IC权重 → 线性加权 → 信号
- 两者是完全不同的系统，ML模型从未被生产使用

**修复方案**: 
- 短期(本次重构): 明确选择**IC加权线性**作为生产方法（简单、可解释、不需要模型artifact）
- 将 `model/pipeline.py` 定位为**研究工具**（用于验证ML是否比线性好），不进入生产路径
- 长期(如果ML证明更好): 加入模型序列化(`joblib.dump`)和版本管理

### 6.2 未复权数据缺口 (HIGH)

**现状**: 仅176/1550只有未复权数据，涨跌停检测对88.6%股票失效
**修复**: 在 Phase B 数据修复中，为所有 data_cache 股票补充未复权日线
**接口**: `ak.stock_zh_a_hist(symbol=code, adjust="")` (adjust=""即未复权)

### 6.3 PEAD缓存过期 (HIGH)

**现状**: 止于2023-12-31，2024-2026无数据
**修复**: 重新运行 `scripts/fetch_events.py` 或专用PEAD fetch，补充2024Q1-2026Q2
**接口**: `ak.stock_yjyg_em(date="20240331")` 等

### 6.4 废弃架构清理 (HIGH)

**现状**: `src/quant/` 是一个完整的平行代码库，6个脚本引用它
**修复**: 
- 整体移入 `archive/src_quant/`
- 6个引用脚本移入 `scripts/archive/`
- 在 README 中说明已废弃

### 6.5 日志框架 (HIGH)

**现状**: 全部 `print()`，无结构化日志
**修复**: 
- 创建 `logger.py`，基于 Python `logging` 模块
- 所有模块改用 `logger.info()` / `logger.warning()` / `logger.error()`
- 输出到 console + `logs/quant_YYYYMMDD.log`
- 关键事件(信号生成、执行、异常)同时写入 `data/events.jsonl`

### 6.6 基准修正 (HIGH)

**现状**: 用"有数据的子集"均值作为基准
**修复**: 
- 获取 CSI1000 指数日线 (Phase B P0任务)
- 回测基准改为 CSI1000 收益率
- 保留全池等权作为辅助参考

---

## 七、更新后的完整缺失清单

按严重度排序：

| # | 严重度 | 缺失项 | 修复阶段 |
|---|--------|--------|----------|
| 1 | CRITICAL | 训练-生产断裂，ML模型无处部署 | Phase A (明确路径选择) |
| 2 | CRITICAL | data_store 上交所缺失 | Phase B |
| 3 | CRITICAL | 无PIT universe (幸存者偏差) | Phase C |
| 4 | CRITICAL | 数据分区矛盾 (config vs 纪律文档) | Phase A |
| 5 | HIGH | 无代码门禁 (gate.py) | Phase A |
| 6 | HIGH | 未复权数据仅11%覆盖 | Phase B |
| 7 | HIGH | PEAD缓存止于2023 | Phase B |
| 8 | HIGH | money_flow / north_flow 为空 | Phase B |
| 9 | HIGH | 无实验追踪系统 | Phase A |
| 10 | HIGH | 无模型持久化 | Phase A (明确不需要) |
| 11 | HIGH | 6个脚本用废弃架构 | Phase A |
| 12 | HIGH | 无日志框架 | Phase A |
| 13 | HIGH | 基准计算违规 | Phase B + C |
| 14 | MEDIUM | fundamental_cache 仅27%覆盖 | Phase B |
| 15 | MEDIUM | 分析师数据仅1天 | Phase C (每日积累) |
| 16 | MEDIUM | 测试覆盖122行/2类 | Phase C |
| 17 | MEDIUM | config.yaml 重复key | Phase A |
| 18 | MEDIUM | 无自动化日更管道 | Phase C |
| 19 | LOW | 无lint/type-check | 延后 |
| 20 | LOW | factor_cache/ 几乎为空 | 非关键路径 |

---

## 八、不做的（明确排除）

- ❌ 付费数据源 (Wind/Choice/Tushare Pro)
- ❌ 实盘交易接口
- ❌ 分钟级回测
- ❌ 多策略聚合 (signal_hub.py 暂不启用)
- ❌ LLM因子 (成本高、不稳定)
- ❌ 港股 (先只做A股)
- ❌ 修复回测结果 (用户明确说"暂时先不进行修复")
