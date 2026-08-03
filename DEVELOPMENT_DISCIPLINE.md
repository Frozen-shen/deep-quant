# 量化开发纪律手册 v2

> 本文件是最高优先级约束。
> **执行方式: 代码强制 (`gate.py`)，不依赖人工自觉。**
> 违反任何一条的结果自动作废，不需要讨论。

---

## 第一条：数据分区（代码强制）

分区定义在 `config.yaml` 的 `data_partition` 字段中（唯一源）。

| 分区 | 日期范围 | 用途 | 限制 |
|------|----------|------|------|
| research | 2015-01-01 → 2024-12-31 | 因子开发、IC计算、Walk-Forward folds 训练/验证 | 可反复使用 |
| development | 2025-01-01 → 2026-06-30 | 终极 TEST（方案C holdout） | **只跑一次** |
| test | 2026-07-01 → 2026-12-31 | 预留验证（数据未到） | **只跑一次** |
| blind | 2027-01-01 → 2027-12-31 | 模拟盘跟踪 (daily_pipeline) | **永不回测** |

**方案C验证协议 (v4, 2026-08-03)**:
- Walk-Forward 内部验证 (research 期内): 5 folds, 训练期逐年扩展
  (Train 2015-2019→Val 2020, ... Train 2015-2023→Val 2024)
- 因子保留标准: |ICIR| ≥ 0.05 在 ≥3/5 folds 中达标 (`run_walkforward_backtest.py` FOLD_MIN_HITS=3)
- 终极 TEST: 稳定因子中位数 ICIR 权重, 2025-01-01 ~ 2026-06-30, 只跑一次
- Blind: 2027-01 起 daily_pipeline 每日跟踪, 永不回测
- universe: 流动性 PIT (全市场 + 上市≥250交易日 + 20日均成交额≥500万),
  由 `data/pit_universe.py` 构建月度快照, 天然无幸存者偏差

**代码强制**: `gate.py` 会阻止使用 "blind" 分区的脚本运行。

```python
# 每个回测脚本开头必须包含:
from gate import check, load_config
config = load_config()
check(partition="research", script_name=__file__, config=config)
```

---

## 第二条：模型参数（代码强制）

| 参数 | 上限 | 理由 |
|------|------|------|
| n_estimators | ≤ 100 | 教训#6: 600树过拟合 |
| max_depth | ≤ 3 | 极端正则化 |
| min_data_in_leaf | ≥ 200 | 防止叶节点过拟合 |

**代码强制**: `gate.py` 检查 config.yaml 中的模型参数。超限 → 脚本被阻止。

**生产路径**: IC加权线性 (`model.type: linear`)，不使用ML模型。
**研究路径**: LightGBM 仅用于验证"ML是否比线性好"，不进入生产。

---

## 第三条：成本模型（代码强制）

总交易成本 (滑点 + 双边手续费) ≥ 15bp。

当前设定: 滑点30bp + 手续费(2.5bp+7.5bp) = 40bp。合规。

**代码强制**: `gate.py` 计算总成本，低于15bp → 脚本被阻止。

---

## 第四条：实验记录（代码强制）

每次回测/实验必须调用 `experiment_tracker.log_experiment()`。
记录自动写入 `experiments/exp_YYYYMMDD_HHMMSS_XXXX.json`。

记录内容: config hash、完整参数、结果指标、时间戳、备注。

**规则**: 
- 不记录实验就跑回测 = 纪律违反
- 实验记录不可删除、不可修改

---

## 第五条：脚本管理

- 只有 `scripts/active/` 中的 8 个脚本可以使用
- 同类功能只允许一个脚本
- 需要"修复"时修改现有脚本，**不得新建**
- 废弃脚本移入 `scripts/archive/`，在文件头加 `# DEPRECATED`
- `scripts/archive/` 中的脚本运行结果不可信、不可报告

---

## 第六条：基准（唯一且固定）

**主基准 = CSI1000 指数收益率**

- 从 `data/cache/index_csi1000.parquet` 读取
- 不得使用子集等权作为主基准
- 全池等权可作为辅助参考（标注为"辅助"），但报告以CSI1000为准
- 不得更换基准来让数字好看

---

## 第七条：PIT Universe（消除幸存者偏差）

- 每个调仓日只用当时存在的股票
- 使用 `data/pit_universe.py` 的 `get_universe(date)` 获取合法股票池
- 禁止按数据长度/文件大小筛选股票
- 如果无法获取完整退市数据，必须在报告中声明

---

## 第八条：验证门（结果必须通过才能报告）

任何回测结果在报告之前，必须通过以下检验：

1. 换手率 > 5%/调仓期（策略必须在交易）
2. 换手率 < 95%/调仓期（不是随机交易）
3. Sharpe < 2.5（异常高大概率有问题）
4. IR < 1.5（异常高大概率有问题）
5. 必须有年度拆分（不能只看总数）
6. 成本 ≥ 15bp round-trip
7. 持仓数量 > 0
8. **Bootstrap 95% CI 排除零**（年化超额和IR）
9. **t-stat > 1.645**（单尾, 样本外期）或 **> 2.54**（Bonferroni校正, 开发期）
10. **MaxDD < 15%**（收紧, 原20%踩线）

**未通过验证门的结果禁止写入任何报告。**

### Go/No-Go 判定标准 (v2 — 2026-08-02 修正)

| 标准 | 阈值 | 说明 |
|------|------|------|
| 年化超额 | >5% | 覆盖成本后仍有意义 |
| IR | >0.5 | 风险调整后收益 |
| t-stat (OOS) | >1.645 | 单尾 p<0.05 |
| t-stat (Dev) | >2.54 | Bonferroni 校正 (9次试验) |
| Bootstrap 95% CI | 排除0 | 年化超额 + IR 两项 |
| MaxDD | <15% | 收紧 (原20%踩线) |
| 因子因果性 | >60% 通过 | Expanding window test |
| 模拟盘跟踪 | ≥6个月 | 真正的样本外 |

**全部通过 → Go (可启动模拟盘)**
**任一不通过 → No-Go (继续观察, 不宣称alpha)**

---

## 第九条：唯一合法开发流程

```
1. 假设 → 写下: "因子X在Y条件下有Z方向alpha"
2. 因子开发 (research分区) → IC/ICIR验证
3. 组合验证 (development分区) → walk-forward回测
4. 最终测试 (test分区, 只跑一次) → 结果锁定
5. 模拟盘 (blind分区) → 每日运行, 永不回测
```

**禁止事项:**
- ❌ 在blind分区做任何回测
- ❌ 在test分区迭代参数
- ❌ 新建回测脚本
- ❌ 不记录实验就跑回测
- ❌ 修改config.yaml后不验证gate
- ❌ 报告未通过验证门的结果

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
| 9 | 盲测偷看后必须作废 | v4 | 1 | ✓ config.contaminated |
| 10 | 多次实验必须Bonferroni校正 | v4 | 1 | 实验追踪 |

---

## 执行架构

```
gate.py              ← 硬门禁 (脚本入口检查)
experiment_tracker.py ← 实验自动记录
config_validator.py  ← 配置一致性校验 (集成在gate.py中)
logger.py            ← 结构化日志
config.yaml          ← 唯一配置源
scripts/active/      ← 唯一合法脚本
```

---

## 第十条：模块准入（防止平行代码库复发）

新增模块必须满足其一才能合并：
- 被 `scripts/active/` 或 `scripts/daily_pipeline.py` 引用；或
- 被 `web/api/` 或 `web/` 前端引用；或
- 有对应 `tests/` 测试覆盖。

否则不得进入根目录（放 `archive/` 或独立研究目录）。
