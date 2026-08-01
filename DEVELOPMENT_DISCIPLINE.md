# 量化开发纪律手册

> 本文件是最高优先级约束。任何回测、实验、报告必须遵守以下规则。
> 违反任何一条的结果自动作废，不需要讨论。

---

## 第一条：数据分区（不可违反）

```
全量数据: 2018-01-01 → 2026-07-10

训练集 (Train):  2018-01-01 → 2023-12-31  (参数学习、IC计算)
验证集 (Val):    2024-01-01 → 2025-06-30  (策略选择、超参调优)
测试集 (Test):   2025-07-01 → 2025-12-31  (最终评估，只跑一次)
盲测集 (Blind):  2026-01-01 → 2026-07-10  (部署后跟踪，永远不用于决策)
```

**规则:**
- 测试集只允许跑一次。跑完后锁定，不得修改参数后重跑。
- 盲测集在任何情况下不得用于回测、参数选择、策略设计。
- 每个脚本必须在开头声明使用哪个分区，并assert日期范围。

```python
# 每个回测脚本必须包含:
PARTITION = "val"  # train / val / test / blind
DATE_RANGES = {
    "train": ("2018-01-01", "2023-12-31"),
    "val":   ("2024-01-01", "2025-06-30"),
    "test":  ("2025-07-01", "2025-12-31"),
    "blind": ("2026-01-01", "2026-07-10"),
}
start, end = DATE_RANGES[PARTITION]
assert PARTITION != "blind", "盲测集禁止用于回测"
```

---

## 第二条：基准（唯一且固定）

**基准 = 全池等权（所有可交易股票的等权组合）**

- 不得使用子集（top-30、文件最大、任何筛选后的切片）作为基准
- 不得更换基准来让数字好看
- 基准在每个回测脚本中自动计算，不允许手动指定

```python
# 基准计算（唯一合法方式）:
benchmark_return_t = mean(all_stock_returns_on_date_t)
# 其中 all_stocks = 当日所有可交易股票（非ST、非停牌、上市>60天）
```

---

## 第三条：Universe（消除幸存者偏差）

- 不得使用"当前存在的股票"回测过去
- 必须使用Point-in-Time (PIT) 成分：每个调仓日只用当时存在的股票
- 如果无法获取完整退市数据，必须在报告中声明："本回测存在潜在幸存者偏差"
- 禁止按数据长度/文件大小筛选股票

---

## 第四条：验证门（结果必须通过才能报告）

任何回测结果在报告之前，必须通过以下自动化检验：

```python
def validate_result(result) -> bool:
    checks = []
    
    # 1. 换手率不能为0（策略必须在交易）
    checks.append(result.avg_turnover > 0.05)  # >5%/rb
    
    # 2. 换手率不能过高（不是随机交易）
    checks.append(result.avg_turnover < 0.95)  # <95%/rb
    
    # 3. Sharpe不能异常高（>2.5大概率有问题）
    checks.append(result.sharpe < 2.5)
    
    # 4. IR不能异常高（>1.5大概率有问题）
    checks.append(result.ir < 1.5)
    
    # 5. 必须有年度拆分（不能只看总数）
    checks.append(len(result.yearly_returns) >= 3)
    
    # 6. 年度胜率必须报告
    win_rate = sum(1 for y in result.yearly_returns.values() if y > 0) / len(result.yearly_returns)
    checks.append(True)  # 只要求报告，不要求通过
    
    # 7. 成本必须>15bp round-trip（不能放水）
    checks.append(result.cost_rate >= 0.0015)
    
    # 8. 持仓数量必须>0
    checks.append(result.n_positions > 0)
    
    return all(checks)
```

**未通过验证门的结果禁止写入任何报告或commit message。**

---

## 第五条：实验记录（可复现）

每次回测必须记录：

```json
{
    "experiment_id": "exp_20260731_001",
    "timestamp": "2026-07-31T12:00:00",
    "partition": "val",
    "universe": "all_1372",
    "benchmark": "full_ew",
    "strategy": "ic_linear_63d_embargo",
    "parameters": {
        "top_k": 30,
        "ic_lookback": 252,
        "decay_halflife": 126,
        "embargo_days": 63,
        "cost_model": "period_dependent"
    },
    "results": {
        "ann_return": 0.155,
        "ann_excess": -0.072,
        "sharpe": 0.71,
        "ir": -0.845,
        "max_drawdown": -0.299,
        "avg_turnover": 0.0,
        "yearly": {"2024": 0.05, "2025": -0.12}
    },
    "validation": {
        "passed": false,
        "failures": ["turnover_too_low"]
    },
    "notes": "换手控制bug导致0%换手，结果无效"
}
```

存储位置: `experiments/YYYY-MM-DD_exp_NNN.json`

---

## 附录：已确认的教训清单

| # | 教训 | 首次犯错 | 重犯次数 |
|---|------|---------|---------|
| 1 | 不按文件大小选池 | v2.0 | 1 |
| 2 | 不用子集当基准 | v2.0 | 1 |
| 3 | IC必须有embargo | v2.0 | 1 |
| 4 | 不碾数据分区 | v2.0 | 2 |
| 5 | 不放水成本 | v2.0 | 1 |
| 6 | 不用600树ML | v1.0 | 0 |
| 7 | 换手=0必须报警 | v4 | 首次 |
| 8 | IR>1.5必须质疑 | v3 | 首次 |

---

## 执行方式

1. 将验证门代码写入 `src/quant/evaluation/gate.py`
2. 所有回测脚本 import 并在输出前调用
3. CI/CD中: 如果commit message包含回测数字但没有对应experiment JSON → 拒绝
4. 每月审查: 对比paper trading实际收益 vs 回测预期，偏差>2% → 策略审查
