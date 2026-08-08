# 波动率分层 × Regime 乘数（股票池分域）设计文档

- 日期: 2026-08-09
- 状态: 已批准（用户 2026-08-09 确认方案 B 软偏好）
- 关联实验: v6 (+2.84%) / v8-v9 (+2.54%, aux 因子有效但强度弱) / v8.1 行业中性化 (-8.2%, 已回退)

## 1. 背景与动机

### 1.1 项目现状
- 稳定 40 因子组合的 alpha 来源是**全市场小市值/低波/低流动性反转风格暴露**（v8.1 行业中性化实验证实：削除风格暴露 = 削除 alpha，fold 均值 +2.54% → -8.2%）
- 新增 7 个辅助因子（两融/解禁/龙虎榜/大宗）中 2 个统计有效（aux_margin_change_5d、aux_lockup_pressure_30d，5/5 folds 命中）但 ICIR 强度不足以进入稳定集，FOLD_MAX_FACTORS 40→50 无效果（v9 与 v8 完全相同）——弱因子在 ICIR 权重体系下天然被压制
- regime_detector 已有双变量状态检测：`detect_v2(date) -> (regime, vol_pct)`，其中 vol_pct = 市场 60 日已实现波动率百分位（0-1）

### 1.2 动机
用户提出：交易前对股票做波动率分层（高/低波池），根据市场风格（regime）在股票池中抉择。行业对应手法为 volatility-managed portfolio / style timing / volatility-sorted portfolios。**选股层的波动率偏好是一条不依赖因子 ICIR 的独立通道**——因子权重层面无法表达的弱因子信息，可以通过"市场状态 → 选股池偏好"表达。

## 2. 设计决策

### 2.1 机制：软偏好（加权）而非硬过滤
- 硬过滤（高波动市场删高波股）会导致调仓边界股票池剧变、换手率飙升触发 50% 上限拒单、回测失真、易过拟合
- 软偏好平滑、与现有 regime 乘数机制同构（`get_weight_multipliers` 已对 momentum/reversal/value/quality 类别做乘数调整）
- 行业实证（Moreira & Muir 2017 等）标准做法为加权而非硬切换

### 2.2 分层：波动率三分位而非 K-means 聚类
- 候选股按自身 60 日收益波动率（日线 close pct_change std）跨截面分位：
  - 低波档: vol < 30 分位
  - 中波档: 30 ≤ vol ≤ 70 分位
  - 高波档: vol > 70 分位
- 分位分层无聚类噪声、可解释、防过拟合、计算零成本

### 2.3 乘数：作用于选股分（score_stocks 输出后、ranker 前）
- 与 v9b 的 inv_vol（权重层，正交）不冲突，可叠加
- 默认乘数（config 可调）：

| 市场状态 | 低波档 | 中波档 | 高波档 |
|---------|:---:|:---:|:---:|
| 高波动 (vol_pct > 0.70) | ×1.5 | ×1.0 | ×0.5 |
| 中性 (0.30 ≤ vol_pct ≤ 0.70) | ×1.0 | ×1.0 | ×1.0 |
| 低波动 (vol_pct < 0.30) | ×0.8 | ×1.0 | ×1.2 |

### 2.4 与现有 regime 机制的关系
- 现有 `get_weight_multipliers` 调整**因子权重**（momentum/reversal/value/quality 类别）
- 新增 pool_filter 调整**股票分数**（按波动率档）
- 两者独立、正交、可同时启用

## 3. 架构与组件

### 3.1 新增组件
- `pool_filter.py`（新模块，被 run_walkforward_backtest 引用，符合模块准入规则）：
  - `vol_bucket(scores, all_data, today) -> dict`：按候选股 60 日波动率分位返回档位标签 {sym: 'low'|'mid'|'high'}
  - `apply_pool_filter(scores, buckets, vol_pct, multipliers) -> dict`：分数 × 档位乘数
- `run_walkforward_backtest.py` 改动：
  - `run_backtest` 增加 `pool_filter: str = "none"` 参数
  - 调仓日 score_stocks 之后、ranker.rank 之前调用 apply_pool_filter
- `config.yaml` 新增：
  ```yaml
  pool_filter:
    enabled: false        # v10 实验时置 true
    low_vol_mult: 1.5     # 高波动市场低波档乘数
    high_vol_mult: 0.5    # 高波动市场高波档乘数
    low_vol_up: 0.8       # 低波动市场低波档乘数
    high_vol_up: 1.2      # 低波动市场高波档乘数
  ```

### 3.2 PIT 安全
- 股票波动率只用 ≤ today 的日线数据（无前视）
- vol_pct 由 regime_detector 在 T 日检测（历史窗口，无前视）
- 与现有 score_stocks/regime 乘数同一时点应用

### 3.3 错误处理
- 波动率不足（<10 个观测）：该股票不参与分档（按中波 ×1.0 处理）
- 候选股全部无波动率数据：pool_filter 静默跳过（行为等于 none）
- vol_pct 缺失（regime_detector 数据不足）：跳过调整

## 4. 测试计划

### 4.1 单元测试（tests/test_pool_filter.py）
1. config 读取默认值
2. vol_bucket 分位正确（构造已知波动率股票）
3. apply_pool_filter 乘数正确（高波动市场低波股分上调）
4. 边界：波动率不足股票不崩溃
5. disabled 时行为与 none 完全一致

### 4.2 回测验证（v10）
- `pool_filter.enabled: true`，其余配置 = v9 基线
- folds-only 跑 5-fold，对比 v9（+2.54%）：
  - 通过标准：fold 均值提升 > 0 且 bootstrap 95% CI 不显著变差
  - 附加观察：换手率变化（不应大幅上升）、高波/低波档入选比例变化

### 4.3 叠加验证（v10b，可选）
- pool_filter + inv_vol 权重同时启用，对比单独效果

## 5. 风险与回退
- 乘数过强 → 过度偏袒低波股，风格更单一：乘数温和（1.5/0.5），config 可调
- 与 v8.1 行业中性化失败教训一致的风险（偏好极端化）：软偏好 + bootstrap 佐证防过拟合
- 不触碰 TEST 锁（folds-only 验证）
- 回退：config enabled: false 即完全恢复 v9 行为

## 6. 成功标准
1. v10 fold 均值 > v9（+2.54%），bootstrap 95% CI 不显著变差
2. 换手率无剧变（< 基线的 1.5 倍）
3. 单元测试全过
4. 不消耗 TEST 锁
