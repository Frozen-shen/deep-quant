# 分钟因子独立叠加层（方案B）设计文档

> 日期: 2026-08-04
> 状态: 待审查

## 1. 背景与问题

分钟频因子（10 个，`min_*`）ICIR 0.24-0.72（强），但数据仅 2022-01 起（baostock 限制）。

方案C fold 结构中，fold 1-2 训练期（2015-2020/2021）无分钟数据 → 分钟因子在这些 fold 的 ICIR 无法计算 → 命中 2/5 < FOLD_MIN_HITS=3 → **被稳定因子筛选排除**。

v5 实测：分钟因子在**有数据的 fold（fold 4-5）中 2/2 = 100% 有效**。问题纯粹是"fold 结构 vs 数据时间跨度"不匹配，而非因子无效。

## 2. 方案选择

**方案B（独立分钟因子验证层）**——用户已确认：

- **不改主 fold 筛选逻辑**（40 个稳定因子体系不动，纪律不污染）
- **新增独立通道**：分钟因子用 fold 4-5 验证期独立验证，作为"增强层"叠加到打分
- **可归因、可调参、可回退**（config 开关 + 权重参数）

放弃方案A（按可用fold计算命中——放宽全局筛选标准风险高）和方案C（数据前移——东财接口不可靠，作为未来可选项）。

## 3. 设计

### 3.1 架构

```
主通道（不变）: 182 因子 → fold 1-5 筛选 → 40 稳定因子 → ICIR 权重 → score_stocks
                                                                      ↓
分钟通道（新增）: 10 个 min_* 因子 → fold 4-5 独立验证 → 验证 ICIR 权重 ──→ 线性叠加
                                                                      （λ 可调）
```

### 3.2 组件

#### A. 分钟因子独立验证（新函数 `validate_minute_factors`）

- **验证窗口**：fold 4-5 训练期（2022-01 ~ 2023-12 / 2022-01 ~ 2024-12），因为只有这些训练期含分钟数据
- **方法**：对每个 min_* 因子，在验证窗口内计算 ICIR（与主通道 `compute_icir_weights` 同方法）
- **保留标准**：|ICIR| ≥ 0.3（比主通道 0.05 严格，因为样本少——fold 4-5 训练期 ~2-3 年，观测点 ~30-40 个）
- **输出**：`{min_factor: validated_icir}` 字典

#### B. 打分叠加（修改 `score_stocks`）

- 签名增加 `minute_weights: dict | None = None` 和 `minute_lambda: float = 0.3`
- 逻辑：主因子分计算后，若 `minute_weights` 非空且当日有分钟数据，则：
  ```
  综合分 = 主因子分 + minute_lambda × 分钟因子加权分
  分钟因子加权分 = Σ(min_icir × z_score(min_factor)) / Σ|min_icir|
  ```
- 分钟因子分独立 z-score（不参与主因子的归一化），避免尺度污染

#### C. 配置（config.yaml 新增）

```yaml
# ── 分钟因子叠加层 (方案B, 2026-08-04) ──
minute_layer:
  enabled: true        # 总开关
  lambda: 0.3          # 叠加权重 (主分:分钟分 = 1:λ)
  min_icir: 0.3        # 分钟因子保留的 ICIR 门槛
  validate_folds: [4, 5]  # 独立验证使用的 fold 训练期
```

### 3.3 数据流

```
precompute_factor_panels (已有, 合并 min_* 面板)
  → run_fold_analysis (已有, 主通道筛选, 分钟因子不参与)
  → validate_minute_factors (新增, fold 4-5 训练期验证分钟因子)
  → run_fold_test (修改, TEST 权重 = 主权重 + 分钟叠加层)
  → run_backtest (已有, score_stocks 内部叠加)
```

### 3.4 纪律约束

- 分钟因子**不进入**稳定因子列表（40 个）——它始终是"叠加层"，主组合结构不变
- 分钟因子权重来自 fold 4-5 训练期（2022-2024），TEST（2025-2026）是真正的样本外
- 无前视：分钟因子 PIT 计算（`compute_minute_factors` 的 as_of_date 参数天然截止）
- 可回退：`minute_layer.enabled: false` 即回到 v5 行为

## 4. 测试

1. `test_validate_minute_factors`: 用 mock 数据验证 ICIR 计算和保留标准
2. `test_score_stocks_with_minute`: 验证叠加逻辑（主分 + λ×分钟分）
3. `test_minute_layer_disabled`: enabled=false 时行为与 v5 完全一致
4. 全量回归: `py -m pytest tests/ -q`

## 5. 验证指标

重跑全量后对比：

| 指标 | v5 (无分钟层) | v6 (有分钟层) | 目标 |
|------|--------------|--------------|------|
| TEST 超额 | -9.6% | ? | 改善 |
| TEST IR | -0.29 | ? | 改善 |
| TEST Sharpe | 0.96 | ? | ≥ 保持 |
| 分钟因子贡献 | 0 | 可归因 | 正贡献 |

## 6. 风险与缓解

| 风险 | 缓解 |
|------|------|
| fold 4-5 样本少 → ICIR 不稳 | ICIR 门槛 0.3（严格）；结果只作为叠加层权重，不改变主组合 |
| 叠加层引入过拟合 | λ 可调（默认 0.3 保守）；enabled 开关可完全回退 |
| 分钟数据覆盖 66% → 部分股票无分钟分 | 无分钟数据的股票分钟分为 0，仅主分生效（自然降级） |
