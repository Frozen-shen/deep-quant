# 5min 上线价值探索：风控层校准 + 执行层 VWAP（设计）

> 2026-08-11 · 承接 5min 上线价值探索（v19/v20/v21b/v22 实验后）
> 前置结论：5m 因子在月频选股无增益（3 次确认）；但实盘价值在**风控层**与**执行层**

## 背景与动机

1. **风控层（方案A）**：v20 用 rv_5m（市场截面已实现波动率）替代 daily vol_pct 时，
   fold_3（2022 熊市）+6pp 年化转正，但模拟考 -2.2pp。
   事后定位：**rv_5m 与 daily 的分布量纲不同**（daily vol60 均值 0.238 / std 0.071，
   rv_5m 均值 0.310 / std 0.044），pool_filter/vol_target 的固定阈值
   （0.30 低波 / 0.70 高波 / 0.85 极高波）是为 daily 分布校准的，
   rv_5m 上 0.70 阈值只对应 **0.21 分位** → 2025-26 几乎全程误判"低波弹性"，
   组合更激进 → 模拟考恶化。**失败根因是阈值标定，不是 rv_5m 信息无效**。

2. **执行层（方案B）**：年化单边换手 326%，滑点(30bps)+佣金总成本 ≈ 2.28%/年。
   实盘用 5m/15m 数据做 VWAP 拆单可把滑点从 30bps 降到 ~10-15bps（残差），
   年化节省 0.33~0.65%。这是 5m 数据在实盘中最扎实的用途（头部私募标准做法），
   与选股正交，纯增益。

## 方案A：rv_5m 滚动分位校准（regime_detector 修复）

### 设计
- 现状（regime_detector.py detect_v2）：rv_5m 路径 `vol_pct = (roll60 <= vol60).mean()`
  用的是**全历史分位**，但 rv_5m 序列只有 2022+ 历史，且分布与 daily 不同 →
  阈值错配。
- 修复：**滚动窗口分位**。`vol_pct = 当前值在过去 252 交易日中的百分位`
  （rolling 252 天，min_periods 60），使 vol_pct 语义为"近一年波动位置"，
  与 daily 路径的阈值语义对齐（0.70 = 近一年 70 分位）。
- daily 路径保持不变（保 v15 基线可比），只改 rv_5m 路径。
- config `regime.vol_source` 已有开关（daily/rv_5m），无需新增配置。

### 改动点
- `regime_detector.py`：
  - `_load_market_rv()` 增加滚动分位序列预计算（`rolling(252).apply(percentile)` 向量化）
  - `detect_v2()` rv_5m 分支：取当日滚动分位值（≤today 最后一个），不再用全历史分位
- 实验：`vol_source: rv_5m` → 全链路回测 v23 → 对比 v15 基线
- 预期：保留 fold_3 熊市改善，修复模拟考恶化（若两者都成立 → 进生产）

### 验证标准
- fold_3（2022）超额 ≥ v15 的 +5.9%（熊市防守不退化）
- 模拟考超额 > v15 的 +0.7%（修复阈值错配）
- 两者同时满足才进生产；否则维持 daily

## 方案B：执行层 VWAP 拆单（SimpleBacktest 执行价改造）

### 设计（v24 实施版修正, 2026-08-11）
- **数据源修正**：初版设计用分钟数据算 VWAP (Σclose×vol/Σvol)，实测发现
  分钟数据（baostock 前复权）与日线（腾讯 qfq）**复权基准不同**（比值 0.943），
  且 2022 前无分钟数据。最终改用：**未复权日线 amount/volume（真实 VWAP）×
  复权因子（复权close/未复权close）** → 全历史覆盖（2018 起）、100% 落在
  复权 [low,high] 区间内验证通过。
- **残差滑点设计修正**：初版把残差直接减在 VWAP 上（方向错误，买卖都减）。
  修正：`_exec_price` 只返回 VWAP 基准价，残差由 `_apply_slippage` 的
  slippage_bps 承担（vwap 模式下 slippage = vwap_residual_bps，买上浮/卖下沉）。
- **VWAP vs open 实测**：日度偏差平均仅 +0.03%（各年 ±0.1-0.2%）→
  VWAP 拆单价值不在"价格时点选择"，而在**滑点假设**（30bps 一次性开盘单
  → 10-15bps 拆单残差）。
- 配置：`execution.execution_price: "open"|"vwap"` + `vwap_residual_bps: 10`

### 改动点
- `model/engine.py` SimpleBacktest：`_exec_price`（open/vwap 基准）+ 残差滑点
- `scripts/active/run_walkforward_backtest.py`：`_build_vwap_panel`
  （未复权×复权因子，懒加载缓存）+ bt_config 接线
- 实验：`execution_price: vwap` → 全链路回测 v24 → 对比 v15（唯一差异=执行价）
- 预期：年化 +0.3~0.65%（滑点 30→10bps），fold/模拟考整体上移

### 验证标准
- 模拟考超额 ≥ v15 +0.7%（纯执行增益，选股逻辑不变）
- 若 fold 均值/模拟考均 ≥ v15 → 进生产

## 风险与边界
- 方案A：滚动 252 分位在小样本早期（2022 前 60 天）回退 daily；阈值对齐是近似，
  若仍不显著则维持 daily
- 方案B：VWAP 是当日实现价（与 open 同为 T+1 可知），无前视；
  残差滑点 10bps 是中性假设，可做 5/10/15 敏感性
- 两方案独立，可分别进出生产

## 验收
- 透传门禁（check_param_passthrough.py）+ 冒烟（--sample 50）+ pytest 61 项
- 结果存档 walkforward_results_v2x_*.json + experiment_tracker 记录
