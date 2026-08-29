# 预期差感知层 + 行业轮动层 设计文档 (2026-08-17)

## 背景与动机

v27（50 稳定因子，小市值/低流动/低波/价值风格）在 2024-2026 AI 硬件成长行情中
系统性缺席（持仓 0 只科技龙头），年化超额 +29.2% 虽不差但暴露单一风格锚定问题。

多风格 sleeve 实验（2026-08-16，分支已合并，`styles.enabled=false` 保持 v27 生产）：
- 实验 A（对照组）：与 v27 逐项一致，门禁 PASS（代码可复现性已验证）
- 实验 B（60/25/15）：**两门禁均未过**（folds 均值 +4.9% < 5.8%，extend DD -16.2% < -15%，
  extend 超额 -2.5% vs v27 +29.2%）→ 结论：**裸增速/裸动量因子给预算会稀释核心 alpha**；
  动量 sleeve 选入的是 ST/题材类投机动量而非时代主线（AI 硬件交集为空）
- C/D 提前终止（用户决策），预算网格不再继续

**核心教训**：市场奖励的是"**比预期好**"，不是"增速高"。裸增速因子缺"预期"锚，
历史 IC 弱（fold 0/5）。正确方向 = 用现有季度财报构造**预期差**信号 + **行业景气**信号。

## 目标

给选股逻辑装两个"时代感知"通道，全部使用现有历史数据（可立即回测、生产可持续）：
1. 预期差感知层（主）：SUE / 盈利加速 / 公告漂移（PEAD）
2. 行业轮动层（辅）：行业动量叠加

## 决策（用户已确认）

- sleeve 架构已合回 main（`styles.enabled=false`，生产零变化），本轮在
  `feature/earnings-surprise` 分支开发
- 新信号接入方式：**成长 sleeve 换料**——替换 growth sleeve 中的裸增速因子
- 主攻方向：预期差感知层为主，行业轮动层为辅

## 模块 1：预期差感知层（主）

### 信号构造（全部 PIT-safe，只用公告日前数据）

**SUE（标准化未预期盈利）**
- 预期：去年同期 EPS（季节性随机游走）；无去年同季数据时回退近 4 季均值
- 意外：surprise = (实际 EPS − 预期 EPS) / 过去 8 季 EPS 变化的标准差
- 数据：`data/fundamental_cache/{sym}.parquet` 的 `摊薄每股收益(元)` 季度序列
  （2017Q1 起，约 37 季）；报告期列 `日期`
- PIT 规则：公告日 = 报告期 + 45 天（`fundamental.py` 现有 `PIT_LAG_DAYS=45` 口径，
  与其保持一致），公告日前该季度数据不可用

**盈利加速**
- eps_yoy = 本季 EPS / 去年同季 EPS − 1；accel = eps_yoy − 上季 eps_yoy
- 数据同上

**PEAD（公告漂移）**
- 因子值 = 公告日后 20 个交易日的累计超额收益（相对等权市场）；公告日来自
  THS 层 `data/fundamental/{sym}.parquet` 的 `announce_date` 列
- 注入规则：公告日当天起该因子进入截面，保持到下一公告（事件驱动，PIT-safe）

### 因子落点

- 新因子名：`sue_std`、`earn_accel`、`pead_20d`
- 面板构建：新增模块 `earnings_surprise.py`（被回测脚本引用 + tests 覆盖，
  满足模块准入），产出 {因子: DataFrame(日期×股票)}，由回测脚本在
  `_merge_fundamental_panels` 同层合并（走与 fund_* 相同的 PIT 管线）
- 接入：config `styles.sleeves.growth.factors` 替换为
  `[sue_std, earn_accel, pead_20d, fund_ocf_ps, fund_ocf_yield]`
  （移除 fund_profit_growth/fund_profit_growth_ded/fund_revenue_growth/aux_yjkb_profit_growth）

## 模块 2：行业轮动层（辅）

### 构造

- 行业指数：`data_store/aux_industry/industry_map.parquet`（覆盖约 48% 股票）
  + 日线收益 → 行业等权日收益序列（PIT：只用当日已有股票）
- 行业动量：过去 60 日行业收益排名 → 截面 z-score（`ind_mom_60`）
- 覆盖缺失的股票：行业动量贡献 0（自然降级）

### 接入

- 个股综合分 + λ × ind_mom_60_z（λ 由 config `styles.industry_lambda` 控制，
  默认 0.10；复用 score_stocks 的 N 通道机制）
- 实现落点：`score_stocks` 增加 industry 通道（与 minute_layer 同模式）

## 实验流程（预注册）

1. **IC 诊断（离线）**：新因子在 folds 训练窗口（2015-2023 逐折）的 IC/ICIR
   与覆盖率，输出诊断表（不设淘汰门槛，只记录；诊断脚本用现有
   `scripts/active/run_factor_ic.py` 或回测内诊断模式）
2. **全量验证**：
   - 配置 X = 成长 sleeve 换料（budget 0.15）+ 行业 λ=0（纯预期差效果）
   - 配置 Y = X + 行业 λ=0.10（叠加效果）
   - 对照 v27 存档
3. **门禁（预注册）**：folds 均值超额 ≥ +5.8% 且 extend 最大回撤 ≥ -15%
   且 extend Sharpe ≥ v27 的 2.17 − 0.3；观察项（非硬性）：extend 持仓出现
   成长/科技类标的
4. **选优**：满足门禁的配置中取 extend Sharpe 最高者；无档满足 → 本轮不通过，
   v27 继续生产，负结果如实记录

## 工程前置（Phase 0）

- fundamental_cache 双 schema 统一读取：新模块需处理英文列（report_date）
  与中文列（日期）两种格式，复用 `data/fundamental.py` 现有兼容逻辑
- 验证 THS 层 announce_date 覆盖率（PEAD 数据基础）
- 合并结果归档 + 上轮实验收尾报告（并入最终结果文档）

## 明确不做

- 分析师一致预期历史采购（本轮仅记录渠道调研结论，不实现）
- ML 组合器 / 提高换手 / 北向资金（生产断供）
- 不再跑裸增速/裸动量预算网格（B 已证伪）

## 交付物

1. `earnings_surprise.py` 新模块 + 单元测试（PIT 正确性、schema 兼容）
2. 行业动量通道 + 测试
3. IC 诊断结果
4. 配置 X/Y 全量 folds+extend 结果（版本化存档 + experiment_tracker 登记）
5. 训练结果报告（含上轮 sleeve 收尾结论 + 本轮结论 + 最终生产版本定稿）
6. AGENTS.md 更新 + 合回 main
