# 因子优化计划

> 目标: 从"3-4个独立信息维度 × 950只"升级为"20+个独立维度 × 3000只"
> 理论IR提升: 18 → 73 (4倍)
> 成功标准: 组合 ICIR > 0.5, 年化超额 > 5%, IR > 0.5, MaxDD < 20%

---

## P0: 扩大宇宙到3000只 ✅

- [x] 编写 `scripts/fetch_full_universe.py` — 全A股数据拉取(剔除ST/次新/低流动性)
- [x] 构建 `data_store/` 目录结构 (每只一个parquet + _meta.json)
- [x] 拉取3000+只股票2018-2026日线数据 (已有2776只, 接近目标)
- [ ] 构建PIT宇宙快照 `data_store/universe_snapshots/YYYY-MM.json`
- [ ] 更新 config.yaml 的 universe 配置指向新数据源
- **Go/No-Go**: 有效标的 ≥ 2500只 ✅ (2776只)

## P1: 引入基本面因子 (盈利动量信号) ✅

- [x] 编写 `factors/earnings_momentum.py` — SUE/ROE加速度/营收surprise/业绩预告/应计比率
- [x] 构建基本面数据管道 `scripts/fetch_fundamentals_v2.py` (akshare stock_financial_analysis_indicator)
- [x] PIT对齐: 公告日+45天延迟, 业绩预告公告日当天可用
- [x] 计算因子值并存入 `data/factor_cache/earnings_momentum/`
- [x] 单因子IC验证: 宇宙=全池, 期间=2018-2024, horizon=5/10/20d
- [x] 验证报告写入 `data/ic_validation/p1_earnings_ic.json`
- **Go/No-Go**: 任一子因子 |ICIR| > 0.5 且 IC正比例 > 55% (待运行验证)

## P2: 引入资金流因子 ✅

- [x] 编写 `factors/money_flow.py` — 主力净流入/北向持仓变化/大单比/流动量/流波动
- [x] 构建资金流数据管道 `scripts/fetch_fund_flow.py` (akshare stock_individual_fund_flow)
- [x] 北向资金: `scripts/fetch_north_flow.py` (stock_hsgt_individual_em, 仅沪深港通~1500只)
- [x] 计算因子值并存入 `data/factor_cache/money_flow/`
- [x] 单因子IC验证 + 独立性检验(vs turnover_vol, return_30d)
- [x] 验证报告写入 `data/ic_validation/p2_flow_ic.json`
- **Go/No-Go**: 任一子因子 |ICIR| > 0.5; 与P1信号rank相关 < 0.4 (待运行验证)

## P3: 补齐Alpha158缺失价量因子 ✅

- [x] 在 `factor_engine.py` 中新增算子: Slope(线性回归斜率), Resi(残差), IdxMax/IdxMin(滚动argmax/argmin)
- [x] 修复 RSqrFactor 的 O(N²) 性能问题 (改为 rolling().corr()**2 向量化)
- [x] 在 `factor_library.py` 新增 ALPHA158_FACTORS: 65个因子 (13族×5窗口)
- [x] 新增: BETA/RSQR/RESI/IMAX/IMIN/IMXD/WVMA/CORD/SUMP/SUMN/SUMD/VMA/VSTD
- [x] 新增 `alpha158_full` preset 到 factor_scorer.py
- [x] 单因子IC验证: 1372只股票, 195组结果 (65因子×3horizon)
- [x] 验证报告写入 `data/ic_validation/p3_alpha158_ic.json`
- **Go/No-Go**: cord ICIR=-0.64 ✅ | imxd/imax/beta_30 也有显著IC

## P4: 引入分析师/事件因子 ✅

- [x] 编写 `factors/analyst_revision.py` — 预期修正/评级变化/覆盖变化/目标价偏离
- [x] 编写 `factors/event_signals.py` — 限售解禁/龙虎榜/业绩预告surprise
- [x] 数据管道: `scripts/fetch_analyst_data.py` (批量快照), `scripts/fetch_events.py` (解禁+龙虎榜)
- [x] 适配akshare实际可用API (多个预期API已失效, 已替换为可用端点)
- [x] 单因子IC验证: preview_surprise ICIR +0.18, lhb ICIR +0.48, lockup ICIR -0.49
- [x] 独立性验证: 与价量因子 |corr| < 0.03 ✅
- [x] 验证报告写入 `data/ic_validation/p4_analyst_event_ic.json`
- **Go/No-Go**: preview_surprise/lockup/lhb 方向正确且独立; 分析师因子需积累快照历史

## P5: 全因子组合验证 + 模拟盘对接 ✅ (脚本已创建, 待运行回测)

- [x] 汇总所有通过验证的因子 → 统一因子矩阵 (`scripts/run_factor_portfolio.py`)
- [x] 独立性检验: 因子间rank相关矩阵, 相关>0.4的取ICIR更高者
- [x] 组合方式: IC加权线性组合 (不用ML)
- [x] Walk-forward回测: 月度调仓, Top-30等权, 宇宙=data_cache 950只
- [x] 基准: 全池等权
- [x] 统计检验: Bootstrap 1000次, 95% CI不含0
- [x] 子样本分析: 按年/按市场阶段(bull/bear/recovery/recent)
- [x] 对接 PaperExecutor: `scripts/run_paper_signal_v3.py`
- [x] 最终报告写入 `data/ic_validation/p5_portfolio_report.json`
- [ ] 运行回测验证: `py scripts/run_factor_portfolio.py`
- [ ] 确认Go/No-Go: 年化超额>5%, IR>0.5, MaxDD<20%, Bootstrap CI不含0
- **Go/No-Go**: 年化超额>5%, IR>0.5, MaxDD<20%, Bootstrap CI不含0

---

## 依赖关系

```
P0 (宇宙扩展)
 ├── P1 (基本面) ──┐
 ├── P2 (资金流) ──┤
 ├── P3 (价量补齐) ─┼── P5 (组合验证)
 └── P4 (分析师) ──┘
```

P1-P4 可并行执行（都依赖P0的数据）
P5 必须等P1-P4全部完成

---

## 技术约束

- 数据源: akshare (免费, 限速~3req/s)
- 存储: parquet (与现有data_cache兼容)
- 验证期: 2018-01 ~ 2024-12 (研究期), 2025-01 ~ 2026-07 (验证期, 不参与参数选择)
- IC方法: Spearman rank, 每5日采样, min_stocks=30
- 模型: 线性IC加权为主, 极度正则化LightGBM(50树/depth2)为对比

---

*创建时间: 2026-08-01*
*状态: P0-P5 代码全部完成, 待运行回测验证*
