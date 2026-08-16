# 执行定价模块优化设计（L0/L1/L2）

> 日期: 2026-08-15
> 前置文档: `docs/EXECUTION_MODEL_PROBLEM.md`（问题梳理）、当日对话中的数据分析

## 已确认决策

1. **范围**：L0（确定性定价+路径统一+缓存修复）+ L1（执行质量指标）+ L2（日内规则实验框架）。**不跑回测**（另一 agent 正在用 30 万/top_k=10 跑 walkforward，文件改动不影响其已加载进程）。
2. **小订单定价**：纯 POV 低参与率（删除随机分支），300 股订单前 2-3 根 5m bar 自然成交完，落在开盘附近。
3. **滑点对齐**：模拟盘算法单（pov/vwap/twap）用 `vwap_residual_bps`（10bp）；open/close 算法仍用 `slippage_bps`（30bp）。回测不变（POV 10bp 残差已正确）。
4. **日内规则集**：全套三个（缺口等待 / 前 30 分钟动量过滤 / 量比过滤），全部默认关闭，参数走 config 兜底默认值。

## 实证依据（当日用本地 5m 数据测量，295/296 笔）

- v24e 已发布结果的成交价实际 = 全天 VWAP+10bp（买入）/ VWAP（卖出），`fill_times` 的 "09:35" 是装饰性的 → 已发布数字含前视偏差，与当前代码脱节。
- 调仓日日内漂移：BUY 股 close-vs-open 均值 -17.6bp、SELL 股 +25.8bp，std ~225bp，t<1.3 → 无可利用的日内 alpha。
- 完美择时上限（每笔买最低/卖最高）：median +138bp/笔、mean +169bp → 日内择时理论天花板，因果规则预期只能拿一小部分。
- 定价约定切换影响：open vs close 相差 ~65bp 资本/验证期 → 影响绝对水平，不影响策略排序。

## L0：确定性定价与路径统一（`data/minute_fetcher.py`）

1. 删除 `get_pov_fills` 小订单随机分支（现 414-429 行），小订单走 POV 循环（ρ=max(0.001, qty/pred_vol)）。
2. 统一重复代码：`get_pov_price` 委托 `get_pov_fills` 取 price（单一实现）。
3. 修 `_local_cache` 截断 bug：缓存键从 symbol 改为 `(symbol, end_date)`，否则首个 end_date 截断后的 df 会让后续调仓日全部取空。
4. 新增 `start_bar` 参数（默认 0）供 L2 的 overlay 指定执行起点。

`model/engine.py` 无需改动（`_pov_price` 已委托 `get_pov_fills`）。

## L0b：滑点对齐（`execution/paper_executor.py` + `scripts/active/run_paper_signal.py`）

- `PaperExecutor` 新增 `residual_bps` 参数（默认 = `slippage_bps`，向后兼容）。
- `_get_execution_price` 返回后按算法类型选择滑点：`minute_mode` 且 algo∈{pov,vwap,twap} → `residual_bps`；否则 `slippage_bps`。
- `run_paper_signal.py` 传入 `residual_bps=config["execution"].get("vwap_residual_bps", slippage)`。
- config.yaml 不改（daily_pipeline 运行中）。

## L1：执行质量指标（新模块 `execution/exec_quality.py`）

`fill_quality(trades, mf=None) -> dict`：
- 对每笔 trade：取当日 5m bar，计算 VWAP / 到达价（首 bar 收盘）/ 完美价（BUY=当日最低、SELL=当日最高）。
- 符号化偏差：BUY 的偏差 = fill/ref - 1，SELL 的偏差 = ref/fill - 1（正 = 亏）。
- 输出按 BUY/SELL 及汇总的 mean/median/p95（bps）与完美择时上限（perfect_gain）。
- 接入：`run_walkforward_backtest.py` 结果 JSON 增加 `execution_quality` 字段（只新增段落）；`paper_executor.py` 日终写当日成交质量（本设计先落模块+测试，报告接入随下一次回测自然生效）。

模块准入合规：被 active 脚本 + paper_executor 引用 + tests 覆盖。

## L2：日内规则实验框架（新模块 `execution/execution_overlay.py`）

三个因果规则（只允许用决策时点之前的数据），每条带超时上限、超时强制执行：

| 规则 | 逻辑 | 默认参数 |
|------|------|---------|
| gap_wait | 开盘缺口 >gap_bps 时等回撤至缺口一半以内或超时 | gap_bps=300, timeout=12 bars |
| momentum_wait | 前 30 分钟 |涨跌幅| >mom_bps 时等回落或超时 | mom_bps=200, timeout=12 bars |
| volume_wait | 量比 <vol_min 时等放量或超时 | vol_min=0.5, timeout=6 bars |

- `decide_start_bar(day_bars, side, rules) -> int`：返回开始执行的 bar index。
- 默认关闭：`overlay=None` 时行为与现在完全一致。
- 验证：`replay_trades(trades, rules, mf)` 对 296 笔历史成交重放，比较规则后成交价 vs 基准（首bar/POV）的 fill_quality。用 138bp 完美上限校准预期。
- 不新建 scripts/（纪律）；验证入口在模块函数内 + tests。

## 测试计划（tests/test_execution_pricing.py）

- 小订单 POV：确定性（两次调用同结果）、成交时间落在开盘附近、fill_times 无 "市价@"。
- 缓存修复：同一 symbol 两个不同 end_date 分别取到各自日期的 bar。
- fill_quality：构造已知 bar 数据的合成 trade，验证符号化偏差与完美上限的数值。
- overlay：gap 规则在缺口 > 阈值时返回 >0 的 start_bar；无缺口时返回 0；超时强制。
- 滑点选择：算法单用 residual_bps、open/close 用 slippage_bps。

## 实施结果（2026-08-15）

- 全部 17 个新测试通过；全量 `py -m pytest tests/ -q` = 78 passed。
- 296 笔重放验证（只读，`execution_overlay.replay_trades`）结论：三个因果规则
  **没有带来执行改善**。首 bar 基准 vs VWAP 偏差 mean +1.0bp / median 0.0bp；
  全套规则后 mean +12.7bp / median +4.1bp（更差），其中 momentum_wait 最差
  （mean +13.2bp）。与事前预期一致——日内漂移是噪声，无可收割 alpha；
  框架的价值在于把该结论变成了可复算的数字，未来任何新规则都用同一入口验证。
- 滑点对齐：模拟盘算法单（pov/vwap/twap）从 30bp 降至 vwap_residual_bps(10bp)，
  open/close 保持 30bp；`run_paper_signal.py` 已传参。config.yaml 未改动
  （overlay 默认关闭，启用需在 config execution 段加 overlay 参数）。
- 未跑回测（遵守约定）；正在运行的 30 万/top_k=10 回测进程加载的是旧代码，
  其结果不受本次改动影响，但下一次回测将自动使用确定性 POV 定价并输出
  `execution_quality` 字段。
