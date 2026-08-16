# 数据获取离线治理设计 (2026-08-16)

## 背景

2026-08-15 daily_pipeline 曾在"信号生成"步骤卡死 3 小时（网络层）。排查发现信号生成
存在网络泄漏点，且分钟数据缺档一周无任何告警。用户决策：训练/回测/信号生成严格离线，
网络访问统一前置到数据获取阶段。

## 现状盘点（排查结论）

- 回测/训练路径已 100% 离线：日线、分钟因子（minute_15m）、aux、基本面、行业映射、
  基准曲线全部读本地 parquet，无网络库引用。
- 唯一网络泄漏：`run_paper_signal.py:475-482`，`minute_mode: true` 时信号步骤现场
  `MinuteFetcher().fetch_batch()` 拉 5 天分钟数据（8-15 卡死主嫌疑）。
- 分钟数据缺档：`minute_5m`（POV 执行定价）与 `minute_15m`（分钟因子）均停在
  2026-08-07，增量更新靠手工脚本，未进管线。
- 目录现状：`minute_5m`（baostock 全历史）、`minute_15m`（因子）、`minute/`
  （MinuteFetcher 滚动缓存，含 5 个旧格式残留文件）。
- 无守卫机制：未来任何代码往训练/回测路径加网络调用不会被拦截。

## 设计决策（用户已确认）

1. **缺失行为分场景**：单股/单因子缺失 = 正常（维持现有 NaN 跳过/覆盖率过滤）；
   整批数据缺档/陈旧 = 异常（回测硬失败，信号告警+标注，不静默决策）。
2. **不重拉历史数据**：历史混源已由单位自动检测/复权对齐消化，本轮只立增量规则。
3. **获取阶段是唯一网络入口**：复用现有 fetch_* 模块，不重构。

## 方案（选定 A: netgate 离线守卫）

### 变更 1: netgate.py 离线守卫（新模块，根目录）

- `set_offline_mode(on: bool)` / `is_offline() -> bool` / `OfflineViolation(Exception)`。
- 无任何依赖（避免循环 import）。
- `data/minute_fetcher.py` `fetch()`：网络分支条件加 `netgate.is_offline()`
  （离线等同 `allow_network=False`，返回本地缓存或 None，走既有回退）。
- `fundamental_fetcher.fetch_one` / `flow_fetcher.fetch_money_flow_snapshot` /
  `smart_money_fetcher` 网络入口：离线模式下抛 `OfflineViolation`。
- 开启点：`run_walkforward_backtest.main()`、`run_full_ic_validation`、
  `run_ic_monitor`、`run_paper_signal.generate_signal_v3()`。
- 不开启：fetch_* 获取脚本、`paper_executor`（盘中 T+1 例外，web 后端进程默认关闭）。

### 变更 2: 删除信号生成的网络预拉取

`run_paper_signal.py:475-482` 的 `MinuteFetcher().fetch_batch()` 块移除；
`generate_signal_v3()` 入口 `set_offline_mode(True)`（同一进程内的 paper_executor
随之离线，分钟数据新鲜度由管线数据更新步骤保证；本地缺失时走既有回退路径）。

### 变更 3: 管线"数据更新"步骤补分钟增量

- `fetch_baostock_minute.py` 新增 `--since N`（交易日）增量模式：每只股票读已有
  parquet 找最大日期，从最大日期往前缓冲 10 天拉到今天，合并去重后写回；
  不带 `--since` 时保持全量行为。另加 `--symbols` 指定股票子集。
- `daily_pipeline.step_data_fetch`：日线更新后调用分钟增量（freq 5 与 15，
  流动性池+持仓+候选，约数百只）。失败不阻塞管线，记入步骤结果并告警。

### 变更 4: 结构性缺失守卫（陈旧度检查）

- `data/minute_fetcher.py` 增加模块级函数：
  `latest_local_minute_date(data_dir) -> Optional[date]`。
- 回测：启动时校验日线/分钟数据最大日期覆盖本次运行所需区间末尾
  （extend_val end / folds end），不满足 → 硬失败退出（实验必须可复现）。
- 信号生成：分钟数据最大日期落后信号日超过 5 个交易日 → alerter 告警 +
  信号报告标注"数据截至 YYYY-MM-DD"（不阻断，但明示）。

### 变更 5: 清理 + 文档

- `data_store/minute/` 5 个旧格式残留文件备份后移走（目录保留，paper_executor
  盘中滚动缓存仍用）。
- AGENTS.md 更新数据流约定。

## 明确不做

- 不重拉历史数据；不改 paper_executor 盘中网络；不重构 fetch 模块；
- 不处理 daily_pipeline 网络层超时（akshare/eastmoney_proxy 内部挂起，
  另立专项）。

## 测试（TDD，先红后绿）

1. `tests/test_netgate.py`：开关往返、默认关闭。
2. netgate 拦截：MinuteFetcher(allow_network=True) + 离线模式 → 无网络调用
   （monkeypatch akshare 断言未触发）；三个 fetcher 离线模式抛 OfflineViolation。
3. 信号生成无网络：删除预拉取后，generate_signal_v3 路径 monkeypatch 网络库断言不触发。
4. `fetch_baostock_minute` 增量合并：合成已有帧 + 新帧 → 按时间去重、排序、保留最新。
5. daily_pipeline 分钟步骤失败容忍：fetch 抛异常 → step 返回告警文案不中断。
6. 陈旧度守卫：模拟缺档数据 → 回测 raise / 信号告警路径。

## 实施顺序

1. netgate.py + 测试
2. 四个 fetcher 接入 netgate + 测试
3. 信号生成删预拉取 + 离线开启 + 测试
4. fetch_baostock_minute 增量合并 + 测试
5. daily_pipeline 分钟步骤 + 测试
6. 陈旧度守卫（回测 + 信号）+ 测试
7. minute/ 残留清理 + AGENTS.md 更新 + 全量回归
