# 前端全面重构设计 — 实验为中心的量化看板

日期: 2026-08-15
状态: 已获用户批准 ("进行全面开发吧")
调研依据: QuantStats/pyfolio/Qlib/backtrader/QuantConnect/聚宽/米筐/QMT/雪球（调研报告已归档于会话）

## 目标

1. 前端展示最新实验内容（v24e 及后续 v25+），可插拔——新实验 JSON 零代码接入
2. 交易详情完整展示（回测逐笔 + 模拟盘实盘，POV 拆单时段）
3. 个股盈亏：回测实验（FIFO 配对）与模拟盘实盘都要
4. 学习业界优秀展示（净值三件套/月度热力图/买卖标记/盈亏贡献图）

## 用户已确认的决策

- 信息架构: **实验为中心 + 四区导航**
- 可插拔: **配置驱动**——后端注册表 API 扫描 JSON 目录，前端通用渲染
- 后端: 加注册表 API
- 个股盈亏: 回测 + 模拟盘两者都要
- 未定项默认值（用户未反对，按推荐执行）: 基准指数用 **中证1000**（策略 universe 对齐, `data/cache/index_csi1000.parquet` 已有; 回测 excess 即 vs 中证1000, 口径一致）

## 信息架构

```
Header: quant-starter [实验选择器 Dropdown ▼]
Sider:  仪表盘 / 研究(因子+实验) / 交易(组合+信号+成交) / 数据(个股+数据状态)
```

- **全局实验上下文**: React Context + URL `?exp=<id>`。选中实验影响: 实验详情页、成交页"回测"Tab、个股页买卖标记
- **仪表盘不随实验切换**——它始终是模拟盘实盘运行状态
- 四区映射: 总览→仪表盘; 因子+实验→研究; 组合+信号+交易监控→交易; 个股+数据状态→数据

## 可插拔 Schema（核心机制）

后端扫描两类目录:
- `data/ic_validation/walkforward_results_v*.json` (kind=walkforward)
- `experiments/exp_*.json` (kind=experiment, 旧 KV 格式)

统一输出三类结构:

```jsonc
// GET /api/experiments/registry → [{ id, kind, name, generated_at, summary }]
// GET /api/experiments/{id} →
{
  "meta": {"id","kind","name","generated_at","description"},
  "metrics": [{"key","label","value","format":"pct|num|money","better":"high|low"}],
  "series": [{"name","type":"line|bar","x":[],"y":[]}],          // 净值/超额/回撤/基准
  "folds": [{"name","excess_annual","sharpe","max_drawdown","ir","avg_turnover"}],
  "stock_pnl": [{"symbol","name","total_pnl","pnl_pct","win_rate","n_round_trips","avg_hold_days"}],
  "trades": [{"date","symbol","action","price","qty","commission","fill_times"}],
  "equity_curve": [{"date","equity"}],
  "benchmark_curve": [{"date","close"}]
}
```

规则: 新实验脚本产出标准 JSON（meta 自描述）→ 前端零改动自动出现。旧 exp_*.json 无此结构 → 前端 KV 表格兜底。

## 后端改动

1. `routers/experiments.py` 重写:
   - `GET /api/experiments/registry`: 扫描两类目录, 解析 meta 生成注册表 (坏 JSON 静默跳过)
   - `GET /api/experiments/{id}`: 完整 schema 解析。walkforward 结果 → metrics (excess_annual/sharpe/max_drawdown/total_return/annual_return/calmar/ir/avg_turnover) + series (equity_curve→净值线, daily_active_returns→累计超额, 回撤序列) + folds + trades + stock_pnl
   - `GET /api/experiments/{id}/compare` 或前端多次调用 registry 条目即可 (对比由前端拉多个 id 组合)
2. **个股盈亏聚合** (后端函数 `aggregate_stock_pnl(trades)`): FIFO 配对买卖 → 已实现盈亏/胜率/持有天数。回测实验从 JSON trades; 模拟盘从 SQLite trades (同函数)
3. `routers/portfolio.py`: 补现价 (data_store 日线最后一根 close, 与 market_value 同日) + 盈亏率
4. 基准序列: `data/cache/index_csi1000.parquet` 收盘 → 实验 schema 的 benchmark_curve (按实验区间裁剪)
5. 新路由 `routers/paper.py` 或复用 broker: `GET /api/paper/stock-pnl` — 模拟盘个股盈亏 (SQLite trades FIFO + 持仓浮盈亏)

## 前端改动

- `src/experiment-context.tsx` (新): Context + URL 同步, `useExperiment()` hook
- `src/App.tsx`: 四区导航 (antd Menu 子菜单) + Header 实验选择器
- `src/components/EquityTriptych.tsx` (新): 净值+基准+回撤阴影三件套 (通用, 仪表盘/实验页/成交页复用)
- `src/components/MonthlyHeatmap.tsx` (新): 月度收益热力图 (从 daily returns 聚合)
- `src/components/PnlBarChart.tsx` (新): 盈亏贡献横向条形图
- `src/components/ExperimentReport.tsx` (新): 通用实验报告渲染器 (metrics 卡行 → series 图 → folds 表 → stock_pnl 榜 → trades 表)
- `src/pages/Overview.tsx` → 仪表盘: 指标卡 + 三件套 + 热力图 + 模拟盘盈亏贡献
- `src/pages/Experiments.tsx`: 实验卡片列表 (registry) + 详情 (ExperimentReport) + 对比模式 (勾选 2-3 → 指标对比表 + 净值叠加)
- `src/pages/Trading.tsx`: 成交双 Tab (回测实验成交[随全局选择器] / 模拟盘实盘成交); 调仓图点击下钻
- `src/pages/Portfolio.tsx`: 现价列 + 盈亏率列 + 盈亏贡献图
- `src/pages/Stocks.tsx`: K 线叠加买卖标记 (markPoint scatter, 买入红▲/卖出绿▼)
- `src/api.ts`: 新端点 + 类型定义

## 错误处理与测试

- 注册表扫描: 坏 JSON `try/except continue` (现有模式)
- 旧格式实验: KV 兜底渲染
- 后端 pytest (tests/ 或 web/api/tests/): FIFO 配对、schema 解析、注册表扫描、stock_pnl 聚合
- 前端: ErrorBoundary 已有 + 各页 isError 分支; 构建 `npm run build` 验证 TS
- 验收: 启动前后端, Playwright 截图 4 区核心页面

## 明确不做 (YAGNI)

- 雷达图 (主流平台不用)、蒙特卡洛、收益分布直方图 (样本少)、微前端/插件框架、动态表单系统、多租户/权限
