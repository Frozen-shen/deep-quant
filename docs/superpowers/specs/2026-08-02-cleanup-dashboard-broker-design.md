# 2026-08-02 架构整理 + 金融看板 + 券商仿真盘接入 设计文档

> 状态: 已与用户确认（2026-08-02）
> 关联: restructure-design.md §6.4（src/quant 清理，本文档执行其方案）

## 一、背景与目标

quant-starter 已从"研究项目"演进为"待上线验证的生产系统"，当前存在三类问题：

1. **双代码库**: 根目录模块栈（活主线，~40 模块）与 `src/quant/` 平行包（~1 万行，100% 死代码，60% 重复）并存
2. **前端瘫痪**: `dashboard.py` 主数据源 `test_results/` 不存在且不可再生，内容硬编码过时，与新体系（模拟盘/信号/实验/IC 验证）完全脱节
3. **执行层缺口**: 8/3 模拟盘开跑，后续需接券商仿真环境（QMT/miniQMT）验证模型；当前只有自研撮合 `execution/paper_executor.py`

**目标**: 清理死代码 → 搭建 FastAPI + React 全功能看板（服务 11/3 毕业评估）→ 预留券商仿真盘执行适配层。分钟级因子决策集成留扩展缺口（IC 验证结论未出）。

**约束**: 8/3 模拟盘上线优先；整理只动死代码，不碰活链路；本机 localhost 部署，无鉴权。

## 二、§1 架构整理方案（Phase 1）

### 2.1 步骤

1. **checkpoint 提交**: 当前 30+ 未提交文件（alpha_decay.py、factor_factory/、IC 验证脚本等 untracked + 若干 modified）先 commit 入库，确保清理过程零丢失
2. **归档死代码**（`git mv` 已跟踪文件 / `mv` 未跟踪文件，全部进 `archive/`）:
   - `src/quant/` → `archive/src_quant/`（执行 restructure-design.md §6.4 已定方案）
   - `configs/` → `archive/configs/`（schema 与 config.yaml 不兼容的死配置）
   - `blind_test.py`、`portfolio.py`、`scheduler.py`、`risk_model.py`、`regime_advanced.py`、`alpha_decay.py`、`alpha_enhancement.py`、`research_rigor.py`、`execution_optimizer.py`、`factor_factory/` → `archive/legacy/`
   - **保留原地**: `blind_results/`（毕业看板"纪律展示"数据资产）、`storage.py`、根 `gate.py`（与 `evaluation/gate.py` 同名但分工不同，均保留）
3. **脚本分流**（消除"第三类脚本"）:
   - → `scripts/active/`: `run_full_ic_validation.py`、`run_holdout_test.py`、`run_fundamental_ic_validation.py`、`run_northbound_ic_validation.py`、`run_relative_ic_validation.py`、`run_minute_ic_validation.py`、`fetch_baostock_minute.py`、`fetch_flow_data.py`、`fetch_index_data.py`、`fetch_minute_data.py`、`fetch_smart_money.py`、`fetch_unadjusted_batch.py`、`daily_snapshot.py`
   - → `scripts/archive/`: `run_p5_portfolio_validation.py`（已被 `active/run_research_backtest.py` 取代，输出同一 `p5_portfolio_report.json`）
   - **保留根目录**: `daily_pipeline.py`（生产编排主入口，README 注明其地位）
4. **修复 `pyproject.toml`**: 删除 `[project.scripts]` 中指向不存在 `quant.cli` 的 `run_backtest` / `fetch_data` 两个入口（`pip install .` 当前会失败）
5. **文档同步**:
   - README.md: 补"已废弃/已归档"说明（src/quant、configs、archive/ 用途）
   - 工作区根 `AGENTS.md`（C:\Users\Frozen\ZCodeProject\AGENTS.md）已严重过时（仍描述 main.py/strategy.py 时代结构），重写为当前真实结构
   - `DEVELOPMENT_DISCIPLINE.md`: 增加"新增模块必须被 active/ 脚本或包内代码引用，否则不合并"防复发条款

### 2.2 验证

- `pytest tests/` 全绿（归档后 import 链不变）
- `python scripts/daily_pipeline.py --check-only`（如支持）或 `run_paper_signal.py --dry-run` 可运行
- 归档后 `grep -r "src.quant\|from src" --include="*.py" scripts/active/ src 2>/dev/null` 无活引用

## 三、§2 权益数据 + FastAPI 后端（Phase 2）

### 3.1 权益数据现状（已确认）

- `PaperExecutor.snapshot()` 每日调用 `storage.log_equity()` 写入 `quant.db` 的 `equity_log` 表（`run_paper_signal.py:402` 已接线）
- `equity_log` 当前为空属正常（模拟盘 8/3 正式开跑），**无需新建持久化**；看板需处理"暂无数据"空态
- 8/3 首跑后验证 equity_log 有数据

### 3.2 后端结构

```
web/api/
├── main.py          # FastAPI 应用入口 (uvicorn :8000)
├── config.py        # 路径/常量（BASE_DIR、DB_PATH 等）
├── cache.py         # lru_cache + TTL 缓存装饰器
└── routers/
    ├── graduation.py  # 8 项毕业指标计算 + 达标状态
    ├── portfolio.py   # 持仓/交易/现金
    ├── equity.py      # 净值曲线 + 回撤（equity_log）
    ├── signals.py     # data/paper_signals_v3.jsonl
    ├── experiments.py # experiments/*.json
    ├── factors.py     # ic_validation/*.json（p3/p6-p9，含分钟因子 p9）
    ├── universe.py    # 股票池/名称/板块
    ├── stocks.py      # 个股 OHLC + 买卖点（parquet 按需读）
    └── broker.py      # 券商执行状态（Phase 4 启用）
```

### 3.3 端点清单

| 端点 | 数据源 | 说明 |
|---|---|---|
| `GET /api/health` | — | 存活 + 各数据源可用性 |
| `GET /api/graduation` | equity_log + ic_validation + signals | 8 项毕业指标（运行时长/年化超额>5%/IR>0.5/MaxDD<15%/信号实现率>80%/IC衰减/Sharpe>0.8/月胜率>55%），每项 value/threshold/status |
| `GET /api/portfolio` | storage.positions + trades + paper_trade/portfolio.json | 持仓、现金、市值、最近交易 |
| `GET /api/equity` | storage.equity_log | 净值曲线、日收益、回撤序列 |
| `GET /api/signals` | data/paper_signals_v3.jsonl | 每日信号、实现率统计 |
| `GET /api/experiments` | experiments/*.json | 实验列表、verdict、参数 |
| `GET /api/factors/ic` | data/ic_validation/*.json | 各来源 IC 结果（p3 价量/p6 基本面/p7 相对/p8 北向/p9 分钟） |
| `GET /api/universe` | data_cache/stock_names.json + a_sectors.json | 名称/板块映射（3021 只） |
| `GET /api/stocks/{symbol}` | data_store/{symbol}.parquet | OHLC + 该股买卖点（从 signals 匹配） |
| `GET /api/universe/search?q=` | universe | 代码/名称搜索（3021 只） |

数据读取复用根模块（`storage`、`data_cache`），大表按需读 + `lru_cache`；CORS 放行 `http://localhost:5173`；无鉴权。

## 四、§3 React 前端（Phase 2-4）

### 4.1 技术栈

Vite + React 18 + TypeScript + Ant Design 5 + ECharts + TanStack Query（服务端状态），dev proxy `/api` → `localhost:8000`。

### 4.2 页面结构

| 路由 | 页面 | 内容 |
|---|---|---|
| `/` | 总览 | 8 项毕业指标卡（达标/未达标着色）+ 净值/回撤曲线 + 风控状态摘要 + 数据更新时间 |
| `/portfolio` | 组合 | 持仓表、现金/市值、交易记录、调仓历史 |
| `/signals` | 信号 | 每日信号列表、信号→成交实现率、被拒原因分布 |
| `/factors` | 因子 | IC 结果表/趋势图（分来源 tab，含 p9 分钟因子）、因子池说明 |
| `/experiments` | 实验 | 实验记录列表、verdict 统计、参数对比 |
| `/stocks` | 个股 | 代码/名称搜索 → K线（ECharts candlestick）+ 买卖点标注 + 基础行情 |
| `/trading` | 交易监控 | **券商仿真盘**：订单状态、成交回报、持仓对齐、账户资金、当日盈亏、风控状态（Phase 4 启用） |
| `/data` | 数据状态 | 数据更新日期、覆盖范围、质量报告 |

### 4.3 设计要点

- 所有页面处理三种状态: loading / 空态（如 equity_log 无数据时明确提示"模拟盘 8/3 开跑后累积"）/ 错误态
- 毕业指标卡: 对应 `docs/PAPER_GRADUATION.md` 的 8 项 AND 条件 + 加分项，未达标显示缺口
- 分钟因子缺口: `/factors` 页已可展示 `p9_minute_ic.json`（数据层已就绪），决策链集成待 IC 结论后补，页面留 tab 位

## 五、§4 券商仿真盘接入（QMT/miniQMT，Phase 3-4）

### 5.1 现状与目标

现状: `execution/paper_executor.py` 自研撮合（T+1、滑点、手续费），无券商对接。

目标: **信号生成链不动**（`run_paper_signal` → factor_scorer → portfolio_ranker → PaperExecutor），在 PaperExecutor 之后增加可插拔的券商执行适配层。仿真环境（xtquant 支持仿真模式）与实盘同接口，未来可无缝切实盘。

### 5.2 BrokerAdapter 抽象

```python
# execution/broker/base.py  （新目录 execution/broker/）
class BrokerAdapter(ABC):
    def connect(self) -> bool                       # 登录/连接状态
    def place_order(self, symbol, side, qty, price_type, price) -> str   # 返回 order_id
    def cancel_order(self, order_id) -> bool
    def get_balance(self) -> dict                   # cash / frozen / total_asset
    def get_positions(self) -> list[dict]           # symbol/qty/avg_cost/market_value
    def get_orders(self, date) -> list[dict]        # 订单状态（未成/部分/全成/已撤）
    def get_trades(self, date) -> list[dict]        # 成交回报
    def get_quotes(self, symbols) -> dict           # 实时/最新价

class QmtAdapter(BrokerAdapter):   # xtquant，仿真模式
class PaperAdapter(BrokerAdapter): # 包装现有 PaperExecutor，回归/降级用
```

### 5.3 每日流程（接入后）

```
daily_pipeline → run_paper_signal（生成信号 + 风控闸门 circuit_breaker）
  → BrokerAdapter.place_order（批量下单）
  → 轮询成交回报 / 收盘后对账 reconcile:
     券商持仓 vs storage.positions → 差异告警（alerter）
  → PaperExecutor.snapshot（权益落盘不变）
```

- 运行前提: QMT 客户端本机登录（xtquant 依赖客户端进程），需券商开通仿真权限
- 状态对账: 每日收盘后比对券商持仓与本地 positions，差异 > 阈值触发告警
- 降级: QMT 不可用时回退 PaperAdapter（模拟盘模式）

### 5.4 前端/后端补位

- 后端 `/api/broker/*`（health/orders/positions/account/trades），Phase 4 启用
- 前端 `/trading` 交易监控页（见 4.2），Phase 4 启用

### 5.5 时间线说明

8/3 模拟盘开跑（PaperExecutor 路径，不依赖 QMT）；QMT 适配器作为独立开发项，在 Phase 3-4 完成，不影响 8/3 上线。QMT 客户端/权限开通由用户侧确认。

## 六、分钟级因子缺口（预留）

- 现状: `minute_factors.py` 已实现 10 个因子；`data/ic_validation/p9_minute_ic.json` 已产出（可能数据深度不足）
- **决策链缺口**: 分钟因子是否进入 `factor_scorer` 因子列表/权重未定（待 IC 验证结论 + 超额贡献评估）
- **已就绪部分**: 后端 `/api/factors/ic` 已支持 p9；前端 `/factors` 页已预留分钟因子 tab
- **待补部分**（标记 TODO，不阻塞本期）: factor_library 注册 + run_paper_signal 评分链集成 + 权重配置

## 七、实施顺序（Phase 0-4）

| Phase | 内容 | 验证 | 预计 |
|---|---|---|---|
| 0 | 环境: `pip install fastapi`；`npm create vite` 初始化 web/ui | 两端可启动 | 0.5h |
| 1 | **架构整理**（§2.1 全部步骤，每步独立 commit） | pytest 绿 + grep 无活引用 + 活脚本 dry-run | 0.5-1 天 |
| 2 | 后端骨架 + `/api/graduation` + `/api/equity` + `/api/portfolio`；前端总览页 | curl 端点 + 浏览器走查总览 | 1-2 天 |
| 3 | 信号/因子/实验/个股页 + 其余端点；**QMT 适配器**（execution/broker/） | pytest API + build + 走查 | 2-3 天 |
| 4 | `/trading` 交易监控页 + 对账机制 + 全量打磨（loading/空态/错误态） | playwright 全页面走查 | 1-2 天 |

**并行约束**: 8/3 模拟盘上线优先，Phase 1 不碰活链路；Phase 2-4 与模拟盘运行并行，后端/前端只读数据，不干扰每日管道。

## 八、风险与开放问题

| 风险 | 缓解 |
|---|---|
| 8/3 上线被整理操作干扰 | checkpoint 先行 + 归档只动死代码 + 每步验证 |
| equity_log 首日无数据 | 前端空态提示；8/3 后验证 snapshot 生效 |
| QMT 登录态/断连/成交回报丢失 | 对账机制 + alerter 告警 + PaperAdapter 降级 |
| 分钟因子决策缺口悬置 | 已留数据层/展示层扩展位，决策集成独立跟进 |
| 3021 只 parquet 查询性能 | 按需读 + lru_cache；个股页单 symbol 查询 |

**开放问题**（不阻塞本期）:
1. QMT 券商/账号开通状态（用户侧确认后填 adapter 配置）
2. 毕业指标计算口径细节（按 PAPER_GRADUATION.md，实现时对照确认）
3. 分钟因子 IC 结论（决定是否进决策链）
