# 看板前端全面开发 — 设计文档

日期: 2026-08-12
状态: 已确认（用户批准设计）

## 1. 背景与目标

quant-starter 看板前端（`web/ui`，React 19 + Vite + Antd v6 + TanStack Query + ECharts）已有 9 个页面，均能拉取真实数据，但打磨不足。本次全面开发的目标：

1. **Experiments 页展示最新实验的操作详情**（参数/结果/决策备注）—— 核心诉求
2. 全部页面表格支持排序/分页/筛选，列名中文化、数字格式化
3. 统一空态、轮询策略、交互质量（防抖、Tag 着色、页面联动）

后端基本不动：`/api/experiments` 已全量返回每条记录的 `parameters`/`results`/`notes`，抽屉详情无需新增端点。

## 2. 现状评估

### 前端 9 页（完整度评级）
| 页面 | 评级 | 主要短板 |
|---|---|---|
| Overview | 基本 | threshold/daily_return/summary 字段未展示；回撤前端硬算 |
| Portfolio | 基本 | 无分页排序筛选；无盈亏；数字无格式化 |
| Signals | 简陋 | 动态列裸英文；无筛选排序；Alert 空态；rowKey 用索引 |
| Factors | 基本 | 无排序；列名英文；meta 只显示 description |
| Experiments | 简陋 | 5 列写死；无分页筛选；config_hash 对用户无意义 |
| Stocks | 基本 | 无防抖；无均线/成交量；搜索框与选中态不同步 |
| Trading | 基本 | 成交表无分页；无 Tag 着色；未连接无告警样式 |
| DataStatus | 简陋 | 仅布尔状态；无自动刷新 |
| Placeholder | 占位 | 死代码，未注册路由 |

共性：所有表格无排序；列名裸英文；空态 Empty/Alert 混用；无页面联动；轮询不统一。

### 后端 API
- 端点齐全（19 个），全部带 TTL 缓存，vite 代理已配置（`/api` → `:8000`）
- `/api/experiments` 返回 `{count, experiments[≤100 全量记录], by_script, by_partition}` —— 详情数据已在列表中
- 无单条详情端点（**不需要**，前端直接用列表数据）

## 3. 架构设计

纯前端改造。新增共享层：

```
web/ui/src/
├── lib/
│   ├── format.ts        # 数字/价格/百分比/日期时间格式化
│   ├── labels.ts        # 字段中文映射 + 枚举值映射（信号方向/action → 中文+Tag色）
│   └── columns.tsx      # 通用表格列工具（中文标题 + sorter + 格式化）
├── components/
│   ├── DetailDrawer.tsx # 通用详情抽屉（标题 + 键值表格 + 自定义块）
│   └── StatCard.tsx     # 统计卡（数字 + 标签 + 状态色）
├── api.ts               # 补全全部端点封装（现仅 2 个）
└── pages/*.tsx          # 逐页改造
```

## 4. 分阶段实施

### Phase 1 — 公共基建 + Experiments
- 新建共享层（format/labels/columns/DetailDrawer/StatCard）；空态统一 `Empty`；api.ts 补全封装
- **Experiments 页**：
  - 统计卡补「分区分布」（后端 `by_partition` 已有，前端未显示）
  - 表格中文列：实验ID/时间/脚本/分区/备注摘要；全部可排序；分页 20 条/页；脚本+分区下拉筛选
  - 最新记录行高亮 + 「最新」Tag
  - 点击行 → 右侧 `DetailDrawer`：parameters 键值表、results 键值表、notes 全文、config_hash、ID 可复制
- 验证：`npm run build`（tsc + vite）+ `npm run lint`

### Phase 2 — 表格类页面
- **Overview**：毕业指标卡补 threshold；新增 summary 卡（总收益/最大回撤/波动率/夏普，数据已有）
- **Portfolio**：中文列名 + 市值格式化 + 盈亏列（市值−成本）+ 汇总卡（现金/总市值/总盈亏）+ 排序分页
- **Signals**：固定中文列（符号/名称/时间/方向/价格/评分/原因）+ 方向 Tag 着色 + 排序分页筛选 + 稳定 rowKey + Empty 空态
- **Factors**：中文列名 + IC 列排序 + 完整 meta + 每 tab 均值摘要
- 验证：同上

### Phase 3 — 交互类页面 + 联动
- **Stocks**：搜索防抖 + 选中显示股票名 + K 线 MA5/MA10/MA20 均线 + 成交量副图 + loading 态 + 支持 `?symbol=` URL 参数
- **Trading**：成交表分页 + action Tag 着色 + 未连接 Alert 告警 + 持仓盈亏列
- **DataStatus**：自动刷新 60s + 数据源计数统计（health 保持布尔，不增强后端）
- **页面联动**：Signals/Portfolio 行点击 → `/stocks?symbol=xxx`
- 删除死代码 `Placeholder.tsx`
- 验证：build + lint + 全流程浏览器验证

## 5. 数据流

React Query 各页查询 → 后端 TTL 缓存 → 共享层格式化 → 渲染。抽屉与联动**不新增请求**（数据已在列表内）。

## 6. 错误处理

- 保留各页 isError → `Alert` 分支（这是错误态，非空态，保持 Alert 合理）
- 全局 ErrorBoundary 已有，不动
- 统一轮询节奏：Overview/Trading 已有；Phase 3 给 DataStatus 加 60s

## 7. 测试策略

前端无测试框架（package.json 无 test 脚本），验证手段：
1. `npm run build` — tsc 严格类型检查 + vite 构建
2. `npm run lint` — oxlint
3. 浏览器真机验证：启动 uvicorn(:8000) + vite(:5173)，用浏览器工具验证关键交互（抽屉详情、排序/分页/筛选、K线均线、联动跳转）

## 8. 非目标（明确不做）

- 不改后端（health 布尔增强、experiments 详情端点均不做）
- 不引入测试框架、不做暗色主题、不动路由结构（App.tsx 仅必要时注册 404）
- 不做多实验对比、不做手动下单功能
