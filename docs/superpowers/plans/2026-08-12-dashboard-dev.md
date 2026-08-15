# 看板前端全面开发 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 全面打磨 quant-starter 看板前端：Experiments 页展示最新实验操作详情（抽屉），全部页面表格支持中文列名/排序/分页/筛选，统一空态与格式化，补充页面联动。

**Architecture:** 纯前端改造（后端不动）。新增共享层 `lib/`（格式化/中文映射/通用列）与 `components/`（DetailDrawer/StatCard），9 个页面逐页接入。分 3 个 Phase 推进，每 Phase 结束可独立验证。

**Tech Stack:** React 19 + TypeScript ~6.0 + Antd v6 + TanStack Query v5 + ECharts 6 + Vite 8

**规格文档:** `docs/superpowers/specs/2026-08-12-dashboard-dev-design.md`

---

## 文件结构

| 文件 | 操作 | 职责 |
|---|---|---|
| `web/ui/src/lib/format.ts` | 新建 | 数字/百分比/时间/截断格式化 |
| `web/ui/src/lib/labels.ts` | 新建 | 字段中文映射 + 枚举值→中文/Tag色 |
| `web/ui/src/lib/columns.tsx` | 新建 | 通用表格列工厂 `col()` |
| `web/ui/src/components/DetailDrawer.tsx` | 新建 | 通用详情抽屉 |
| `web/ui/src/components/StatCard.tsx` | 新建 | 统计卡 |
| `web/ui/src/api.ts` | 修改 | 补全全部端点封装 |
| `web/ui/src/pages/Experiments.tsx` | 重写 | 抽屉详情+排序分页筛选+最新高亮 |
| `web/ui/src/pages/Overview.tsx` | 修改 | threshold 展示 + summary 卡 |
| `web/ui/src/pages/Portfolio.tsx` | 重写 | 盈亏+汇总卡+排序分页 |
| `web/ui/src/pages/Signals.tsx` | 重写 | 固定中文列+Tag+排序分页筛选+联动 |
| `web/ui/src/pages/Factors.tsx` | 修改 | 中文列+排序+meta+摘要 |
| `web/ui/src/pages/Stocks.tsx` | 重写 | 防抖+MA均线+成交量+URL参数 |
| `web/ui/src/pages/Trading.tsx` | 修改 | Tag着色+分页+告警+盈亏列 |
| `web/ui/src/pages/DataStatus.tsx` | 修改 | 中文标签+自动刷新+统计 |
| `web/ui/src/pages/Placeholder.tsx` | 删除 | 死代码 |
| `web/ui/src/index.css` | 修改 | 追加最新行高亮样式 |

## 环境注意（重要）

当前执行 shell 为受限环境：**无 node/npm/python/git**。所有 `npm run build`/`npm run lint` 验证命令，若执行时报 `'npm' 不是内部或外部命令`，请改为在用户系统终端（Windows Terminal / VS Code 集成终端，`cd C:\Users\Frozen\ZCodeProject\quant-starter\web\ui`）执行同一命令，确认通过后继续下一任务。代码文件写入不受影响。

---

# Phase 1 — 公共基建 + Experiments

## Task 1: 格式化与中文映射工具

**Files:**
- Create: `web/ui/src/lib/format.ts`
- Create: `web/ui/src/lib/labels.ts`

- [ ] **Step 1: 创建 `web/ui/src/lib/format.ts`**

```ts
/** 数字千分位格式化；null/undefined/NaN → '—' */
export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return v.toLocaleString('zh-CN', { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

/** 小数比例 → 百分比字符串；0.056 → '5.60%' */
export function fmtPct(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return `${(v * 100).toFixed(digits)}%`
}

/** ISO 时间戳 → 'YYYY-MM-DD HH:mm'；解析失败原样返回 */
export function fmtDateTime(iso?: string): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`
}

/** 长文本截断（表格摘要列用），null/undefined → '—' */
export function truncate(s: string | null | undefined, n = 60): string {
  if (!s) return '—'
  return s.length > n ? `${s.slice(0, n)}…` : s
}
```

- [ ] **Step 2: 创建 `web/ui/src/lib/labels.ts`**

```ts
/** 字段名 → 中文标题（表格列、抽屉键值展示通用） */
export const fieldLabels: Record<string, string> = {
  experiment_id: '实验ID', timestamp: '时间', script: '脚本', partition: '分区', config_hash: '配置哈希',
  notes: '备注', parameters: '参数', results: '结果',
  symbol: '代码', name: '名称', qty: '数量', avg_cost: '成本价', market_value: '市值', entry_date: '建仓日',
  signal_date: '信号日期', mode: '模式', buy: '买入', sell: '卖出', hold: '持有', n_factors: '因子数',
  verdict: '结论', circuit_breaker: '熔断', date: '日期', action: '方向', price: '价格',
  commission: '佣金', reason: '原因',
  factor: '因子', ic_mean: '平均IC', icir: 'ICIR', ic_std: 'IC标准差', n_days: '样本天数', pos_ratio: '正占比',
}

/** 信号模式 → 中文+Tag色 */
export const modeMap: Record<string, { text: string; color: string }> = {
  live: { text: '实盘', color: 'blue' },
  dry_run: { text: '模拟', color: 'default' },
}

/** 信号结论 → 中文+Tag色 */
export const verdictMap: Record<string, { text: string; color: string }> = {
  SKIP: { text: '跳过', color: 'default' },
  CONDITIONAL: { text: '有条件', color: 'orange' },
  EXECUTE: { text: '执行', color: 'green' },
}

/** 交易方向 → 中文+Tag色（A股红涨绿跌：买入红、卖出绿） */
export const actionMap: Record<string, { text: string; color: string }> = {
  buy: { text: '买入', color: 'red' },
  sell: { text: '卖出', color: 'green' },
  BUY: { text: '买入', color: 'red' },
  SELL: { text: '卖出', color: 'green' },
}

/** 大小写不敏感的交易方向 Tag */
export function actionTag(action: unknown): { text: string; color: string } {
  const key = String(action ?? '').toLowerCase()
  return actionMap[key] ?? { text: String(action ?? '—'), color: 'default' }
}

/** 毕业指标状态 → 中文+Tag色 */
export const statusMap: Record<string, { text: string; color: string }> = {
  pass: { text: '达标', color: 'green' },
  fail: { text: '未达标', color: 'red' },
  pending: { text: '待数据', color: 'orange' },
}

/** 数据分区 → Tag色 */
export const partitionColors: Record<string, string> = {
  research: 'blue', val: 'cyan', test: 'purple',
  development: 'geekblue', blind: 'volcano',
}

/** 数据源键 → 中文 */
export const healthSourceLabels: Record<string, string> = {
  equity_log: '净值日志', portfolio: '组合持仓', signals: '信号记录',
  experiments: '实验记录', ic_results: 'IC 验证', data_store: '行情数据',
}
```

- [ ] **Step 3: 验证类型**

Run: `cd /d C:\Users\Frozen\ZCodeProject\quant-starter\web\ui && npx tsc --noEmit -p tsconfig.app.json`
Expected: 无输出（0 错误）。若 npm/npx 不可用，改在系统终端执行；至少用编辑器确认无语法错误。

- [ ] **Step 4: 记录完成**

在实施日志中记录 Task 1 完成（本环境无 git，不做 commit）。

## Task 2: 通用列工厂与共享组件

**Files:**
- Create: `web/ui/src/lib/columns.tsx`
- Create: `web/ui/src/components/DetailDrawer.tsx`
- Create: `web/ui/src/components/StatCard.tsx`

- [ ] **Step 1: 创建 `web/ui/src/lib/columns.tsx`**

```tsx
import type { ReactNode } from 'react'
import { fieldLabels } from './labels'

function cmp(a: unknown, b: unknown): number {
  if (typeof a === 'number' && typeof b === 'number') return a - b
  return String(a ?? '').localeCompare(String(b ?? ''), 'zh-CN')
}

/** 通用表格列：自动中文标题 + 可选排序/宽度/省略/自定义渲染 */
export function col<T>(key: keyof T & string, opts: {
  title?: string
  sorter?: boolean
  defaultSortOrder?: 'ascend' | 'descend'
  width?: number
  ellipsis?: boolean
  render?: (v: unknown, row: T) => ReactNode
} = {}) {
  return {
    title: opts.title ?? fieldLabels[key] ?? key,
    dataIndex: key,
    width: opts.width,
    ellipsis: opts.ellipsis,
    sorter: opts.sorter ? (a: T, b: T) => cmp(a[key], b[key]) : undefined,
    defaultSortOrder: opts.defaultSortOrder,
    render: opts.render,
  }
}
```

- [ ] **Step 2: 创建 `web/ui/src/components/DetailDrawer.tsx`**

```tsx
import type { ReactNode } from 'react'
import { Drawer, Descriptions } from 'antd'

export interface KV { label: string; value: ReactNode }

/** 通用详情抽屉：标题 + 副标题 + 键值表 + 自定义附加块 */
export default function DetailDrawer({ open, title, subtitle, kvs, extra, onClose }: {
  open: boolean
  title: ReactNode
  subtitle?: ReactNode
  kvs: KV[]
  extra?: ReactNode
  onClose: () => void
}) {
  return (
    <Drawer title={title} open={open} onClose={onClose} width={560}>
      {subtitle && <div style={{ marginBottom: 12 }}>{subtitle}</div>}
      {kvs.length > 0 && (
        <Descriptions column={1} size="small" bordered
          items={kvs.map((kv) => ({ key: kv.label, label: kv.label, children: kv.value }))} />
      )}
      {extra && <div style={{ marginTop: 16 }}>{extra}</div>}
    </Drawer>
  )
}
```

- [ ] **Step 3: 创建 `web/ui/src/components/StatCard.tsx`**

```tsx
import { Card, Statistic } from 'antd'

/** 统计卡：数字 + 可选前后缀/精度/颜色 */
export default function StatCard({ title, value, precision = 0, suffix, prefix, color }: {
  title: string
  value?: number | string | null
  precision?: number
  suffix?: string
  prefix?: string
  color?: string
}) {
  return (
    <Card size="small">
      <Statistic title={title} value={value as number} precision={precision}
        suffix={suffix} prefix={prefix} valueStyle={color ? { color } : undefined} />
    </Card>
  )
}
```

- [ ] **Step 4: 验证类型**

Run: `cd /d C:\Users\Frozen\ZCodeProject\quant-starter\web\ui && npx tsc --noEmit -p tsconfig.app.json`
Expected: 无输出（0 错误）。

## Task 3: api.ts 补全端点封装

**Files:**
- Modify: `web/ui/src/api.ts`（整体重写为下方内容）

- [ ] **Step 1: 重写 `web/ui/src/api.ts`**

```ts
import axios from 'axios'

export const api = axios.create({ baseURL: '/api' })

export interface GraduationMetric {
  key: string
  name: string
  value: number | null
  threshold: number | string | null
  status: 'pass' | 'fail' | 'pending'
  detail: string
}
export interface EquityPoint { date: string; total_equity: number; daily_return: number | null }
export interface EquitySummary {
  total_return: number | null
  max_drawdown: number | null
  volatility: number | null
  sharpe: number | null
}

export async function fetchGraduation(): Promise<{ metrics: GraduationMetric[]; overall: string }> {
  const { data } = await api.get('/graduation')
  return data
}
export async function fetchEquity(): Promise<{ curve: EquityPoint[]; summary: EquitySummary | null }> {
  const { data } = await api.get('/equity')
  return data
}
export async function fetchExperiments(): Promise<any> {
  const { data } = await api.get('/experiments')
  return data
}
export async function fetchPortfolio(): Promise<any> {
  const { data } = await api.get('/portfolio')
  return data
}
export async function fetchSignals(): Promise<any> {
  const { data } = await api.get('/signals')
  return data
}
export async function fetchFactorsIc(): Promise<any> {
  const { data } = await api.get('/factors/ic')
  return data
}
export async function fetchBroker(): Promise<any> {
  const { data } = await api.get('/broker/status')
  return data
}
export async function fetchHealth(): Promise<{ status: string; data_sources: Record<string, boolean> }> {
  const { data } = await api.get('/health')
  return data
}
export async function searchUniverse(q: string): Promise<{ stocks: Array<{ symbol: string; name: string; sector: string }> }> {
  const { data } = await api.get('/universe/search', { params: { q } })
  return data
}
export async function fetchStock(symbol: string): Promise<any> {
  const { data } = await api.get(`/stocks/${symbol}`)
  return data
}
```

- [ ] **Step 2: 验证类型**

Run: `cd /d C:\Users\Frozen\ZCodeProject\quant-starter\web\ui && npx tsc --noEmit -p tsconfig.app.json`
Expected: 无输出（0 错误）。

## Task 4: Experiments 页（Phase 1 核心）

**Files:**
- Modify: `web/ui/src/pages/Experiments.tsx`（整体重写）
- Modify: `web/ui/src/index.css`（末尾追加一条样式）

- [ ] **Step 1: 重写 `web/ui/src/pages/Experiments.tsx`**

```tsx
import { useMemo, useState } from 'react'
import { Card, Col, Row, Table, Typography, Spin, Alert, Empty, Tag, Select, Space, Descriptions } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { fetchExperiments } from '../api'
import { col } from '../lib/columns'
import { fmtDateTime, truncate } from '../lib/format'
import { partitionColors } from '../lib/labels'
import DetailDrawer from '../components/DetailDrawer'

interface Exp {
  experiment_id: string
  timestamp: string
  script: string
  partition: string
  config_hash: string
  parameters?: Record<string, unknown>
  results?: Record<string, unknown>
  notes?: string
}

/** 递归键值块：值仍为对象时嵌套展示 */
function KVBlock({ title, obj }: { title: string; obj?: Record<string, unknown> }) {
  if (!obj || Object.keys(obj).length === 0) return null
  const items = Object.entries(obj).map(([k, v]) => ({
    key: k,
    label: k,
    children: v !== null && typeof v === 'object'
      ? <KVBlock title={k} obj={v as Record<string, unknown>} />
      : String(v),
  }))
  return (
    <div style={{ marginBottom: 16 }}>
      <Typography.Text strong>{title}</Typography.Text>
      <Descriptions column={1} size="small" bordered items={items} style={{ marginTop: 8 }} />
    </div>
  )
}

export default function Experiments() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['experiments'], queryFn: fetchExperiments,
  })
  const [script, setScript] = useState<string>()
  const [partition, setPartition] = useState<string>()
  const [selected, setSelected] = useState<Exp | null>(null)

  const exps = useMemo(() => (data?.experiments ?? []) as Exp[], [data])
  const scripts = useMemo(() => [...new Set(exps.map((e) => e.script))].sort(), [exps])
  const partitions = useMemo(() => [...new Set(exps.map((e) => e.partition))].sort(), [exps])
  const rows = useMemo(() => exps.filter((e) =>
    (!script || e.script === script) && (!partition || e.partition === partition)),
  [exps, script, partition])
  const latestId = useMemo(() => {
    if (!rows.length) return undefined
    return rows.reduce((a, b) => (b.timestamp > a.timestamp ? b : a)).experiment_id
  }, [rows])

  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="加载失败" />

  const columns = [
    col<Exp>('experiment_id', { title: '实验ID', width: 220, ellipsis: true, sorter: true }),
    col<Exp>('timestamp', { title: '时间', width: 150, sorter: true, defaultSortOrder: 'descend',
      render: (v) => fmtDateTime(v as string) }),
    col<Exp>('script', { title: '脚本', sorter: true }),
    col<Exp>('partition', { title: '分区', width: 130,
      render: (v) => <Tag color={partitionColors[String(v)] ?? 'default'}>{String(v)}</Tag> }),
    col<Exp>('notes', { title: '备注摘要', ellipsis: true, render: (v) => truncate(String(v ?? ''), 60) }),
  ]

  return (
    <div>
      <Typography.Title level={4}>实验记录</Typography.Title>
      <Row gutter={12}>
        <Col span={8}><Card size="small">总数：{data?.count ?? 0}</Card></Col>
        <Col span={8}><Card size="small">脚本分布：{Object.entries(data?.by_script ?? {}).map(([k, v]) => `${k}×${v}`).join('、')}</Card></Col>
        <Col span={8}><Card size="small">分区分布：{Object.entries(data?.by_partition ?? {}).map(([k, v]) => `${k}×${v}`).join('、')}</Card></Col>
      </Row>
      <Space style={{ marginTop: 16 }}>
        <Select allowClear placeholder="按脚本筛选" style={{ width: 260 }} value={script} onChange={setScript}
          options={scripts.map((s) => ({ value: s, label: s }))} />
        <Select allowClear placeholder="按分区筛选" style={{ width: 180 }} value={partition} onChange={setPartition}
          options={partitions.map((p) => ({ value: p, label: p }))} />
      </Space>
      {rows.length ? (
        <Table rowKey="experiment_id" dataSource={rows} columns={columns} size="small"
          style={{ marginTop: 16 }}
          pagination={{ pageSize: 20, showSizeChanger: true, showTotal: (t) => `共 ${t} 条` }}
          rowClassName={(r) => (r.experiment_id === latestId ? 'row-latest' : '')}
          onRow={(r) => ({ onClick: () => setSelected(r), style: { cursor: 'pointer' } })}
        />
      ) : (
        <Empty description="暂无实验记录" style={{ marginTop: 24 }} />
      )}
      <DetailDrawer
        open={!!selected}
        onClose={() => setSelected(null)}
        title={selected ? (
          <Space>{selected.experiment_id}
            {selected.experiment_id === latestId && <Tag color="green">最新</Tag>}
          </Space>
        ) : ''}
        subtitle={selected ? (
          <Space>脚本 {selected.script} · 分区 {selected.partition} · {fmtDateTime(selected.timestamp)}</Space>
        ) : null}
        kvs={selected ? [
          { label: '配置哈希', value: <Typography.Text copyable>{selected.config_hash}</Typography.Text> },
        ] : []}
        extra={selected ? (
          <>
            <KVBlock title="参数" obj={selected.parameters} />
            <KVBlock title="结果" obj={selected.results} />
            {selected.notes && (
              <div>
                <Typography.Text strong>决策备注</Typography.Text>
                <Typography.Paragraph style={{ marginTop: 8, whiteSpace: 'pre-wrap' }}>{selected.notes}</Typography.Paragraph>
              </div>
            )}
          </>
        ) : null}
      />
    </div>
  )
}
```

- [ ] **Step 2: `web/ui/src/index.css` 末尾追加最新行高亮样式**

```css
/* 最新实验行高亮（Experiments 页） */
.row-latest td { background: #f6ffed !important; }
```

- [ ] **Step 3: 构建验证**

Run: `cd /d C:\Users\Frozen\ZCodeProject\quant-starter\web\ui && npm run build`
Expected: tsc 通过 + vite 构建成功输出 `dist/`。若 npm 不可用 → 系统终端执行，确认后继续。

- [ ] **Step 4: 记录 Phase 1 完成**

Phase 1 完成标志：Experiments 页点击行弹出抽屉（参数/结果/备注/哈希），表格默认按时间降序、可排序/分页/筛选，最新行绿色高亮。

---

# Phase 2 — 表格类页面

## Task 5: Overview 页

**Files:**
- Modify: `web/ui/src/pages/Overview.tsx`

- [ ] **Step 1: 修改 `web/ui/src/pages/Overview.tsx`**（保持图表逻辑不变，改指标卡与新增 summary 行）

替换 `statusColor/statusText` 两行为 import 自 labels；指标卡加 threshold；图表下方加 summary 卡：

```tsx
import { Card, Col, Row, Tag, Spin, Alert, Empty } from 'antd'
import { useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import { fetchGraduation, fetchEquity } from '../api'
import type { GraduationMetric } from '../api'
import { fmtNum, fmtPct } from '../lib/format'
import { statusMap } from '../lib/labels'
import StatCard from '../components/StatCard'

/** 按指标键区分单位：比例类用百分比，其余原样 */
const pctKeys = new Set(['excess_return', 'max_drawdown', 'fill_rate', 'monthly_win_rate'])

function valueText(m: GraduationMetric): string {
  if (m.value === null || m.value === undefined) return '—'
  return pctKeys.has(m.key) ? fmtPct(m.value) : fmtNum(m.value, m.key === 'runtime_days' ? 0 : 2)
}
function thresholdText(m: GraduationMetric): string {
  if (m.threshold === null || m.threshold === undefined) return '—'
  if (typeof m.threshold === 'string') return m.threshold
  return pctKeys.has(m.key) ? fmtPct(m.threshold, 0) : String(m.threshold)
}

export default function Overview() {
  const g = useQuery({ queryKey: ['graduation'], queryFn: fetchGraduation, refetchInterval: 60_000 })
  const eq = useQuery({ queryKey: ['equity'], queryFn: fetchEquity, refetchInterval: 60_000 })

  if (g.isLoading || eq.isLoading) return <Spin size="large" style={{ display: 'block', margin: '80px auto' }} />
  if (g.isError || eq.isError) return <Alert type="error" showIcon message="后端不可用" description="请确认 web/api 已启动 (uvicorn :8000)" />

  const metrics = g.data?.metrics ?? []
  const curve = eq.data?.curve ?? []
  const summary = eq.data?.summary

  const equityOption = {
    tooltip: { trigger: 'axis' },
    legend: { data: ['净值', '回撤'] },
    xAxis: { type: 'category', data: curve.map((p) => p.date) },
    yAxis: [{ type: 'value', name: '净值' }, { type: 'value', name: '回撤', axisLabel: { formatter: '{value}%' } }],
    series: [
      { name: '净值', type: 'line', data: curve.map((p) => p.total_equity), showSymbol: false },
      { name: '回撤', type: 'line', yAxisIndex: 1,
        data: (() => {
          let peak = -Infinity
          return curve.map((p) => {
            peak = Math.max(peak, p.total_equity)
            return Number(((p.total_equity / peak - 1) * 100).toFixed(2))
          })
        })(),
        showSymbol: false, lineStyle: { color: '#cf1322' } },
    ],
  }

  return (
    <div>
      <h2>毕业指标（目标 2026-11-03）</h2>
      {metrics.length ? (
      <Row gutter={[12, 12]}>
        {metrics.map((m) => (
          <Col key={m.key} xs={12} md={6}>
            <Card size="small" title={m.name}>
              <div style={{ fontSize: 20, fontWeight: 600 }}>
                {valueText(m)}
                <Tag color={statusMap[m.status]?.color ?? 'default'} style={{ marginLeft: 8 }}>
                  {statusMap[m.status]?.text ?? m.status}
                </Tag>
              </div>
              <div style={{ color: '#888', fontSize: 12 }}>阈值 {thresholdText(m)}</div>
              <div style={{ color: '#888', fontSize: 12 }}>{m.detail}</div>
            </Card>
          </Col>
        ))}
      </Row>
      ) : (
        <Empty description="毕业指标待计算（模拟盘 8/3 开跑后产生）" />
      )}
      <Card title="模拟盘净值与回撤" style={{ marginTop: 16 }}>
        {curve.length === 0 ? (
          <Empty description="模拟盘 8/3 开跑后每日累积权益数据" />
        ) : (
          <ReactECharts option={equityOption} style={{ height: 360 }} />
        )}
      </Card>
      <Row gutter={12} style={{ marginTop: 16 }}>
        <Col span={6}><StatCard title="累计收益" value={fmtPct(summary?.total_return)} /></Col>
        <Col span={6}><StatCard title="最大回撤" value={fmtPct(summary?.max_drawdown)} color="#cf1322" /></Col>
        <Col span={6}><StatCard title="年化波动" value={fmtPct(summary?.volatility)} /></Col>
        <Col span={6}><StatCard title="夏普比率" value={summary?.sharpe != null ? fmtNum(summary.sharpe, 2) : '—'} /></Col>
      </Row>
    </div>
  )
}
```

- [ ] **Step 2: 构建验证**

Run: `cd /d C:\Users\Frozen\ZCodeProject\quant-starter\web\ui && npm run build` → tsc + vite 通过。

## Task 6: Portfolio 页

**Files:**
- Modify: `web/ui/src/pages/Portfolio.tsx`（整体重写）

- [ ] **Step 1: 重写 `web/ui/src/pages/Portfolio.tsx`**

```tsx
import { useMemo } from 'react'
import { Card, Row, Col, Spin, Alert, Table, Typography, Empty } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { fetchPortfolio } from '../api'
import { col } from '../lib/columns'
import { fmtNum, fmtPct } from '../lib/format'
import StatCard from '../components/StatCard'

interface Position { symbol: string; qty: number; avg_cost: number; market_value: number; entry_date?: string }

export default function Portfolio() {
  const navigate = useNavigate()
  const { data, isLoading, isError } = useQuery({
    queryKey: ['portfolio'], queryFn: fetchPortfolio, refetchInterval: 30_000,
  })
  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="加载失败" />

  const positions = useMemo(() => (data?.positions ?? []) as Position[], [data])
  const pnl = (p: Position) => (p.market_value ?? 0) - (p.avg_cost ?? 0) * (p.qty ?? 0)
  const pnlPct = (p: Position) => (p.avg_cost ?? 0) * (p.qty ?? 0) > 0
    ? pnl(p) / ((p.avg_cost ?? 0) * (p.qty ?? 0)) : 0
  const totalMv = positions.reduce((s, p) => s + (p.market_value ?? 0), 0)
  const totalCost = positions.reduce((s, p) => s + (p.avg_cost ?? 0) * (p.qty ?? 0), 0)
  const totalPnl = totalMv - totalCost
  const pnlColor = (v: number) => (v > 0 ? '#cf1322' : v < 0 ? '#3f8600' : undefined)

  const columns = [
    col<Position>('symbol', { title: '代码', width: 100, sorter: true }),
    col<Position>('qty', { title: '数量', width: 100, sorter: true, render: (v) => fmtNum(v as number, 0) }),
    col<Position>('avg_cost', { title: '成本价', width: 120, sorter: true, render: (v) => fmtNum(v as number, 2) }),
    col<Position>('market_value', { title: '市值', width: 140, sorter: true, render: (v) => fmtNum(v as number, 2) }),
    { title: '浮动盈亏', dataIndex: 'pnl', width: 160,
      sorter: (a: Position, b: Position) => pnl(a) - pnl(b),
      render: (_v: unknown, r: Position) => (
        <span style={{ color: pnlColor(pnl(r)) }}>
          {fmtNum(pnl(r), 2)}（{fmtPct(pnlPct(r))}）
        </span>
      ) },
    col<Position>('entry_date', { title: '建仓日', width: 120, sorter: true }),
  ]

  return (
    <div>
      <Typography.Title level={4}>模拟盘组合</Typography.Title>
      <Row gutter={12}>
        <Col span={6}><StatCard title="现金" value={fmtNum(data?.cash, 0)} /></Col>
        <Col span={6}><StatCard title="总市值" value={fmtNum(totalMv, 0)} /></Col>
        <Col span={6}><StatCard title="总成本" value={fmtNum(totalCost, 0)} /></Col>
        <Col span={6}><StatCard title="总盈亏" value={fmtNum(totalPnl, 0)} color={pnlColor(totalPnl)} /></Col>
      </Row>
      {positions.length ? (
        <Table rowKey="symbol" dataSource={positions} columns={columns} size="small"
          style={{ marginTop: 16 }}
          pagination={{ pageSize: 10, showSizeChanger: true, showTotal: (t) => `共 ${t} 条` }}
          onRow={(r) => ({ onClick: () => navigate(`/stocks?symbol=${r.symbol}`), style: { cursor: 'pointer' } })}
        />
      ) : (
        <Empty description="暂无持仓（模拟盘 8/3 开跑后产生）" style={{ marginTop: 24 }} />
      )}
    </div>
  )
}
```

- [ ] **Step 2: 构建验证**：`npm run build` 通过。

## Task 7: Signals 页

**Files:**
- Modify: `web/ui/src/pages/Signals.tsx`（整体重写）

- [ ] **Step 1: 重写 `web/ui/src/pages/Signals.tsx`**

```tsx
import { useMemo, useState } from 'react'
import { Table, Typography, Spin, Alert, Empty, Tag, Select, Space } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { fetchSignals } from '../api'
import { col } from '../lib/columns'
import { fmtDateTime } from '../lib/format'
import { modeMap, verdictMap } from '../lib/labels'

interface Signal {
  timestamp: string
  signal_date: string
  mode: string
  buy: string[]
  sell: string[]
  hold: string[]
  n_factors?: number
  verdict?: string
  circuit_breaker?: string
  execution?: { buy_filled?: number; buy_rejected?: number; sell_filled?: number; sell_rejected?: number }
}

/** 股票代码列表（最多 3 个 Tag，点击跳个股页） */
function TagList({ codes, onCode }: { codes: string[]; onCode: (c: string) => void }) {
  if (!codes || codes.length === 0) return <Typography.Text type="secondary">—</Typography.Text>
  const shown = codes.slice(0, 3)
  return (
    <Space size={4} wrap>
      {shown.map((c) => (
        <Tag key={c} color="blue" style={{ cursor: 'pointer' }}
          onClick={(e) => { e.stopPropagation(); onCode(c) }}>{c}</Tag>
      ))}
      {codes.length > 3 && <Typography.Text type="secondary">等{codes.length}只</Typography.Text>}
    </Space>
  )
}

export default function Signals() {
  const navigate = useNavigate()
  const { data, isLoading, isError } = useQuery({
    queryKey: ['signals'], queryFn: fetchSignals, refetchInterval: 60_000,
  })
  const [mode, setMode] = useState<string>()
  const [verdict, setVerdict] = useState<string>()

  const sigs = useMemo(() => (data?.signals ?? []) as Signal[], [data])
  const rows = useMemo(() => sigs.filter((s) =>
    (!mode || s.mode === mode) && (!verdict || s.verdict === verdict)),
  [sigs, mode, verdict])

  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="加载失败" />

  const columns = [
    col<Signal>('timestamp', { title: '时间', width: 150, sorter: true, defaultSortOrder: 'descend',
      render: (v) => fmtDateTime(v as string) }),
    col<Signal>('signal_date', { title: '信号日期', width: 110, sorter: true }),
    col<Signal>('mode', { title: '模式', width: 80,
      render: (v) => { const m = modeMap[String(v)]; return <Tag color={m?.color}>{m?.text ?? String(v)}</Tag> } }),
    { title: '买入', dataIndex: 'buy',
      render: (_v: unknown, r: Signal) => <TagList codes={r.buy} onCode={(c) => navigate(`/stocks?symbol=${c}`)} /> },
    { title: '卖出', dataIndex: 'sell',
      render: (_v: unknown, r: Signal) => <TagList codes={r.sell} onCode={(c) => navigate(`/stocks?symbol=${c}`)} /> },
    col<Signal>('n_factors', { title: '因子数', width: 90, sorter: true }),
    col<Signal>('verdict', { title: '结论', width: 100,
      render: (v) => { const m = verdictMap[String(v)] ?? verdictMap[String(v).toUpperCase()]; return <Tag color={m?.color}>{m?.text ?? String(v)}</Tag> } }),
    col<Signal>('circuit_breaker', { title: '熔断', width: 90,
      render: (v) => (String(v) === 'active' ? <Tag color="red">熔断中</Tag> : '—') }),
    { title: '成交', dataIndex: 'execution', width: 150,
      render: (_v: unknown, r: Signal) => {
        const e = r.execution
        if (!e) return '—'
        return `买${e.buy_filled ?? 0}/${e.buy_rejected ?? 0} 卖${e.sell_filled ?? 0}/${e.sell_rejected ?? 0}`
      } },
  ]

  return (
    <div>
      <Typography.Title level={4}>每日信号（共 {data?.count ?? 0} 条，显示最近 200）</Typography.Title>
      <Space style={{ marginBottom: 12 }}>
        <Select allowClear placeholder="模式" style={{ width: 140 }} value={mode} onChange={setMode}
          options={[{ value: 'live', label: '实盘' }, { value: 'dry_run', label: '模拟' }]} />
        <Select allowClear placeholder="结论" style={{ width: 140 }} value={verdict} onChange={setVerdict}
          options={Object.entries(verdictMap).map(([k, v]) => ({ value: k, label: v.text }))} />
      </Space>
      {rows.length ? (
        <Table rowKey={(r) => `${r.timestamp}-${r.signal_date}`} dataSource={rows} columns={columns} size="small"
          pagination={{ pageSize: 20, showSizeChanger: true, showTotal: (t) => `共 ${t} 条` }} />
      ) : (
        <Empty description="暂无信号（模拟盘 8/3 开跑后产生）" />
      )}
    </div>
  )
}
```

- [ ] **Step 2: 构建验证**：`npm run build` 通过。

## Task 8: Factors 页

**Files:**
- Modify: `web/ui/src/pages/Factors.tsx`（整体重写）

- [ ] **Step 1: 重写 `web/ui/src/pages/Factors.tsx`**

```tsx
import { Tabs, Table, Typography, Spin, Alert, Empty, Space, Tag } from 'antd'
import type { ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchFactorsIc } from '../api'
import { col } from '../lib/columns'
import { fmtNum } from '../lib/format'

const sourceLabel: Record<string, string> = {
  p3_full_ic: '价量因子 (P3)', p6_fundamental_ic: '基本面 (P6)',
  p7_relative_ic: '相对因子 (P7)', p8_northbound_ic: '北向 (P8)', p9_minute_ic: '分钟因子 (P9)',
}

interface IcRow { factor: string; ic_mean?: number; icir?: number; ic_std?: number; n_days?: number; pos_ratio?: number }

export default function Factors() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['factors-ic'], queryFn: fetchFactorsIc,
  })
  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="加载失败" />
  const sources = data?.sources ?? []
  const resultOf = (s: string) => data?.results?.[s] ?? { results: [], meta: {} }

  const cols = [
    col<IcRow>('factor', { title: '因子', sorter: true, ellipsis: true }),
    col<IcRow>('ic_mean', { title: '平均IC', sorter: true, defaultSortOrder: 'descend',
      render: (v) => fmtNum(v as number, 4) }),
    col<IcRow>('icir', { title: 'ICIR', sorter: true, render: (v) => fmtNum(v as number, 4) }),
    col<IcRow>('ic_std', { title: 'IC标准差', sorter: true, render: (v) => fmtNum(v as number, 4) }),
    col<IcRow>('n_days', { title: '样本天数', sorter: true, render: (v) => fmtNum(v as number, 0) }),
    col<IcRow>('pos_ratio', { title: '正占比', sorter: true, render: (v) => fmtNum(v as number, 4) }),
  ]

  /** meta 全量键值（不只 description） */
  const metaTags = (s: string) => Object.entries(resultOf(s).meta ?? {})
    .map(([k, v]) => <Tag key={k}>{k}: {String(v)}</Tag>)

  /** 均值摘要 */
  const summaryOf = (s: string) => {
    const rs = resultOf(s).results ?? []
    if (!rs.length) return null
    const avg = (k: 'ic_mean' | 'icir') => rs.reduce((a, r) => a + (r[k] ?? 0), 0) / rs.length
    return { n: rs.length, ic: avg('ic_mean'), icir: avg('icir') }
  }

  return (
    <div>
      <Typography.Title level={4}>因子 IC 验证结果</Typography.Title>
      {sources.length === 0 ? (
        <Empty description="暂无因子 IC 数据（数据积累中）" />
      ) : (
      <Tabs
        items={sources.map((s: string) => {
          const sum = summaryOf(s)
          return {
            key: s, label: sourceLabel[s] ?? s,
            children: (
              <div>
                <Space wrap style={{ marginBottom: 8 }}>
                  {metaTags(s)}
                  {sum && <Tag color="blue">因子 {sum.n} 个 · 平均IC {fmtNum(sum.ic, 4)} · 平均ICIR {fmtNum(sum.icir, 4)}</Tag>}
                </Space>
                <Table rowKey="factor" size="small" columns={cols}
                  dataSource={resultOf(s).results ?? []}
                  pagination={{ pageSize: 20, showSizeChanger: true, showTotal: (t) => `共 ${t} 条` }} />
              </div>
            ),
          }
        })}
      />
      )}
    </div>
  )
}
```

- [ ] **Step 2: 构建验证**：`npm run build` 通过。

- [ ] **Step 3: 记录 Phase 2 完成**

Phase 2 完成标志：4 个页面全部中文列名、表格可排序分页；Overview 显示阈值与 summary 卡；Portfolio 显示盈亏与汇总；Signals 有 Tag/筛选；Factors 有排序/摘要/meta。

---

# Phase 3 — 交互类页面 + 联动

## Task 9: Stocks 页

**Files:**
- Modify: `web/ui/src/pages/Stocks.tsx`（整体重写）

- [ ] **Step 1: 重写 `web/ui/src/pages/Stocks.tsx`**

```tsx
import { useEffect, useMemo, useState } from 'react'
import { Card, AutoComplete, Spin, Alert, Typography, Empty } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import ReactECharts from 'echarts-for-react'
import { searchUniverse, fetchStock } from '../api'

interface Ohlc { date: string; open: number; high: number; low: number; close: number; volume?: number }

/** 简单移动平均（前 n-1 位为 null） */
function ma(data: Ohlc[], n: number): (number | null)[] {
  return data.map((_, i) => {
    if (i < n - 1) return null
    let s = 0
    for (let j = i - n + 1; j <= i; j++) s += data[j].close
    return Number((s / n).toFixed(2))
  })
}

export default function Stocks() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [input, setInput] = useState('')
  const [q, setQ] = useState('')
  const [symbol, setSymbol] = useState<string | null>(searchParams.get('symbol'))
  const [selectedLabel, setSelectedLabel] = useState<string | null>(null)

  // 400ms 防抖
  useEffect(() => {
    const t = setTimeout(() => setQ(input.trim()), 400)
    return () => clearTimeout(t)
  }, [input])

  const search = useQuery({
    queryKey: ['search', q], queryFn: () => searchUniverse(q),
    enabled: q.length >= 1,
  })
  const detail = useQuery({
    queryKey: ['stock', symbol], queryFn: () => fetchStock(symbol as string),
    enabled: !!symbol,
  })

  const options = useMemo(() =>
    (search.data?.stocks ?? []).map((s) => ({
      value: s.symbol, label: `${s.symbol} ${s.name}`,
    })), [search.data])

  const ohlc = (detail.data?.ohlc ?? []) as Ohlc[]
  const candleOption = {
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    legend: { data: ['K线', 'MA5', 'MA10', 'MA20', '成交量'] },
    grid: [
      { left: 60, right: 20, top: 30, height: '55%' },
      { left: 60, right: 20, top: '72%', height: '18%' },
    ],
    xAxis: [
      { type: 'category', data: ohlc.map((o) => o.date) },
      { type: 'category', gridIndex: 1, data: ohlc.map((o) => o.date), axisLabel: { show: false } },
    ],
    yAxis: [
      { scale: true },
      { gridIndex: 1, scale: true, axisLabel: { show: false } },
    ],
    dataZoom: [{ type: 'inside' }],
    series: [
      { name: 'K线', type: 'candlestick', data: ohlc.map((o) => [o.open, o.close, o.low, o.high]) },
      { name: 'MA5', type: 'line', data: ma(ohlc, 5), showSymbol: false, smooth: true },
      { name: 'MA10', type: 'line', data: ma(ohlc, 10), showSymbol: false, smooth: true },
      { name: 'MA20', type: 'line', data: ma(ohlc, 20), showSymbol: false, smooth: true },
      { name: '成交量', type: 'bar', xAxisIndex: 1, yAxisIndex: 1,
        data: ohlc.map((o) => ({
          value: o.volume ?? 0,
          itemStyle: { color: o.close >= o.open ? '#cf1322' : '#3f8600' },
        })) },
    ],
  }

  return (
    <div>
      <Typography.Title level={4}>个股行情</Typography.Title>
      <AutoComplete
        style={{ width: 320 }} options={options} value={selectedLabel ?? symbol ?? input}
        loading={search.isFetching}
        onSearch={setInput}
        onSelect={(v, o) => {
          setSymbol(v)
          setSelectedLabel(String(o.label))
          setSearchParams({ symbol: v })
        }}
        placeholder="输入代码或名称（如 600519 / 茅台）"
      />
      {search.isError && <Alert type="error" message="股票搜索服务不可用" style={{ marginTop: 16 }} />}
      {detail.isLoading && <Spin style={{ marginTop: 24, display: 'block' }} />}
      {detail.isError && <Alert type="error" message="股票不存在或数据缺失" style={{ marginTop: 16 }} />}
      {detail.data && (ohlc.length ? (
        <Card title={`${detail.data.symbol} ${detail.data.name}`} style={{ marginTop: 16 }}>
          <ReactECharts option={candleOption} style={{ height: 480 }} />
        </Card>
      ) : (
        <Empty description="该股票暂无行情数据" style={{ marginTop: 24 }} />
      ))}
    </div>
  )
}
```

- [ ] **Step 2: 构建验证**：`npm run build` 通过。

## Task 10: Trading 页

**Files:**
- Modify: `web/ui/src/pages/Trading.tsx`（整体重写）

- [ ] **Step 1: 重写 `web/ui/src/pages/Trading.tsx`**

```tsx
import { Card, Col, Row, Table, Typography, Spin, Alert, Empty, Tag } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { fetchBroker } from '../api'
import { col } from '../lib/columns'
import { fmtNum, fmtDateTime } from '../lib/format'
import { actionTag } from '../lib/labels'
import StatCard from '../components/StatCard'

interface Position { symbol: string; qty: number; avg_cost: number; market_value: number }
interface Trade { date: string; symbol: string; action: string; qty: number; price: number; commission?: number; reason?: string }

export default function Trading() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['broker'], queryFn: fetchBroker, refetchInterval: 30_000,
  })
  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="执行状态不可用" />

  const connected = !!data?.connected
  const positions = (data?.positions ?? []) as Position[]
  const trades = (data?.trades ?? []) as Trade[]
  const pnl = (p: Position) => (p.market_value ?? 0) - (p.avg_cost ?? 0) * (p.qty ?? 0)

  const posCols = [
    col<Position>('symbol', { title: '代码', width: 100 }),
    col<Position>('qty', { title: '数量', width: 100, render: (v) => fmtNum(v as number, 0) }),
    col<Position>('avg_cost', { title: '成本价', width: 120, render: (v) => fmtNum(v as number, 2) }),
    col<Position>('market_value', { title: '市值', width: 140, render: (v) => fmtNum(v as number, 2) }),
    { title: '浮动盈亏', dataIndex: 'pnl', width: 160,
      render: (_v: unknown, r: Position) => (
        <span style={{ color: pnl(r) >= 0 ? '#cf1322' : '#3f8600' }}>{fmtNum(pnl(r), 2)}</span>
      ) },
  ]
  const tradeCols = [
    col<Trade>('date', { title: '时间', width: 150, render: (v) => fmtDateTime(v as string) }),
    col<Trade>('symbol', { title: '代码', width: 100 }),
    col<Trade>('action', { title: '方向', width: 90,
      render: (v) => { const t = actionTag(v); return <Tag color={t.color}>{t.text}</Tag> } }),
    col<Trade>('qty', { title: '数量', width: 100, render: (v) => fmtNum(v as number, 0) }),
    col<Trade>('price', { title: '价格', width: 110, render: (v) => fmtNum(v as number, 2) }),
    col<Trade>('commission', { title: '佣金', width: 100, render: (v) => fmtNum(v as number, 2) }),
    col<Trade>('reason', { title: '原因', ellipsis: true }),
  ]

  return (
    <div>
      <Typography.Title level={4}>交易监控（{data?.adapter} · {connected ? '已连接' : '未连接'}）</Typography.Title>
      {!connected && (
        <Alert type="warning" showIcon message="模拟盘执行器未连接"
          description="请检查 execution/paper_executor 是否运行，或后端 broker 适配器配置" style={{ marginBottom: 16 }} />
      )}
      <Row gutter={12}>
        <Col span={8}><StatCard title="现金" value={fmtNum(data?.balance?.cash, 0)} /></Col>
        <Col span={8}><StatCard title="持仓数" value={positions.length} /></Col>
        <Col span={8}><StatCard title="今日成交" value={trades.length} /></Col>
      </Row>
      {positions.length ? (
        <Table rowKey="symbol" dataSource={positions} columns={posCols} size="small" style={{ marginTop: 16 }}
          pagination={{ pageSize: 10, showTotal: (t) => `共 ${t} 条` }} />
      ) : (
        <Empty description="暂无持仓" style={{ marginTop: 24 }} />
      )}
      <Typography.Title level={5} style={{ marginTop: 24 }}>最近成交</Typography.Title>
      {trades.length ? (
        <Table rowKey={(_r, i) => `${i}`} dataSource={trades} columns={tradeCols} size="small"
          pagination={{ pageSize: 10, showSizeChanger: true, showTotal: (t) => `共 ${t} 条` }} />
      ) : (
        <Empty description="暂无成交记录" />
      )}
    </div>
  )
}
```

- [ ] **Step 2: 构建验证**：`npm run build` 通过。

## Task 11: DataStatus 页 + 删除死代码

**Files:**
- Modify: `web/ui/src/pages/DataStatus.tsx`（整体重写）
- Delete: `web/ui/src/pages/Placeholder.tsx`

- [ ] **Step 1: 重写 `web/ui/src/pages/DataStatus.tsx`**

```tsx
import { Card, List, Tag, Typography, Spin, Alert, Empty, Row, Col } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { fetchHealth } from '../api'
import { healthSourceLabels } from '../lib/labels'
import StatCard from '../components/StatCard'

export default function DataStatus() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['health'], queryFn: fetchHealth, refetchInterval: 60_000,
  })
  if (isError) return <Alert type="error" message="后端不可用" description="请确认 web/api 已启动 (uvicorn :8000)" />
  const sources = Object.entries(data?.data_sources ?? {})
  const okCount = sources.filter(([, v]) => v).length
  return (
    <div>
      <Typography.Title level={4}>数据状态</Typography.Title>
      <Row gutter={12} style={{ marginBottom: 16 }}>
        <Col span={8}><StatCard title="可用数据源" value={okCount} /></Col>
        <Col span={8}><StatCard title="数据源总数" value={sources.length} /></Col>
        <Col span={8}><StatCard title="后端状态" value={data?.status === 'ok' ? '正常' : data?.status ?? '—'} /></Col>
      </Row>
      <Card>
        {isLoading ? <Spin /> : (sources.length ? (
          <List
            dataSource={sources}
            renderItem={([k, v]) => (
              <List.Item>{healthSourceLabels[k] ?? k}（{k}）：
                <Tag color={v ? 'green' : 'orange'}>{v ? '可用' : '待积累'}</Tag>
              </List.Item>
            )}
          />
        ) : (
          <Empty description="暂无数据源状态" />
        ))}
      </Card>
    </div>
  )
}
```

- [ ] **Step 2: 删除 `web/ui/src/pages/Placeholder.tsx`**

用 `del "C:\Users\Frozen\ZCodeProject\quant-starter\web\ui\src\pages\Placeholder.tsx"` 删除（未被任何文件 import，已确认）。

- [ ] **Step 3: 构建验证**

Run: `cd /d C:\Users\Frozen\ZCodeProject\quant-starter\web\ui && npm run build`
Expected: tsc 通过（删除 Placeholder 后无引用错误）+ vite 构建成功。

## Task 12: 全量验证

- [ ] **Step 1: lint 全量检查**

Run: `cd /d C:\Users\Frozen\ZCodeProject\quant-starter\web\ui && npm run lint`
Expected: oxlint 无 error（warning 可接受）。若报 'npm' 不可用 → 系统终端执行。

- [ ] **Step 2: 生产构建**

Run: `cd /d C:\Users\Frozen\ZCodeProject\quant-starter\web\ui && npm run build`
Expected: tsc 0 错误 + vite 输出 dist/。

- [ ] **Step 3: 浏览器真机验证**（需完整环境：python + node）

1. 启动后端：系统终端执行
   ```
   cd C:\Users\Frozen\ZCodeProject\quant-starter
   python -m uvicorn web.api.main:app --port 8000
   ```
2. 启动前端：系统终端执行
   ```
   cd C:\Users\Frozen\ZCodeProject\quant-starter\web\ui
   npm run dev
   ```
3. 打开 http://localhost:5173 ，逐页验证：
   - **总览**：指标卡显示阈值；summary 卡 4 项数值与净值图下方
   - **组合**：持仓表含盈亏列、可排序分页；顶部 4 张汇总卡
   - **信号**：中文列 + 模式/结论 Tag + 筛选下拉 + 排序分页；点买入/卖出股票 Tag 跳个股页并加载该股 K 线
   - **因子**：中文列 + 平均IC 默认降序 + 摘要 Tag + meta Tag
   - **实验**：最新行绿色高亮在最上；点任意行 → 抽屉显示 参数/结果/决策备注/配置哈希(可复制)；脚本/分区筛选；分页
   - **个股**：输入"600519"搜索出现联想；选中后 K 线含 MA5/10/20 与成交量副图；从信号页跳转带 ?symbol= 自动加载
   - **交易监控**：成交表方向 Tag 着色 + 分页；未连接时显示告警条；持仓盈亏列
   - **数据状态**：3 张统计卡 + 中文数据源列表 + 60s 自动刷新

- [ ] **Step 4: 总结交付**

在会话总结中报告：改动文件清单、验证结果（build/lint/浏览器逐项）、遗留问题（如有）。

---

## Self-Review 记录

- **规格覆盖**：Experiments 抽屉详情(规格§4 Phase1) → Task 4；排序/分页/筛选 → Tasks 4/6/7/8；Overview threshold+summary → Task 5；Portfolio 盈亏+汇总 → Task 6；Signals 固定列+Tag+联动 → Task 7/9/11；Factors 排序+meta → Task 8；Stocks 防抖+MA+成交量+URL → Task 9；Trading Tag+分页+告警 → Task 10；DataStatus 刷新+统计 → Task 11；删除 Placeholder → Task 11；空态统一 Empty → 各 Task（错误态保留 Alert）；api.ts 补全 → Task 3。无遗漏。
- **占位符扫描**：无 TBD/TODO；每个代码步骤含完整代码。
- **类型一致性**：`col()` 泛型签名、`DetailDrawer` 的 `KV` 接口、`StatCard` props、`labels.ts` 导出（modeMap/verdictMap/actionTag/statusMap/partitionColors/healthSourceLabels）、`format.ts` 导出（fmtNum/fmtPct/fmtDateTime/truncate）、`api.ts` 封装函数名（fetchExperiments/fetchPortfolio/fetchSignals/fetchFactorsIc/fetchBroker/fetchHealth/searchUniverse/fetchStock）在全部 Task 中一致使用。
