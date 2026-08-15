import { useMemo, useState } from 'react'
import { Card, Col, Row, Table, Tag, Checkbox, Spin, Alert, Empty, Button, Typography, Collapse, Space } from 'antd'
import { useQuery, useQueries } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import { fetchExperimentDetail, fetchUniverse, fetchExperiments } from '../api'
import type { ExperimentRegistryItem } from '../api'
import { fmtNum, fmtPct, fmtDateTime, truncate } from '../lib/format'
import { partitionColors } from '../lib/labels'
import { useExperiment } from '../experiment-context'
import ExperimentReport from '../components/ExperimentReport'
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

export default function Experiments() {
  const { expId, setExpId, detail, detailLoading, registry } = useExperiment()
  const [compareIds, setCompareIds] = useState<string[]>([])
  const uni = useQuery({ queryKey: ['universe'], queryFn: fetchUniverse, staleTime: 300_000 })
  const legacy = useQuery({ queryKey: ['experiments'], queryFn: fetchExperiments })
  const [selected, setSelected] = useState<Exp | null>(null)

  const nameOf = useMemo(() => {
    const m = new Map<string, string>()
    for (const s of uni.data?.stocks ?? []) m.set(s.symbol, s.name)
    return (sym: string) => m.get(sym) ?? sym
  }, [uni.data])

  const exps: ExperimentRegistryItem[] = registry?.experiments ?? []

  // 对比: 勾选的实验详情 (useQueries 支持动态数量)
  const compareQueries = useQueries({
    queries: compareIds.map(id => ({ queryKey: ['exp-detail', id], queryFn: () => fetchExperimentDetail(id), staleTime: 60_000 })),
  })
  const compareData = compareIds.map((id, i) => ({ id, q: compareQueries[i] }))

  const compareMetrics = [
    ['excess_annual', '年化超额', 'pct'], ['total_return', '总收益', 'pct'],
    ['sharpe', 'Sharpe', 'num'], ['max_drawdown', '最大回撤', 'pct'],
    ['calmar', 'Calmar', 'num'], ['avg_turnover', '平均换手', 'pct'],
  ] as const

  const compareTableData = compareMetrics.map(([key, label]) => {
    const row: any = { key, label }
    let best: number | null = null
    let bestId = ''
    for (const { id, q } of compareData) {
      const m = q.data?.metrics?.find(mm => mm.key === key)
      row[id] = m?.value ?? null
      if (typeof m?.value === 'number' && m.value !== null) {
        const better = m.better === 'low' ? 'min' : 'max'
        if (best === null || (better === 'max' && m.value > best) || (better === 'min' && m.value < best)) {
          best = m.value; bestId = id
        }
      }
    }
    row._best = bestId
    return row
  })

  const compareSeries: any[] = compareData.map(({ id, q }) => {
    const eq = q.data?.series?.find(s => s.name === '组合净值')
    return { id, name: id, x: eq?.x ?? [], y: eq?.y ?? [] }
  })

  // 旧 exp_*.json 记录 (保留原有能力, 折叠区)
  const legacyExps = useMemo(() => (legacy.data?.experiments ?? []) as Exp[], [legacy.data])
  const legacyCols = [
    { title: '实验ID', dataIndex: 'experiment_id', width: 220, ellipsis: true, sorter: (a: Exp, b: Exp) => a.experiment_id.localeCompare(b.experiment_id) },
    { title: '时间', dataIndex: 'timestamp', width: 150, sorter: (a: Exp, b: Exp) => a.timestamp.localeCompare(b.timestamp), defaultSortOrder: 'descend' as const, render: (v: string) => fmtDateTime(v) },
    { title: '脚本', dataIndex: 'script', sorter: (a: Exp, b: Exp) => a.script.localeCompare(b.script) },
    { title: '分区', dataIndex: 'partition', width: 130, render: (v: string) => <Tag color={partitionColors[v] ?? 'default'}>{v}</Tag> },
    { title: '备注摘要', dataIndex: 'notes', ellipsis: true, render: (v: string) => truncate(v ?? '', 60) },
  ]

  return (
    <div>
      <Typography.Title level={4}>实验（可插拔注册表）</Typography.Title>
      <Typography.Paragraph type="secondary">
        新实验产出标准 JSON 后自动出现在此列表（无需改前端）。勾选最多 3 个进行对比。
      </Typography.Paragraph>

      {/* 对比勾选 */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {exps.map(e => (
            <Checkbox key={e.id} checked={compareIds.includes(e.id)}
              disabled={!compareIds.includes(e.id) && compareIds.length >= 3}
              onChange={(ev) => {
                setCompareIds(prev => ev.target.checked
                  ? [...prev, e.id] : prev.filter(x => x !== e.id))
              }}>
              {e.name}
            </Checkbox>
          ))}
        </div>
        {compareIds.length >= 2 && (
          <>
            <Card size="small" title="指标对比（绿色=最优）" style={{ marginTop: 12 }}>
              <Table size="small" rowKey="key" pagination={false}
                dataSource={compareTableData}
                columns={[
                  { title: '指标', dataIndex: 'label' },
                  ...compareIds.map(id => ({
                    title: id, dataIndex: id,
                    render: (v: number | null, r: any) => {
                      if (v === null || v === undefined) return '—'
                      const key = r.key as string
                      const isPct = compareMetrics.find(c => c[0] === key)?.[2] === 'pct'
                      const text = isPct ? fmtPct(v) : fmtNum(v, 2)
                      return <span style={{ color: r._best === id ? '#389e0d' : undefined, fontWeight: r._best === id ? 600 : 400 }}>{text}</span>
                    },
                  })),
                ]} />
            </Card>
            <Card size="small" title="净值曲线叠加" style={{ marginTop: 12 }}>
              <ReactECharts option={{
                tooltip: { trigger: 'axis' },
                legend: { data: compareSeries.map(s => s.name) },
                grid: { left: 60, right: 30, top: 40, bottom: 30 },
                xAxis: { type: 'category', data: compareSeries[0]?.x ?? [] },
                yAxis: { type: 'value', scale: true },
                series: compareSeries.map(s => ({ name: s.name, type: 'line', data: s.y, showSymbol: false })),
              }} style={{ height: 320 }} />
            </Card>
          </>
        )}
      </Card>

      {/* 实验卡片列表 */}
      <Row gutter={[12, 12]}>
        {exps.map(e => (
          <Col key={e.id} xs={24} md={12} xl={8}>
            <Card size="small" hoverable
              style={{ borderColor: expId === e.id ? '#1677ff' : undefined }}
              onClick={() => setExpId(e.id)}
              title={<span>{e.name} {e.kind === 'walkforward' ? <Tag color="blue">回测</Tag> : <Tag>实验</Tag>}</span>}
              extra={<Button type="link" size="small" onClick={(ev) => { ev.stopPropagation(); setExpId(e.id) }}>查看报告</Button>}>
              <div style={{ fontSize: 12, color: '#888' }}>{e.generated_at}</div>
              {e.summary?.excess_annual != null && (
                <div style={{ marginTop: 8 }}>
                  年化超额 <b>{fmtPct(e.summary.excess_annual)}</b> · Sharpe <b>{e.summary.sharpe != null ? fmtNum(e.summary.sharpe, 2) : '—'}</b> · 回撤 <b>{e.summary.max_drawdown != null ? fmtPct(e.summary.max_drawdown) : '—'}</b>
                </div>
              )}
            </Card>
          </Col>
        ))}
        {!exps.length && <Empty description="注册表为空" />}
      </Row>

      {/* 当前选中实验的报告视图 */}
      {expId && (
        <Card title={`实验报告: ${expId}`} style={{ marginTop: 24 }}>
          {detailLoading ? <Spin /> : detail
            ? <ExperimentReport detail={detail} nameOf={nameOf} />
            : <Alert type="error" message="加载失败" />}
        </Card>
      )}

      {/* 旧实验记录 (exp_*.json KV) */}
      <Collapse style={{ marginTop: 24 }}
        items={[{
          key: 'legacy', label: `历史实验记录 (exp_*.json, ${legacy.data?.count ?? 0} 条)`,
          children: legacyExps.length ? (
            <Table rowKey="experiment_id" dataSource={legacyExps} columns={legacyCols} size="small"
              pagination={{ pageSize: 15, showSizeChanger: true, showTotal: (t) => `共 ${t} 条` }}
              onRow={(r) => ({ onClick: () => setSelected(r), style: { cursor: 'pointer' } })}
            />
          ) : <Empty description="暂无历史实验记录" />,
        }]}
      />

      <DetailDrawer
        open={!!selected}
        onClose={() => setSelected(null)}
        title={selected?.experiment_id ?? ''}
        subtitle={selected ? (
          <Space>脚本 {selected.script} · 分区 {selected.partition} · {fmtDateTime(selected.timestamp)}</Space>
        ) : null}
        kvs={selected ? [{ label: '配置哈希', value: <Typography.Text copyable>{selected.config_hash}</Typography.Text> }] : []}
        extra={selected ? (
          <>
            {Object.keys(selected.parameters ?? {}).length > 0 && (
              <DescriptionsBlock title="参数" obj={selected.parameters} />
            )}
            {Object.keys(selected.results ?? {}).length > 0 && (
              <DescriptionsBlock title="结果" obj={selected.results} />
            )}
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

/** 递归键值块：值仍为对象时 JSON 序列化展示 */
function DescriptionsBlock({ title, obj }: { title: string; obj?: Record<string, unknown> }) {
  if (!obj || Object.keys(obj).length === 0) return null
  return (
    <div style={{ marginBottom: 16 }}>
      <Typography.Text strong>{title}</Typography.Text>
      <pre style={{ marginTop: 8, fontSize: 12, background: '#fafafa', padding: 8, borderRadius: 4, overflow: 'auto' }}>
        {JSON.stringify(obj, null, 2)}
      </pre>
    </div>
  )
}
