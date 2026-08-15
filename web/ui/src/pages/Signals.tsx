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
  const latestId = useMemo(() => {
    if (!rows.length) return undefined
    return rows.reduce((a, b) => (b.timestamp > a.timestamp ? b : a)).timestamp
  }, [rows])

  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="加载失败" />

  const columns = [
    { title: '', dataIndex: 'latest', width: 60,
      render: (_v: unknown, r: Signal) => (r.timestamp === latestId ? <Tag color="green">最新</Tag> : '') },
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
          rowClassName={(r) => (r.timestamp === latestId ? 'row-latest' : '')}
          pagination={{ pageSize: 20, showSizeChanger: true, showTotal: (t) => `共 ${t} 条` }} />
      ) : (
        <Empty description="暂无信号（模拟盘 8/3 开跑后产生）" />
      )}
    </div>
  )
}
