import { useMemo } from 'react'
import { Card, Row, Col, Spin, Alert, Table, Typography, Empty } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { fetchPortfolio, fetchUniverse } from '../api'
import { col } from '../lib/columns'
import { fmtNum, fmtPct } from '../lib/format'
import StatCard from '../components/StatCard'
import PnlBarChart from '../components/PnlBarChart'

interface Position { symbol: string; qty: number; avg_cost: number; market_value: number; entry_date?: string; current_price?: number | null; pnl?: number; pnl_pct?: number | null }

export default function Portfolio() {
  const navigate = useNavigate()
  const { data, isLoading, isError } = useQuery({
    queryKey: ['portfolio'], queryFn: fetchPortfolio, refetchInterval: 30_000,
  })
  const uni = useQuery({ queryKey: ['universe'], queryFn: fetchUniverse, staleTime: 300_000 })
  const nameOf = useMemo(() => {
    const m = new Map<string, string>()
    for (const s of uni.data?.stocks ?? []) m.set(s.symbol, s.name)
    return (sym: string) => m.get(sym) ?? sym
  }, [uni.data])

  const positions = useMemo(() => (data?.positions ?? []) as Position[], [data])
  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="加载失败" />

  const pnl = (p: Position) => p.pnl ?? (p.market_value ?? 0) - (p.avg_cost ?? 0) * (p.qty ?? 0)
  const pnlPct = (p: Position) => p.pnl_pct ?? ((p.avg_cost ?? 0) * (p.qty ?? 0) > 0 ? pnl(p) / ((p.avg_cost ?? 0) * (p.qty ?? 0)) : 0)
  const totalMv = positions.reduce((s, p) => s + (p.market_value ?? 0), 0)
  const totalCost = positions.reduce((s, p) => s + (p.avg_cost ?? 0) * (p.qty ?? 0), 0)
  const totalPnl = totalMv - totalCost
  const pnlColor = (v: number) => (v > 0 ? '#cf1322' : v < 0 ? '#3f8600' : undefined)

  const columns = [
    col<Position>('symbol', { title: '代码', width: 110, sorter: true, render: (v) => `${nameOf(v as string)} ${v}` }),
    col<Position>('qty', { title: '数量', width: 90, sorter: true, render: (v) => fmtNum(v as number, 0) }),
    col<Position>('avg_cost', { title: '成本价', width: 100, sorter: true, render: (v) => fmtNum(v as number, 2) }),
    col<Position>('current_price', { title: '现价', width: 100, sorter: true, render: (v) => (v == null ? '—' : fmtNum(v as number, 2)) }),
    col<Position>('market_value', { title: '市值', width: 120, sorter: true, render: (v) => fmtNum(v as number, 2) }),
    { title: '浮动盈亏', dataIndex: 'pnl', width: 170,
      sorter: (a: Position, b: Position) => pnl(a) - pnl(b),
      render: (_v: unknown, r: Position) => (
        <span style={{ color: pnlColor(pnl(r)) }}>
          {fmtNum(pnl(r), 2)}（{fmtPct(pnlPct(r))}）
        </span>
      ) },
    col<Position>('entry_date', { title: '建仓日', width: 110, sorter: true }),
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
        <>
          <Card size="small" title="盈亏贡献" style={{ marginTop: 16 }}>
            <PnlBarChart items={positions.map(p => ({ symbol: p.symbol, total_pnl: pnl(p) }))} nameOf={nameOf} />
          </Card>
          <Table rowKey="symbol" dataSource={positions} columns={columns} size="small"
            style={{ marginTop: 16 }}
            pagination={{ pageSize: 10, showSizeChanger: true, showTotal: (t) => `共 ${t} 条` }}
            onRow={(r) => ({ onClick: () => navigate(`/stocks?symbol=${r.symbol}`), style: { cursor: 'pointer' } })}
          />
        </>
      ) : (
        <Empty description="暂无持仓（模拟盘 8/3 开跑后产生）" style={{ marginTop: 24 }} />
      )}
    </div>
  )
}
