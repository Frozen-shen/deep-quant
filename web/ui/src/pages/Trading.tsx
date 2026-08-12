import { useMemo, useState } from 'react'
import { Card, Col, Row, Table, Typography, Spin, Alert, Empty, Tag, Select, Space } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { fetchBroker, fetchBrokerTrades, fetchBacktestTrades, fetchUniverse } from '../api'
import { col } from '../lib/columns'
import { fmtNum, fmtDateTime } from '../lib/format'
import { actionTag } from '../lib/labels'
import StatCard from '../components/StatCard'

interface Position { symbol: string; qty: number; avg_cost: number; market_value: number }
interface Trade { date: string; symbol: string; action: string; qty: number; price: number; commission?: number; reason?: string }

export default function Trading() {
  const [year, setYear] = useState<number>()
  const { data, isLoading, isError } = useQuery({
    queryKey: ['broker'], queryFn: fetchBroker, refetchInterval: 30_000,
  })
  const universe = useQuery({
    queryKey: ['universe'], queryFn: fetchUniverse, staleTime: 300_000,
  })
  const tradesQuery = useQuery({
    queryKey: ['broker-trades', year], queryFn: () => fetchBrokerTrades(year),
    enabled: !!year,
  })
  /** v24b 最优实验 (EXTEND 模拟考) 调仓记录 */
  const btQuery = useQuery({
    queryKey: ['backtest-trades'], queryFn: fetchBacktestTrades, staleTime: 300_000,
  })
  /** symbol → 股票名称 映射 */
  const nameOf = useMemo(() => {
    const m = new Map<string, string>()
    for (const s of universe.data?.stocks ?? []) m.set(s.symbol, s.name)
    return m
  }, [universe.data])
  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="执行状态不可用" />

  const connected = !!data?.connected
  const positions = (data?.positions ?? []) as Position[]
  /** 选中年份时用该年全部成交（回测 2021-2024 / 模拟盘 2026），否则用 status 最近 50 条 */
  const trades = (year ? tradesQuery.data?.trades ?? [] : (data?.trades ?? [])) as Trade[]
  const pnl = (p: Position) => (p.market_value ?? 0) - (p.avg_cost ?? 0) * (p.qty ?? 0)

  const posCols = [
    col<Position>('symbol', { title: '代码', width: 90 }),
    { title: '名称', dataIndex: 'symbol', width: 120,
      render: (_v: unknown, r: Position) => nameOf.get(r.symbol) ?? '—' },
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
    col<Trade>('symbol', { title: '代码', width: 90 }),
    { title: '名称', dataIndex: 'symbol', width: 130,
      render: (_v: unknown, r: Trade) => nameOf.get(r.symbol) ?? '—' },
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

      {/* ★ v24b 最优实验 (主视图): 指标 + 换仓记录 */}
      <Typography.Title level={5} style={{ marginTop: 16 }}>v24b 最优实验（EXTEND 模拟考 2025-01~2026-06）</Typography.Title>
      {btQuery.isLoading ? <Spin /> : btQuery.data?.available === false ? (
        <Empty description={btQuery.data?.note ?? '实验数据不可用'} />
      ) : (
        <>
          <Row gutter={12} style={{ marginBottom: 12 }}>
            <Col span={6}><StatCard title="超额年化" value={btQuery.data?.excess_annual != null ? `${(btQuery.data.excess_annual as number).toFixed(1)}%` : '—'} /></Col>
            <Col span={6}><StatCard title="Sharpe" value={btQuery.data?.sharpe ?? '—'} /></Col>
            <Col span={6}><StatCard title="最大回撤" value={btQuery.data?.max_drawdown != null ? `${(btQuery.data.max_drawdown as number).toFixed(1)}%` : '—'} /></Col>
            <Col span={6}><StatCard title="换仓笔数" value={btQuery.data?.count ?? 0} /></Col>
          </Row>
          <Alert type="info" showIcon style={{ marginBottom: 12 }}
            message={`v24b（VWAP执行+10bps残差）· ${btQuery.data?.period ?? ''} · ${btQuery.data?.n_rebalances ?? 0} 次调仓`}
            description="数据还原自回测 JSON（walkforward_results_v24b_vwap.json），非实盘成交" />
          <Table rowKey={(_r, i) => `${i}`} dataSource={btQuery.data?.trades ?? []} columns={tradeCols} size="small"
            pagination={{ pageSize: 20, showSizeChanger: true, showTotal: (t) => `共 ${t} 条` }} />
        </>
      )}

      {/* 模拟盘实盘: 持仓 + 最近成交 */}
      <Typography.Title level={5} style={{ marginTop: 32 }}>模拟盘实盘</Typography.Title>
      <Row gutter={12}>
        <Col span={8}><StatCard title="现金" value={fmtNum(data?.balance?.cash, 0)} /></Col>
        <Col span={8}><StatCard title="持仓数" value={positions.length} /></Col>
        <Col span={8}><StatCard title="最近成交" value={(data?.trades ?? []).length} /></Col>
      </Row>
      {positions.length ? (
        <Table rowKey="symbol" dataSource={positions} columns={posCols} size="small" style={{ marginTop: 16 }}
          pagination={{ pageSize: 10, showTotal: (t) => `共 ${t} 条` }} />
      ) : (
        <Empty description="暂无持仓" style={{ marginTop: 24 }} />
      )}
      <Typography.Title level={5} style={{ marginTop: 24 }}>最近成交</Typography.Title>
      <Space style={{ marginBottom: 12 }}>
        <Select allowClear placeholder="按年份筛选" style={{ width: 160 }} value={year} onChange={setYear}
          options={[
            { value: 2021, label: '2021（回测）' },
            { value: 2022, label: '2022（回测）' },
            { value: 2023, label: '2023（回测）' },
            { value: 2024, label: '2024（回测）' },
            { value: 2025, label: '2025（TEST① 无交易）' },
            { value: 2026, label: '2026（模拟盘）' },
          ]} />
        {year && <Typography.Text type="secondary">该年成交 {tradesQuery.data?.count ?? 0} 条</Typography.Text>}
        {year === 2025 && <Tag color="orange">TEST① 分区（2025-01~2026-06）纪律禁止交易，无成交记录</Tag>}
      </Space>
      {trades.length ? (
        <Table rowKey={(_r, i) => `${i}`} dataSource={trades} columns={tradeCols} size="small"
          pagination={{ pageSize: 10, showSizeChanger: true, showTotal: (t) => `共 ${t} 条` }} />
      ) : (
        <Empty description="暂无成交记录" />
      )}
    </div>
  )
}
