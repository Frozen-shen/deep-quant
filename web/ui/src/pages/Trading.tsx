import { useMemo, useState } from 'react'
import { Col, Row, Table, Typography, Spin, Alert, Empty, Tag, Space, Card, Segmented, Tooltip } from 'antd'
import { useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import { fetchBroker, fetchBacktestTrades, fetchUniverse } from '../api'
import { col } from '../lib/columns'
import { fmtNum, fmtDateTime } from '../lib/format'
import { actionTag } from '../lib/labels'
import StatCard from '../components/StatCard'

interface Position { symbol: string; qty: number; avg_cost: number; market_value: number }
interface Trade { date: string; symbol: string; action: string; qty: number; price: number; commission?: number; reason?: string; fill_times?: string[] }

export default function Trading() {
  /** 换仓明细按实验年份筛选: all / 2025 / 2026 (EXTEND 覆盖区间) */
  const [btYear, setBtYear] = useState<'all' | '2025' | '2026'>('all')
  const { data, isLoading, isError } = useQuery({
    queryKey: ['broker'], queryFn: fetchBroker, refetchInterval: 30_000,
  })
  const universe = useQuery({
    queryKey: ['universe'], queryFn: fetchUniverse, staleTime: 300_000,
  })
  /** v24b 最优实验 (EXTEND 模拟考): 指标 + 净值曲线 + 调仓记录 */
  const btQuery = useQuery({
    queryKey: ['backtest-trades'], queryFn: fetchBacktestTrades, staleTime: 300_000,
  })
  /** symbol → 股票名称 映射 */
  const nameOf = useMemo(() => {
    const m = new Map<string, string>()
    for (const s of universe.data?.stocks ?? []) m.set(s.symbol, s.name)
    return m
  }, [universe.data])

  const bt = btQuery.data
  const curve = bt?.equity_curve ?? []
  const btTrades = bt?.trades ?? []

  /** 净值曲线图: 净值(左轴) + 累计超额(右轴) */
  const equityOption = useMemo(() => {
    const dates = curve.map((p) => p.date)
    const equities = curve.map((p) => p.equity)
    // 累计收益: 从净值反推 (起始=0)
    const base = equities[0] ?? 100000
    const cumExcess = equities.map((e) => (e / base - 1) * 100)
    return {
      tooltip: { trigger: 'axis' },
      legend: { data: ['组合净值', '累计收益 %'] },
      grid: { left: 60, right: 60, top: 40, bottom: 40 },
      xAxis: { type: 'category', data: dates, boundaryGap: false },
      yAxis: [
        { type: 'value', name: '净值', scale: true },
        { type: 'value', name: '累计%', axisLabel: { formatter: '{value}%' } },
      ],
      series: [
        { name: '组合净值', type: 'line', data: equities, showSymbol: false, lineStyle: { width: 2 } },
        { name: '累计收益 %', type: 'line', yAxisIndex: 1, data: cumExcess, showSymbol: false,
          lineStyle: { width: 1.5, type: 'dashed' } },
      ],
    }
  }, [curve])

  /** 调仓时间线: 每次调仓的买卖笔数 */
  const rebalanceOption = useMemo(() => {
    const byDate = new Map<string, { buy: number; sell: number }>()
    for (const t of btTrades) {
      const e = byDate.get(t.date) ?? { buy: 0, sell: 0 }
      if (t.action === 'BUY') e.buy += 1
      else e.sell += 1
      byDate.set(t.date, e)
    }
    const dates = [...byDate.keys()].sort()
    return {
      tooltip: { trigger: 'axis' },
      legend: { data: ['买入', '卖出'] },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: 'category', data: dates },
      yAxis: { type: 'value', name: '笔数' },
      series: [
        { name: '买入', type: 'bar', stack: 't', data: dates.map((d) => byDate.get(d)!.buy),
          itemStyle: { color: '#cf1322' } },
        { name: '卖出', type: 'bar', stack: 't', data: dates.map((d) => byDate.get(d)!.sell),
          itemStyle: { color: '#3f8600' } },
      ],
    }
  }, [btTrades])

  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="执行状态不可用" />

  const connected = !!data?.connected
  const positions = (data?.positions ?? []) as Position[]
  const pnl = (p: Position) => (p.market_value ?? 0) - (p.avg_cost ?? 0) * (p.qty ?? 0)

  /** 换仓明细按实验年份筛选 (EXTEND 2025-01~2026-06) */
  const filteredTrades = btYear === 'all'
    ? btTrades
    : btTrades.filter((t: Trade) => t.date.startsWith(btYear))

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
    { title: '时间', dataIndex: 'date', width: 170,
      render: (v: string, r: Trade) => {
        const ft = (r as any).fill_times as string[] | undefined
        const base = fmtDateTime(v as string)
        if (ft && ft.length > 1) {
          // POV 多段拆单: 显示首末时段 + 段数
          return <Tooltip title={`POV 拆单 ${ft.length} 段: ${ft.join(', ')}`}><span>{base} <Tag color="blue">{ft.length}段</Tag></span></Tooltip>
        }
        if (ft && ft.length === 1) {
          // 单段成交 (小订单 VWAP): 显示成交时段
          return <Tooltip title={`成交时段 ${ft[0]}`}><span>{base} <Tag color="green">{ft[0]}</Tag></span></Tooltip>
        }
        return base
      } },
    col<Trade>('symbol', { title: '代码', width: 90 }),
    { title: '名称', dataIndex: 'symbol', width: 130,
      render: (_v: unknown, r: Trade) => nameOf.get(r.symbol) ?? '—' },
    col<Trade>('action', { title: '方向', width: 90,
      render: (v) => { const t = actionTag(v); return <Tag color={t.color}>{t.text}</Tag> } }),
    col<Trade>('qty', { title: '数量', width: 100, render: (v) => fmtNum(v as number, 0) }),
    col<Trade>('price', { title: '价格', width: 110, render: (v) => fmtNum(v as number, 2) }),
    col<Trade>('commission', { title: '佣金', width: 100, render: (v) => fmtNum(v as number, 2) }),
    { title: '原因', dataIndex: 'reason', ellipsis: true,
      render: (_v: unknown, r: Trade) => r.reason ?? (r.action === 'BUY' ? '建仓/加仓' : '调出/减仓') },
  ]

  return (
    <div>
      <Typography.Title level={4}>交易监控（{data?.adapter} · {connected ? '已连接' : '未连接'}）</Typography.Title>
      {!connected && (
        <Alert type="warning" showIcon message="模拟盘执行器未连接"
          description="请检查 execution/paper_executor 是否运行，或后端 broker 适配器配置" style={{ marginBottom: 16 }} />
      )}

      {/* ═══ v24b 最优实验 (主视图) ═══ */}
      <Card size="small" title="v24b 最优实验（EXTEND 模拟考 2025-01 ~ 2026-06）"
        extra={<Tag color="blue">{bt?.version ?? 'v24b'}</Tag>} style={{ marginBottom: 16 }}>
        {btQuery.isLoading ? <Spin /> : bt?.available === false ? (
          <Empty description='实验数据不可用' />
        ) : (
          <>
            <Row gutter={[12, 12]}>
              <Col span={6}><StatCard title="超额年化" value={bt?.excess_annual != null ? `${(bt.excess_annual as number).toFixed(1)}%` : '—'} /></Col>
              <Col span={6}><StatCard title="Sharpe" value={bt?.sharpe ?? '—'} /></Col>
              <Col span={6}><StatCard title="最大回撤" value={bt?.max_drawdown != null ? `${(bt.max_drawdown as number).toFixed(1)}%` : '—'} /></Col>
              <Col span={6}><StatCard title="换仓笔数" value={bt?.count ?? 0} /></Col>
            </Row>
            <Typography.Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 4 }}>
              {bt?.period ?? ''} · {bt?.n_rebalances ?? 0} 次调仓 · 回测数据（非实盘成交）
              {bt?.source === 'real'
                ? <Tag color="green" style={{ marginLeft: 8 }}>真实逐笔成交</Tag>
                : <Tag color="orange" style={{ marginLeft: 8 }}>近似重建（旧 JSON 无逐笔）</Tag>}
            </Typography.Paragraph>
            <Row gutter={16}>
              <Col span={14}>
                <Typography.Text strong>净值曲线</Typography.Text>
                <ReactECharts option={equityOption} style={{ height: 280 }} notMerge />
              </Col>
              <Col span={10}>
                <Typography.Text strong>调仓分布（买卖笔数/次）</Typography.Text>
                <ReactECharts option={rebalanceOption} style={{ height: 280 }} notMerge />
              </Col>
            </Row>
          </>
        )}
      </Card>

      {/* ═══ 换仓明细 (按实验年份筛选) ═══ */}
      <Card size="small" title="换仓明细"
        extra={<Space><Tag color="blue">{bt?.period ?? ''}</Tag><Tag>{bt?.n_rebalances ?? 0} 次调仓</Tag></Space>}
        style={{ marginBottom: 16 }}>
        <Space style={{ marginBottom: 12 }}>
          <Segmented
            options={[
              { label: '全部', value: 'all' },
              { label: '2025', value: '2025' },
              { label: '2026', value: '2026' },
            ]}
            value={btYear} onChange={setBtYear} />
          <Typography.Text type="secondary">
            筛选后 {filteredTrades.length} 笔（共 {btTrades.length} 笔，POV 执行·含佣金，悬停时间列看成交时段）
          </Typography.Text>
        </Space>
        {filteredTrades.length ? (
          <Table rowKey={(_r, i) => `${i}`} dataSource={filteredTrades} columns={tradeCols} size="small"
            pagination={{ pageSize: 20, showSizeChanger: true, showTotal: (t) => `共 ${t} 条` }} />
        ) : (
          <Empty description="该年份无换仓记录" />
        )}
      </Card>

      {/* ═══ 模拟盘实盘 ═══ */}
      <Card size="small" title="模拟盘实盘" style={{ marginBottom: 16 }}>
        <Row gutter={12} style={{ marginBottom: 12 }}>
          <Col span={8}><StatCard title="现金" value={fmtNum(data?.balance?.cash, 0)} /></Col>
          <Col span={8}><StatCard title="持仓数" value={positions.length} /></Col>
          <Col span={8}><StatCard title="最近成交" value={(data?.trades ?? []).length} /></Col>
        </Row>
        {positions.length ? (
          <Table rowKey="symbol" dataSource={positions} columns={posCols} size="small"
            pagination={{ pageSize: 10, showTotal: (t) => `共 ${t} 条` }} />
        ) : (
          <Empty description="暂无持仓" />
        )}
      </Card>
    </div>
  )
}
