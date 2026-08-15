import { useMemo, useState } from 'react'
import { Col, Row, Table, Typography, Spin, Alert, Empty, Tag, Space, Card, Segmented, Tooltip, Tabs } from 'antd'
import { useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import { fetchBroker, fetchUniverse, fetchPaperStockPnl } from '../api'
import { col } from '../lib/columns'
import { fmtNum, fmtPct, fmtDateTime } from '../lib/format'
import { actionTag } from '../lib/labels'
import StatCard from '../components/StatCard'
import EquityTriptych from '../components/EquityTriptych'
import PnlBarChart from '../components/PnlBarChart'
import { useExperiment } from '../experiment-context'

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
  const paperPnl = useQuery({ queryKey: ['paper-stock-pnl'], queryFn: fetchPaperStockPnl, refetchInterval: 60_000 })
  /** 当前选中实验 (全局 Context, URL ?exp= 同步) */
  const { expId, detail, detailLoading } = useExperiment()

  /** symbol → 股票名称 映射 */
  const nameOf = useMemo(() => {
    const m = new Map<string, string>()
    for (const s of universe.data?.stocks ?? []) m.set(s.symbol, s.name)
    return (sym: string) => m.get(sym) ?? sym
  }, [universe.data])

  const btTrades = (detail?.trades ?? []) as Trade[]
  const curve = detail?.equity_curve ?? []
  const bench = detail?.benchmark_curve ?? []
  const metrics = detail?.metrics ?? []

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

  const metricOf = (key: string) => metrics.find(m => m.key === key)?.value

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
      render: (_v: unknown, r: Trade) => nameOf(r.symbol) },
    col<Trade>('action', { title: '方向', width: 90,
      render: (v) => { const t = actionTag(v); return <Tag color={t.color}>{t.text}</Tag> } }),
    col<Trade>('qty', { title: '数量', width: 100, render: (v) => fmtNum(v as number, 0) }),
    col<Trade>('price', { title: '价格', width: 110, render: (v) => fmtNum(v as number, 2) }),
    col<Trade>('commission', { title: '佣金', width: 100, render: (v) => fmtNum(v as number, 2) }),
    { title: '原因', dataIndex: 'reason', ellipsis: true,
      render: (_v: unknown, r: Trade) => r.reason ?? (r.action === 'BUY' ? '建仓/加仓' : '调出/减仓') },
  ]

  /** Tab 1: 回测实验成交 (跟随全局实验选择器) */
  const backtestTab = (
    <div>
      {detailLoading ? <Spin /> : !detail ? (
        <Empty description="未选择实验（请在顶部选择器选择）" />
      ) : (
        <>
          <Row gutter={[12, 12]}>
            <Col span={6}><StatCard title="超额年化" value={metricOf('excess_annual') != null ? `${(metricOf('excess_annual') as number).toFixed(1)}%` : '—'} /></Col>
            <Col span={6}><StatCard title="Sharpe" value={metricOf('sharpe') ?? '—'} /></Col>
            <Col span={6}><StatCard title="最大回撤" value={metricOf('max_drawdown') != null ? `${(metricOf('max_drawdown') as number).toFixed(1)}%` : '—'} /></Col>
            <Col span={6}><StatCard title="成交笔数" value={btTrades.length} /></Col>
          </Row>
          <Row gutter={16} style={{ marginTop: 16 }}>
            <Col span={14}>
              <Typography.Text strong>净值与回撤（vs 中证1000）</Typography.Text>
              <EquityTriptych equity={curve} benchmark={bench} />
            </Col>
            <Col span={10}>
              <Typography.Text strong>调仓分布（买卖笔数/次）</Typography.Text>
              <ReactECharts option={rebalanceOption} style={{ height: 280 }} notMerge />
            </Col>
          </Row>
        </>
      )}

      <Card size="small" title="换仓明细" style={{ marginTop: 16 }}>
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
    </div>
  )

  /** Tab 2: 模拟盘实盘 */
  const paperTab = (
    <div>
      <Row gutter={12} style={{ marginBottom: 12 }}>
        <Col span={8}><StatCard title="现金" value={fmtNum(data?.balance?.cash, 0)} /></Col>
        <Col span={8}><StatCard title="持仓数" value={positions.length} /></Col>
        <Col span={8}><StatCard title="最近成交" value={(data?.trades ?? []).length} /></Col>
      </Row>
      <Card size="small" title="模拟盘个股盈亏（已实现+浮动）" style={{ marginBottom: 16 }}>
        <PnlBarChart items={(paperPnl.data?.items ?? []).map(i => ({ symbol: i.symbol, total_pnl: i.total_pnl + (i.unrealized_pnl ?? 0) }))} nameOf={nameOf} />
      </Card>
      <Card size="small" title="个股盈亏明细" style={{ marginBottom: 16 }}>
        <Table size="small" rowKey="symbol" dataSource={paperPnl.data?.items ?? []} pagination={{ pageSize: 15 }}
          columns={[
            { title: '代码', dataIndex: 'symbol', render: (v) => `${nameOf(v)} ${v}` },
            { title: '已实现盈亏', dataIndex: 'total_pnl', sorter: (a: any, b: any) => a.total_pnl - b.total_pnl,
              render: (v: number) => <span style={{ color: v > 0 ? '#cf1322' : v < 0 ? '#3f8600' : undefined }}>{fmtNum(v, 2)}</span> },
            { title: '浮动盈亏', dataIndex: 'unrealized_pnl', render: (v) => v == null ? '—' : <span style={{ color: v > 0 ? '#cf1322' : v < 0 ? '#3f8600' : undefined }}>{fmtNum(v, 2)}</span> },
            { title: '回合数', dataIndex: 'n_round_trips' },
            { title: '胜率', dataIndex: 'win_rate', render: (v) => v == null ? '—' : fmtPct(v) },
          ]} />
      </Card>
      <Card size="small" title="当前持仓">
        {positions.length ? (
          <Table rowKey="symbol" dataSource={positions} size="small"
            pagination={{ pageSize: 10, showTotal: (t) => `共 ${t} 条` }}
            columns={[
              col<Position>('symbol', { title: '代码', width: 90 }),
              { title: '名称', dataIndex: 'symbol', width: 120,
                render: (_v: unknown, r: Position) => nameOf(r.symbol) },
              col<Position>('qty', { title: '数量', width: 100, render: (v) => fmtNum(v as number, 0) }),
              col<Position>('avg_cost', { title: '成本价', width: 120, render: (v) => fmtNum(v as number, 2) }),
              col<Position>('market_value', { title: '市值', width: 140, render: (v) => fmtNum(v as number, 2) }),
              { title: '浮动盈亏', dataIndex: 'pnl', width: 160,
                render: (_v: unknown, r: Position) => (
                  <span style={{ color: pnl(r) >= 0 ? '#cf1322' : '#3f8600' }}>{fmtNum(pnl(r), 2)}</span>
                ) },
            ]} />
        ) : (
          <Empty description="暂无持仓" />
        )}
      </Card>
    </div>
  )

  return (
    <div>
      <Typography.Title level={4}>成交明细{expId ? `（实验: ${expId}）` : ''}</Typography.Title>
      {!connected && (
        <Alert type="warning" showIcon message="模拟盘执行器未连接"
          description="请检查 execution/paper_executor 是否运行，或后端 broker 适配器配置" style={{ marginBottom: 16 }} />
      )}
      <Tabs defaultActiveKey="backtest" items={[
        { key: 'backtest', label: '回测实验成交', children: backtestTab },
        { key: 'paper', label: '模拟盘实盘成交', children: paperTab },
      ]} />
    </div>
  )
}
