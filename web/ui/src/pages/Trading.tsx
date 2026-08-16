import { useMemo, useState } from 'react'
import { Col, Row, Table, Typography, Spin, Alert, Empty, Tag, Space, Card, Segmented, Tooltip, Tabs, AutoComplete } from 'antd'
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
interface Trade { date: string; symbol: string; action: string; qty: number; price: number; commission?: number; reason?: string; fill_times?: string[]; segment?: string }

export default function Trading() {
  /** 换仓明细按年份筛选 (动态: 从实验 trades 提取实际年份, 默认全部) */
  const [btYear, setBtYear] = useState<string>('all')
  /** 净值图阶段切换: fold_1..5 / extend_val (默认模拟考=最新) */
  const [segKey, setSegKey] = useState<string>('extend_val')
  /** 换仓明细按股票搜索筛选 (代码/名称, 空=全部) */
  const [btSymbol, setBtSymbol] = useState<string>('')
  /** 搜索框输入文本 (独立于筛选态, 支持自由输入) */
  const [btInput, setBtInput] = useState<string>('')
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
  const segments = detail?.segments ?? []

  /** 当前选中阶段的净值/基准曲线 (segments 由后端提供, 各 fold 独立) */
  const seg = segments.find(s => s.key === segKey) ?? segments[segments.length - 1]
  const segCurve = seg?.equity?.length ? seg.equity : curve
  const segBench = seg?.benchmark?.length ? seg.benchmark : bench

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

  /** 股票搜索框选项: 仅列出该实验实际交易过的股票 (避免无关干扰) */
  const tradeSymbols = useMemo(() => {
    const set = new Set<string>()
    for (const t of btTrades) set.add(t.symbol)
    return [...set].sort()
  }, [btTrades])
  const stockOptions = useMemo(() => tradeSymbols.map(s => ({
    value: s, label: `${s} ${nameOf(s)}`,
  })), [tradeSymbols, nameOf])

  /** 动态年份选项: 从实验 trades 提取实际覆盖年份 (2020-2026) */
  const yearOptions = useMemo(() => {
    const years = new Set<string>()
    for (const t of btTrades) years.add(t.date.slice(0, 4))
    return ['all', ...[...years].sort()]
  }, [btTrades])

  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="执行状态不可用" />

  const connected = !!data?.connected
  const positions = (data?.positions ?? []) as Position[]
  const pnl = (p: Position) => (p.market_value ?? 0) - (p.avg_cost ?? 0) * (p.qty ?? 0)

  /** 换仓明细筛选: 年份 (动态) + 股票 (代码/名称) 叠加 */
  const filteredTrades = btTrades.filter((t: Trade) => {
    if (btYear !== 'all' && !t.date.startsWith(btYear)) return false
    if (btSymbol && t.symbol !== btSymbol) return false
    return true
  })

  /** 阶段标签映射: fold_1..5 → 2020..2024 验证期, extend_val → 模拟考 */
  const segmentTag = (seg?: string) => {
    if (!seg) return null
    const map: Record<string, { text: string; color: string }> = {
      fold_1: { text: 'Fold1·2020', color: 'default' },
      fold_2: { text: 'Fold2·2021', color: 'default' },
      fold_3: { text: 'Fold3·2022', color: 'default' },
      fold_4: { text: 'Fold4·2023', color: 'default' },
      fold_5: { text: 'Fold5·2024', color: 'default' },
      extend_val: { text: '模拟考', color: 'blue' },
    }
    const m = map[seg]
    return m ? <Tag color={m.color} style={{ marginLeft: 4 }}>{m.text}</Tag> : null
  }

  const metricOf = (key: string) => metrics.find(m => m.key === key)?.value

  /** 时间列渲染: POV 多段拆单 / 随机时点市价(小订单) / 全天VWAP(旧数据) / 单时段 */
  const timeRender = (v: string, r: Trade) => {
    const ft = (r as any).fill_times as string[] | undefined
    const base = fmtDateTime(v as string)
    if (ft && ft.length > 1) {
      return <Tooltip title={`POV 拆单 ${ft.length} 段: ${ft.join(', ')}`}><span>{base} <Tag color="blue">{ft.length}段</Tag></span></Tooltip>
    }
    if (ft && ft.length === 1 && ft[0] === '全天VWAP') {
      // 旧数据 (v24e 及之前): 全天均价成交, 无单一时刻
      return <Tooltip title="旧实验假设: 按全天 VWAP 均价成交"><span>{base} <Tag color="orange">全天VWAP(旧)</Tag></span></Tooltip>
    }
    if (ft && ft.length === 1 && ft[0].startsWith('市价@')) {
      // 小订单 (<0.1% 日成交量): 随机时点市价单 — 模拟执行时间的不确定性
      const t = ft[0].slice(3)
      return <Tooltip title={`订单小于日成交量 0.1%, 模拟随机时点市价单成交 (${t}, 固定种子可复现)`}><span>{base} <Tag color="green">{t}</Tag></span></Tooltip>
    }
    if (ft && ft.length === 1) {
      return <Tooltip title={`成交时段 ${ft[0]}`}><span>{base} <Tag color="green">{ft[0]}</Tag></span></Tooltip>
    }
    return base
  }

  const tradeCols = [
    { title: '时间', dataIndex: 'date', width: 220, sorter: (a: Trade, b: Trade) => a.date.localeCompare(b.date),
      render: (v: string, r: Trade) => (
        <span>{timeRender(v, r)}{segmentTag(r.segment)}</span>
      ) },
    col<Trade>('symbol', { title: '代码', width: 90, sorter: true }),
    { title: '名称', dataIndex: 'symbol', width: 130, sorter: (a: Trade, b: Trade) => nameOf(a.symbol).localeCompare(nameOf(b.symbol), 'zh-CN'),
      render: (_v: unknown, r: Trade) => nameOf(r.symbol) },
    col<Trade>('action', { title: '方向', width: 90, sorter: true,
      render: (v) => { const t = actionTag(v); return <Tag color={t.color}>{t.text}</Tag> } }),
    col<Trade>('qty', { title: '数量', width: 100, sorter: true, render: (v) => fmtNum(v as number, 0) }),
    col<Trade>('price', { title: '价格', width: 110, sorter: true, render: (v) => fmtNum(v as number, 2) }),
    col<Trade>('commission', { title: '佣金', width: 100, sorter: true, render: (v) => fmtNum(v as number, 2) }),
    { title: '成交后净值', dataIndex: 'equity_after', width: 130, sorter: (a: any, b: any) => (a.equity_after ?? 0) - (b.equity_after ?? 0),
      render: (_v: unknown, r: any) => r.equity_after != null ? fmtNum(r.equity_after, 0) : '—' },
    { title: '原因', dataIndex: 'reason', ellipsis: true, sorter: (a: Trade, b: Trade) => String(a.reason ?? '').localeCompare(String(b.reason ?? '')),
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
              <Space style={{ marginBottom: 8 }} wrap>
                <Typography.Text strong>净值与回撤（vs 中证1000）</Typography.Text>
                {segments.length > 1 && (
                  <Segmented
                    size="small"
                    options={segments.map(s => ({ label: s.label, value: s.key }))}
                    value={segKey}
                    onChange={(v) => setSegKey(String(v))} />
                )}
              </Space>
              <EquityTriptych equity={segCurve} benchmark={segBench} />
            </Col>
            <Col span={10}>
              <Typography.Text strong>调仓分布（买卖笔数/次）</Typography.Text>
              <ReactECharts option={rebalanceOption} style={{ height: 280 }} notMerge />
            </Col>
          </Row>
        </>
      )}

      <Card size="small" title="换仓明细" style={{ marginTop: 16 }}>
        <Space style={{ marginBottom: 12 }} wrap>
          <Segmented
            options={[
              { label: '全部', value: 'all' },
              ...yearOptions.filter(y => y !== 'all').map(y => ({ label: y, value: y })),
            ]}
            value={btYear} onChange={(v) => setBtYear(String(v))} />
          <AutoComplete
            style={{ width: 260 }}
            placeholder="按股票代码/名称筛选（如 600519 或 茅台）"
            value={btInput}
            options={stockOptions}
            filterOption={(input, option) =>
              (option?.label ?? '').toLowerCase().includes(input.toLowerCase()) ||
              (option?.value ?? '').includes(input)
            }
            onSelect={(v) => {
              setBtSymbol(v)
              setBtInput(`${v} ${nameOf(v)}`)
            }}
            onChange={(v) => {
              setBtInput(v)
              const t = v.trim()
              if (!t) { setBtSymbol(''); return }
              // 精确 6 位代码 → 直接筛选
              if (/^\d{6}$/.test(t)) { setBtSymbol(t); return }
              // 名称模糊匹配: 唯一命中 → 直接筛选; 多选/无 → 仅联想
              const hit = tradeSymbols.filter(s => nameOf(s).includes(t))
              setBtSymbol(hit.length === 1 ? hit[0] : '')
            }}
            allowClear
          />
          {btSymbol && (
            <Tag closable color="blue" onClose={() => setBtSymbol('')}>
              {btSymbol} {nameOf(btSymbol)}
            </Tag>
          )}
          <Typography.Text type="secondary">
            筛选后 {filteredTrades.length} 笔（共 {btTrades.length} 笔；小订单=随机时点市价单，大订单=POV 拆单；悬停时间列看明细）
          </Typography.Text>
        </Space>
        {filteredTrades.length ? (
          <Table rowKey={(r) => `${r.date}-${r.symbol}-${r.action}-${r.price}-${r.qty}`} dataSource={filteredTrades} columns={tradeCols} size="small"
            pagination={{ pageSize: 20, showSizeChanger: true, showTotal: (t) => `共 ${t} 条` }} />
        ) : (
          <Empty description={btSymbol ? `该筛选条件下无换仓记录（${btSymbol} ${nameOf(btSymbol)}）` : '该年份无换仓记录'} />
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
