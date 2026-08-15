import { Card, Col, Row, Tag, Spin, Alert, Empty } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { useMemo } from 'react'
import { fetchGraduation, fetchEquity, fetchPaperStockPnl, fetchUniverse } from '../api'
import type { GraduationMetric } from '../api'
import { fmtNum, fmtPct } from '../lib/format'
import { statusMap } from '../lib/labels'
import StatCard from '../components/StatCard'
import EquityTriptych from '../components/EquityTriptych'
import MonthlyHeatmap from '../components/MonthlyHeatmap'
import PnlBarChart from '../components/PnlBarChart'

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
  const pnl = useQuery({ queryKey: ['paper-stock-pnl'], queryFn: fetchPaperStockPnl, refetchInterval: 60_000 })
  const uni = useQuery({ queryKey: ['universe'], queryFn: fetchUniverse, staleTime: 300_000 })

  const nameOf = useMemo(() => {
    const m = new Map<string, string>()
    for (const s of uni.data?.stocks ?? []) m.set(s.symbol, s.name)
    return (sym: string) => m.get(sym) ?? sym
  }, [uni.data])

  if (g.isLoading || eq.isLoading) return <Spin size="large" style={{ display: 'block', margin: '80px auto' }} />
  if (g.isError || eq.isError) return <Alert type="error" showIcon message="后端不可用" description="请确认 web/api 已启动 (uvicorn :8000)" />

  const metrics = g.data?.metrics ?? []
  const curve = eq.data?.curve ?? []
  const summary = eq.data?.summary
  const dailyReturns = curve
    .filter(p => p.daily_return !== null && p.daily_return !== undefined)
    .map(p => ({ date: p.date, ret: p.daily_return as number }))

  return (
    <div>
      <h2>模拟盘实盘总览</h2>
      <Row gutter={12}>
        <Col span={6}><StatCard title="累计收益" value={fmtPct(summary?.total_return)} /></Col>
        <Col span={6}><StatCard title="最大回撤" value={fmtPct(summary?.max_drawdown)} color="#cf1322" /></Col>
        <Col span={6}><StatCard title="年化波动" value={fmtPct(summary?.volatility)} /></Col>
        <Col span={6}><StatCard title="夏普比率" value={summary?.sharpe != null ? fmtNum(summary.sharpe, 2) : '—'} /></Col>
      </Row>

      {metrics.length ? (
        <Row gutter={[12, 12]} style={{ marginTop: 12 }}>
          {metrics.map((m) => (
            <Col key={m.key} xs={12} md={6}>
              <Card size="small" title={m.name}>
                <div style={{ fontSize: 20, fontWeight: 600 }}>
                  {valueText(m)}
                  <Tag color={statusMap[m.status]?.color ?? 'default'} style={{ marginLeft: 8 }}>
                    {statusMap[m.status]?.text ?? m.status}
                  </Tag>
                </div>
                <div style={{ color: '#888', fontSize: 12 }}>{m.detail}</div>
                {!m.detail.startsWith('阈值') && <div style={{ color: '#888', fontSize: 12 }}>阈值 {thresholdText(m)}</div>}
              </Card>
            </Col>
          ))}
        </Row>
      ) : (
        <Empty description="毕业指标待计算（模拟盘 8/3 开跑后产生）" style={{ marginTop: 16 }} />
      )}

      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col span={16}>
          <Card title="模拟盘净值与回撤">
            {curve.length === 0 ? <Empty description="模拟盘 8/3 开跑后每日累积权益数据" />
              : <EquityTriptych equity={curve.map(p => ({ date: p.date, equity: p.total_equity }))} />}
          </Card>
        </Col>
        <Col span={8}>
          <Card title="月度收益热力图">
            {dailyReturns.length === 0 ? <Empty description="暂无日收益数据" />
              : <MonthlyHeatmap dailyReturns={dailyReturns} />}
          </Card>
        </Col>
      </Row>

      <Card title="模拟盘个股盈亏贡献（已实现+浮动）" style={{ marginTop: 16 }}>
        {(pnl.data?.items?.length ?? 0) === 0 ? <Empty description="暂无已实现盈亏" />
          : <PnlBarChart items={(pnl.data?.items ?? []).map(i => ({ symbol: i.symbol, total_pnl: i.total_pnl + (i.unrealized_pnl ?? 0) }))} nameOf={nameOf} />}
      </Card>
    </div>
  )
}
