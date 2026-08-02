import { Card, Col, Row, Tag, Spin, Alert, Empty } from 'antd'
import { useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import { fetchGraduation, fetchEquity } from '../api'

const statusColor: Record<string, string> = { pass: 'green', fail: 'red', pending: 'orange' }
const statusText: Record<string, string> = { pass: '达标', fail: '未达标', pending: '待数据' }

export default function Overview() {
  const g = useQuery({ queryKey: ['graduation'], queryFn: fetchGraduation, refetchInterval: 60_000 })
  const eq = useQuery({ queryKey: ['equity'], queryFn: fetchEquity, refetchInterval: 60_000 })

  if (g.isLoading || eq.isLoading) return <Spin size="large" style={{ display: 'block', margin: '80px auto' }} />
  if (g.isError || eq.isError) return <Alert type="error" showIcon message="后端不可用" description="请确认 web/api 已启动 (uvicorn :8000)" />

  const metrics = g.data?.metrics ?? []
  const curve = eq.data?.curve ?? []

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
                {m.value ?? '—'}
                <Tag color={statusColor[m.status]} style={{ marginLeft: 8 }}>{statusText[m.status]}</Tag>
              </div>
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
    </div>
  )
}
