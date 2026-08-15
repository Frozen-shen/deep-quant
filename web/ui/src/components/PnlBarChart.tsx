import ReactECharts from 'echarts-for-react'

interface Props { items: Array<{ symbol: string; total_pnl: number }>; nameOf: (s: string) => string }

/** 盈亏贡献横向条形图: 按盈亏额排序, 红正绿负。 */
export default function PnlBarChart({ items, nameOf }: Props) {
  if (!items.length) return null
  const sorted = [...items].sort((a, b) => a.total_pnl - b.total_pnl)
  const labels = sorted.map(i => `${nameOf(i.symbol)} ${i.symbol}`)
  const vals = sorted.map(i => Number(i.total_pnl.toFixed(0)))
  return <ReactECharts option={{
    tooltip: { trigger: 'axis' },
    grid: { left: 110, right: 30, top: 10, bottom: 30 },
    xAxis: { type: 'value' },
    yAxis: { type: 'category', data: labels },
    series: [{ type: 'bar', data: vals,
      itemStyle: { color: (p: any) => (p.value >= 0 ? '#cf1322' : '#3f8600') } }],
  }} style={{ height: Math.max(200, labels.length * 26) }} />
}
