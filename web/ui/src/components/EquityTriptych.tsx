import ReactECharts from 'echarts-for-react'

interface Props { equity: Array<{ date: string; equity: number }>; benchmark?: Array<{ date: string; close: number }> }

/** 净值三件套: 组合净值(左轴) + 基准(左轴虚线) + 回撤阴影(右轴)。 */
export default function EquityTriptych({ equity, benchmark }: Props) {
  if (!equity.length) return null
  const dates = equity.map(p => p.date)
  const eq = equity.map(p => p.equity)
  // 回撤序列 (水下)
  let peak = -Infinity
  const dd = equity.map(p => {
    peak = Math.max(peak, p.equity)
    return Number(((p.equity / peak - 1) * 100).toFixed(2))
  })
  const series: any[] = [
    { name: '组合净值', type: 'line', data: eq, showSymbol: false, lineStyle: { width: 2 } },
    { name: '回撤 %', type: 'line', yAxisIndex: 1, data: dd, showSymbol: false,
      lineStyle: { width: 1, color: '#cf1322' },
      areaStyle: { color: 'rgba(207,19,34,0.12)' } },
  ]
  if (benchmark?.length) {
    const bMap = new Map(benchmark.map(p => [p.date, p.close]))
    series.push({
      name: '基准(中证1000)', type: 'line', data: dates.map(d => bMap.get(d) ?? null),
      showSymbol: false, lineStyle: { width: 1.5, type: 'dashed', color: '#888' },
    })
  }
  return <ReactECharts option={{
    tooltip: { trigger: 'axis' },
    legend: { data: series.map(s => s.name) },
    grid: { left: 60, right: 60, top: 40, bottom: 30 },
    xAxis: { type: 'category', data: dates, boundaryGap: false },
    yAxis: [
      { type: 'value', name: '净值', scale: true },
      { type: 'value', name: '回撤%', max: 0, axisLabel: { formatter: '{value}%' } },
    ],
    series,
  }} style={{ height: 360 }} />
}
