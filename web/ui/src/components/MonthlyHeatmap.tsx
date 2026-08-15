import ReactECharts from 'echarts-for-react'

interface Props { dailyReturns: Array<{ date: string; ret: number }> }

/** 月度收益热力图: 年(行) × 月(列), 红正绿负 (QuantStats 风格)。 */
export default function MonthlyHeatmap({ dailyReturns }: Props) {
  if (!dailyReturns.length) return null
  const byYm = new Map<string, number>()
  for (const d of dailyReturns) {
    const ym = d.date.slice(0, 7)
    byYm.set(ym, (byYm.get(ym) ?? 1) * (1 + d.ret) - 1)
  }
  const years = [...new Set([...byYm.keys()].map(k => k.slice(0, 4)))].sort()
  const months = Array.from({ length: 12 }, (_, i) => String(i + 1).padStart(2, '0'))
  const data: Array<[number, number, number]> = []
  years.forEach((y, yi) => {
    months.forEach((m, mi) => {
      const v = byYm.get(`${y}-${m}`)
      if (v !== undefined) data.push([mi, yi, Number((v * 100).toFixed(2))])
    })
  })
  const maxAbs = Math.max(1, ...data.map(d => Math.abs(d[2])))
  return <ReactECharts option={{
    tooltip: { formatter: (p: any) => `${years[p.value[1]]}-${months[p.value[0]]}: ${p.value[2]}%` },
    grid: { left: 50, right: 20, top: 10, bottom: 50 },
    xAxis: { type: 'category', data: months.map(m => `${Number(m)}月`) },
    yAxis: { type: 'category', data: years },
    visualMap: { min: -maxAbs, max: maxAbs, calculable: true, orient: 'horizontal',
      left: 'center', bottom: 0, inRange: { color: ['#3f8600', '#f0f0f0', '#cf1322'] } },
    series: [{ type: 'heatmap', data,
      label: { show: true, formatter: (p: any) => `${p.value[2] > 0 ? '+' : ''}${p.value[2]}%` } }],
  }} style={{ height: 220 }} />
}
