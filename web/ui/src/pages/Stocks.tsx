import { useMemo, useState } from 'react'
import { Card, AutoComplete, Spin, Alert, Typography, Empty } from 'antd'
import { useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import { api } from '../api'

export default function Stocks() {
  const [q, setQ] = useState('')
  const [symbol, setSymbol] = useState<string | null>(null)
  const search = useQuery({
    queryKey: ['search', q], queryFn: async () => (await api.get('/universe/search', { params: { q } })).data,
    enabled: q.length >= 1,
  })
  const detail = useQuery({
    queryKey: ['stock', symbol], queryFn: async () => (await api.get(`/stocks/${symbol}`)).data,
    enabled: !!symbol,
  })
  const options = useMemo(() =>
    (search.data?.stocks ?? []).map((s: any) => ({
      value: s.symbol, label: `${s.symbol} ${s.name}`,
    })), [search.data])

  const ohlc = detail.data?.ohlc ?? []
  const candleOption = {
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: ohlc.map((o: any) => o.date) },
    yAxis: { scale: true },
    dataZoom: [{ type: 'inside' }],
    series: [{
      type: 'candlestick',
      data: ohlc.map((o: any) => [o.open, o.close, o.low, o.high]),
    }],
  }
  return (
    <div>
      <Typography.Title level={4}>个股行情</Typography.Title>
      <AutoComplete
        style={{ width: 320 }} options={options} onSearch={setQ}
        onSelect={(v) => setSymbol(v)} placeholder="输入代码或名称（如 600519 / 茅台）"
      />
      {search.isError && <Alert type="error" message="股票搜索服务不可用" style={{ marginTop: 16 }} />}
      {detail.isLoading && <Spin style={{ marginTop: 24, display: 'block' }} />}
      {detail.isError && <Alert type="error" message="股票不存在或数据缺失" style={{ marginTop: 16 }} />}
      {detail.data && (ohlc.length ? (
        <Card title={`${detail.data.symbol} ${detail.data.name}`} style={{ marginTop: 16 }}>
          <ReactECharts option={candleOption} style={{ height: 420 }} />
        </Card>
      ) : (
        <Empty description="该股票暂无行情数据" style={{ marginTop: 24 }} />
      ))}
    </div>
  )
}
