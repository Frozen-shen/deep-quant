import { useEffect, useMemo, useState } from 'react'
import { Card, AutoComplete, Spin, Alert, Typography, Empty, Tag, Space } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import ReactECharts from 'echarts-for-react'
import { searchUniverse, fetchStock, fetchUniverse } from '../api'
import { useExperiment } from '../experiment-context'

interface Ohlc { date: string; open: number; high: number; low: number; close: number; volume?: number }

/** 简单移动平均（前 n-1 位为 null） */
function ma(data: Ohlc[], n: number): (number | null)[] {
  return data.map((_, i) => {
    if (i < n - 1) return null
    let s = 0
    for (let j = i - n + 1; j <= i; j++) s += data[j].close
    return Number((s / n).toFixed(2))
  })
}

export default function Stocks() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [input, setInput] = useState('')
  const [q, setQ] = useState('')
  const [symbol, setSymbol] = useState<string | null>(searchParams.get('symbol'))

  // 400ms 防抖
  useEffect(() => {
    const t = setTimeout(() => setQ(input.trim()), 400)
    return () => clearTimeout(t)
  }, [input])

  const search = useQuery({
    queryKey: ['search', q], queryFn: () => searchUniverse(q),
    enabled: q.length >= 1,
  })
  const universe = useQuery({
    queryKey: ['universe'], queryFn: fetchUniverse, staleTime: 300_000,
  })
  const detail = useQuery({
    queryKey: ['stock', symbol], queryFn: () => fetchStock(symbol as string),
    enabled: !!symbol,
  })
  /** 当前选中实验的该股买卖点 (全局 Context) */
  const { detail: expDetail } = useExperiment()

  const tradesOfStock = useMemo(
    () => (expDetail?.trades ?? []).filter((t: any) => t.symbol === symbol),
    [expDetail, symbol],
  )

  // 从 URL 参数进入（信号/持仓联动）时，行情加载后回填输入框
  useEffect(() => {
    if (symbol && !input && detail.data?.name) setInput(`${symbol} ${detail.data.name}`)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol, detail.data])

  const options = useMemo(() =>
    (search.data?.stocks ?? []).map((s) => ({
      value: s.symbol, label: `${s.symbol} ${s.name}${s.sector ? ` · ${s.sector}` : ''}`,
    })), [search.data])

  const ohlc = (detail.data?.ohlc ?? []) as Ohlc[]
  /** 实验买卖标记 (backtrader 风格: 买入红▲ / 卖出绿▼) */
  const buyMarks = tradesOfStock.filter((t: any) => t.action === 'BUY').map((t: any) => ({
    name: `买入 ${t.date}`, coord: [t.date, t.price], itemStyle: { color: '#cf1322' },
  }))
  const sellMarks = tradesOfStock.filter((t: any) => t.action === 'SELL').map((t: any) => ({
    name: `卖出 ${t.date}`, coord: [t.date, t.price], itemStyle: { color: '#3f8600' }, symbolRotate: 180,
  }))
  const candleOption = {
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    legend: { data: ['K线', 'MA5', 'MA10', 'MA20', '成交量'] },
    grid: [
      { left: 60, right: 20, top: 30, height: '55%' },
      { left: 60, right: 20, top: '72%', height: '18%' },
    ],
    xAxis: [
      { type: 'category', data: ohlc.map((o) => o.date) },
      { type: 'category', gridIndex: 1, data: ohlc.map((o) => o.date), axisLabel: { show: false } },
    ],
    yAxis: [
      { scale: true },
      { gridIndex: 1, scale: true, axisLabel: { show: false } },
    ],
    dataZoom: [{ type: 'inside' }],
    series: [
      { name: 'K线', type: 'candlestick', data: ohlc.map((o) => [o.open, o.close, o.low, o.high]),
        markPoint: (buyMarks.length || sellMarks.length) ? {
          symbol: 'triangle', symbolSize: 11,
          data: [...buyMarks, ...sellMarks],
        } : undefined },
      { name: 'MA5', type: 'line', data: ma(ohlc, 5), showSymbol: false, smooth: true },
      { name: 'MA10', type: 'line', data: ma(ohlc, 10), showSymbol: false, smooth: true },
      { name: 'MA20', type: 'line', data: ma(ohlc, 20), showSymbol: false, smooth: true },
      { name: '成交量', type: 'bar', xAxisIndex: 1, yAxisIndex: 1,
        data: ohlc.map((o) => ({
          value: o.volume ?? 0,
          itemStyle: { color: o.close >= o.open ? '#cf1322' : '#3f8600' },
        })) },
    ],
  }

  return (
    <div>
      <Typography.Title level={4}>个股行情</Typography.Title>
      <AutoComplete
        style={{ width: 320 }} options={options} value={input}
        onSearch={(v) => setInput(v)}
        onSelect={(v, o) => {
          setSymbol(v)
          setInput(String(o.label))
          setSearchParams({ symbol: v })
        }}
        placeholder="输入代码或名称（如 600519 / 茅台）"
      />
      {search.isError && <Alert type="error" message="股票搜索服务不可用" style={{ marginTop: 16 }} />}
      {detail.isLoading && <Spin style={{ marginTop: 24, display: 'block' }} />}
      {detail.isError && <Alert type="error" message="股票不存在或数据缺失" style={{ marginTop: 16 }} />}
      {detail.data && (ohlc.length ? (
        <Card title={(
          <Space>
            <span>{detail.data.symbol} {detail.data.name}</span>
            {(() => {
              const hit = (search.data?.stocks ?? []).find((s) => s.symbol === detail.data.symbol)
                ?? (universe.data?.stocks ?? []).find((s) => s.symbol === detail.data.symbol)
              return hit?.sector ? <Tag color="geekblue">{hit.sector}</Tag> : null
            })()}
          </Space>
        )} style={{ marginTop: 16 }}
        extra={tradesOfStock.length > 0 ? (
          <Tag color="blue">{expDetail?.meta?.id ?? '实验'}: 买 {buyMarks.length} / 卖 {sellMarks.length}</Tag>
        ) : undefined}>
          <ReactECharts option={candleOption} style={{ height: 480 }} />
        </Card>
      ) : (
        <Empty description="该股票暂无行情数据" style={{ marginTop: 24 }} />
      ))}
    </div>
  )
}
