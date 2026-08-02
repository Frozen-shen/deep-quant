import { Card, Col, Row, Table, Typography, Spin, Alert, Empty } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'

export default function Trading() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['broker'], queryFn: async () => (await api.get('/broker/status')).data,
    refetchInterval: 30_000,
  })
  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="执行状态不可用" />
  const posCols = ['symbol', 'qty', 'avg_cost', 'market_value'].map((k) => ({ title: k, dataIndex: k }))
  const tradeCols = ['date', 'symbol', 'action', 'qty', 'price', 'commission', 'reason'].map((k) => ({ title: k, dataIndex: k }))
  return (
    <div>
      <Typography.Title level={4}>交易监控（{data.adapter} · {data.connected ? '已连接' : '未连接'}）</Typography.Title>
      <Row gutter={12}>
        <Col span={8}><Card size="small" title="账户">现金 {data.balance?.cash?.toLocaleString()}</Card></Col>
        <Col span={8}><Card size="small" title="持仓数">{data.positions?.length}</Card></Col>
        <Col span={8}><Card size="small" title="今日成交">{data.trades?.length}</Card></Col>
      </Row>
      {data.positions?.length ? (
        <Table rowKey="symbol" dataSource={data.positions} columns={posCols} size="small" style={{ marginTop: 16 }}
          pagination={{ pageSize: 10 }} />
      ) : (
        <Empty description="暂无持仓" style={{ marginTop: 24 }} />
      )}
      <Typography.Title level={5} style={{ marginTop: 24 }}>最近成交</Typography.Title>
      {data.trades?.length ? (
        <Table rowKey={(_r, i) => `${i}`} dataSource={data.trades} columns={tradeCols} size="small" />
      ) : (
        <Empty description="暂无成交记录" />
      )}
    </div>
  )
}
