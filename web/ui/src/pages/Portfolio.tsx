import { Card, Spin, Alert, Table, Typography } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'

export default function Portfolio() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['portfolio'], queryFn: async () => (await api.get('/portfolio')).data,
  })
  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="加载失败" />
  const cols = [
    { title: '代码', dataIndex: 'symbol' }, { title: '数量', dataIndex: 'qty' },
    { title: '成本', dataIndex: 'avg_cost' }, { title: '市值', dataIndex: 'market_value' },
    { title: '建仓日', dataIndex: 'entry_date' },
  ]
  return (
    <div>
      <Typography.Title level={4}>模拟盘组合</Typography.Title>
      <Card>现金 {data.cash?.toLocaleString()} / 初始 {data.initial_capital?.toLocaleString()} / 起始 {data.inception_date}</Card>
      <Table rowKey="symbol" dataSource={data.positions ?? []} columns={cols} pagination={false} style={{ marginTop: 16 }} />
    </div>
  )
}
