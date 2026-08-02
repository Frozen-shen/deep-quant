import { Card, Table, Spin, Alert, Typography } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'

export default function Signals() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['signals'], queryFn: async () => (await api.get('/signals')).data,
  })
  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="加载失败" />
  const sigs = data?.signals ?? []
  const cols = sigs.length
    ? Object.keys(sigs[0]).filter((k) => k !== 'metadata').map((k) => ({ title: k, dataIndex: k }))
    : []
  return (
    <div>
      <Typography.Title level={4}>每日信号（共 {data?.count ?? 0} 条，显示最近 200）</Typography.Title>
      <Card>
        {sigs.length ? <Table rowKey={(_r, i) => `${i}`} dataSource={sigs} columns={cols} size="small" pagination={{ pageSize: 20 }} />
          : <Alert type="info" message="暂无信号（模拟盘 8/3 开跑后产生）" />}
      </Card>
    </div>
  )
}
