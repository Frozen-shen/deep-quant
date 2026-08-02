import { Card, List, Tag, Typography, Spin } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'

export default function DataStatus() {
  const { data, isLoading } = useQuery({
    queryKey: ['health'], queryFn: async () => (await api.get('/health')).data,
  })
  return (
    <div>
      <Typography.Title level={4}>数据状态</Typography.Title>
      <Card>
        {isLoading ? <Spin /> : (
          <List
            dataSource={Object.entries(data?.data_sources ?? {})}
            renderItem={([k, v]) => (
              <List.Item>数据源 {k}：<Tag color={v ? 'green' : 'orange'}>{v ? '可用' : '待积累'}</Tag></List.Item>
            )}
          />
        )}
      </Card>
    </div>
  )
}
