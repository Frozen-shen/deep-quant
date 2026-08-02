import { Card, List, Tag, Typography, Spin, Alert, Empty } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'

export default function DataStatus() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['health'], queryFn: async () => (await api.get('/health')).data,
  })
  if (isError) return <Alert type="error" message="后端不可用" description="请确认 web/api 已启动 (uvicorn :8000)" />
  const sources = Object.entries(data?.data_sources ?? {})
  return (
    <div>
      <Typography.Title level={4}>数据状态</Typography.Title>
      <Card>
        {isLoading ? <Spin /> : (sources.length ? (
          <List
            dataSource={sources}
            renderItem={([k, v]) => (
              <List.Item>数据源 {k}：<Tag color={v ? 'green' : 'orange'}>{v ? '可用' : '待积累'}</Tag></List.Item>
            )}
          />
        ) : (
          <Empty description="暂无数据源状态" />
        ))}
      </Card>
    </div>
  )
}
