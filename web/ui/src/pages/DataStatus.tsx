import { Card, List, Tag, Typography, Spin, Alert, Empty, Row, Col } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { fetchHealth } from '../api'
import { healthSourceLabels } from '../lib/labels'
import StatCard from '../components/StatCard'

export default function DataStatus() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['health'], queryFn: fetchHealth, refetchInterval: 60_000,
  })
  if (isError) return <Alert type="error" message="后端不可用" description="请确认 web/api 已启动 (uvicorn :8000)" />
  const sources = Object.entries(data?.data_sources ?? {})
  const okCount = sources.filter(([, v]) => v).length
  return (
    <div>
      <Typography.Title level={4}>数据状态</Typography.Title>
      <Row gutter={12} style={{ marginBottom: 16 }}>
        <Col span={8}><StatCard title="可用数据源" value={okCount} /></Col>
        <Col span={8}><StatCard title="数据源总数" value={sources.length} /></Col>
        <Col span={8}><StatCard title="后端状态" value={data?.status === 'ok' ? '正常' : data?.status ?? '—'} /></Col>
      </Row>
      <Card>
        {isLoading ? <Spin /> : (sources.length ? (
          <List
            dataSource={sources}
            renderItem={([k, v]) => (
              <List.Item>{healthSourceLabels[k] ?? k}（{k}）：
                <Tag color={v ? 'green' : 'orange'}>{v ? '可用' : '待积累'}</Tag>
              </List.Item>
            )}
          />
        ) : (
          <Empty description="暂无数据源状态" />
        ))}
      </Card>
    </div>
  )
}
