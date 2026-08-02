import { Card, Col, Row, Table, Typography, Spin, Alert, Empty } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'

export default function Experiments() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['experiments'], queryFn: async () => (await api.get('/experiments')).data,
  })
  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="加载失败" />
  const exps = data?.experiments ?? []
  const cols = ['experiment_id', 'timestamp', 'script', 'partition', 'config_hash'].map((k) => ({ title: k, dataIndex: k }))
  return (
    <div>
      <Typography.Title level={4}>实验记录</Typography.Title>
      <Row gutter={12}>
        <Col span={8}><Card size="small">总数：{data?.count ?? 0}</Card></Col>
        <Col span={16}><Card size="small">
          脚本分布：{Object.entries(data?.by_script ?? {}).map(([k, v]) => `${k}×${v}`).join('、')}
        </Card></Col>
      </Row>
      {exps.length ? (
        <Table rowKey="experiment_id" dataSource={exps} columns={cols} size="small" style={{ marginTop: 16 }} />
      ) : (
        <Empty description="暂无实验记录" style={{ marginTop: 24 }} />
      )}
    </div>
  )
}
