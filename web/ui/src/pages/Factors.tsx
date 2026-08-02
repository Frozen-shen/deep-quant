import { Tabs, Table, Typography, Spin, Alert } from 'antd'
import type { ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'

const sourceLabel: Record<string, string> = {
  p3_full_ic: '价量因子 (P3)', p6_fundamental_ic: '基本面 (P6)',
  p7_relative_ic: '相对因子 (P7)', p8_northbound_ic: '北向 (P8)', p9_minute_ic: '分钟因子 (P9)',
}

export default function Factors() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['factors-ic'], queryFn: async () => (await api.get('/factors/ic')).data,
  })
  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="加载失败" />
  const sources = data?.sources ?? []
  const resultOf = (s: string) => data?.results?.[s] ?? { results: [], meta: {} }
  const cols = ['factor', 'ic_mean', 'icir', 'ic_std', 'n_days', 'pos_ratio'].map((k) => ({
    title: k, dataIndex: k,
    render: (v: unknown): ReactNode => (typeof v === 'number' ? Number(v).toFixed(4) : (v as ReactNode)),
  }))
  return (
    <div>
      <Typography.Title level={4}>因子 IC 验证结果</Typography.Title>
      {sources.length === 0 ? (
        <Alert type="info" message="暂无因子 IC 数据（数据积累中）" />
      ) : (
      <Tabs
        items={sources.map((s: string) => ({
          key: s, label: sourceLabel[s] ?? s,
          children: (
            <div>
              <Typography.Paragraph type="secondary">
                {resultOf(s).meta?.description ?? ''}
              </Typography.Paragraph>
              <Table rowKey="factor" size="small" columns={cols}
                dataSource={resultOf(s).results ?? []} pagination={{ pageSize: 20 }} />
            </div>
          ),
        }))}
      />
      )}
    </div>
  )
}
