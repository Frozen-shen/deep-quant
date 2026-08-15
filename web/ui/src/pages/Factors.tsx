import { Tabs, Table, Typography, Spin, Alert, Empty, Space, Tag } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { fetchFactorsIc } from '../api'
import { col } from '../lib/columns'
import { fmtNum } from '../lib/format'

const sourceLabel: Record<string, string> = {
  p3_full_ic: '价量因子 (P3)', p6_fundamental_ic: '基本面 (P6)',
  p7_relative_ic: '相对因子 (P7)', p8_northbound_ic: '北向 (P8)', p9_minute_ic: '分钟因子 (P9)',
  p10_minute_ic_decay_5m: '分钟IC衰减 5m (P10)', p10_minute_ic_decay_15m: '分钟IC衰减 15m (P10)',
}

interface IcRow { factor: string; ic_mean?: number; icir?: number; ic_std?: number; n_days?: number; pos_ratio?: number }

/** p10 分钟IC衰减矩阵列（按预测周期 horizon） */
function p10Columns(horizons: number[]) {
  const cols = [col<Record<string, unknown>>('factor', { title: '因子', sorter: true, ellipsis: true })]
  for (const h of horizons) {
    const key = String(h)
    cols.push({
      title: `H+${h}`, dataIndex: key, width: 110, ellipsis: true, defaultSortOrder: undefined,
      sorter: (a: Record<string, unknown>, b: Record<string, unknown>) => {
        const av = (a[key] as Record<string, unknown> | undefined)?.ic_mean as number | undefined
        const bv = (b[key] as Record<string, unknown> | undefined)?.ic_mean as number | undefined
        return (av ?? 0) - (bv ?? 0)
      },
      render: (_v: unknown, r: Record<string, unknown>) => {
        const d = r[key] as Record<string, unknown> | undefined
        if (!d) return '—'
        return <span>{fmtNum(d.ic_mean as number, 4)}<Typography.Text type="secondary" style={{ marginLeft: 4 }}>({fmtNum(d.icir as number, 2)})</Typography.Text></span>
      },
    })
  }
  return cols
}

export default function Factors() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['factors-ic'], queryFn: fetchFactorsIc,
  })
  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="加载失败" />
  const sources = data?.sources ?? []
  const resultOf = (s: string) => data?.results?.[s] ?? { results: [], meta: {} }

  /** p10 源：结构为 {freq, horizons, n_stocks, summary, factors:{因子:{horizon:{ic_mean,icir,n,pos_ratio}}}} */
  const isP10 = (s: string) => s.startsWith('p10_')

  const cols = [
    col<IcRow>('factor', { title: '因子', sorter: true, ellipsis: true }),
    col<IcRow>('ic_mean', { title: '平均IC', sorter: true, defaultSortOrder: 'descend',
      render: (v) => fmtNum(v as number, 4) }),
    col<IcRow>('icir', { title: 'ICIR', sorter: true, render: (v) => fmtNum(v as number, 4) }),
    col<IcRow>('ic_std', { title: 'IC标准差', sorter: true, render: (v) => fmtNum(v as number, 4) }),
    col<IcRow>('n_days', { title: '样本天数', sorter: true, render: (v) => fmtNum(v as number, 0) }),
    col<IcRow>('pos_ratio', { title: '正占比', sorter: true, render: (v) => fmtNum(v as number, 4) }),
  ]

  /** meta 全量键值（不只 description） */
  const metaTags = (s: string) => Object.entries(resultOf(s).meta ?? {})
    .map(([k, v]) => <Tag key={k}>{k}: {String(v)}</Tag>)

  /** 均值摘要 */
  const summaryOf = (s: string) => {
    const rs = resultOf(s).results ?? []
    if (!rs.length) return null
    const avg = (k: 'ic_mean' | 'icir') => rs.reduce((a: number, r: any) => a + (r[k] ?? 0), 0) / rs.length
    return { n: rs.length, ic: avg('ic_mean'), icir: avg('icir') }
  }

  return (
    <div>
      <Typography.Title level={4}>因子 IC 验证结果</Typography.Title>
      {sources.length === 0 ? (
        <Empty description="暂无因子 IC 数据（数据积累中）" />
      ) : (
      <Tabs
        items={sources.map((s: string) => {
          const sum = summaryOf(s)
          const p10 = isP10(s) ? resultOf(s) : null
          return {
            key: s, label: sourceLabel[s] ?? s,
            children: p10 ? (
              <div>
                <Space wrap style={{ marginBottom: 8 }}>
                  <Tag color="blue">频率 {p10.freq} 分钟 · 股票 {p10.n_stocks} 只 · 周期 {String(p10.horizons ?? []).replace(/,/g, ' / ')} 天</Tag>
                  {Object.entries(p10.summary ?? {}).map(([k, v]) => (
                    <Tag key={k}>{k}: {String(v)}</Tag>
                  ))}
                </Space>
                <Table rowKey="factor" size="small" columns={p10Columns(p10.horizons ?? [])}
                  dataSource={Object.entries(p10.factors ?? {}).map(([f, v]) => ({ factor: f, ...(v as object) }))}
                  pagination={{ pageSize: 20, showSizeChanger: true, showTotal: (t) => `共 ${t} 条` }} />
              </div>
            ) : (
              <div>
                <Space wrap style={{ marginBottom: 8 }}>
                  {metaTags(s)}
                  {sum && <Tag color="blue">因子 {sum.n} 个 · 平均IC {fmtNum(sum.ic, 4)} · 平均ICIR {fmtNum(sum.icir, 4)}</Tag>}
                </Space>
                <Table rowKey="factor" size="small" columns={cols}
                  dataSource={resultOf(s).results ?? []}
                  pagination={{ pageSize: 20, showSizeChanger: true, showTotal: (t) => `共 ${t} 条` }} />
              </div>
            ),
          }
        })}
      />
      )}
    </div>
  )
}
