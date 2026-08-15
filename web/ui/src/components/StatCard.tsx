import { Card, Statistic } from 'antd'

/** 统计卡：数字 + 可选前后缀/精度/颜色 */
export default function StatCard({ title, value, precision = 0, suffix, prefix, color }: {
  title: string
  value?: number | string | null
  precision?: number
  suffix?: string
  prefix?: string
  color?: string
}) {
  return (
    <Card size="small">
      <Statistic title={title} value={value as number} precision={precision}
        suffix={suffix} prefix={prefix} valueStyle={color ? { color } : undefined} />
    </Card>
  )
}
