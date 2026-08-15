import type { ReactNode } from 'react'
import { Drawer, Descriptions } from 'antd'

export interface KV { label: string; value: ReactNode }

/** 通用详情抽屉：标题 + 副标题 + 键值表 + 自定义附加块 */
export default function DetailDrawer({ open, title, subtitle, kvs, extra, onClose }: {
  open: boolean
  title: ReactNode
  subtitle?: ReactNode
  kvs: KV[]
  extra?: ReactNode
  onClose: () => void
}) {
  return (
    <Drawer title={title} open={open} onClose={onClose} width={560}>
      {subtitle && <div style={{ marginBottom: 12 }}>{subtitle}</div>}
      {kvs.length > 0 && (
        <Descriptions column={1} size="small" bordered
          items={kvs.map((kv) => ({ key: kv.label, label: kv.label, children: kv.value }))} />
      )}
      {extra && <div style={{ marginTop: 16 }}>{extra}</div>}
    </Drawer>
  )
}
