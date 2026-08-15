import type { ReactNode } from 'react'
import { fieldLabels } from './labels'

function cmp(a: unknown, b: unknown): number {
  if (typeof a === 'number' && typeof b === 'number') return a - b
  return String(a ?? '').localeCompare(String(b ?? ''), 'zh-CN')
}

/** 通用表格列：自动中文标题 + 可选排序/宽度/省略/自定义渲染 */
export function col<T>(key: keyof T & string, opts: {
  title?: string
  sorter?: boolean
  defaultSortOrder?: 'ascend' | 'descend'
  width?: number
  ellipsis?: boolean
  render?: (v: unknown, row: T) => ReactNode
} = {}) {
  return {
    title: opts.title ?? fieldLabels[key] ?? key,
    dataIndex: key,
    width: opts.width,
    ellipsis: opts.ellipsis,
    sorter: opts.sorter ? (a: T, b: T) => cmp(a[key], b[key]) : undefined,
    defaultSortOrder: opts.defaultSortOrder,
    render: opts.render,
  }
}
