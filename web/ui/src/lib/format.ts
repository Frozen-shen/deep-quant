/** 数字千分位格式化；null/undefined/NaN → '—' */
export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return v.toLocaleString('zh-CN', { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

/** 小数比例 → 百分比字符串；0.056 → '5.60%' */
export function fmtPct(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return `${(v * 100).toFixed(digits)}%`
}

/**
 * 时间格式化:
 *  - 纯日期 "2025-01-03" → 原样返回 (回测成交只有日期无时刻, VWAP=全天均价;
 *    new Date 会把 ISO 纯日期当 UTC 午夜, +8 时区后错误显示 08:00)
 *  - ISO 时间戳 "2026-08-11T08:00:00" → 'YYYY-MM-DD HH:mm'
 *  - 解析失败原样返回
 */
export function fmtDateTime(iso?: string): string {
  if (!iso) return '—'
  if (/^\d{4}-\d{2}-\d{2}$/.test(iso)) return iso  // 纯日期: 不做时区转换
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`
}

/** 长文本截断（表格摘要列用），null/undefined → '—' */
export function truncate(s: string | null | undefined, n = 60): string {
  if (!s) return '—'
  return s.length > n ? `${s.slice(0, n)}…` : s
}
