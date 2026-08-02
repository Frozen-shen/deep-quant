import axios from 'axios'

export const api = axios.create({ baseURL: '/api' })

export interface GraduationMetric {
  key: string
  name: string
  value: number | null
  threshold: number | string
  status: 'pass' | 'fail' | 'pending'
  detail: string
}
export interface EquityPoint { date: string; total_equity: number; daily_return: number | null }

export async function fetchGraduation(): Promise<{ metrics: GraduationMetric[]; overall: string }> {
  const { data } = await api.get('/graduation')
  return data
}
export async function fetchEquity(): Promise<{ curve: EquityPoint[]; summary: any }> {
  const { data } = await api.get('/equity')
  return data
}
