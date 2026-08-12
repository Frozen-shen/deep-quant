import axios from 'axios'

export const api = axios.create({ baseURL: '/api' })

export interface GraduationMetric {
  key: string
  name: string
  value: number | null
  threshold: number | string | null
  status: 'pass' | 'fail' | 'pending'
  detail: string
}
export interface EquityPoint { date: string; total_equity: number; daily_return: number | null }
export interface EquitySummary {
  total_return: number | null
  max_drawdown: number | null
  volatility: number | null
  sharpe: number | null
}

export async function fetchGraduation(): Promise<{ metrics: GraduationMetric[]; overall: string }> {
  const { data } = await api.get('/graduation')
  return data
}
export async function fetchEquity(): Promise<{ curve: EquityPoint[]; summary: EquitySummary | null }> {
  const { data } = await api.get('/equity')
  return data
}
export async function fetchExperiments(): Promise<any> {
  const { data } = await api.get('/experiments')
  return data
}
export async function fetchPortfolio(): Promise<any> {
  const { data } = await api.get('/portfolio')
  return data
}
export async function fetchSignals(): Promise<any> {
  const { data } = await api.get('/signals')
  return data
}
export async function fetchFactorsIc(): Promise<any> {
  const { data } = await api.get('/factors/ic')
  return data
}
export async function fetchBroker(): Promise<any> {
  const { data } = await api.get('/broker/status')
  return data
}
export async function fetchHealth(): Promise<{ status: string; data_sources: Record<string, boolean> }> {
  const { data } = await api.get('/health')
  return data
}
export async function searchUniverse(q: string): Promise<{ stocks: Array<{ symbol: string; name: string; sector: string }> }> {
  const { data } = await api.get('/universe/search', { params: { q } })
  return data
}
export async function fetchStock(symbol: string): Promise<any> {
  const { data } = await api.get(`/stocks/${symbol}`)
  return data
}
export async function fetchUniverse(): Promise<{ total: number; stocks: Array<{ symbol: string; name: string; sector: string }> }> {
  const { data } = await api.get('/universe')
  return data
}
export async function fetchBrokerTrades(year?: number): Promise<{ year: number | null; count: number; trades: any[] }> {
  const { data } = await api.get('/broker/trades', { params: { year, limit: 2000 } })
  return data
}
export async function fetchBacktestTrades(): Promise<{
  available: boolean; version?: string; period?: string; excess_annual?: number;
  sharpe?: number; max_drawdown?: number; n_rebalances?: number; count: number; trades: any[];
  equity_curve?: Array<{ date: string; equity: number }>;
  active_returns?: number[];
  rebalances?: Array<{ date: string; n_positions: number }>;
}> {
  const { data } = await api.get('/broker/backtest-trades')
  return data
}
