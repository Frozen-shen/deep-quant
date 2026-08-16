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
  source?: 'real' | 'reconstructed';
  equity_curve?: Array<{ date: string; equity: number }>;
  active_returns?: number[];
  rebalances?: Array<{ date: string; n_positions: number }>;
}> {
  const { data } = await api.get('/broker/backtest-trades')
  return data
}

// ── 实验注册表 (可插拔核心) ──
export interface ExperimentRegistryItem {
  id: string
  kind: 'walkforward' | 'experiment'
  name: string
  generated_at: string
  has_trades: boolean
  summary: { excess_annual?: number | null; sharpe?: number | null; max_drawdown?: number | null; total_return?: number | null }
}
export interface MetricItem { key: string; label: string; value: number | string | null; format: 'pct' | 'num' | 'money' | 'str'; better: 'high' | 'low' }
export interface SeriesItem { name: string; type: 'line' | 'bar'; x: string[]; y: number[] }
export interface FoldResult { name: string; train?: string; val?: string; excess_annual?: number | null; sharpe?: number | null; max_drawdown?: number | null; ir?: number | null; avg_turnover?: number | null }
export interface StockPnlItem { symbol: string; total_pnl: number; realized_pnl: number; n_round_trips: number; win_rate: number | null; buy_count: number; sell_count: number; open_qty: number; current_price?: number | null; unrealized_pnl?: number | null; avg_cost?: number | null }
export interface ExperimentDetail {
  meta: { id: string; kind: string; name: string; generated_at: string; description?: string }
  metrics: MetricItem[]
  series: SeriesItem[]
  folds: FoldResult[]
  stock_pnl: StockPnlItem[]
  trades: any[]
  equity_curve: Array<{ date: string; equity: number }>
  benchmark_curve: Array<{ date: string; close: number }>
  segments: Array<{
    key: string
    label: string
    val: string
    equity: Array<{ date: string; equity: number }>
    benchmark: Array<{ date: string; close: number }>
    excess_annual?: number | null
    sharpe?: number | null
    max_drawdown?: number | null
    n_trades: number
  }>
}

export async function fetchExperimentRegistry(): Promise<{ count: number; experiments: ExperimentRegistryItem[] }> {
  const { data } = await api.get('/experiments/registry')
  return data
}
export async function fetchExperimentDetail(id: string): Promise<ExperimentDetail> {
  const { data } = await api.get(`/experiments/${id}`)
  return data
}
export async function fetchPaperStockPnl(): Promise<{ items: StockPnlItem[]; count: number }> {
  const { data } = await api.get('/paper/stock-pnl')
  return data
}
