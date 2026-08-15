import { Card, Col, Row, Table, Empty, Tag } from 'antd'
import type { ExperimentDetail, MetricItem } from '../api'
import { fmtNum, fmtPct } from '../lib/format'
import EquityTriptych from './EquityTriptych'
import PnlBarChart from './PnlBarChart'

interface Props { detail: ExperimentDetail; nameOf: (s: string) => string }

function metricText(m: MetricItem): string {
  if (m.value === null || m.value === undefined) return '—'
  // 注意: 后端 walkforward JSON 的 pct 指标是百分数原值 (如 -0.2 = -0.2%),
  // 不能走 fmtPct (它会 ×100)。fmtPct 仅用于小数比率 (如 daily_return)。
  if (m.format === 'pct') return `${(m.value as number).toFixed(2)}%`
  if (m.format === 'str') return String(m.value)
  return fmtNum(m.value as number, 2)
}

/** 通用实验报告渲染器: metrics 卡 → 净值三件套 → folds 表 → 个股盈亏 → 逐笔成交。 */
export default function ExperimentReport({ detail, nameOf }: Props) {
  const { metrics, series, folds, stock_pnl, trades, equity_curve } = detail
  const eqSeries = series.find(s => s.name === '组合净值')
  const benchSeries = series.find(s => s.name.includes('基准'))
  const equity = eqSeries && eqSeries.x.length
    ? eqSeries.x.map((d, i) => ({ date: d, equity: eqSeries.y[i] }))
    : equity_curve
  const benchmark = benchSeries && benchSeries.x.length
    ? benchSeries.x.map((d, i) => ({ date: d, close: benchSeries.y[i] }))
    : undefined

  const pnlColor = (v: number) => (v > 0 ? '#cf1322' : v < 0 ? '#3f8600' : undefined)

  return (
    <div>
      <Row gutter={[12, 12]}>
        {metrics.map(m => (
          <Col key={m.key} xs={12} md={6} xl={4}>
            <Card size="small" title={m.label}>
              <div style={{ fontSize: 18, fontWeight: 600, color: m.better === 'low'
                ? (typeof m.value === 'number' && m.value < 0 ? '#cf1322' : undefined)
                : undefined }}>
                {metricText(m)}
              </div>
            </Card>
          </Col>
        ))}
      </Row>

      <Card title="净值与回撤" style={{ marginTop: 16 }}>
        <EquityTriptych equity={equity} benchmark={benchmark} />
      </Card>

      {folds.length > 0 && (
        <Card title="Walk-Forward 各折成绩" style={{ marginTop: 16 }}>
          <Table size="small" rowKey="name" pagination={false}
            dataSource={folds}
            columns={[
              { title: 'Fold', dataIndex: 'name' },
              { title: '验证期', dataIndex: 'val' },
              { title: '年化超额', dataIndex: 'excess_annual', render: (v) => v == null ? '—' : `${(v as number).toFixed(1)}%` },
              { title: 'Sharpe', dataIndex: 'sharpe', render: (v) => v == null ? '—' : fmtNum(v, 2) },
              { title: '最大回撤', dataIndex: 'max_drawdown', render: (v) => v == null ? '—' : `${(v as number).toFixed(1)}%` },
              { title: 'IR', dataIndex: 'ir', render: (v) => v == null ? '—' : fmtNum(v, 2) },
            ]} />
        </Card>
      )}

      {stock_pnl.length > 0 && (
        <Card title={`个股盈亏（${stock_pnl.length} 只）`} style={{ marginTop: 16 }}>
          <PnlBarChart items={stock_pnl.map(s => ({ symbol: s.symbol, total_pnl: s.total_pnl }))} nameOf={nameOf} />
          <Table size="small" rowKey="symbol" style={{ marginTop: 8 }} pagination={{ pageSize: 15 }}
            dataSource={stock_pnl}
            columns={[
              { title: '代码', dataIndex: 'symbol', render: (v) => `${nameOf(v)} ${v}` },
              { title: '已实现盈亏', dataIndex: 'total_pnl', sorter: (a: any, b: any) => a.total_pnl - b.total_pnl,
                render: (v: number) => <span style={{ color: pnlColor(v) }}>{fmtNum(v, 0)}</span> },
              { title: '回合数', dataIndex: 'n_round_trips' },
              { title: '胜率', dataIndex: 'win_rate', render: (v) => v == null ? '—' : fmtPct(v) },
            ]} />
        </Card>
      )}

      {trades.length > 0 && (
        <Card title={`逐笔成交（${trades.length} 笔）`} style={{ marginTop: 16 }}>
          <Table size="small" rowKey={(r) => `${r.date}-${r.symbol}-${r.action}-${r.price}`}
            pagination={{ pageSize: 20 }} dataSource={trades}
            columns={[
              { title: '日期', dataIndex: 'date' },
              { title: '代码', dataIndex: 'symbol', render: (v) => `${nameOf(v)} ${v}` },
              { title: '方向', dataIndex: 'action', render: (v) => <Tag color={v === 'BUY' ? 'red' : 'green'}>{v === 'BUY' ? '买入' : '卖出'}</Tag> },
              { title: '价格', dataIndex: 'price', render: (v) => fmtNum(v, 2) },
              { title: '数量', dataIndex: 'qty' },
              { title: '佣金', dataIndex: 'commission', render: (v) => fmtNum(v, 2) },
            ]} />
        </Card>
      )}

      {!metrics.length && !trades.length && (
        <Empty description="该实验无结构化结果（旧格式实验，展示参数）" />
      )}
    </div>
  )
}
