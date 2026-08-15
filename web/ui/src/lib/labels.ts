/** 字段名 → 中文标题（表格列、抽屉键值展示通用） */
export const fieldLabels: Record<string, string> = {
  experiment_id: '实验ID', timestamp: '时间', script: '脚本', partition: '分区', config_hash: '配置哈希',
  notes: '备注', parameters: '参数', results: '结果',
  symbol: '代码', name: '名称', qty: '数量', avg_cost: '成本价', market_value: '市值', entry_date: '建仓日',
  signal_date: '信号日期', mode: '模式', buy: '买入', sell: '卖出', hold: '持有', n_factors: '因子数',
  verdict: '结论', circuit_breaker: '熔断', date: '日期', action: '方向', price: '价格',
  commission: '佣金', reason: '原因',
  factor: '因子', ic_mean: '平均IC', icir: 'ICIR', ic_std: 'IC标准差', n_days: '样本天数', pos_ratio: '正占比',
}

/** 信号模式 → 中文+Tag色 */
export const modeMap: Record<string, { text: string; color: string }> = {
  live: { text: '实盘', color: 'blue' },
  dry_run: { text: '模拟', color: 'default' },
}

/** 信号结论 → 中文+Tag色 */
export const verdictMap: Record<string, { text: string; color: string }> = {
  SKIP: { text: '跳过', color: 'default' },
  CONDITIONAL: { text: '有条件', color: 'orange' },
  EXECUTE: { text: '执行', color: 'green' },
  'v24b-prod': { text: 'v24b 生产', color: 'blue' },
}

/** 交易方向 → 中文+Tag色（A股红涨绿跌：买入红、卖出绿） */
export const actionMap: Record<string, { text: string; color: string }> = {
  buy: { text: '买入', color: 'red' },
  sell: { text: '卖出', color: 'green' },
  BUY: { text: '买入', color: 'red' },
  SELL: { text: '卖出', color: 'green' },
}

/** 大小写不敏感的交易方向 Tag */
export function actionTag(action: unknown): { text: string; color: string } {
  const key = String(action ?? '').toLowerCase()
  return actionMap[key] ?? { text: String(action ?? '—'), color: 'default' }
}

/** 毕业指标状态 → 中文+Tag色 */
export const statusMap: Record<string, { text: string; color: string }> = {
  pass: { text: '达标', color: 'green' },
  fail: { text: '未达标', color: 'red' },
  pending: { text: '待数据', color: 'orange' },
}

/** 数据分区 → Tag色 */
export const partitionColors: Record<string, string> = {
  research: 'blue', val: 'cyan', test: 'purple',
  development: 'geekblue', blind: 'volcano',
}

/** 数据源键 → 中文 */
export const healthSourceLabels: Record<string, string> = {
  equity_log: '净值日志', portfolio: '组合持仓', signals: '信号记录',
  experiments: '实验记录', ic_results: 'IC 验证', data_store: '行情数据',
}
