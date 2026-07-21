# 日内数据接入 TODO

> 状态: 计划阶段 | 预计工时: 待调研

## 数据源调研

### 分钟级数据 (5/15/30/60分钟)

| 接口 | 数据源 | 网络 | 备注 |
|------|------|:--:|------|
| `stock_zh_a_hist_min_em` | 东方财富 push2 | ❌ 被封 | 最常用,但网络不通 |
| `stock_zh_a_tick_tx` | 腾讯 | ⚠️ 待验证 | Tick级,可能可用 |
| 新浪分时API | 新浪 | ⚠️ 待验证 | curl可用,需解析 |

### Tick级数据

| 接口 | 说明 |
|------|------|
| `stock_zh_a_tick_tx_js` | 腾讯Tick (JSON格式) |

## 实现计划

### Phase 1: 验证数据源
- [ ] 测试 `stock_zh_a_tick_tx` 网络可达性
- [ ] 测试新浪分时API
- [ ] 确定可用的最小时间粒度

### Phase 2: 存储方案
- [ ] Parquet 按日分区存储 (`intraday/{symbol}/{YYYYMMDD}.parquet`)
- [ ] 设计表结构: datetime, open, high, low, close, volume, amount

### Phase 3: 因子扩展
- [ ] 日内波动率 (high-low spread within day)
- [ ] VWAP 偏离度 (close vs VWAP)
- [ ] 尾盘效应 (最后30分钟涨跌幅)
- [ ] 开盘跳空 (open vs prev_close)
- [ ] 日内量价关系 (上午vs下午成交量比)

### Phase 4: 回测引擎
- [ ] 复用 `BacktestEngine`,设置 `freq="5min"`
- [ ] 处理日内停牌 (某分钟无数据)
- [ ] T+1约束: 日内可买卖但当日不能卖出

## 预估工时
- Phase 1: 1小时
- Phase 2: 1小时
- Phase 3: 2小时
- Phase 4: 2小时

## 参考资料
- Qlib `qlib/backtest/executor.py`: 多频率嵌套回测 (yield from 模式)
- Qlib `qlib/backtest/exchange.py`: limit_threshold 分钟级检查
