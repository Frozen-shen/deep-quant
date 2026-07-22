# Phase 1 TODO: 评测体系升级 + A股真实交易约束

## 1. evaluator.py — 新增7个核心评测指标
- [ ] 1.1 DSR (Deflated Sharpe Ratio)
- [ ] 1.2 Skew / Kurtosis
- [ ] 1.3 Capture Ratio Up/Down
- [ ] 1.4 Ulcer Index + UPI
- [ ] 1.5 SQN (System Quality Number)
- [ ] 1.6 Rolling Sharpe 6M/12M
- [ ] 1.7 重构 analyze_window()

## 2. A股真实交易约束
- [ ] 2.1 真实费率模型 (MARKET_CONFIG + calc_commission)
- [ ] 2.2 trading_rules.py (涨跌停/停牌/ST/一字板)
- [ ] 2.3 fetch_stock_status.py (ST数据拉取)
- [ ] 2.4 test_rolling_v3.py 注入交易约束

## 3. Dashboard 可视化升级
- [ ] 3.1 月度收益热力图
- [ ] 3.2 水下曲线 (Underwater Plot)
- [ ] 3.3 滚动指标线图

## 4. 重跑评测
- [ ] 4.1 运行 test_rolling_v3.py (新费率+约束)
- [ ] 4.2 查看新 ReportCard 评级
