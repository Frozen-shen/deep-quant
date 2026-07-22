# Phase 2 TODO: 风控系统 + 因子扩展

## 1. risk_manager.py (新模块)
- [ ] 1.1 RiskManager 预交易检查 (单票≤25%, 总敞口≤95%, 日内熔断)
- [ ] 1.2 StopLoss/StopTrail/TakeProfit (扩展position_entry)
- [ ] 1.3 Kelly/ATR 仓位计算

## 2. 因子扩展
- [ ] 2.1 factor_engine.py: 新增 Corr 算子 (价量相关性)
- [ ] 2.2 factor_engine.py: 新增 RSqr 算子 (趋势拟合度)
- [ ] 2.3 factor_library.py: 激活 Qlib K-bar 9因子
- [ ] 2.4 fundamental_cache.py: PE/PB/ROE 数据管道

## 3. test_rolling_v3.py 集成
- [ ] 3.1 注入 RiskManager 到回测循环
- [ ] 3.2 注入止损系统

## 4. 重跑评测
- [ ] 4.1 运行 test_rolling_v3.py (新风控+因子)
- [ ] 4.2 查看新 ReportCard
