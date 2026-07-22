# Phase 3 TODO: 模型升级 + 归因分析

## 1. ml_ranker.py — DEnsemble 迭代重训练
- [ ] 1.1 fit() 添加 sample_weight 参数
- [ ] 1.2 难样本检测: 组内排名偏差 → 权重放大
- [ ] 1.3 DEnsembleRanker 包装类: N模型迭代 → 加权集成

## 2. factor_report_card.py (新模块) — IC分析报告卡
- [ ] 2.1 分组收益曲线 (top quintile vs bottom)
- [ ] 2.2 IC热力图 (月×因子)
- [ ] 2.3 预测自相关 + 信号换手率

## 3. performance_attribution.py (新模块) — 归因分析
- [ ] 3.1 Brinson归因: 行业配置效应 + 选股效应
- [ ] 3.2 补齐 A_SECTORS 映射 (12/30 → 30/30)

## 4. Dashboard 更新
- [ ] 4.1 IC热力图展示
- [ ] 4.2 分位数组收益曲线
