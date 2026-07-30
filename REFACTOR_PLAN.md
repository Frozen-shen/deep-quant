# Deep Quant 全面重构蓝图

> IC 证据：价量因子在 CSI300 无 alpha (最高 ICIR 0.30)。战略：A(换矿脉) + C(沉淀底盘)并行。

## P0: 数据地基 + 纪律机制 — ✅ 部分完成

### 盲测锁 + 实验记账
- [x] scripts/run_blind_test.py: locked=true 拒绝重跑, config hash 写入 git tag
- [x] model/experiment.py: 每次 run 落盘 (config hash + git hash + 全指标)
- [x] .github/workflows/ci.yml: CI 锁检 + 单测

### 待做
- [ ] 真 PIT 宇宙 (历史成分股 + 退市 + ST)
- [ ] 双价格体系 (原始未复权判板 + 后复权算收益)
- [ ] 交易日历
- [ ] 数据快照版本化 (manifest per parquet)

## P1: 因子与模型 + 回测补丁

- [ ] 基线阶梯 L0-L3 (等权 → 单因子 → IC加权 → LightGBM)
- [ ] IC 流水线 Newey-West 修正 + 自动产权重
- [ ] 回测引擎: 结果对象化 + 容量约束 + 滑点 + 双基准
- [ ] 参数敏感性报告

## P2: 信息扩展

- [ ] PIT 财报因子 (公告日对齐)
- [ ] 资金流数据

## P3: 工程结构

- [ ] src/deepquant/ 包化
- [ ] pyproject.toml + CI 完整

---
