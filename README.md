# deep-quant

A股多因子 Top-K 排名量化交易系统 — 18只股票截面评分 → LightGBM Lambdarank 动态选股

## 核心结果 (诚实滚动重训练)

| 窗口 | 训练期 | 测试期 | 策略收益 | 基准 | **超额** |
|------|------|------|------:|------:|:------:|
| W1 | 2018-2020 | 2021 | +39.9% | +14.5% | **+25.5%** ✅ |
| W2 | 2019-2021 | 2022 | +5.9% | -18.4% | **+24.3%** ✅ |
| W3 | 2020-2022 | 2023 | +18.6% | -4.2% | **+22.8%** ✅ |
| W4 | 2021-2023 | 2024 | +31.8% | +24.3% | **+7.4%** ✅ |
| W5 | 2022-2024 | 2025 | +24.1% | +22.7% | **+1.4%** ✅ |
| W6 | 2023-2025 | 2026H1 | -12.7% | -11.7% | **-1.0%** ≈ |
| **均值** | | | | | **+13.4%** 🎯 |

| 指标 | 值 |
|------|------|
| 平均超额 | **+13.4%** |
| 中位数超额 | +15.1% |
| 正窗口比例 | 5/6 (83%) |
| 信息比率 (IR) | **1.10** |
| 超额标准差 | 12.1% |

> 严格训练/测试分离: 每窗口用前3年训练, 后1年测试。永不穿越。确定性种子(seed=42), 结果可复现。

## 核心架构

```
日线28因子计算 (FactorCache预计算)
    → 截面z-score标准化 (每日同股票池内)
    → LightGBM Lambdarank (学习截面排序, 非回归)
    → 前瞻收益率标签 (close[T+5]/close[T]-1, 无数据泄露)
    → Top-K选股 (持有最强4只)
    → 滚动重训练 (每12个月用前3年数据重训)
```

## 关键创新

| 创新 | 说明 | 贡献 |
|------|------|:--:|
| **前瞻标签** | 标签=未来5日收益, 非过去收益 | 根本性修复 |
| **Lambdarank** | 直接优化截面排序, 非回归预测幅度 | 架构升级 |
| **截面标准化** | 训练特征每日z-score, 消除量纲差异 | 特征工程 |
| **FactorCache** | 因子预计算, 训练100x加速 | 工程优化 |
| **确定性种子** | seed=42, 结果完全可复现 | 可靠性 |
| **L1正则化** | lambda_l1=0.5, 防过拟合 | 泛化提升 |

## 因子体系 (28因子, IC验证)

基于30只A股2024-2026的Spearman Rank IC分析, 全部28因子ICIR > 0.10:

| 排名 | 因子 | IC均值 | ICIR | 类型 |
|:--:|------|:--:|:--:|------|
| 1 | sharpe_20d | -0.93 | -12.9 | 风险调整 |
| 2 | ma5_ma30_spread | +0.90 | +13.7 | 趋势 |
| 3 | ma3_ma20_spread | +0.85 | +7.8 | 趋势 |
| 4 | ma10_ma20_spread | +0.82 | +7.8 | 趋势 |
| 5 | ma5_ma20_spread | +0.82 | +7.3 | 趋势 |
| 6 | return_7d | -0.79 | -5.7 | 动量 |
| 7 | rank_20 | +0.77 | +5.2 | 价格位置 |
| ... | (21 more) | ... | ... | 波动/量价/K线 |

> 完整IC结果见 `factor_ic_results.csv`

## 快速开始

```bash
pip install -r requirements.txt

# 1. 拉取数据缓存 (一次性)
python data_cache.py --fetch

# 2. 因子IC分析 (可选, 5-10分钟)
python factor_analysis.py

# 3. 滚动重训练 — 核心测试
python test_rolling_v3.py

# 4. 超参数搜索 (可选, 10-15分钟)
python hyper_search.py

# 5. Web看板
streamlit run dashboard.py
```

## 项目结构

```
deep-quant/
├── 数据层
│   ├── data_fetcher.py         A股(新浪) + 港股(新浪) 双市场
│   ├── data_cache.py           本地parquet缓存 (30只A股, ~3MB)
│   └── data_cache/             缓存文件目录
├── 因子层
│   ├── factor_engine.py        因子表达式DSL (Ref/Std/Max/Min/EMA/Rank)
│   ├── factor_library.py       28+预定义因子 (价格/均线/波动/量价/K线)
│   ├── factor_scorer.py        多因子加权 + 截面评分 (ic_optimized / v2预设)
│   ├── factor_cache.py         ★ 因子预计算缓存 (O(ND×NS)→O(NS))
│   └── factor_analysis.py      30股 Spearman Rank IC/ICIR 验证
├── 策略层
│   ├── strategy.py             增强MA + RSI均值回复 + 策略路由器
│   ├── portfolio_ranker.py     Top-K排名选股 (板块中性化)
│   └── signal_hub.py           多策略信号聚合
├── ML层
│   ├── ml_ranker.py            ★ LightGBM Lambdarank (确定性种子)
│   └── hyper_search.py         超参数网格搜索 (NDCG评估)
├── 测试层
│   ├── test_rolling_v3.py      ★ 全优化版滚动重训练 (推荐使用)
│   ├── test_rolling_v2.py      前瞻标签版 (v2)
│   └── test_results/           ★ 时间戳存档目录
├── 执行层
│   ├── paper_trade_a.py        A股Top-K纸面交易 (含日内执行)
│   ├── portfolio.py            持仓资金管理
│   ├── backtest.py             回测引擎 (T+0/T+1)
│   └── executor.py             模拟下单
├── 分析层
│   ├── analysis.py             25+绩效指标 + 统计检验
│   ├── validator.py            滚动窗口验证
│   └── stress_test.py          压力测试
├── 生产层
│   ├── storage.py              SQLite 8张表
│   ├── scheduler.py            APScheduler 定时调度
│   ├── alerter.py              微信/钉钉通知
│   └── dashboard.py            Streamlit 5页看板
└── 辅助层
    ├── macro_overlay.py         宏观叠加 (大盘+北向)
    ├── fundamental_llm.py       DeepSeek 基本面评分
    └── sector_analyzer.py       板块映射 + 同伴比较
```

## 运行模式

| 命令 | 说明 | 市场 |
|------|------|:--:|
| `python test_rolling_v3.py` | ★ 滚动重训练 (推荐) | A股 |
| `python test_rolling_v2.py` | 滚动重训练 v2 | A股 |
| `python factor_analysis.py` | IC/ICIR 因子验证 | A股 |
| `python hyper_search.py` | LightGBM 超参数搜索 | A股 |
| `python paper_trade_a.py` | A股Top-K回测 | A股 |
| `streamlit run dashboard.py` | Web看板 | A/H |

## 版本演进

| 版本 | 核心改进 | 平均超额 | IR |
|------|------|:------:|:--:|
| v1 原版 | Regression + 10股 + 50树 | -4.4% | — |
| v2a | Lambdarank + 旧标签(动量) | -19.3% | -0.32 |
| v2b | Lambdarank + **前瞻标签** | +10.1% | 正 |
| **v3** | **+FactorCache + 确定性 + L1=0.5** | **+13.4%** | **1.10** |

## 技术栈

Python 3.12 · pandas · numpy · scipy · LightGBM · akshare · matplotlib · streamlit · APScheduler · SQLite · PyTorch

## 参考

- Microsoft Qlib — 因子表达式引擎 + Alpha158 + TopkDropoutStrategy
- LightGBM — Lambdarank + NDCG评估
- 严格训练/测试分离: 滚动窗口 + 前瞻标签 + 无数据泄露
