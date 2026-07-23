# deep-quant

A股多因子 Top-K 排名量化系统 — 30只股票 × 39因子 → LightGBM Lambdarank 动态选股

## 核心结果 (v4 全优化, 诚实8窗口滚动重训练)

| 窗口 | 训练期 | 测试期 | 策略收益 | 基准 | **超额** |
|------|------|------|------:|------:|:------:|
| W1 | 2019H2-2020 | 2021H1 | +71.4% | +6.2% | **+65.1%** ✅ |
| W2 | 2020H1-2021H2 | 2021H2-2022H1 | +24.6% | -10.3% | **+34.9%** ✅ |
| W3 | 2021H1-2022H1 | 2022H2-2023H1 | +32.6% | +5.2% | **+27.5%** ✅ |
| W4 | 2021H2-2023H1 | 2023H2-2024H1 | -29.6% | -19.4% | **-10.2%** ❌ |
| W5 | 2022H2-2023 | 2024H1-2024Q3 | +15.2% | +12.2% | **+3.0%** ✅ |
| W6 | 2023H1-2024H2 | 2024Q4-2025H1 | -4.0% | -7.7% | **+3.7%** ✅ |
| W7 | 2024H1-2025H1 | 2025Q3-2026Q1 | +4.9% | +12.6% | **-7.8%** ≈ |
| W8 | 2024Q4-2026H1 | 2026Q2-2026Q3 | +2.0% | +8.5% | **-6.5%** ≈ |
| **均值** | | | | | **+13.7%** 🎯 |

| 指标 | 值 |
|------|------|
| 平均超额 | **+13.7%** |
| 中位数超额 | +3.4% |
| 正窗口比例 | 5/8 (62.5%) |
| 信息比率 (IR) | **0.52** |
| 最差窗口 | -10.2% |
| 超额标准差 | 26.6% |

> 严格训练/测试分离: 每窗口用前1.5年训练, 后9个月测试。时间衰减权重(半衰期0.7年)。永不穿越。

## 核心架构

```
日线39因子计算 (FactorCache预计算)
    → 截面z-score标准化 (每日同股票池内)
    → LightGBM Lambdarank (学习截面排序, 非回归)
    → 时间衰减样本权重 (近期数据权重更高)
    → 前瞻收益率标签 (close[T+5]/close[T]-1, 无数据泄露)
    → Top-K选股 (持有最强4只, 换手缓冲+成本门槛)
    → Regime检测 (MA60+ADX → 牛/熊/震荡自适应)
    → 稳健性评测 (交叉窗口IR + 单窗亏损惩罚)
```

## v4 关键改进 (相对 v3)

| Phase | 改进 | 效果 |
|-------|------|------|
| **P0** | 修复5个Bug (DSL优先级/Kelly/费率等) | 基础可靠性 |
| **P1** | 换手率控制 (缓冲+确认期+成本门槛) | 交易降低63% |
| **P2** | 因子增强 (39因子含反转/流动性/波动率) | 信息维度扩展 |
| **P3** | Regime检测 (市场状态自适应参数) | 熊市防御↑ |
| **Phase 1** | 训练窗口3年→1.5年 + 时间衰减权重 | 更快适应市场 |
| **Phase 2** | 股票池18→30只 (含CSI 300扩展管线) | 截面统计意义↑ |
| **Phase 3** | 参数重校准 + Regime方向修正 | 稳健性↑ |

## 版本演进

| 版本 | 核心改进 | 平均超额 | IR | 正窗口 |
|------|------|:------:|:--:|:--:|
| v1 原版 | Regression + 10股 + 50树 | -4.4% | — | — |
| v2b | Lambdarank + **前瞻标签** | +10.1% | 正 | — |
| v3 | FactorCache + 确定性 + L1正则 | +13.4% | 1.10 | 5/6 |
| **v4** | **P0-P3 + Phase1-3 全优化** | **+13.7%** | **0.52** | **5/8** |

## 因子体系 (39因子)

基于优化后的 ic_optimized preset，涵盖6大类:

| 类别 | 因子数 | 代表性因子 |
|------|:------:|------|
| 趋势/均线 | 10 | ma5_ma20_spread, ma20_ma60_spread, ma_bullish |
| 波动率 | 6 | volatility_20d, vol_regime, boll_width |
| 动量/反转 | 5 | return_7d, reversal_1d, rev_mom_spread |
| 量价 | 7 | turnover_trend, vol_price_sync, liq_ratio |
| K线形态 | 6 | amplitude_5d, klen, cntd_20 |
| 风险调整 | 5 | sharpe_20d, skew_20d, rsqr_20 |

## 快速开始

```bash
pip install -r requirements.txt

# 1. 拉取数据缓存 (一次性)
python data_cache.py --fetch

# 2. 扩展至CSI300 (可选, 需网络)
python data_cache.py --fetch-index 000300

# 3. 滚动重训练 — 核心测试
python test_rolling_v3.py

# 4. Web看板
streamlit run dashboard.py
```

## 项目结构

```
deep-quant/
├── 数据层
│   ├── data_fetcher.py         A股/港股 + 指数成分股获取 (CSI300/500/1000)
│   ├── data_cache.py           本地parquet缓存 (动态股票池)
│   ├── trading_rules.py        ★ A股真实交易约束 (费率/涨跌停/停牌)
│   └── data_cache/             缓存文件 + 行业映射
├── 因子层
│   ├── factor_engine.py        因子表达式DSL (Ref/Std/Corr/RSqr/一元负号)
│   ├── factor_library.py       39+预定义因子 (6大类: 趋势/波动/动量/量价/K线/风险)
│   ├── factor_scorer.py        多因子加权 (ic_optimized preset)
│   ├── factor_cache.py         ★ 因子预计算缓存
│   └── factor_analysis.py      Spearman Rank IC/ICIR 验证
├── 策略层
│   ├── strategy.py             增强MA + RSI均值回复 + 策略路由器
│   ├── portfolio_ranker.py     ★ Top-K排名选股 (换手缓冲/成本门槛/Regime/行业中性化)
│   ├── regime_detector.py      ★ 市场状态检测 (MA60+ADX → 牛/熊/震荡)
│   └── signal_hub.py           多策略信号聚合
├── ML层
│   ├── ml_ranker.py            ★ LightGBM Lambdarank + DEnsembleRanker
│   └── hyper_search.py         超参数网格搜索
├── 测试层
│   ├── test_rolling_v3.py      ★ 全优化版滚动重训练 (v4)
│   └── test_results/           时间戳存档目录
├── 执行层
│   ├── portfolio.py            持仓资金管理
│   ├── backtest.py             回测引擎 (统一真实费率)
│   ├── risk_manager.py         风控系统 (止损/熔断/Kelly仓位)
│   └── executor.py             模拟下单
├── 分析层
│   ├── evaluator.py            ★ 17指标评测 + 稳健性惩罚 (DSR/滚动Sharpe)
│   ├── analysis.py             绩效分析 + 统计检验
│   ├── sector_analyzer.py      A股/港股行业分类
│   └── stress_test.py          压力测试
├── 生产层
│   ├── storage.py              SQLite持久化
│   ├── scheduler.py            定时调度
│   ├── alerter.py              告警通知
│   └── dashboard.py            Streamlit 5页看板
└── TODO.md                    优化路线图 + 完整记录
```

## 运行模式

| 命令 | 说明 | 市场 |
|------|------|:--:|
| `python test_rolling_v3.py` | ★ 滚动重训练 (推荐) | A股 |
| `python data_cache.py --fetch-index 000300` | 拉取CSI300成分股 | A股 |
| `python factor_analysis.py` | IC/ICIR 因子验证 | A股 |
| `python hyper_search.py` | LightGBM 超参数搜索 | A股 |
| `streamlit run dashboard.py` | Web看板 | A/H |

## 技术栈

Python 3.12 · pandas · numpy · scipy · LightGBM · akshare · matplotlib · streamlit · APScheduler · SQLite

## 参考

- Microsoft Qlib — 因子表达式引擎 + Alpha158 + TopkDropoutStrategy + DEnsembleModel
- LightGBM — Lambdarank + NDCG评估 + sample_weight
- 严格训练/测试分离: 滚动窗口 + 前瞻标签 + 时间衰减权重 + 无数据泄露
- Harvey & Liu 2015 — Deflated Sharpe Ratio (多重测试校正)
