# deep-quant

A股多因子 Top-K 排名量化交易系统 — 20只股票截面评分 → 动态选股

## 核心结果

| 配置 | 股票池 | 策略收益 | 基准收益 | 超额收益 |
|------|:--:|------|------|------|
| IC优化 (ic_optimized) | 10只A股 | +636% | +150% | **+485.7%** |
| IC优化 (ic_optimized) | 20只A股 | +167% | +72% | **+94.6%** |

> 回测区间: 2024-01 ~ 2026-07 | 0 Token | 纯本地计算

## 策略体系

```
日线因子打分 (8个IC验证因子)
    → 截面排名 (z-score标准化)
    → Top-K选股 (持有最强4只)
    → 日内执行 (VWAP成交 + 信号确认 + 止损)
    → 风控保护 (时间止损30天 + 回撤熔断15%)
```

## 因子体系 (IC优化)

| 因子 | 权重 | IC(20d) | 说明 |
|------|:--:|:--:|------|
| volatility_20d | +0.25 | +0.076 | 波动率(正向) |
| ma5_ma20_spread | +0.20 | +0.052 | 短期均线偏离 |
| ma10_ma20_spread | +0.15 | +0.059 | 中期均线偏离 |
| ma20_ma60_spread | +0.10 | +0.044 | 长期趋势确认 |
| ma5_cross_ma20 | +0.10 | +0.022 | 金叉死叉 |
| vol_ratio | +0.10 | +0.021 | 放量确认 |
| ma_bullish | +0.05 | - | 多头排列 |
| position_20d | +0.05 | - | 价格位置 |

## 日内交易四层

```
L1 自适应执行: 强信号抢筹(10min) / 弱信号VWAP(全天)
L2 信号确认:   开盘跳空+量比 → 仓位调整0.5~1.2x
L3 日内风控:   11:00时间止损 + 14:50尾盘清仓 + 5min跟踪止损
L4 Alpha因子:  开盘跳空/早盘量比/午后反转/VWAP位置/大单异动
```

## 快速开始

```bash
pip install -r requirements.txt

# A股回测
python paper_trade_a.py

# A/B对比测试
python test_a_share.py

# IC因子分析
python factor_analysis.py

# Web看板
streamlit run dashboard.py
```

## 项目结构

```
deep-quant/
├── 数据层
│   ├── data_fetcher.py         A股(新浪)+港股(新浪) 双市场
│   ├── intraday_fetcher.py     新浪5分钟K线 + 日内因子
│   └── event_fetcher.py        公司公告采集
├── 因子层
│   ├── factor_engine.py        因子表达式DSL
│   ├── factor_library.py       43+6个预定义因子
│   ├── factor_scorer.py        多因子加权 + 截面评分 + IC优化预设
│   ├── factor_analysis.py      Spearman Rank IC/ICIR 验证
│   └── indicators.py           9个技术指标
├── 策略层
│   ├── strategy.py             增强MA + RSI均值回复 + 策略路由器
│   ├── portfolio_ranker.py     Top-K排名选股 (板块中性化)
│   └── signal_hub.py           多策略信号聚合
├── 执行层
│   ├── paper_trade_a.py        A股Top-K纸面交易 (含日内执行)
│   ├── intraday_executor.py    日内四层 (VWAP/确认/风控/Alpha)
│   ├── portfolio.py            持仓资金管理
│   ├── backtest.py             回测引擎 (T+0/T+1, ATR止损)
│   └── executor.py             模拟下单
├── ML/DL层
│   ├── ml_ranker.py            LightGBM 回归排序器
│   ├── dl_models.py            PyTorch LSTM + Transformer
│   └── llm_weight_optimizer.py LLM定制因子权重
├── 分析层
│   ├── test_a_share.py         A/B对比测试框架
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
    ├── sector_analyzer.py       板块映射 + 同伴比较
    └── alt_data.py              北向资金 + 融资融券因子
```

## 运行模式

| 命令 | 说明 | Token |
|------|------|:--:|
| `python paper_trade_a.py` | A股Top-K回测 | 0 |
| `python test_a_share.py` | A/B四组对比 | 0 |
| `python factor_analysis.py` | IC/ICIR因子验证 | 0 |
| `streamlit run dashboard.py` | Web看板 | 0 |

## 技术栈

Python · pandas · numpy · akshare · scipy · matplotlib · streamlit · APScheduler · SQLite · PyTorch · LightGBM · DeepSeek API

## 参考

- Microsoft Qlib — 因子表达式引擎 + Alpha158 + TopkDropoutStrategy
- Backtrader — Cerebro引擎 + 122指标
- VN.PY — 事件驱动 + 中国市场
