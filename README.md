# deep-quant

港股多因子 Top-K 排名交易系统 — 12只港股截面评分 → 动态选股 → +32.35%超额收益

## 一句话

不再是"每只股票该不该买"，而是"12只里哪3只最强"——永远持有排名最高的K只。

## 核心结果

```
策略收益: +90.56%  |  等权基准: +58.22%  |  超额: +32.35%
回测区间: 2024-01 ~ 2026-07  |  交易日: 618天  |  交易: 126笔
```

## 架构

```
deep-quant/
├── 数据层
│   ├── data_fetcher.py       A股(新浪)+港股(新浪) 双市场
│   ├── event_fetcher.py      公司公告/研报采集
│   └── news_fetcher.py       财经新闻采集
├── 因子层
│   ├── factor_engine.py      因子表达式DSL ("Mean($close,5)/$close-1")
│   ├── factor_library.py     43个预定义因子
│   ├── factor_scorer.py      截面评分 + 多因子加权
│   └── indicators.py         9个技术指标 (RSI/MACD/BOLL/ATR/ADX/KDJ/OBV)
├── 策略层
│   ├── strategy.py           增强MA交叉 + RSI均值回复 + 策略路由器
│   ├── portfolio_ranker.py   Top-K排名选股 (参考 Qlib TopkDropoutStrategy)
│   └── signal_hub.py          多策略信号聚合
├── 执行层
│   ├── paper_trade_portfolio.py  多股票组合纸面交易
│   ├── portfolio.py              持仓/资金管理
│   ├── executor.py               模拟下单
│   └── backtest.py               回测引擎 (T+0/T+1, ATR止损, 次日开盘价)
├── 分析层
│   ├── analysis.py            绩效指标 (Sharpe/Sortino/VaR/统计检验)
│   ├── validator.py           滚动窗口验证 + 参数扫描
│   └── stress_test.py         崩盘回放 + 手续费敏感性
├── LLM层
│   ├── llm_factor.py          DeepSeek事件评分 (4个后端)
│   ├── fundamental_llm.py     DeepSeek基本面评分
│   ├── llm_weight_optimizer.py LLM定制因子权重
│   ├── macro_overlay.py       宏观叠加 (大盘+北向资金)
│   └── validate_llm.py        LLM事件预测准确率验证
├── 生产层
│   ├── storage.py             SQLite 8张表
│   ├── scheduler.py           APScheduler 定时调度
│   ├── alerter.py             微信/钉钉通知
│   └── dashboard.py           Streamlit 4页看板
├── 辅助层
│   ├── sector_analyzer.py     板块映射+同伴比较
│   ├── alt_data.py            北向资金+融资融券因子
│   ├── stock_filter.py        选股过滤器
│   └── quick_validate.py      快速参数扫描
└── 测试层
    └── test_data/
        ├── datasets/          18个CSV (5只股票 × 8年历史)
        ├── generate.py        一键生成测试数据
        ├── loader.py          便捷数据加载
        ├── verify.py          独立算法交叉验证
        └── scenarios.py       场景配置
```

## 快速开始

```bash
# 安装
pip install -r requirements.txt

# 单股票回测
MARKET=hk python main.py

# Top-3 排名制组合交易 (推荐)
python paper_trade_portfolio.py

# 查看结果
streamlit run dashboard.py
```

## 运行模式

| 命令 | 说明 |
|------|------|
| `python paper_trade_portfolio.py` | Top-3排名制, 12只港股 |
| `MARKET=hk python main.py` | 单股票回测 (指定标的) |
| `python paper_trade.py --symbol 01810` | 单股票纸面交易 |
| `python scheduler.py --once` | 每日信号生成 |
| `streamlit run dashboard.py` | Web看板 |

## 市场支持

| | A股 | 港股 |
|------|:--:|:--:|
| 数据源 | 新浪 | 新浪 |
| 交易制度 | T+1 | T+0 |
| 手续费 | 3bp | 14bp(逐项) |
| 无风险利率 | 2% | 3.5% |

## LLM 配置

```bash
# DeepSeek
LLM_BACKEND=openai OPENAI_API_KEY=sk-xxx python validate_llm.py

# 本地模型
LLM_BACKEND=ollama LLM_MODEL=qwen2.5:7b python main.py
```

## 技术栈

Python · pandas · numpy · akshare · scipy · matplotlib · streamlit · APScheduler · SQLite · DeepSeek API

## 参考

- Qlib TopkDropoutStrategy (Microsoft)
- Backtrader Cerebro引擎
- VN.PY 事件驱动架构
