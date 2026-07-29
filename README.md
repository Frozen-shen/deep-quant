# Deep Quant v2

A股多因子量化系统 — Point-in-Time 成分股 × 无泄露训练 × 诚实盲测

## 核心改进 (相对 v1)

| 改进 | v1 (已废弃) | v2 |
|------|------------|-----|
| 股票池 | 2026年手挑牛股回测2021年 | Point-in-Time CSI300每月成分股 |
| 因子筛选 | 用2024-2026数据选因子 | 因子冻结在研究期(2018-2022) |
| 训练标签 | 穿越测试期(无embargo) | 训练截止到test_start - 20天 |
| 成交价 | 当天收盘价 | T+1开盘价 |
| 涨跌停 | can_buy/can_sell从未接线 | 每次交易前检查并执行 |
| 盲测 | 看过结果后继续改代码 | 参数冻结, trial计数, 只跑一次 |
| 参数管理 | 散落各文件全局变量 | config.yaml统一收口 |

## 项目结构

```
deep-quant/
├── config.yaml              ← 唯一参数源
├── data/
│   └── universe.py          ← Point-in-Time成分股管理
├── factors/
│   ├── engine.py            ← 因子DSL解析 (保留)
│   ├── library.py           ← 因子定义 (保留)
│   ├── scorer.py            ← 因子筛选 (绑定研究期数据)
│   └── cache.py             ← 因子预计算 (保留)
├── model/
│   ├── ranker.py            ← LightGBM Lambdarank (保留)
│   └── pipeline.py          ← ★ 无泄露训练管道 (新建)
├── execution/
│   ├── rules.py             ← A股真实交易规则 (保留)
│   ├── portfolio.py         ← 持仓管理 (保留)
│   └── portfolio_ranker.py  ← Top-K排名选股 (保留)
├── evaluation/
│   ├── evaluator.py         ← dev/blind分离评分 (保留)
│   └── regime_detector.py   ← 市场状态检测 (保留)
├── scripts/
│   ├── run_backtest.py      ← 开发期walk-forward验证
│   └── run_blind_test.py    ← ★ 盲测 (参数冻结, 一次跑)
└── dashboard/
    └── app.py               ← Streamlit看板
```

## 快速开始

```bash
pip install -r requirements.txt

# 1. 拉取数据
python scripts/fetch_data.py

# 2. 开发期验证 (可反复运行, 可调参)
python scripts/run_backtest.py

# 3. 盲测 (★ 参数冻结后只跑一次)
#    跑之前: 在 config.yaml 中确认所有参数不再修改
python scripts/run_blind_test.py

# 4. 看板
streamlit run dashboard/app.py
```

## 数据分区

```
2018-01 ─────── 2022-12 ─────── 2024-06 ─────── 2026-07
   │                 │                 │              │
   └─ 因子研究期 ────┘                  │              │
                     └─ 模型开发期 ────┘              │
                                      └─ ★盲测期 ────┘
```

- **研究期**: 做因子IC分析、选因子。从此"ic_optimized"因子集被冻结
- **开发期**: 训练模型、调超参。可用 `run_backtest.py` 反复运行
- **盲测期**: ★ 参数冻结后只跑一次。结果即为最终答案。`trial_count` 会自动 +1

## 诚实评估

- **开发集结果**: 仅供参考（参数根据这些数据选出）
- **盲测集结果**: ★ 最终成绩单（未参与任何开发决策）
- **trial_count**: 盲测运行次数。DSR 用真实 trial 数校正多重测试偏差

## 技术栈

Python 3.12 · LightGBM · pandas · numpy · scipy · akshare · streamlit · PyYAML
