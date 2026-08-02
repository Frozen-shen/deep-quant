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
│   └── universe.py          ← 成分股管理 (当前使用快照, PIT待完善)
├── factor_engine.py         ← 因子DSL解析
├── factor_library.py        ← 因子定义
├── factor_scorer.py         ← 因子筛选 (ic_top20预设)
├── factor_cache.py          ← 因子预计算
├── ml_ranker.py             ← LightGBM Lambdarank
├── model/
│   └── pipeline.py          ← ★ 无泄露训练管道
├── trading_rules.py         ← A股真实交易规则
├── portfolio.py             ← 持仓管理
├── portfolio_ranker.py      ← Top-K排名选股
├── evaluator.py             ← dev/blind分离评分
├── regime_detector.py       ← 市场状态检测
├── data_cache.py            ← 数据缓存
├── data_fetcher.py          ← 数据获取
├── storage.py               ← SQLite持久化
├── sector_analyzer.py       ← A股行业分类
├── scripts/
│   ├── run_backtest.py      ← 开发期walk-forward验证
│   └── run_blind_test.py    ← ★ 盲测 (参数冻结)
├── dashboard.py             ← Streamlit看板
├── tests/
│   └── test_core.py         ← 单元测试 (12 pass)
└── data_cache/              ← 股票数据缓存 (parquet)
```

### 已归档（archive/）
- `archive/src_quant/` — 早期平行代码库（2026-08-01 重写时弃用），保留作参考
- `archive/configs/`、`archive/legacy/` — 死配置与孤死模块
- `scripts/archive/` — 禁止运行，结果不可信（历史研究脚本）

## 快速开始

```bash
pip install -r requirements.txt

# 1. 拉取数据 (首次)
python data_cache.py --fetch-index 000300

# 2. 开发期验证 (可反复运行, 可调参)
python scripts/run_backtest.py

# 3. 盲测 (★ 参数冻结后只跑一次)
python scripts/run_blind_test.py

# 4. 单元测试
python -m unittest tests.test_core -v

# 5. 看板
streamlit run dashboard.py --server.headless true
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

## 已知限制 (诚实声明)

- **PIT成分股**: 当前使用CSI300快照覆盖所有月份（akshare历史API限制），尚未包含退市股
- **因子冻结**: `ic_top20`为手工精选，尚无自动化IC分析脚本绑定研究期数据
- **盲测**: Trial#1 已完成，结果: **策略+18.9%，超额-19.1%（1/3正窗口）**。此后盲测期因参数修改被污染，Trial#2-3 作废
- **数据快照**: 仓库含30只parquet(基础集), 本地已拉取176只. 需运行 `data_cache.py --fetch-index 000300` 扩展

## 技术栈

Python 3.12 · LightGBM · pandas · numpy · scipy · akshare · streamlit · PyYAML
