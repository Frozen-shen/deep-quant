# 合法脚本 (active/)

只有此目录中的脚本可以使用。同类功能只允许存在一个脚本。

| 脚本 | 用途 | 运行频率 |
|------|------|----------|
| `run_research_backtest.py` | 研究阶段回测 (唯一) | 按需 |
| `run_factor_ic.py` | 因子IC验证 (唯一) | 按需 |
| `run_paper_signal.py` | 模拟盘信号生成 (唯一) | 每日 |
| `fetch_daily_data.py` | 全市场日线数据获取 | 每周/按需 |
| `fetch_fundamentals.py` | 基本面数据获取 | 每月 |
| `run_ic_monitor.py` | IC衰减监控 | 每周 |
| `export_equity_curve.py` | 净值曲线导出 | 按需 |
| `init_paper_account.py` | 模拟盘账户初始化 | 一次性 |

## 规则

1. **禁止新建回测脚本** — 修改 `run_research_backtest.py`
2. **禁止在 archive/ 中运行脚本** — 结果不可信
3. **每个脚本开头必须有 gate.check() 调用**
