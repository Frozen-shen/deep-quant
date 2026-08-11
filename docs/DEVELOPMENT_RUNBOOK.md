# 开发运行手册（防低级失误 Checklist）

> 2026-08-09 建立。本轮开发暴露 5 类低级失误（参数名错、漏 --liquid、面板区间未覆盖、
> 结果覆盖无备份、config 编辑误伤）。以下机制为强制门禁。
> 2026-08-11 增补：参数透传遗漏 3 连发（weight_mode/vol_target/trend_timing），
> 新增第 7 节 AST 透传门禁。

## 1. 改代码后：冒烟测试门禁（必做）

任何修改 `run_walkforward_backtest.py` / 因子模块 / 面板逻辑后，**先跑冒烟**再全量：

```bash
py scripts/active/check_param_passthrough.py        # 透传完整性 (第7节)
py scripts/active/run_walkforward_backtest.py --folds --folds-only --liquid --sample 50
```

**检查日志 4 个关键行**（缺一不可）：
1. `universe: 流动性PIT(全市场+过滤)` ← 防漏 --liquid（PIT 应为 3000+ 非 800）
2. `因子面板所需日期: ... (~2026-07-31)` ← 面板覆盖 extend/TEST 区间
3. `实验状态` 段的 pool_filter/分区打印 ← 防 config 遗留
4. 回测出现 `调仓日 ... 买入/持仓` ← 防空仓（scores/面板缺失）

## 2. 全量运行前：命令基线核对

标准实验命令（与上次成功的 diff 检查）：

```bash
# fold 验证 + 扩展模拟考 (TEST① 毕业段)
py scripts/active/run_walkforward_backtest.py --folds --folds-only --liquid --extend-val 2025-01-01 2026-06-30

# 纯 fold 验证
py scripts/active/run_walkforward_backtest.py --folds --folds-only --liquid

# 终极 TEST② (数据完备后, 会消耗锁)
py scripts/active/run_walkforward_backtest.py --folds --liquid
```

核对项：`--liquid` 必带 | `--folds-only` 决定是否碰 TEST | `--extend-val` 区间与面板 needed_dates 匹配

## 3. 结果文件管理

- 脚本启动时自动备份 `walkforward_results.json` → `walkforward_results_bak_<时间戳>.json`（已内置）
- 重要实验的 JSON 手动备份命名：`walkforward_results_<版本>_<特征>.json`
- **不要**覆盖式引用旧版本结果文件（如把 v10 结果当 v9 基线引用）

## 4. Config 编辑纪律

- 用 `py - <<EOF` 精确替换 + 断言（`assert old in s`），不用 sed 无断言替换
- 编辑后必跑：`py -c "import yaml; yaml.safe_load(open('config.yaml',encoding='utf-8'))"`
- 启动时 gate 完整性校验（重复键检测）已内置，失败即拒绝启动

## 5. 实验开关复位检查

- 实验间检查 config 遗留状态（启动日志"实验状态"段自动打印）：
  - `pool_filter.enabled`（实验后复位 false，生产启用时 true）
  - `neutralization.industry_neutral`（实验后复位 false）
  - `minute_factors.freq`（生产 15）
  - `fold.max_factors`（生产 50）

## 6. 纪律红线（不可违反）

- TEST②（2026-07+）只跑一次，`--force-partial-test` 仅用于数据完备性确认
- TEST①（2025-2026）已毕业：可作扩展模拟考（--extend-val），**不进训练**（fold 结构保持 2015-2023 训练）
- blind（2027+）永不回测，只走 daily_pipeline 模拟盘

## 7. 参数透传完整性门禁（2026-08-11 增补）

**背景**：2026-08-09~11 三个实验参数（weight_mode / vol_target_cfg / trend_timing_cfg）
在 `main → run_fold_analysis / run_fold_test → run_backtest` 链路中遗漏 3 次，
参数被静默吞掉退化为默认值，导致实验结论失真。均为"中间层签名/调用漏改"。

**门禁**（任何修改主脚本后必跑，退出码 0 才继续）：

```bash
py scripts/active/check_param_passthrough.py
```

检查逻辑（AST 静态分析，不执行代码）：
1. 签名检查：`run_backtest` / `run_fold_analysis` / `run_fold_test` 必须声明全部关键参数
2. 调用点检查：每个调用点必须显式传入签名中的全部关键参数（禁静默省略 → 默认值）
3. main 豁免签名检查（CLI 入口从 config 读参），但其调用点仍被检查

**新增实验参数时必须**：登记到 `check_param_passthrough.py` 的 `DEFAULT_KEY_PARAMS`，
否则门禁覆盖不到。遗漏可致实验结论失真（代价：一整轮回测作废）。
