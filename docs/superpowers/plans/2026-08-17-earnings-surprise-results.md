# 预期差感知层 + 行业轮动层 — 实验结果报告

> 设计文档: `docs/superpowers/specs/2026-08-17-earnings-surprise-and-sector-momentum-design.md`
> 实施计划: `docs/superpowers/plans/2026-08-17-earnings-surprise-implementation.md`
> 分支: `feature/earnings-surprise` | 完成日期: 2026-08-29

## 结论速览

| 实验 | 配置 | folds 超额均值 | extend 总/年化超额 | Sharpe | 回撤 | 门禁 |
|---|---|---|---|---|---|---|
| v27 基线 (生产) | styles off, λ=0 | +5.8% | +93.6% / +29.2% | 2.17 | -10.8% | — |
| X | growth 换预期差料 0.15 | +0.06% | +57.5% / +7.8% | 0.84 | -22.5% | ❌ 惨败 |
| Y | styles off, λ=0.10 | +9.2% | +111.0% / +39.2% | 2.06 | -13.6% | ✅ 通过(带保留) |

## X — 预期差 sleeve 换料: 负结果

SUE/盈利加速/PEAD 替换 growth 因子, momentum 预算 0.05 保留。全部门禁失败。
folds 逐年 -15.1/+5.5/+7.2/-13.8/+16.5 — 高方差, 信号在多数年份是噪声。
**注意混杂变量**: 复盘发现 X 运行时 config 里 trend_timing/orthogonalize 处于误开的
true 状态(HEAD 为 false, 2026-08-17 配置翻案事故残留)。X 实际动了 3 个变量而非 1 个,
"预期差料必然拖垮组合"的归因强度下降; 但方向仍为负, 不推翻。
如要翻案, 需补跑"仅换料、其余全 v27"的干净复测。

## Y — 行业动量 λ 通道: 通过, 但归因有保留

v27 基线 + `industry_lambda: 0.10` (composite += 0.10 × ind_mom_60 截面 z-score)。
收益端全面改善 (2021 +6.2→+20.2, 2024 -2.9→+6.1, extend 超额 +29.2→+39.2),
风险端轻微劣化 (Sharpe 2.06 vs 2.17, 回撤 -13.6 vs -10.8) —
典型的方向性暴露放大, 不是彩票 (换手仅 4.3→4.7, 无单一年份主导)。

**归因保留 (2026-08-29 定稿前发现)**:
1. ind_mom_60 命中 fold 门槛 2/5 (<3/5), 未进入稳定因子与 extend 核心权重 → extend 区间作用通道纯净 (仅 λ)。
2. 但 fold_3 (2022) / fold_5 (2024) 的核心回测权重里 ind_mom_60 以 ≈-0.18 入选,
   与 λ 的 +0.10 方向相反 → folds 数字是"核心反向 + λ 正向"的混合, 不能单独归功于行业动量。
   fold_3 恰好也是 5 折中唯一下滑的一折 (+10.2→+2.9), 与此自洽。
3. λ=0.10 是预设值, 未扫参 → 无 λ 维度过拟合, 这是结论干净的部分。

## 接线修复 (Y 启动前)

`ind_mom_60` 并入 factor_names 原被锁在 styles.enabled 门控内 — 若 Y 直接跑,
λ>0 但面板缺失, 行业通道静默失效 (假阴性)。提取为 `fold_extra_factor_names`
纯函数 + 5 单测 (commit c824cb2)。λ 透传链路 (fold/extend) 本从原始 config 读取, 未动。

## 运行事故记录

- X run3 "崩溃" 实为系统休眠挂起进程 71 分钟 (非代码/内存问题, --sample 300 全管道通过验证)。
- Y 全程 keep-awake 保活 (SetThreadExecutionState), 1h48m 无中断跑完。
- 本机 `python` 不可用统一用 `py`; 长回测建议继续带保活。

## v28 定稿 (2026-08-29)

- 生产配置: v27 全部不变 + `styles: {enabled: false, industry_lambda: 0.10}`
- 存档: `walkforward_results_v28_industry_lambda010.json` + `surprise_Y_industry010.json`
- 预期管理: folds 增益归因不纯, extend 增益归因干净; 真正决策依据是 extend 模拟考
- 后续候选实验 (未排期): Z1 = 核心池剔除 ind_mom_60 + 纯 λ 通道复跑 (净化 folds 归因);
  Z2 = trend_timing=false 下重跑 X (翻案检查); λ∈{0.05,0.15} 敏感性 (确认 0.10 非孤点)
