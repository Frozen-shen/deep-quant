"""
scripts/active/run_walkforward_backtest.py — Walk-Forward 回测 (动态IC权重)

背景:
  run_corrected_backtest.py 使用 2018-2022 的固定 IC 权重 (P5),
  因子衰减导致 2026 年样本外失败。本脚本在每个调仓日用
  滚动 (trailing) 数据重新估计因子 ICIR, 实现 walk-forward。

方法论 (每个调仓日 T):
  1. 回看 252 个交易日窗口 [T-252, T-21], 每 21 天取一个月度观测点
     (标签为 21 日前瞻收益, 在 T 时刻已全部实现 → 无未来函数)
  2. 每个因子计算截面 Spearman Rank IC
  3. ICIR = mean(IC) / std(IC), 仅保留 |ICIR| > 0.02 的因子
  4. 用动态 ICIR 权重对当日股票打分, 选 top_k
  5. 每 20 个交易日调仓一次

保障:
  - PIT universe: 每个调仓日用 CSI300+CSI1000 月度成分
  - 参数统一: top_k/成本/调仓频率来自 config.yaml
  - 日期守卫: gate.py DateRangeGuard 拦截盲测期访问
  - T+1 执行: 调仓日信号, 次日开盘价成交

用法:
  py scripts/active/run_walkforward_backtest.py                 # Development + TEST
  py scripts/active/run_walkforward_backtest.py --dev-only      # 仅 Development
  py scripts/active/run_walkforward_backtest.py --test-only     # 仅 TEST
  py scripts/active/run_walkforward_backtest.py --sample 200    # 抽样200只快速测试

输出:
  data/ic_validation/walkforward_results.json
"""

import os
import sys
import json
import time
import gc
import argparse
import warnings
import faulthandler
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
faulthandler.enable()  # 段错误/OOM 时打印 Python 栈到 stderr


def _rss_gb() -> float:
    """当前进程工作集内存 (GB); Windows 用 ctypes, 失败返回 -1。"""
    try:
        import ctypes

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong),
                        ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = _PMC()
        pmc.cb = ctypes.sizeof(pmc)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
        return pmc.WorkingSetSize / 1024 ** 3
    except Exception:
        return -1.0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from logger import get_logger
from gate import load_config, check_date_range, DateRangeGuard, GateViolation
from data.pit_universe import get_universe

log = get_logger("walkforward_bt")

IC_DIR = os.path.join(BASE_DIR, "data", "ic_validation")
OUTPUT_PATH = os.path.join(IC_DIR, "walkforward_results.json")
BENCH_PATH = os.path.join(BASE_DIR, "data", "cache", "index_csi1000.parquet")

# ── Walk-forward 参数 ──
TRAIL_DAYS = 252          # 回看窗口 (交易日)

# 调仓周期 (路线C v22, 2026-08-11): 由 config execution.rebalance_days 控制
# (默认 20=月频; 5=周频实验)。LABEL_HORIZON/IC_STEP 与调仓周期联动,
# 保证 IC 估计的持有期与换手频率匹配。
# ★ 语义保持 v15: 标签 = 调仓 + 1 (月频 20+1=21, 与 v15 硬编码 21 完全一致,
#   2026-08-11 曾误改为 max(5, REBALANCE_DAYS)=20 导致 IC 窗口漂移, stable 集
#   从 50/50 一致性变为 40/50 → 已修复)。
def _load_rebalance_days() -> int:
    try:
        cfg = load_config(os.path.join(BASE_DIR, "config.yaml"))
        return int(cfg.get("execution", {}).get("rebalance_days", 20))
    except Exception:
        return 20


REBALANCE_DAYS = _load_rebalance_days()
LABEL_HORIZON = max(5, REBALANCE_DAYS + 1)  # 前瞻收益标签长度 (交易日)
IC_STEP = LABEL_HORIZON                     # IC观测间隔 (交易日)
MIN_ICIR = 0.02           # |ICIR| 入选阈值
MIN_IC_OBS = 6            # 因子最少月度IC观测数
MIN_CROSS_SECTION = 30    # 单次截面IC最少股票数

# ── 分钟因子 (方案C v5, 数据 2022 起) ──
def _minute_dir() -> str:
    """分钟数据目录, 频率由 config.yaml minute_factors.freq 决定 (15/5)."""
    try:
        cfg = load_config(os.path.join(BASE_DIR, "config.yaml"))
        freq = str(cfg.get("minute_factors", {}).get("freq", "15"))
    except Exception:
        freq = "15"
    return os.path.join(BASE_DIR, "data_store", f"minute_{freq}m")


MINUTE_DIR = _minute_dir()
MINUTE_DATA_START = "2022-01-01"  # 分钟数据起点 (该日期前无数据 → NaN 跳过)

# 模块级懒加载缓存: sym -> (按日升序 DataFrame, 唯一日 datetime64 数组) 或 None。
# 每只股票只读一次 parquet (2606 只 × 多次读取会爆内存/慢)。
_minute_df_cache: dict = {}


def _load_partitions() -> dict:
    """从 config.yaml 读取数据分区 (硬编码仅作 fallback)。"""
    try:
        cfg = load_config(os.path.join(BASE_DIR, "config.yaml"))
        dp = cfg["data_partition"]
        return {
            "development": (dp["development"]["start"], dp["development"]["end"]),
            "test": (dp["test"]["start"], dp["test"]["end"]),
        }
    except Exception:
        return {
            "development": ("2026-07-01", "2026-12-31"),
            "test": ("2026-07-01", "2026-12-31"),
        }


PARTITIONS = _load_partitions()


def load_bt_config() -> dict:
    """从 config.yaml 加载执行参数 (与 run_corrected_backtest 统一)。"""
    cfg = load_config(os.path.join(BASE_DIR, "config.yaml"))
    return {
        "rebalance_days": REBALANCE_DAYS,
        "top_k": cfg["execution"]["top_k"],
        "initial_capital": cfg["execution"]["initial_capital"],
        "lot_size": cfg["execution"]["lot_size"],
        "slippage_bps": cfg["execution"]["slippage_bps"],
        "commission_buy": cfg["execution"]["commission_buy"],
        "commission_sell": cfg["execution"]["commission_sell"],
        "vol_source": str(cfg.get("regime", {}).get("vol_source", "daily")),
        # 执行价模式 (方案B v24): open=次日开盘 / vwap=次日VWAP拆单
        "execution_price": str(cfg.get("execution", {}).get("execution_price", "open")),
        "vwap_residual_bps": int(cfg.get("execution", {}).get("vwap_residual_bps", 0)),
        # 组合层换手参数 (2026-08-16: config portfolio 段 bt_* 键覆盖,
        # 缺省保持原硬编码值, 供换手压降实验用)
        "n_drop": int(cfg.get("portfolio", {}).get("bt_n_drop", 10)),
        "hold_thresh": int(cfg.get("portfolio", {}).get("bt_hold_thresh", 0)),
        "sell_rank_buffer": int(cfg.get("portfolio", {}).get("bt_sell_rank_buffer", 3)),
        "buy_confirm_days": int(cfg.get("portfolio", {}).get("bt_buy_confirm_days", 1)),
        "cost_threshold": float(cfg.get("portfolio", {}).get("bt_cost_threshold", 0.08)),
        # 风格预算 (v29): 同家族因子 |权重| 之和上限
        "style_cap": float(cfg.get("fold", {}).get("style_cap", 0.4)),
    }


# ═══════════════════════════════════════════════════════════
#  面板构建
# ═══════════════════════════════════════════════════════════

def build_calendar(all_data: dict, min_coverage: int = 100) -> list:
    """全局交易日历 (所有股票日期的并集, 过滤后排序)。

    ★ 2026-08-15: 双重过滤脏日期 —
    ① 周末 (A股不存在周末交易);
    ② 覆盖 <min_coverage 只的日期: 春节等假期只有退市股有脏行
    (如 2026-02-19/20/23 仅 000540/002450 有数据), 正常交易日 4900+。
    """
    from collections import Counter
    date_counter = Counter()
    for df in all_data.values():
        dts = pd.to_datetime(df["date"]).dt.normalize().tolist()
        date_counter.update(d for d in dts if d.weekday() < 5)
    return sorted(d for d, n in date_counter.items() if n >= min_coverage)


def turnover_period_cap(monthly_cap: float, rebalance_days: int) -> float:
    """期换手上限。

    ★ 2026-08-15 修复 (选项A): 月频 (>=20 交易日) 的"期"≈一个月,
    期上限直接用月上限, 不再乘 rebalance_days/21 自我压线 — 旧公式对
    月频(20)给出 0.5×20/21≈47.6%, 与策略正常换手 50% 恰好冲突, 导致
    每个调仓日全部被换手约束跳过 (2026-08-15 的 folds 回测只交易了
    建仓日一天)。短周期 (周频等) 仍按天数比例缩放, 保持公平对比。
    """
    if rebalance_days >= 20:
        return monthly_cap
    return monthly_cap * (rebalance_days / 21.0)


def safe_mark_to_market(equity_fn, positions: dict, close_prices: dict,
                        prev_equity: float) -> float:
    """防御性计价: 有持仓但当日无任何有效收盘价时沿用前值。

    ★ 2026-08-15 修复: 原逻辑 close_prices 为空时持仓按 0 计价,
    净值塌陷为纯现金 (日历污染日)。空仓时仍正常计价 (cash 场景)。
    """
    if not close_prices and positions:
        return prev_equity
    return equity_fn(close_prices)


def build_close_panel(all_data: dict, calendar: list) -> pd.DataFrame:
    """收盘价面板: index=日历, columns=股票代码。不交易日为 NaN (不前填)。"""
    panel = pd.DataFrame(
        {sym: df.set_index(pd.to_datetime(df["date"]))["close"]
         for sym, df in all_data.items()}
    )
    return panel.reindex(pd.DatetimeIndex(calendar))


_industry_map_cache: dict | None = None


def _load_industry_map() -> dict:
    """加载行业映射 {6位代码: industry} (新浪行业快照, 键去 sh/sz 前缀)。

    近似 PIT: 行业分类为最新快照 (换行业股票占比极小, 影响可接受)。
    模块级缓存 (调仓日重复调用避免反复读盘)。
    """
    global _industry_map_cache
    if _industry_map_cache is not None:
        return _industry_map_cache
    path = os.path.join(BASE_DIR, "data_store", "aux_industry", "industry_map.parquet")
    if not os.path.exists(path):
        _industry_map_cache = {}
        return _industry_map_cache
    try:
        df = pd.read_parquet(path)
        codes = (df["code"].astype(str)
                 .str.replace("sh", "", regex=False)
                 .str.replace("sz", "", regex=False))
        _industry_map_cache = dict(zip(codes, df["industry"].astype(str)))
    except Exception:
        _industry_map_cache = {}
    return _industry_map_cache


def _industry_cap_weights(weights: dict, industry_map: dict,
                          max_industry_pct: float) -> dict:
    """行业权重封顶: 同行业权重之和超上限时按比例缩减。

    ★ 2026-08-16 (v28): 行业中性从"记日志"改为"实际执行"。
    无行业映射的股票不参与约束 (保留原权重)。
    """
    if not weights or max_industry_pct <= 0:
        return weights
    out = dict(weights)
    by_ind: dict = {}
    for s, w in out.items():
        ind = industry_map.get(s)
        if ind:
            by_ind.setdefault(ind, []).append(s)
    for syms in by_ind.values():
        total = sum(out[s] for s in syms)
        if total > max_industry_pct:
            scale = max_industry_pct / total
            for s in syms:
                out[s] *= scale
    return out


def _style_budget_weights(weights: dict, style_cap: float = 0.4) -> dict:
    """风格预算: 同因子家族 (名称去 `_Nd` 后缀) 的 |权重| 之和超 style_cap
    时按比例缩减, 符号保留。

    ★ 2026-08-16 (v29): 防一族同质因子 (如 amihud_*/volatility_*) 垄断
    权重 — 2020 年小盘风格逆风失效的根因。
    """
    import re
    if not weights or style_cap <= 0:
        return weights
    out = dict(weights)
    fam: dict = {}
    for f in out:
        fam.setdefault(re.sub(r'_\d+d$', '', f), []).append(f)
    for fns in fam.values():
        total = sum(abs(out[f]) for f in fns)
        if total > style_cap:
            scale = style_cap / total
            for f in fns:
                out[f] *= scale
    return out


# ── 多风格 sleeve 配置 (2026-08-16, config styles 段) ──
SLEEVE_BUDGET_ORDER = ("momentum", "growth")


def load_styles_config(config: dict) -> dict | None:
    """styles 段: enabled=false/缺失 → None (行为与 v27 完全一致)。"""
    cfg = config.get("styles") or {}
    if not cfg.get("enabled"):
        return None
    budgets = cfg.get("budgets") or {}
    for name, raw in budgets.items():
        try:
            b = float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"styles.budgets.{name} 必须为数值且在 [0,1], got {raw!r}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"styles.budgets.{name} 必须在 [0,1], got {b}")
    for name in (cfg.get("sleeves") or {}):
        if name not in budgets:
            raise ValueError(f"styles.budgets 缺少 sleeve '{name}' 的预算")
    if sum(float(v) for v in budgets.values()) > 1.0:
        raise ValueError("styles.budgets 合计不得超过 1.0 (core 隐含 1-Σ)")
    for name, scfg in (cfg.get("sleeves") or {}).items():
        if not scfg.get("factors"):
            raise ValueError(f"styles.sleeves.{name}.factors 为空")
    return cfg


def assert_sleeve_mode_allowed(styles_cfg, args_folds: bool, args_folds_only: bool) -> None:
    """sleeve 启用时禁止消耗一次性 TEST: folds 且非 folds-only → RuntimeError。"""
    if styles_cfg and args_folds and not args_folds_only:
        raise RuntimeError("多风格 sleeve 模式仅支持 --folds-only + --extend-val")


def split_sleeve_factors(factor_names: list, styles_cfg: dict):
    """核心池 = 全部因子 - sleeve 因子; 返回 (core_names, {name: [factors]})。

    接受完整 config (含 styles 键) 或 styles 子段 (load_styles_config 返回值)。
    """
    if "sleeves" not in styles_cfg:
        styles_cfg = styles_cfg.get("styles") or {}
    sleeves = {}
    sleeve_set = set()
    for name, scfg in (styles_cfg.get("sleeves") or {}).items():
        fs = [f for f in scfg.get("factors", []) if f in factor_names]
        sleeves[name] = fs
        sleeve_set.update(fs)
    core = [f for f in factor_names if f not in sleeve_set]
    return core, sleeves


def fold_extra_factor_names(styles_cfg: dict) -> set:
    """fold 模式下需额外并入 factor_names 的因子名集合 (styles 段驱动)。

    - enabled: 并入全部 sleeve 因子 (precompute 对该列无数据自动跳过,
      面板由 merge_surprise_panels 补)
    - industry_lambda>0: 并入 ind_mom_60 —— 独立于 sleeve 开关。
      行业 λ 是叠加通道而非 sleeve 分池, Y 实验要求 "v27 基线
      (styles.enabled=false) + 行业通道" 的单变量归因, 禁用时仍须生效。
    """
    styles_cfg = styles_cfg or {}
    extra = set()
    if styles_cfg.get("enabled"):
        for scfg in (styles_cfg.get("sleeves") or {}).values():
            extra |= set(scfg.get("factors") or [])
    try:
        lam = float(styles_cfg.get("industry_lambda", 0.0) or 0.0)
    except (TypeError, ValueError):
        lam = 0.0
    if lam > 0:
        extra.add("ind_mom_60")
    return extra


def parse_budget_combos(s: str) -> list:
    """'0.25/0.15,0.2/0.2' → [[('momentum',0.25),('growth',0.15)], ...]"""
    combos = []
    for part in s.split(","):
        vals = [p.strip() for p in part.split("/")]
        if len(vals) != len(SLEEVE_BUDGET_ORDER):
            raise ValueError(
                f"预算组合需 {len(SLEEVE_BUDGET_ORDER)} 个值 (mom/growth): '{part}'")
        combo = [(SLEEVE_BUDGET_ORDER[i], float(vals[i]))
                 for i in range(len(SLEEVE_BUDGET_ORDER))]
        if any(not 0.0 <= v <= 1.0 for _, v in combo):
            raise ValueError(f"预算值须在 [0,1]: '{part}'")
        if sum(v for _, v in combo) > 1.0:
            raise ValueError(f"预算合计不得超过 1.0: '{part}'")
        combos.append(combo)
    if not combos:
        raise ValueError("预算组合列表不能为空")
    return combos


def _industry_neutralize(df: pd.DataFrame, industry_map: dict) -> pd.DataFrame:
    """行业截面中性化: 每个日期, 行业内 z-score (组内标准化)。

    未覆盖股票 (无行业映射) 保留原值, 不参与任何行业组。
    """
    ind_codes: dict = {}
    for code, ind in industry_map.items():
        ind_codes.setdefault(ind, []).append(code)

    out = df.copy()
    for ind, codes in ind_codes.items():
        cols = [c for c in codes if c in out.columns]
        if len(cols) < 5:
            continue
        sub = out[cols].to_numpy(dtype=np.float64)
        mu = np.nanmean(sub, axis=1, keepdims=True)
        sd = np.nanstd(sub, axis=1, keepdims=True)
        sd = np.where(sd < 1e-12, np.nan, sd)
        normed = (sub - mu) / sd
        # 显式构造 DataFrame 赋值 (避免 loc 2D 赋值的列序歧义)
        out[cols] = pd.DataFrame(normed.astype(np.float32),
                                 index=out.index, columns=cols)
    return out


def neutralize_factor(df: pd.DataFrame, k: float = 3.0,
                      industry_map: dict | None = None) -> pd.DataFrame:
    """
    前置中性化: MAD 去极值 + z-score 标准化 (逐列/逐因子)。
    处理 NaN (保留为 NaN, 不参与统计)。

    industry_map 提供时: z-score 步骤替换为行业截面中性化
    (行业组内 z-score), 用于消除行业风格暴露 (v8.1)。
    """
    # 保持输入 dtype (float32): 全面板 astype(float64) 内存翻倍 (~9.4GB→~19GB) 会 OOM。
    # 计算阶段逐列提升为 float64 (单列很小), 写回时显式降回 float32。
    out = df.copy()
    for col in out.columns:
        vals = out[col]
        m = vals.notna()
        if m.sum() < 10:
            continue
        x = vals[m].to_numpy(dtype=np.float64)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad < 1e-12:
            mad = np.std(x)
        if mad < 1e-12:
            continue
        # MAD 去极值: |x - med| > k * 1.4826 * mad → 截断
        limit = k * 1.4826 * mad
        x = np.clip(x, med - limit, med + limit)
        # pandas 2.x 下 float64 数组直接写入 float32 列会抛 LossySetitemError,
        # 必须显式 .astype(np.float32) (数值差异 max ~5.7e-8, 可接受)
        out.loc[m, col] = x.astype(np.float32)

    if industry_map:
        # 行业截面中性化 (替换原时序 z-score)
        return _industry_neutralize(out, industry_map)

    # 原逻辑: 逐股票时序 z-score
    for col in out.columns:
        vals = out[col]
        m = vals.notna()
        if m.sum() < 10:
            continue
        x = vals[m].to_numpy(dtype=np.float64)
        mu, sd = np.mean(x), np.std(x)
        if sd < 1e-12:
            continue
        out.loc[m, col] = ((x - mu) / sd).astype(np.float32)
    return out


def apply_portfolio_constraints(scores: dict, constraints: dict) -> dict:
    """
    组合后置约束: 单票仓位上限 (等权分仓目标权重)。

    输入: 候选股票 dict (仅键有意义, 值可为分数或 1.0)。
    输出: 每只股票的目标仓位权重 — 等权 1/n 天然满足 max_single_pct
          时返回等权; 等权超过上限时每只缩放到上限 (剩余仓位留现金)。

    注意: SimpleBacktest.execute 按等权分配现金且不支持个股权重, 本函数
    作为约束计算的基准实现 (测试钉住此语义); run_backtest 中的约束以
    检查+日志落地, 实际买入等权受 ranker top_k 天然限制。
    """
    if not scores:
        return scores
    n = len(scores)
    ew = 1.0 / n
    max_single = float(constraints.get("max_single_pct", 0.05))
    if ew <= max_single:
        return {k: ew for k in scores}
    log.warning("组合约束: 等权 %.1f%% > 单票上限 %.1f%%, %d 只均缩放到上限 "
                "(剩余仓位留现金)", ew * 100, max_single * 100, n)
    return {k: max_single for k in scores}


def _cap_single_weights(weights: dict, max_single_pct: float) -> dict:
    """个股权重封顶: 超限个股缩到上限, 其余保持不变 (不重归一化, 剩余留现金)。

    ★ 2026-08-16 (v27): inv_vol/risk_parity 已给出相对权重时, 约束只做
    封顶, 不破坏相对比例。
    """
    if not weights or max_single_pct <= 0:
        return weights
    return {k: (min(v, max_single_pct) if v > 0 else v)
            for k, v in weights.items()}


def precompute_factor_panels(all_data: dict, factor_names: list,
                             needed_dates: list,
                             include_fundamental: bool = False,
                             include_aux: bool = False,
                             include_minute: bool = False,
                             minute_lookback: int = 20,
                             neutralize_enabled: bool = False,
                             neutralize_k: float = 3.0,
                             industry_map: dict | None = None) -> dict:
    """
    预计算因子并裁剪到所需日期, 构建 {factor: DataFrame(日期×股票)} 面板。

    内存优化 (方案C v4):
      - float32 存储 (因子值精度足够, 内存减半)
      - 单次全因子计算 (每只股票 compute_factors 只调用一次)
      - 面板构建完成后立即释放 per_stock 中间数据

    include_fundamental=True: 额外合并 fund_* 基本面因子 (PIT-safe)。
      基本面因子由本函数单独计算 (factor_scorer 无法输出非 DSL 的 fund_* 列),
      每只股票只读一次财报缓存, 用 searchsorted 做逐日期 PIT 最近可用查找。

    include_minute=True: 额外合并 min_* 分钟因子 (PIT-safe, 方案C v5)。
      分钟数据自 2022-01 起 (fold 3-5 验证期可用), 按调仓日采样计算 +
      前向填充, 详见 _merge_minute_panels。

    neutralize_enabled=True: 返回前对每个因子面板统一做前置中性化
      (MAD 去极值 + z-score, 逐因子独立), 下游 IC/打分全部消费中性化面板。
    """
    from factor_scorer import FactorScorer
    scorer = FactorScorer.from_preset("full_auto")

    needed_set = set(pd.DatetimeIndex(needed_dates))
    symbols = sorted(all_data.keys())
    idx = pd.DatetimeIndex(needed_dates)
    n_stocks = len(symbols)

    t0 = time.time()
    per_stock = {}   # sym -> DataFrame(needed_dates × factors, float32)
    n_ok = 0
    for i in range(0, len(symbols), 200):
        batch = symbols[i:i + 200]
        for sym in batch:
            df = all_data[sym]
            try:
                full = scorer.compute_factors(df)
                if "date" not in full.columns:
                    full["date"] = df["date"].values
                full["date"] = pd.to_datetime(full["date"])
                sub = full[full["date"].isin(needed_set)]
                if len(sub) == 0:
                    continue
                keep = [c for c in factor_names if c in sub.columns]
                sdf = sub.set_index("date")[keep].astype(np.float32)
                if sdf.shape[1] > 0:
                    per_stock[sym] = sdf
                    n_ok += 1
            except Exception:
                continue
        done = min(i + 200, len(symbols))
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(symbols) - done) / rate if rate > 0 else 0
        log.info("  因子预计算: %d/%d (有效 %d, %.0fs, 剩余~%.0fs)",
                 done, len(symbols), n_ok, elapsed, eta)

    log.info("  构建因子面板 (%d 因子 × %d 日期 × %d 股票, float32)...",
             len(factor_names), len(needed_dates), n_ok)

    # 逐因子面板: dict 收集 → pandas 原生对齐 (C 实现, 快)
    # 内存峰值 = per_stock (9.4GB) + panels (9.4GB) ≈ 19GB < 34GB ✅
    panels = {}
    try:
        for fi, fn in enumerate(factor_names):
            cols = {sym: sdf[fn] for sym, sdf in per_stock.items()
                    if fn in sdf.columns}
            if cols:
                # 对齐到全市场股票列 (与分钟/基本面/aux面板同处理): 个别
                # 股票因子计算失败会缺席该面板 → 必须 reindex 统一列集,
                # 否则 compute_icir_weights 的 column_stack 因列数不一致崩溃
                # (2026-08-13 实测: 4946 vs 4998)。
                panels[fn] = pd.DataFrame(
                    cols, index=idx, dtype=np.float32
                ).reindex(columns=sorted(all_data.keys()))
            if (fi + 1) % 40 == 0 or fi == len(factor_names) - 1:
                log.info("  面板: %d/%d 因子", fi + 1, len(factor_names))
    finally:
        per_stock.clear()
        import gc
        gc.collect()

    # ── 基本面因子合并 (PIT-safe, 由本函数单独计算, 不依赖 scorer) ──
    if include_fundamental:
        t1 = time.time()
        n_fund = _merge_fundamental_panels(panels, all_data, factor_names, idx)
        log.info("  基本面因子面板: %d 个 (%.0fs)", n_fund, time.time() - t1)

    # ── 辅助数据因子合并 (v8, PIT-safe, 由 aux_factors.py 计算) ──
    if include_aux:
        t1a = time.time()
        n_aux = _merge_aux_panels(panels, factor_names, idx, symbols)
        log.info("  辅助数据因子面板: %d 个 (%.0fs)", n_aux, time.time() - t1a)

    # ── 分钟因子合并 (PIT-safe, 方案C v5, 数据2022起) ──
    if include_minute:
        t1m = time.time()
        n_min = _merge_minute_panels(panels, all_data, factor_names, idx,
                                     lookback=minute_lookback)
        log.info("  分钟因子面板: %d 个 (%.0fs)", n_min, time.time() - t1m)

    # ── 前置中性化 (MAD去极值 + z-score, 逐因子) ──
    if neutralize_enabled:
        t2 = time.time()
        for fn in list(panels.keys()):
            panels[fn] = neutralize_factor(panels[fn], k=neutralize_k,
                                           industry_map=industry_map)
        log.info("  中性化完成: %d 因子 (MAD去极值 k=%.1f, 行业中性化=%s, %.0fs)",
                 len(panels), neutralize_k,
                 "开" if industry_map else "关", time.time() - t2)

    log.info("  面板就绪: %d 因子 × %d 日期", len(panels), len(needed_dates))
    return panels


# ═══════════════════════════════════════════════════════════
#  基本面因子面板 (PIT-safe)
# ═══════════════════════════════════════════════════════════

# 价格类因子: fund_因子 -> (财报基数列, 变换模式)
#   div_price = 基数值 / PIT价格 (bp/ep/ocf_yield), price_div = PIT价格 / 基数值 (pb)
_FUND_PRICE_MAP = {
    "fund_bp": ("bvps", "div_price"),
    "fund_ep": ("eps_ttm", "div_price"),
    "fund_pb": ("bvps", "price_div"),
    "fund_ocf_yield": ("ocf_ps", "div_price"),
}


def _fund_report_factors(fin: pd.DataFrame) -> pd.DataFrame | None:
    """
    从单只股票的财报序列 (data/fundamental.py fetch_financials 输出) 预计算
    每期报告的 13 个基本面因子值 (不含价格类分母, 价格在合并时按日 PIT 对齐)。

    Returns: DataFrame index=报告期(日期), columns=因子键(无 fund_ 前缀), 或 None。
    """
    if fin is None or len(fin) == 0 or "日期" not in fin.columns:
        return None
    f = fin.copy()
    f["日期"] = pd.to_datetime(f["日期"])
    f = f.sort_values("日期").reset_index(drop=True)

    def col(name: str) -> pd.Series | None:
        if name not in f.columns:
            return None
        return pd.to_numeric(f[name], errors="coerce")

    roe = col("净资产收益率(%)")
    eps = col("摊薄每股收益(元)")
    bvps = col("每股净资产_调整前(元)")
    ocf = col("每股经营性现金流(元)")
    pg = col("净利润增长率(%)")
    rg = col("主营业务收入增长率(%)")
    dr = col("资产负债率(%)")
    nm = col("销售净利率(%)")
    ded = col("扣除非经常性损益后的净利润(元)")

    out = pd.DataFrame(index=pd.DatetimeIndex(f["日期"]))
    month = f["日期"].dt.month

    # ROE (按报告期季度数年化, 与 fundamental_fetcher 口径一致)
    if roe is not None:
        mult = month.map({3: 4.0, 6: 2.0, 9: 4 / 3}).fillna(1.0)
        out["roe"] = (roe * mult).to_numpy(dtype=np.float32)
        out["roe_ttm"] = roe.rolling(4, min_periods=4).mean().to_numpy(dtype=np.float32)

    # EPS TTM (最近4季度滚动加总; 摊薄EPS为年内累计口径)
    if eps is not None:
        out["eps_ttm"] = eps.rolling(4, min_periods=4).sum().to_numpy(dtype=np.float32)

    # 扣非净利润同比增速 (同季度对比: groupby(quarter).shift(1) 取去年同季)
    if ded is not None:
        q = f["日期"].dt.quarter
        prev = ded.groupby(q).shift(1)
        g = (ded / prev - 1.0).where(prev > 0)
        out["profit_growth_ded"] = g.to_numpy(dtype=np.float32)

    for key, s in [("profit_growth", pg), ("revenue_growth", rg),
                   ("debt_ratio", dr), ("net_margin", nm),
                   ("ocf_ps", ocf), ("bvps", bvps)]:
        if s is not None:
            out[key] = s.to_numpy(dtype=np.float32)

    # 应计利润 (每股口径: (EPS_TTM - OCF_PS) / BVPS, 与 fundamental_fetcher 一致)
    if eps is not None and ocf is not None and bvps is not None:
        out["accruals"] = ((out["eps_ttm"] - ocf.to_numpy(dtype=np.float32))
                           / bvps.to_numpy(dtype=np.float32))
    return out


def _merge_fundamental_panels(panels: dict, all_data: dict,
                              factor_names: list, idx: pd.DatetimeIndex) -> int:
    """
    将 fund_* 基本面因子按日期 PIT 合并进 panels (原地修改)。

    PIT 规则 (与 data/fundamental.py 一致):
      - 每只股票只读一次财报缓存 (fetch_financials)
      - 报告期 + PIT_LAG_DAYS 之后才可用 (财报公布延迟缓冲, 禁止前视)
      - 每个面板日期取 <= date-PIT_LAG_DAYS 的最新一期财报 (searchsorted)
      - 价格类因子 (bp/ep/pb/ocf_yield) 使用面板日期当日及之前最近收盘价
      - 财报缺失/异常 → 静默跳过该股票

    Returns: 实际合并的 fund_* 因子数。
    """
    from data.fundamental import fetch_financials, PIT_LAG_DAYS
    from factor_library import FUNDAMENTAL_FACTORS

    fund_names = [fn for fn in factor_names if fn in FUNDAMENTAL_FACTORS]
    if not fund_names:
        return 0
    fund_keys = {fn: fn.replace("fund_", "") for fn in fund_names}

    dates_arr = idx.to_numpy(dtype="datetime64[ns]")
    lag = np.timedelta64(PIT_LAG_DAYS, "D")

    # searchsorted 要求日期升序; 面板日期非有序时先排序再还原
    if np.all(dates_arr[1:] >= dates_arr[:-1]):
        sorted_dates = dates_arr
        inv = None
    else:
        order = np.argsort(dates_arr, kind="stable")
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        sorted_dates = dates_arr[order]

    cols = {fn: {} for fn in fund_names}
    for sym in all_data:
        try:
            vals = _fund_report_factors(fetch_financials(sym))
            if vals is None or len(vals) == 0:
                continue
            rdates = vals.index.to_numpy(dtype="datetime64[ns]")
            # PIT 可用日期: 报告期 + 财报公布延迟缓冲
            avail = rdates + lag
            pos = np.searchsorted(avail, sorted_dates, side="right") - 1
            valid = pos >= 0
            if not valid.any():
                continue

            # PIT 收盘价: 面板日期当日及之前最近收盘价
            px = all_data[sym]
            # T3 审查修复: searchsorted 要求价格日期升序 — 缓存数据乱序时
            # 先排序, 防止静默产出错误的 PIT 价格 (只读副本, 不改动缓存)
            px_dates = pd.to_datetime(px["date"])
            if not px_dates.is_monotonic_increasing:
                px = px.sort_values("date").reset_index(drop=True)
                px_dates = pd.to_datetime(px["date"])
            pdates = px_dates.to_numpy(dtype="datetime64[ns]")
            closes = px["close"].to_numpy(dtype=np.float64)
            ppos = np.searchsorted(pdates, sorted_dates, side="right") - 1
            pvalid = (ppos >= 0) & (closes[np.clip(ppos, 0, len(closes) - 1)] > 0)
            price = closes[np.clip(ppos, 0, len(closes) - 1)]

            v = vals.to_numpy(dtype=np.float64)  # (n_reports, n_keys)
            rp = np.clip(pos, 0, len(v) - 1)
            for fn in fund_names:
                base_key, mode = _FUND_PRICE_MAP.get(
                    fn, (fund_keys[fn], None))
                if base_key not in vals.columns:
                    continue
                kpos = vals.columns.get_loc(base_key)
                values = np.where(valid, v[rp, kpos], np.nan)
                if mode == "div_price":
                    m = pvalid & (price > 0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        values = np.where(m, values / price, np.nan)
                elif mode == "price_div":
                    m = pvalid & (price > 0) & (values > 0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        values = np.where(m, price / values, np.nan)
                if inv is not None:
                    values = values[inv]
                cols[fn][sym] = values.astype(np.float32)
        except Exception:
            continue

    n_added = 0
    # 对齐到全市场股票列 (与分钟面板同处理): 基本面面板只含有数据的股票,
    # reindex 到全部股票 (缺失填 NaN), 保证 column_stack 列数一致。
    all_syms = sorted(all_data.keys())
    for fn in fund_names:
        if cols[fn]:
            df = pd.DataFrame(cols[fn], index=idx, dtype=np.float32)
            panels[fn] = df.reindex(columns=all_syms)
            n_added += 1
    return n_added


# ═══════════════════════════════════════════════════════════
#  分钟因子面板 (PIT-safe, 方案C v5)
# ═══════════════════════════════════════════════════════════

def _load_minute_df(sym: str):
    """
    懒加载单只股票的分钟数据 (data_store/minute_15m/{sym}.parquet)。

    模块级缓存: 每只股票只读一次 parquet (2606 只 × 多次读取会爆内存/慢)。
    返回 (按日升序 DataFrame, 唯一日 datetime64 数组) 或 None (无文件/异常)。
    """
    if sym in _minute_df_cache:
        return _minute_df_cache[sym]
    df = None
    path = os.path.join(MINUTE_DIR, f"{sym}.parquet")
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            if "day" not in df.columns or len(df) == 0:
                df = None
        except Exception:
            df = None
    if df is not None:
        # searchsorted 要求日期升序; 乱序缓存只排序一次
        if not df["day"].is_monotonic_increasing:
            df = df.sort_values("day").reset_index(drop=True)
        days_arr = np.unique(df["day"].to_numpy(dtype="datetime64[ns]"))
        _minute_df_cache[sym] = (df, days_arr)
    else:
        _minute_df_cache[sym] = None
    return _minute_df_cache[sym]


def _ffill(arr: np.ndarray) -> np.ndarray:
    """前向填充 NaN (C 速度: maximum.accumulate 定位最近有效值)。"""
    mask = ~np.isnan(arr)
    if not mask.any():
        return arr
    last_valid = np.maximum.accumulate(np.where(mask, np.arange(len(arr)), -1))
    out = arr.copy()
    valid = last_valid >= 0
    out[valid] = arr[last_valid[valid]]
    return out


# ── VWAP 面板 (方案B v24, 2026-08-11) ──
# 日 VWAP = (未复权 amount/volume × 单位修正) × 复权因子, 对齐回测日线复权基准。
# 数据源: data_cache/unadjusted/{sym}.parquet (真实量额 + 未复权 close) +
#         data_store/{sym}.parquet (复权 close, 计算因子 close_adj/close_u)。
# ★ 单位自动检测 (2026-08-11 实测发现): 历史 fetch 混用 baostock/腾讯源,
#   volume 单位不一致 (部分股票=股, 部分=手, 如 000001=手/601318=股)。
#   用中位数 amt/vol vs close 量级判定: 100倍 → 手, 否则股。
# 验证: 10/10 股票 VWAP 100% 落在复权 [low,high] 区间 (全历史 2018 起)。
_vwap_cache: dict = {}


def _build_vwap_panel(symbols: list) -> dict:
    """
    构建日 VWAP 面板 {symbol: DataFrame(date 索引 × vwap 列)}。

    每只股票懒加载一次 (模块级缓存 _vwap_cache)。全历史覆盖
    (2018 起, 无 fold 1-2 回退问题)。单位检测失败/数据缺失 → 跳过
    (execute 回退 open)。
    """
    global _vwap_cache
    if _vwap_cache:
        return _vwap_cache
    u_dir = os.path.join(BASE_DIR, "data_cache", "unadjusted")
    adj_dir = os.path.join(BASE_DIR, "data_store")
    n_ok = 0
    for i, sym in enumerate(symbols):
        upath = os.path.join(u_dir, f"{sym}.parquet")
        apath = os.path.join(adj_dir, f"{sym}.parquet")
        if not (os.path.exists(upath) and os.path.exists(apath)):
            continue
        try:
            u = pd.read_parquet(upath, columns=["date", "close", "amount", "volume"])
            a = pd.read_parquet(apath, columns=["date", "close"])
            u["date"] = pd.to_datetime(u["date"])
            a["date"] = pd.to_datetime(a["date"])
            if len(u) == 0 or len(a) == 0:
                continue
            m = u.merge(a, on="date", suffixes=("_u", "_adj"), how="inner")
            m = m.sort_values("date")
            m = m[(m["volume"] > 0) & (m["amount"] > 0) & (m["close_u"] > 0)]
            if len(m) == 0:
                continue
            # 单位检测: 中位数 amt/vol 与 close 同量级 → 股; 100倍 → 手
            per = (m["amount"] / m["volume"]).median()
            med_close = m["close_u"].median()
            if med_close * 50 < per < med_close * 200:
                vol_factor = 100.0  # 手
            else:
                vol_factor = 1.0    # 股
            m["vwap"] = (m["amount"] / (m["volume"] * vol_factor)) * \
                        (m["close_adj"] / m["close_u"])
            m = m[m["vwap"].notna() & (m["vwap"] > 0)]
            if len(m) > 0:
                _vwap_cache[sym] = m[["date", "vwap"]].set_index("date").sort_index()
                n_ok += 1
        except Exception:
            continue
        if (i + 1) % 500 == 0:
            log.info("  VWAP面板: %d/%d 只 (%d 有效)", i + 1, len(symbols), n_ok)
    log.info("  VWAP面板: %d/%d 只 (未复权×复权因子, 单位自动检测)", n_ok, len(symbols))
    return _vwap_cache


def _merge_aux_panels(panels: dict, factor_names: list,
                      idx: pd.DatetimeIndex, symbols: list) -> int:
    """将 aux_* 辅助数据因子按日期 PIT 合并进 panels (原地修改, v8)。

    PIT 规则 (与 _merge_fundamental_panels 同风格):
      - 因子面板由 aux_factors 构建 (归一化已完成)
      - 两融/龙虎榜/大宗当日盘后披露 → 面板日期取 <= 回测日期的最近值
      - 列对齐到主面板股票集 (aux 源含退市股, 需裁剪)

    Returns: 实际合并的 aux_* 因子数。
    """
    from factor_library import AUX_FACTORS
    from aux_factors import (build_margin_panels, build_lockup_panels,
                             build_lhb_panels, build_dzjy_panels,
                             build_gdhs_panels, build_ggcg_panels,
                             build_fhps_panels, build_yjkb_panels,
                             _mktcap_panel)

    aux_names = [fn for fn in factor_names if fn in AUX_FACTORS]
    if not aux_names:
        return 0

    mp = build_margin_panels()  # {fn: DataFrame(date × 两融标的)}
    mktcap = _mktcap_panel()
    mp.update(build_lockup_panels(mktcap))  # 解禁压力 (共享流通市值面板)
    mp.update(build_lhb_panels(mktcap))     # 龙虎榜
    mp.update(build_dzjy_panels())          # 大宗交易
    mp.update(build_gdhs_panels())          # 股东户数 (筹码集中度)
    mp.update(build_ggcg_panels())          # 股东增减持事件
    mp.update(build_fhps_panels())          # 分红送配 (高送转/红利)
    mp.update(build_yjkb_panels())          # 业绩快报
    if not mp:
        return 0

    dates_arr = idx.to_numpy(dtype="datetime64[ns]")
    n_merged = 0
    for fn in aux_names:
        if fn not in mp:
            continue
        src = mp[fn]
        src_dates = src.index.to_numpy(dtype="datetime64[ns]")
        if not np.all(src_dates[1:] >= src_dates[:-1]):
            order = np.argsort(src_dates, kind="stable")
            src_dates = src_dates[order]
            src = src.iloc[order]
        # PIT: 每个回测日期取 <= 该日期的最近披露日
        pos = np.searchsorted(src_dates, dates_arr, side="right") - 1
        valid = pos >= 0
        # 对齐到主面板股票集 (aux 源含退市股 → 裁剪; 缺失股票 → NaN 列)
        cols = {}
        for sym in symbols:
            if sym not in src.columns:
                cols[sym] = np.full(len(dates_arr), np.nan, dtype=np.float64)
                continue
            vals = src[sym].to_numpy(dtype=np.float64)
            rp = np.clip(pos, 0, len(vals) - 1)
            out = np.full(len(dates_arr), np.nan, dtype=np.float64)
            out[valid] = vals[rp[valid]]
            cols[sym] = out
        aligned = pd.DataFrame(cols, index=idx, dtype=np.float32)
        if len(aligned.columns) > 0:
            panels[fn] = aligned
            n_merged += 1
    return n_merged


SURPRISE_FACTORS = ("sue_std", "earn_accel", "pead_20d", "ind_mom_60")


def merge_surprise_panels(panels: dict, factor_names: list,
                          idx: pd.DatetimeIndex, symbols: list,
                          all_data: dict, industry_map: dict) -> int:
    """预期差/行业动量因子面板合并 (2026-08-17, PIT-safe)。

    sue_std/earn_accel 来自季度财报 (公告日生效), pead_20d 事件驱动,
    ind_mom_60 来自日线行业动量。仅合并 factor_names 中含有的因子。
    """
    from earnings_surprise import (sue_panel, earn_accel_panel, pead_panel,
                                   industry_momentum_panel)
    cal = list(idx)
    want = [f for f in SURPRISE_FACTORS if f in factor_names]
    if not want:
        return 0
    n = 0
    if "sue_std" in want:
        panels["sue_std"] = sue_panel(symbols, cal).reindex(index=idx, columns=symbols)
        n += 1
    if "earn_accel" in want:
        panels["earn_accel"] = earn_accel_panel(symbols, cal).reindex(index=idx, columns=symbols)
        n += 1
    if "pead_20d" in want:
        panels["pead_20d"] = pead_panel(symbols, all_data, cal).reindex(index=idx, columns=symbols)
        n += 1
    if "ind_mom_60" in want:
        panels["ind_mom_60"] = industry_momentum_panel(
            symbols, all_data, industry_map, cal).reindex(index=idx, columns=symbols)
        n += 1
    return n


def _merge_minute_panels(panels: dict, all_data: dict,
                         factor_names: list, idx: pd.DatetimeIndex,
                         lookback: int = 20) -> int:
    """
    将 min_* 分钟因子按调仓日采样合并进 panels (PIT-safe, 原地修改)。

    PIT 规则 (与 _merge_fundamental_panels 同风格):
      - compute_minute_factors 的 as_of_date 参数天然是截止日 → 无前视
      - 分钟数据仅 2022-01 起 → 该日期前的面板日期保持 NaN (无数据)
      - 无分钟数据 / 数据不足 / 异常 → 静默跳过该股票

    性能 (方案C): 分钟因子是 lookback 日均值的慢变量, 按 REBALANCE_DAYS
      交易日采样计算 (每月一次, ~55 次/股 而非全面板逐日 ~1100 次/股),
      其余面板日期由最近一次采样值前向填充 (ffill, PIT-safe)。

    Returns: 实际合并的 min_* 因子数。
    """
    from minute_factors import MINUTE_FACTOR_NAMES, compute_minute_factors

    minute_names = [fn for fn in factor_names if fn in MINUTE_FACTOR_NAMES]
    if not minute_names:
        return 0
    if not os.path.isdir(MINUTE_DIR):
        log.warning("  分钟因子: 数据目录缺失, 跳过: %s", MINUTE_DIR)
        return 0

    asof_arr = idx.to_numpy(dtype="datetime64[ns]")
    # searchsorted/ffill 要求日期升序; 面板日期非有序时先排序再还原 (与
    # _merge_fundamental_panels 同模式, 防止静默错位)
    if np.all(asof_arr[1:] >= asof_arr[:-1]):
        sorted_arr = asof_arr
        inv = None
    else:
        order = np.argsort(asof_arr, kind="stable")
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        sorted_arr = asof_arr[order]

    min_start = np.datetime64(MINUTE_DATA_START, "ns")
    grid = np.array([d for d in sorted_arr if d >= min_start])
    if len(grid) == 0:
        return 0
    sample_dates = grid[::REBALANCE_DAYS]
    sample_pos = np.searchsorted(sorted_arr, sample_dates, side="left")

    t0 = time.time()
    cols = {fn: {} for fn in minute_names}
    n_ok = 0
    for i, sym in enumerate(all_data):
        try:
            loaded = _load_minute_df(sym)
            if loaded is None:
                continue
            mdf, days_arr = loaded
            dvals = mdf["day"].to_numpy(dtype="datetime64[ns]")
            per_factor = {fn: [] for fn in minute_names}
            for asof in sample_dates:
                # PIT 窗口: 最近 lookback 个交易日 (<= asof) — 与传全量 df 结果一致
                p = int(np.searchsorted(days_arr, asof, side="right"))
                if p == 0:
                    # 该股分钟数据尚未开始 (如新股), 无历史 → NaN
                    for fn in minute_names:
                        per_factor[fn].append(np.nan)
                    continue
                r1 = int(np.searchsorted(dvals, asof, side="right"))
                r0 = int(np.searchsorted(
                    dvals, days_arr[max(0, p - lookback)], side="left"))
                w = mdf.iloc[r0:r1]
                res = compute_minute_factors(w, pd.Timestamp(asof), lookback)
                for fn in minute_names:
                    per_factor[fn].append(res.get(fn, np.nan))
            for fn in minute_names:
                arr = np.full(len(sorted_arr), np.nan, dtype=np.float32)
                arr[sample_pos] = np.asarray(per_factor[fn], dtype=np.float32)
                arr = _ffill(arr)
                if inv is not None:
                    arr = arr[inv]
                cols[fn][sym] = arr
            n_ok += 1
        except Exception:
            continue
        if (i + 1) % 500 == 0 or i == len(all_data) - 1:
            log.info("  分钟因子: %d/%d 只, 有效 %d (%.0fs)",
                     i + 1, len(all_data), n_ok, time.time() - t0)

    n_added = 0
    # 对齐到全市场股票列: 分钟面板只含有数据的股票,
    # 必须 reindex 到全部股票 (缺失填 NaN), 否则 compute_icir_weights
    # 的 column_stack 会因列数不一致崩溃 (5005 vs 3005)。
    all_syms = sorted(all_data.keys())
    for fn in minute_names:
        if cols[fn]:
            df = pd.DataFrame(cols[fn], index=idx, dtype=np.float32)
            panels[fn] = df.reindex(columns=all_syms)
            n_added += 1
    return n_added


def validate_minute_factors(factor_panels: dict, close_panel: pd.DataFrame,
                            calendar: list, cal_idx: dict, factor_names: list,
                            train_folds: list, min_icir: float = 0.3) -> dict:
    """
    方案B: 分钟因子独立验证层。

    用 fold 4-5 训练期 (2022+) 独立估计 min_* 因子的 ICIR,
    通过 |ICIR| >= min_icir 的因子作为叠加层权重 (各 fold 中位数)。
    与主通道 (40 稳定因子) 完全隔离, 不参与主筛选。

    Returns: {min_factor: validated_icir}
    """
    from minute_factors import MINUTE_FACTOR_NAMES
    minute_names = [fn for fn in factor_names if fn in MINUTE_FACTOR_NAMES]
    if not minute_names or not train_folds:
        return {}
    log.info("  [probe] validate: %d 分钟因子, folds=%s",
             len(minute_names), train_folds)

    # 对每个 fold 训练期计算 ICIR
    fold_icirs = {fn: [] for fn in minute_names}
    for (ts, te) in train_folds:
        # 用训练期末作为 t_date (固定窗口模式)
        t_date = None
        for d in calendar:
            if pd.Timestamp(ts).date() <= d.date() <= pd.Timestamp(te).date():
                t_date = d
        if t_date is None:
            continue
        weights, ic_stats = compute_icir_weights(
            factor_panels, close_panel, calendar, cal_idx,
            t_date, minute_names, train_start=ts, train_end=te)
        for fn in minute_names:
            st = ic_stats.get(fn)
            if st is not None:
                fold_icirs[fn].append(st["icir"])
            else:
                fold_icirs[fn].append(0.0)

    # 取中位数, 过门槛保留
    result = {}
    for fn in minute_names:
        arr = np.array(fold_icirs[fn])
        if len(arr) == 0:
            continue
        med = float(np.median(arr))
        if abs(med) >= min_icir:
            result[fn] = med
    if result:
        log.info("  分钟叠加层: %d/%d 因子通过验证 |ICIR|>=%.2f: %s",
                 len(result), len(minute_names), min_icir,
                 {k: round(v, 3) for k, v in result.items()})
    return result


# ═══════════════════════════════════════════════════════════
#  动态 IC 权重
# ═══════════════════════════════════════════════════════════

def rank_corr_cols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Spearman 秩相关: x 形状 (n_obs, n_factors), y 形状 (n_obs,) 且无 NaN。
    返回每个因子与 y 的秩相关系数 (NaN 表示无法计算)。
    与 scipy.stats.spearmanr 一致: 在 (因子, y) 联合有效观测上秩化。
    """
    from scipy.stats import rankdata
    n_obs, n_f = x.shape
    out = np.full(n_f, np.nan)
    for fi in range(n_f):
        col = x[:, fi]
        mask = ~np.isnan(col)
        if mask.sum() < MIN_CROSS_SECTION:
            continue
        cv, yv = col[mask], y[mask]
        if (cv.max() - cv.min()) < 1e-12 or (yv.max() - yv.min()) < 1e-12:
            continue
        r = np.corrcoef(rankdata(cv), rankdata(yv))[0, 1]
        if not np.isnan(r):
            out[fi] = r
    return out


BENCH_RET_CLIP = 0.5      # 等权基准单日收益截断上限 (±50%, 超过视为脏数据)
BENCH_MIN_COVERAGE = 100  # 日期覆盖股票数下限 (<此数视为假交易日, 剔除)


def _benchmark_daily_ret(all_data: dict, min_coverage: int = BENCH_MIN_COVERAGE):
    """全市场等权日收益序列 (单股单日收益截断 ±50% + 低覆盖日期剔除)。

    ★ 2026-08-15: 退市股脏价格会把等权基准炸飞。
    ① 单股单日收益 clip ±50% (A股真实日涨跌上限 ±30%);
    ② 春节等假期只有 1-2 只退市股有脏行, skipna mean 会把这 2 只当
    "全市场" → 覆盖 < min_coverage 的日期整体剔除 (正常交易日 4900+)。
    """
    daily_rets = {}
    for sym, df in all_data.items():
        s = df.set_index(pd.to_datetime(df["date"]))["close"]
        s = s[s > 0]
        r = s.pct_change()
        daily_rets[sym] = r.clip(-BENCH_RET_CLIP, BENCH_RET_CLIP)
    panel = pd.DataFrame(daily_rets)
    coverage = panel.notna().sum(axis=1)
    panel = panel[coverage >= min_coverage]
    return panel.mean(axis=1, skipna=True)


def _equal_weight_benchmark(start: str, end: str,
                            all_data: dict):
    """
    全市场等权基准 (方案C v4.1): 用回测股票池每日等权收益构建,
    与策略池/流动性过滤天然一致, 避免小盘指数基准错配。

    Returns: pd.Series of close prices (index=date) or None
    """
    try:
        eqw = _benchmark_daily_ret(all_data)
        eqw = eqw[(eqw.index >= pd.Timestamp(start)) &
                  (eqw.index <= pd.Timestamp(end))]
        eqw = eqw.dropna()
        if len(eqw) < 10:
            return None
        # 转价格序列 (从 100 起)
        return 100.0 * (1 + eqw).cumprod()
    except Exception:
        return None


def _nearest_idx(cal_idx: dict, date_str: str):
    """
    Find the nearest calendar index for a requested date.
    (训练期起止日期可能是非交易日, 回退到最近交易日)
    """
    target = pd.Timestamp(date_str)
    if target in cal_idx:
        return cal_idx[target]
    # 找 <= target 的最大交易日
    candidates = [d for d in cal_idx if d <= target]
    if candidates:
        return cal_idx[max(candidates)]
    # target 早于所有交易日 → 取最早交易日
    if cal_idx:
        return cal_idx[min(cal_idx.keys())]
    return None


def compute_icir_weights(factor_panels: dict, close_panel: pd.DataFrame,
                         calendar: list, cal_idx: dict, t_date,
                         factor_names: list,
                         train_start: str | None = None,
                         train_end: str | None = None,
                         universe_fn=None) -> tuple:
    """
    计算因子 ICIR 权重。

    两种窗口模式:
      rolling (默认): 滚动 [T-TRAIL_DAYS, T-LABEL_HORIZON] 窗口, 每个调仓日重估。
      fixed (方案C fold): 传 train_start/train_end, 在固定训练期内估计,
        用于验证期全部调仓日 (避免验证期信息泄漏)。

    窗口: [start_idx, end_idx] (日历索引), 每 IC_STEP 天一个观测点。
    标签: 21 日前瞻收益 close[t+21]/close[t]-1 (在 T 时已全部实现)。

    Returns: (weights: {factor: icir}, ic_stats: {factor: {ic_mean, ic_std, n_obs}})
    """
    t_idx = cal_idx.get(pd.Timestamp(t_date))
    if t_idx is None:
        return {}, {}

    if train_start is not None and train_end is not None:
        # 固定训练窗口 (方案C fold): 训练期内全部观测点
        s_idx = _nearest_idx(cal_idx, train_start)
        e_idx = _nearest_idx(cal_idx, train_end)
        if s_idx is None or e_idx is None:
            return {}, {}
        start_idx = s_idx
        # 严格无泄漏: 训练观测点标签(21日前瞻)必须完全在训练期内
        end_idx = min(e_idx - LABEL_HORIZON, t_idx - LABEL_HORIZON)
        if end_idx <= start_idx:
            return {}, {}
    else:
        # 滚动窗口 (默认)
        start_idx = t_idx - TRAIL_DAYS
        if start_idx < 0:
            start_idx = 0  # 早期数据不足 → 扩展窗口 (expanding)
        end_idx = t_idx - LABEL_HORIZON
    offsets = list(range(start_idx, end_idx + 1, IC_STEP))
    if not offsets:
        return {}, {}
    available = [fn for fn in factor_names if fn in factor_panels]
    log.info("  [probe] ICIR 估计: %s 模式, %d obs × %d 因子 (%s ~ %s)",
             "fixed" if (train_start and train_end) else "rolling",
             len(offsets), len(available),
             calendar[start_idx].date() if start_idx < len(calendar) else "?",
             calendar[end_idx].date() if end_idx < len(calendar) else "?")

    close_vals = close_panel.to_numpy()
    n_dates = len(calendar)
    # 预取每个观测日的因子截面矩阵 (n_stocks, n_factors)
    matrices = []
    labels = []
    for oi in offsets:
        if oi + LABEL_HORIZON >= n_dates or oi < 0:
            continue
        # ★ FIX-C: IC 域 = 回测域 — 每个观测日动态过滤到当时的 universe
        if universe_fn is not None:
            oi_date = str(calendar[oi].date())
            try:
                uni = set(universe_fn(oi_date))
            except Exception:
                uni = None
        else:
            uni = None
        c0 = close_vals[oi, :]
        c1 = close_vals[oi + LABEL_HORIZON, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = c1 / c0 - 1
        valid_px = (c0 > 0) & ~np.isnan(c0) & ~np.isnan(c1)
        if uni is not None:
            col_names = close_panel.columns
            # 只在 universe 内计算 IC
            uidx = [ci for ci, s in enumerate(col_names) if s in uni]
            valid_px = valid_px.copy()
            valid_px[~np.isin(np.arange(len(col_names)), uidx)] = False
        ret = np.where(valid_px, ratio, np.nan)
        if np.sum(~np.isnan(ret)) < MIN_CROSS_SECTION:
            continue
        cols = []
        for fn in available:
            cols.append(factor_panels[fn].loc[calendar[oi]].to_numpy())
        x = np.column_stack(cols)
        # 仅保留有有效标签的行
        valid = ~np.isnan(ret)
        matrices.append(x[valid, :])
        labels.append(ret[valid])

    if not matrices:
        return {}, {}

    ic_records = {fn: [] for fn in available}
    for x, y in zip(matrices, labels):
        ics = rank_corr_cols(x, y)
        for fi, fn in enumerate(available):
            if not np.isnan(ics[fi]):
                ic_records[fn].append(ics[fi])

    weights = {}
    ic_stats = {}
    for fn in available:
        arr = np.array(ic_records[fn])
        if len(arr) < MIN_IC_OBS:
            continue
        mu = float(np.mean(arr))
        sd = float(np.std(arr))
        icir = mu / sd if sd > 1e-9 else 0.0
        ic_stats[fn] = {
            "ic_mean": round(mu, 5),
            "ic_std": round(sd, 5),
            "icir": round(icir, 4),
            "n_obs": int(len(arr)),
        }
        if abs(icir) > MIN_ICIR:
            weights[fn] = icir
    return weights, ic_stats


def _weighted_z_composite(cross: pd.DataFrame, weights: dict) -> np.ndarray:
    """cross (n × 因子) → 因子内 z-score 后按权重加权和, 除以 Σ|w| (长度 n)。"""
    names = [n for n in weights if n in cross.columns]
    vals = cross[names].to_numpy(dtype=np.float64)
    n, m = vals.shape
    w = np.array([weights[n] for n in names], dtype=np.float64)
    comp = np.zeros(n)
    for fi in range(m):
        col = vals[:, fi]
        mask = ~np.isnan(col)
        if mask.sum() < 10:
            continue
        mu = np.nanmean(col)
        sd = np.nanstd(col)
        if sd < 1e-9:
            continue
        z = np.where(mask, (col - mu) / sd, 0.0)
        comp += w[fi] * z
    denom = np.sum(np.abs(w))
    if denom < 1e-9:
        return comp
    return comp / denom


def score_stocks(factor_panels: dict, weights: dict, t_date,
                 sleeve_weights: list | None = None,
                 minute_weights: dict | None = None,
                 minute_lambda: float = 0.3,
                 industry_lambda: float = 0.0) -> dict:
    """ICIR 加权 z-score 打分, 支持多 sleeve 预算混合 (2026-08-16)。

    sleeve_weights: [{"name","weights","budget"}, ...]
      composite = (1-Σbudget)×主分 + Σ budget×sleeve分 (各通道内部分别归一)。
    sleeve_weights=None 时公式与 v27 一致 (float64 计算路径, 与 v27 float32
    的期望差异 ≤1e-7 量级, 排序影响可忽略; 真实 A/B 由实验 A 对照 v27 存档验证)。
    """
    if not weights:
        return {}
    abs_w = np.sum(np.abs(list(weights.values())))
    if abs_w < 1e-9:
        return {}
    factor_names = list(weights.keys())

    cols = {}
    for n in factor_names:
        p = factor_panels[n]
        if t_date in p.index:
            cols[n] = p.loc[t_date]
    if not cols:
        return {}
    cross = pd.DataFrame(cols)

    n_f = cross.shape[1]
    cov = cross.notna().sum(axis=1)
    cross = cross[cov >= n_f * 0.5]
    if len(cross) < 10:
        return {}

    base = _weighted_z_composite(cross, weights)
    core_budget = 1.0
    sleeve_sum = 0.0
    if sleeve_weights:
        for sw in sleeve_weights:
            s_w = sw.get("weights") or {}
            s_budget = float(sw.get("budget", 0.0))
            if not s_w or s_budget <= 0:
                continue
            s_cols = {}
            for n in s_w:
                p = factor_panels.get(n)
                if p is not None and t_date in p.index:
                    s_cols[n] = p.loc[t_date]
            if not s_cols:
                continue
            s_cross = pd.DataFrame(s_cols).reindex(cross.index)
            s_comp = _weighted_z_composite(s_cross, s_w)
            # 无该 sleeve 数据的股票 s_comp=0 → 仅主分生效 (自然降级)
            s_comp = np.nan_to_num(s_comp, nan=0.0)
            sleeve_sum = sleeve_sum + s_budget * s_comp
            core_budget -= s_budget
    composite = core_budget * base + sleeve_sum

    # ── 方案B: 分钟因子叠加层 (语义不变) ──
    if minute_weights:
        m_names = list(minute_weights.keys())
        m_w = np.array([minute_weights[n] for n in m_names])
        m_abs = np.sum(np.abs(m_w))
        if m_abs > 1e-9:
            m_cols = {}
            for n in m_names:
                p = factor_panels.get(n)
                if p is not None and t_date in p.index:
                    m_cols[n] = p.loc[t_date]
            if m_cols:
                m_cross = pd.DataFrame(m_cols).reindex(cross.index)
                m_comp = _weighted_z_composite(m_cross, minute_weights)
                m_comp = np.nan_to_num(m_comp, nan=0.0)
                composite = composite + minute_lambda * m_comp

    # ── 行业动量叠加 (2026-08-17): composite += λ × ind_mom_60 通道分 ──
    if industry_lambda > 0:
        ip = factor_panels.get("ind_mom_60")
        if ip is not None and t_date in ip.index:
            i_vals = ip.loc[t_date].reindex(cross.index).to_numpy(dtype=np.float64)
            mu_i = np.nanmean(i_vals)
            sd_i = np.nanstd(i_vals)
            if sd_i > 1e-9:
                z_i = np.where(~np.isnan(i_vals), (i_vals - mu_i) / sd_i, 0.0)
                composite = composite + industry_lambda * z_i

    return {s: float(v) for s, v in zip(cross.index, composite)
            if not np.isnan(v)}


# ═══════════════════════════════════════════════════════════
#  回测主循环
# ═══════════════════════════════════════════════════════════

def _factor_category(fn: str) -> str:
    """因子类别: momentum / reversal / value / quality / other"""
    if fn.startswith("fund_"):
        if fn in ("fund_bp", "fund_ep", "fund_pb", "fund_sp"):
            return "value"
        if fn in ("fund_roe", "fund_roe_ttm", "fund_net_margin", "fund_accruals"):
            return "quality"
        return "other"
    if any(k in fn for k in ("momentum", "return_")):
        return "momentum"
    if any(k in fn for k in ("vol", "corr", "cord", "amplitude", "skew",
                             "amihud", "turnover", "k_len", "big_", "channel")):
        return "reversal"
    return "other"


def _inv_vol_weights(all_data: dict, buy_list: list, today,
                     lookback: int = 60) -> dict | None:
    """波动率倒数加权 (v9b): w_i ∝ 1/σ_i, σ=过去 lookback 日日收益 std。

    PIT: 只用 <= today 的数据。返回 {sym: weight} (归一化) 或 None。
    """
    vols = {}
    for s in buy_list:
        if s not in all_data:
            continue
        df = all_data[s][all_data[s]["date"] <= today]
        if len(df) < 20:
            continue
        rets = df["close"].pct_change().dropna().tail(lookback)
        if len(rets) < 10:
            continue
        v = float(rets.std())
        if v > 0 and not np.isnan(v):
            vols[s] = v
    if not vols:
        return None
    inv = {s: 1.0 / v for s, v in vols.items()}
    tot = sum(inv.values())
    if tot <= 0:
        return None
    return {s: w / tot for s, w in inv.items()}


def _risk_parity_weights(all_data: dict, buy_list: list, today,
                         lookback: int = 60, shrink: float = 0.5) -> dict | None:
    """风险平价权重 (P0, 2026-08-09): 按协方差风险贡献均等分配。

    简化实现 (小样本友好):
      - 样本协方差 + Ledoit-Wolf 收缩 (shrink 比例)
      - 风险贡献均等: w_i ∝ 1/(ΣΣ w_j σ_ij 的边际贡献) — 用迭代近似:
        权重 ∝ 对角元素倒数 → 迭代 3 次风险平价
    PIT: 只用 <= today 数据。
    """
    rets = {}
    for s in buy_list:
        if s not in all_data:
            continue
        df = all_data[s][all_data[s]["date"] <= today]
        if len(df) < 20:
            continue
        r = df["close"].pct_change().dropna().tail(lookback)
        if len(r) < 10:
            continue
        rets[s] = r.to_numpy(dtype=np.float64)
    if len(rets) < 2:
        return None
    # 对齐长度
    n = min(len(v) for v in rets.values())
    X = np.column_stack([v[-n:] for v in rets.values()])
    syms = list(rets.keys())
    # 样本协方差 + 收缩
    S = np.cov(X, rowvar=False)
    diag = np.diag(S)
    target = np.eye(len(syms)) * np.mean(diag)
    S_shrunk = (1 - shrink) * S + shrink * target
    # 迭代风险平价 (3 次)
    w = 1.0 / np.sqrt(np.diag(S_shrunk))
    w = w / w.sum()
    for _ in range(3):
        port_var = w @ S_shrunk @ w
        mrc = S_shrunk @ w / port_var  # 边际风险贡献
        w = w * (1.0 / np.maximum(mrc, 1e-9))
        w = w / w.sum()
    return {s: float(wi) for s, wi in zip(syms, w) if wi > 0}


def _trend_scale(regime, cfg: dict | None) -> float:
    """趋势择时仓位缩放 (阶段3, 2026-08-10): 保障熊市年化收益为正。

    大盘趋势状态 (RegimeDetector.detect_v2):
      TREND_DOWN → down_scale (防守降仓, 默认 0.4)
      RANGE      → range_scale (中性偏防守, 默认 0.8)
      TREND_UP   → 1.0 (满仓)
    cfg: {"down_scale": 0.4, "range_scale": 0.8}
    """
    if not cfg:
        return 1.0
    rname = getattr(regime, 'name', str(regime))
    if 'DOWN' in rname:
        return float(cfg.get("down_scale", 0.4))
    if 'RANGE' in rname:
        return float(cfg.get("range_scale", 0.8))
    return 1.0


def _vol_target_scale(vol_pct: float, cfg: dict | None) -> float:
    """波动率目标仓位缩放 (P0, Moreira & Muir 2017 简化版)。

    市场已实现波动率百分位 vol_pct (0-1) 高于目标 → 降仓:
      scale = clip(target_pct / (vol_pct + eps), min_scale, max_scale)
    cfg: {"target_pct": 0.7, "max_scale": 1.0, "min_scale": 0.4}
    """
    if not cfg:
        return 1.0
    target = float(cfg.get("target_pct", 0.7))
    mx = float(cfg.get("max_scale", 1.0))
    mn = float(cfg.get("min_scale", 0.4))
    if vol_pct <= 0 or target <= 0:
        return 1.0
    scale = target / max(vol_pct, 1e-6)
    return float(np.clip(scale, mn, mx))


def run_backtest(all_data, factor_panels, close_panel, calendar, cal_idx,
                 factor_names, bt_config, start, end, label="",
                 fixed_weights: dict | None = None,
                 universe_fn=get_universe,
                 use_regime: bool = False,
                 portfolio_constraints: dict | None = None,
                 sleeve_weights: list | None = None,
                 minute_weights: dict | None = None,
                 minute_lambda: float = 0.3,
                 weight_mode: str = "equal",
                 pool_filter_cfg: dict | None = None,
                 vol_target_cfg: dict | None = None,
                 trend_timing_cfg: dict | None = None,
                 industry_lambda: float = 0.0):
    """回测主循环。

    fixed_weights: 若提供, 全程使用该固定权重 (方案C fold 验证期模式,
      权重在训练期估计, 验证期不重估 → 无信息泄漏)。
    universe_fn: universe 提供函数, 默认指数成分, 可传 get_liquid_universe。
    use_regime: 启用风格状态机 (方案C v5), 每个调仓日按 RegimeDetector
      双变量检测结果调整因子权重 (含动量崩溃保护)。
    portfolio_constraints: 组合后置约束 dict (config.yaml portfolio_constraints
      段: max_single_pct/max_industry_pct/max_turnover), None=不启用。
    sleeve_weights: 多风格 sleeve 列表 [{"name","weights","budget"}, ...],
      与 fixed_weights 配套 (fold 模式), None=单通道 (v27 行为)。
    minute_weights: 方案B 分钟叠加层权重 {min_factor: icir}, None=不叠加。
    minute_lambda: 叠加权重 λ (综合分 = 主分 + λ×分钟分)。
    industry_lambda: 行业动量叠加 λ (2026-08-17; 综合分 += λ×ind_mom_60 通道分,
      0=关闭)。
    pool_filter_cfg: 股票池分域配置 (config.yaml pool_filter 段,
      含 enabled/low_vol_mult/high_vol_mult/low_vol_up/high_vol_up)。
      None 或 enabled=false 时行为与 v9 完全一致 (不施加任何乘数)。
    vol_target_cfg: 波动率目标仓位配置 (config.yaml vol_target 段,
      含 enabled/target_pct/max_scale/min_scale)。
      None 或 enabled=false 时行为与之前完全一致 (不缩放仓位)。
    """
    from model.engine import SimpleBacktest
    from trading_rules import TradingRules
    from portfolio_ranker import PortfolioRanker

    # ★ 方案B v24: VWAP 拆单模式下, 滑点假设 = 残差滑点 (拆单无法完美命中
    # VWAP, 但远低于一次性开盘市价单的 30bps; 买上浮/卖下沉由 _apply_slippage
    # 按方向处理)。POV (v24e) 同用残差滑点 (POV 已按市场节奏分散)。
    exec_price = bt_config.get("execution_price", "open")
    slip = bt_config["slippage_bps"]
    if exec_price in ("vwap", "pov"):
        slip = bt_config.get("vwap_residual_bps", slip)
        log.info("  %s执行模式: 残差滑点=%dbps (原开盘+%dbps)",
                 exec_price.upper(), slip, bt_config["slippage_bps"])

    bt = SimpleBacktest(
        initial_capital=bt_config["initial_capital"],
        top_k=bt_config["top_k"],
        lot_size=bt_config["lot_size"],
        slippage_bps=slip,
        turnover_limit_pct=1.0,
        execution_price=exec_price,
        vwap_residual_bps=bt_config.get("vwap_residual_bps", 0),
    )
    # ★ vwap/pov 执行价模式 → 构建日 VWAP 面板 (pov 无分钟数据时回退 vwap)
    if bt.execution_price in ("vwap", "pov") and not bt.vwap_panel:
        bt.vwap_panel = _build_vwap_panel(sorted(all_data.keys()))
    if bt.execution_price in ("vwap", "pov") and not bt.vwap_panel:
        log.warning("  [%s] vwap 面板为空, 回退 open 执行价", label)
    rules = TradingRules()
    # 组合层参数从 config 读 (2026-08-16, 换手压降实验可改 config):
    # hold_thresh 与调仓周期联动 (路线C v22): 月频30天 / 周频~10天,
    # 避免周频下 PortfolioRanker 的 30 天持有锁死换手
    hold_thresh = int(bt_config.get("hold_thresh", 30) or 30)
    if hold_thresh <= 0:
        hold_thresh = max(5, REBALANCE_DAYS + 10)
    ranker = PortfolioRanker(
        top_k=bt_config["top_k"],
        n_drop=int(bt_config.get("n_drop", 10)),
        hold_thresh=hold_thresh,
        sell_rank_buffer=int(bt_config.get("sell_rank_buffer", 3)),
        buy_confirm_days=int(bt_config.get("buy_confirm_days", 1)),
        cost_threshold=float(bt_config.get("cost_threshold", 0.08)),
    )

    # Regime 检测器: use_regime 时在整个回测期只创建一次 (避免每个调仓日
    # 重复 from_benchmark_parquet 读盘), 调仓日仅调用 get_weight_multipliers
    regime_det = None
    if use_regime:
        from regime_detector import RegimeDetector
        if os.path.exists(BENCH_PATH):
            regime_det = RegimeDetector.from_benchmark_parquet(
                BENCH_PATH, profile="conservative",
                vol_source=bt_config.get("vol_source", "daily"))
            log.info("[%s] regime 检测启用 (基准: %s, vol_source=%s)",
                     label, BENCH_PATH, bt_config.get("vol_source", "daily"))
        else:
            log.warning("[%s] use_regime=True 但基准文件缺失: %s, 跳过",
                        label, BENCH_PATH)

    all_dates_set = set(pd.to_datetime(calendar))
    bt_dates = sorted(d for d in all_dates_set
                      if pd.Timestamp(start).date() <= d.date() <= pd.Timestamp(end).date())
    if not bt_dates:
        log.error("[%s] 无交易日", label)
        return None

    log.info("=" * 60)
    log.info("[%s] %s ~ %s (%d 天), top_k=%d, 动态IC权重",
             label, bt_dates[0].date(), bt_dates[-1].date(),
             len(bt_dates), bt_config["top_k"])
    log.info("=" * 60)

    equity_curve = []
    daily_returns = []
    positions_history = []
    weights_history = []
    turnover_history = []
    rebalance_count = 0
    pending = None
    industry_warned = False
    prev_equity = float(bt_config["initial_capital"])
    pit_sizes = []
    trades_history = []   # 逐笔成交 (v24b 后, 2026-08-12): engine.execute 返回

    for di, today in enumerate(bt_dates):
        # T+1 执行
        if pending is not None:
            _, _, trades = bt.execute(pending, today, all_data, rules)
            for t in trades:
                t = dict(t)
                t["label"] = label
                trades_history.append(t)
            pending = None

        # 调仓日
        if di % bt_config["rebalance_days"] == 0:
            pit_stocks = set(universe_fn(str(today.date())))
            pit_sizes.append(len(pit_stocks))

            # ★ 权重: 固定权重 (方案C fold) 或滚动 ICIR (默认)
            if fixed_weights is not None:
                weights = fixed_weights
            else:
                t_ic0 = time.time()
                weights, ic_stats = compute_icir_weights(
                    factor_panels, close_panel, calendar, cal_idx,
                    today, factor_names, universe_fn=universe_fn)
                log.info("  [%s] 调仓日 %s: %d 因子入选 (|ICIR|>%.2f), IC计算 %.1fs",
                         label, today.date(), len(weights), MIN_ICIR,
                         time.time() - t_ic0)

            # ★ Regime 风格轮动 (方案C v5): 按因子类别乘数调整权重
            if regime_det is not None:
                mults = regime_det.get_weight_multipliers(str(today.date()))
                # 拷贝后调整: fold 模式下 fixed_weights 字典跨调仓日复用,
                # 原地修改会导致乘数在多日叠加 (复合错误)
                adjusted = dict(weights)
                n_adj = 0
                for fn in list(adjusted.keys()):
                    cat = _factor_category(fn)
                    if cat in mults:
                        adjusted[fn] *= mults[cat]
                        n_adj += 1
                weights = adjusted
                log.info("  [%s] 调仓日 %s: regime 乘数调整 %d 因子 "
                         "(momentum×%.2f reversal×%.2f value×%.2f quality×%.2f)",
                         label, today.date(), n_adj,
                         mults["momentum"], mults["reversal"],
                         mults["value"], mults["quality"])

            # ★ 风格预算 (v29): 同家族因子 |权重| 之和封顶, 防一族垄断
            # (regime 乘数之后应用, 乘数放大过的家族同样受限)
            weights = _style_budget_weights(
                weights, style_cap=float(bt_config.get("style_cap", 0.4)))

            weights_history.append({
                "date": str(today.date()),
                "n_selected": len(weights),
                "weights": {k: round(v, 4) for k, v in
                            sorted(weights.items(), key=lambda kv: -abs(kv[1]))},
                "top10": [k for k, _ in sorted(weights.items(),
                                                 key=lambda kv: -abs(kv[1]))[:10]],
            })

            # 评分
            scores = score_stocks(factor_panels, weights, today,
                                  sleeve_weights=sleeve_weights,
                                  minute_weights=minute_weights,
                                  minute_lambda=minute_lambda,
                                  industry_lambda=industry_lambda)

            # ★ 股票池分域 (v10): 波动率分层 × regime 乘数 (选股层软偏好)。
            # 乘数表与 vol_pct 严格对应: >0.70 高波动市场 → 避险表 (低波加分),
            # <0.30 低波动市场 → 弹性表 (高波加分), 其余中性 (×1.0)。
            # 分支结构即保证对应关系 (热路径不用断言); 数据不足的股票
            # vol_bucket 按 mid 处理 (×1.0), PIT 由 vol_bucket 内部保证。
            if pool_filter_cfg and pool_filter_cfg.get("enabled"):
                from pool_filter import vol_bucket, apply_pool_filter
                if regime_det is not None:
                    _reg, _vol_pct = regime_det.detect_v2(str(today.date()))
                else:
                    _vol_pct = 0.5
                _buckets = vol_bucket(scores, all_data, today)
                if _vol_pct > 0.70:  # 高波动市场: 避险偏好
                    _mults = {"low": pool_filter_cfg.get("low_vol_mult", 1.5),
                              "mid": 1.0,
                              "high": pool_filter_cfg.get("high_vol_mult", 0.5)}
                elif _vol_pct < 0.30:  # 低波动市场: 弹性偏好
                    _mults = {"low": pool_filter_cfg.get("low_vol_up", 0.8),
                              "mid": 1.0,
                              "high": pool_filter_cfg.get("high_vol_up", 1.2)}
                else:
                    _mults = {"low": 1.0, "mid": 1.0, "high": 1.0}
                scores = apply_pool_filter(scores, _buckets, _vol_pct, _mults)
                log.info("  [%s] 调仓日 %s: pool_filter vol_pct=%.2f (%s)",
                         label, today.date(), _vol_pct,
                         "高波避险" if _vol_pct > 0.70 else
                         ("低波弹性" if _vol_pct < 0.30 else "中性"))

            # PIT 过滤
            scores = {s: v for s, v in scores.items() if s in pit_stocks}

            if scores and len(scores) >= bt_config["top_k"]:
                tradeable = {}
                for sym, sc in scores.items():
                    if sym in all_data:
                        dt = all_data[sym][all_data[sym]["date"] <= today].tail(2)
                        if len(dt) >= 2 and not rules.is_suspended(sym, dt):
                            tradeable[sym] = sc

                if len(tradeable) >= bt_config["top_k"]:
                    holdings = list(bt.positions.keys())
                    decision = ranker.rank(tradeable, holdings)
                    decision["buy"] = [
                        s for s in decision.get("buy", [])
                        if s in all_data and rules.can_buy(
                            s, all_data[s][all_data[s]["date"] <= today].tail(2))]
                    decision["sell"] = [
                        s for s in decision.get("sell", [])
                        if s in all_data and rules.can_sell(
                            s, all_data[s][all_data[s]["date"] <= today].tail(2))]
                    # ★ 组合层权重优化 (v9b): 波动率倒数加权
                    if weight_mode == "inv_vol":
                        w = _inv_vol_weights(all_data, decision.get("buy", []), today)
                        if w:
                            decision["weights"] = w
                    elif weight_mode == "risk_parity":
                        w = _risk_parity_weights(all_data, decision.get("buy", []), today)
                        if w:
                            decision["weights"] = w

                    # ★ 波动率目标仓位 (P0): 按市场波动率缩放总仓位
                    if (vol_target_cfg and vol_target_cfg.get("enabled")
                            and regime_det is not None):
                        _r2, _vp2 = regime_det.detect_v2(str(today.date()))
                        _scale = _vol_target_scale(_vp2, vol_target_cfg)
                        if _scale < 1.0 and decision.get("buy"):
                            # 买入金额按 scale 缩放的实现: 记录 scale 供 execute 用
                            decision["cash_scale"] = _scale
                            log.info("  [%s] 调仓日 %s: vol_target scale=%.2f (vol_pct=%.2f)",
                                     label, today.date(), _scale, _vp2)
                    # ★ 趋势择时 (阶段3): 大盘下跌趋势降仓, 保障熊市年化收益为正
                    if (trend_timing_cfg and trend_timing_cfg.get("enabled")
                            and regime_det is not None):
                        _r3, _vp3 = regime_det.detect_v2(str(today.date()))
                        _tscale = _trend_scale(_r3, trend_timing_cfg)
                        if _tscale < 1.0 and decision.get("buy"):
                            _prev = float(decision.get("cash_scale", 1.0))
                            decision["cash_scale"] = _prev * _tscale
                            log.info("  [%s] 调仓日 %s: trend_timing scale=%.2f (regime=%s, 总仓位=%.2f)",
                                     label, today.date(), _tscale,
                                     getattr(_r3, 'name', _r3),
                                     decision["cash_scale"])
                    pending = decision
                    rebalance_count += 1
                    n_turn = (len(decision.get("sell", [])) +
                              len(decision.get("buy", []))) / (2 * bt_config["top_k"])
                    turnover_history.append(n_turn)

                    # ── 组合后置约束 (方案C v5) ──
                    # ★ 2026-08-16 (v27): 从"检查+日志"改为"实际执行" —
                    # 目标权重写入 decision["weights"], bt.execute 按权重
                    # 分配现金 (engine.py 已支持 weights, 缺省等权兜底)。
                    if portfolio_constraints:
                        prev_w = decision.get("weights")
                        if prev_w:
                            # inv_vol/risk_parity 已设权重 → 只做个股封顶
                            w = _cap_single_weights(
                                prev_w,
                                portfolio_constraints.get("max_single_pct", 0.05))
                        else:
                            # 等权基线 → 等权或整体缩到上限 (原语义)
                            w = apply_portfolio_constraints(
                                {s: 1.0 for s in decision.get("buy", [])},
                                portfolio_constraints)
                        if w:
                            # ★ 行业中性 (v28): 行业权重封顶, 实际执行
                            max_ind = portfolio_constraints.get(
                                "max_industry_pct", 0.25)
                            w = _industry_cap_weights(w, _load_industry_map(),
                                                      max_ind)
                            decision["weights"] = w
                            n_buy = len(decision.get("buy", []))
                            ew_pct = 1.0 / n_buy if n_buy else 0
                            max_single = portfolio_constraints.get(
                                "max_single_pct", 0.05)
                            if ew_pct > max_single:
                                log.info("  [%s] 单票等权 %.1f%% > 上限 %.1f%%, "
                                         "已按权重执行 (剩余留现金)",
                                         label, ew_pct * 100, max_single * 100)
                        # 行业约束: 无行业数据 → 跳过 (每期只记一次 warning)
                        if (portfolio_constraints.get("max_industry_pct")
                                and not industry_warned):
                            industry_warned = True
                            log.warning("  [%s] 行业约束已配置 (max_industry_pct=%.0f%%) "
                                        "但无行业数据, 跳过",
                                        label,
                                        portfolio_constraints["max_industry_pct"] * 100)
                        # 换手约束: 超上限 → 跳过本轮调仓 (pending=None)。
                        # 语义: config max_turnover=0.5 是"月度累计单边换手"上限
                        # (纪律: 月单边≤50%)。★ 2026-08-15 修复: 期上限由
                        # turnover_period_cap 计算 (月频不缩放, 短周期按比例),
                        # 旧公式对月频(20)压线到 47.6% 与正常换手 50% 冲突,
                        # 导致每轮调仓全被跳过。
                        # 建仓期 (无持仓) 跳过检查: 分母 max(n_hold,1) 退化
                        # 为 1 会把首次建仓 (30只) 误判为 1500% 换手而永久卡死
                        n_hold = len(bt.positions)
                        if n_hold > 0:
                            monthly_cap = portfolio_constraints.get(
                                "max_turnover", 0.5)
                            max_turn = turnover_period_cap(monthly_cap,
                                                           REBALANCE_DAYS)
                            turnover = ((len(decision.get("buy", [])) +
                                         len(decision.get("sell", []))) /
                                        (2 * max(n_hold, 1)))
                            if turnover > max_turn:
                                log.info("  [%s] 换手 %.0f%% > 期上限 %.0f%% "
                                         "(月上限%.0f%%, 周期%d天), 跳过本轮调仓",
                                         label, turnover * 100, max_turn * 100,
                                         monthly_cap * 100, REBALANCE_DAYS)
                                pending = None

            positions_history.append({
                "date": str(today.date()),
                "positions": sorted(bt.positions.keys()),
            })

        # Mark-to-market (用面板收盘价, O(1))
        close_prices = {}
        if today in cal_idx:
            crow = close_panel.iloc[cal_idx[today]]
            for sym in list(bt.positions.keys()):
                if sym in crow.index:
                    v = crow[sym]
                    if not (isinstance(v, float) and np.isnan(v)):
                        close_prices[sym] = float(v)
        # ★ 2026-08-15: 无有效收盘价时沿用前值 (防御脏日历日按 0 计价)
        equity = safe_mark_to_market(bt.mark_to_market, bt.positions,
                                     close_prices, prev_equity)
        equity_curve.append({"date": str(today.date()), "equity": equity})
        daily_ret = (equity / prev_equity - 1) if prev_equity > 0 else 0.0
        daily_returns.append(daily_ret)
        prev_equity = equity

    # ── 统计 ──
    eq = np.array([e["equity"] for e in equity_curve])
    rets = np.array(daily_returns)
    total_return = eq[-1] / bt_config["initial_capital"] - 1
    n_years = len(bt_dates) / 252.0
    annual_return = (1 + total_return) ** (1 / max(n_years, 0.1)) - 1

    # 基准 (方案C v4.1: 优先全市场等权, 与策略池一致; CSI1000 仅 fallback)
    bench_annual = 0.0
    bench_daily = np.zeros(len(rets))
    ew_bench = _equal_weight_benchmark(start, end, all_data)
    if ew_bench is not None:
        bs = ew_bench
        if len(bs) > 1:
            bt_total = bs.iloc[-1] / bs.iloc[0] - 1
            bench_annual = (1 + bt_total) ** (1 / max(n_years, 0.1)) - 1
            bench_daily = bs.pct_change().dropna().to_numpy()
    elif os.path.exists(BENCH_PATH):
        bdf = pd.read_parquet(BENCH_PATH)
        bdf["date"] = pd.to_datetime(bdf["date"])
        bdf = bdf.set_index("date")
        rs = pd.Timestamp(start)
        re_ = pd.Timestamp(end)
        bm = (bdf.index >= rs) & (bdf.index <= re_)
        bs = bdf.loc[bm, "close"]
        if len(bs) > 1:
            bt_total = bs.iloc[-1] / bs.iloc[0] - 1
            bench_annual = (1 + bt_total) ** (1 / max(n_years, 0.1)) - 1
            bench_daily = bs.pct_change().dropna().to_numpy()

    excess = annual_return - bench_annual

    # Sharpe (rf=2.5%)
    rf_daily = 0.025 / 252
    excess_daily = rets - rf_daily
    sharpe = (np.mean(excess_daily) / np.std(excess_daily) * np.sqrt(252)
              if np.std(excess_daily) > 0 else 0)

    # IR
    if len(bench_daily) >= len(rets):
        active = rets - bench_daily[:len(rets)]
    else:
        active = rets - np.pad(bench_daily, (0, len(rets) - len(bench_daily)))
    ir = (np.mean(active) / np.std(active) * np.sqrt(252)
          if np.std(active) > 0 else 0)

    # MaxDD
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(np.min(dd))
    calmar = annual_return / abs(max_dd) if abs(max_dd) > 0 else 0

    avg_turnover = np.mean(turnover_history) if turnover_history else 0

    result = {
        "label": label,
        "period": f"{bt_dates[0].date()} ~ {bt_dates[-1].date()}",
        "n_days": len(bt_dates),
        "total_return": round(total_return * 100, 1),
        "annual_return": round(annual_return * 100, 1),
        "benchmark_annual": round(bench_annual * 100, 1),
        "excess_annual": round(excess * 100, 1),
        "sharpe": round(sharpe, 2),
        "ir": round(ir, 2),
        "max_drawdown": round(max_dd * 100, 1),
        "calmar": round(calmar, 2),
        "n_rebalances": rebalance_count,
        "avg_turnover": round(avg_turnover * 100, 1),
        "avg_pit_size": int(np.mean(pit_sizes)) if pit_sizes else 0,
        "avg_n_selected_factors": int(np.mean(
            [w["n_selected"] for w in weights_history])) if weights_history else 0,
        # 日度数据 (供 bootstrap 分析)
        "daily_returns": [round(float(r), 8) for r in rets],
        "daily_active_returns": [round(float(a), 8) for a in active],
        # 明细
        "weights_evolution": weights_history,
        "positions_history": positions_history,
        "equity_curve": equity_curve,
        "trades": trades_history,   # 逐笔成交 (v24b 后 2026-08-12: 真实成交价/数量/佣金)
    }

    log.info("  结果:")
    log.info("    总收益: %+.1f%%", result["total_return"])
    log.info("    年化收益: %+.1f%%", result["annual_return"])
    log.info("    基准年化: %+.1f%%", result["benchmark_annual"])
    log.info("    年化超额: %+.1f%%", result["excess_annual"])
    log.info("    Sharpe: %.2f  IR: %.2f", result["sharpe"], result["ir"])
    log.info("    最大回撤: %.1f%%  Calmar: %.2f",
             result["max_drawdown"], result["calmar"])
    log.info("    调仓: %d 次, 平均PIT: %d 只, 平均入选因子: %d",
             rebalance_count, result["avg_pit_size"],
             result["avg_n_selected_factors"])
    return result


# ═══════════════════════════════════════════════════════════
#  方案C: Walk-Forward Fold 分析 (训练→验证, 固定权重)
# ═══════════════════════════════════════════════════════════

# Fold 划分 (方案C): 训练期逐年扩展, 验证期固定1年
FOLDS = [
    {"train": ("2015-01-01", "2019-12-31"), "val": ("2020-01-01", "2020-12-31")},
    {"train": ("2015-01-01", "2020-12-31"), "val": ("2021-01-01", "2021-12-31")},
    {"train": ("2015-01-01", "2021-12-31"), "val": ("2022-01-01", "2022-12-31")},
    {"train": ("2015-01-01", "2022-12-31"), "val": ("2023-01-01", "2023-12-31")},
    {"train": ("2015-01-01", "2023-12-31"), "val": ("2024-01-01", "2024-12-31")},
]
FOLD_MIN_HITS = 3          # 因子须在 >=3/5 folds 中 |ICIR| 达标才保留
FOLD_ICIR_MIN = 0.05       # fold 内 |ICIR| 入选阈值 (滚动模式下仅作下限)
FOLD_T_STAT_MIN = 1.645    # 统计显著标准: |ICIR|*sqrt(n_obs) >= 1.645 (单尾5%)
FOLD_MAX_FACTORS = 40      # 稳定因子数量上限 (防止噪音因子稀释组合); 可由 config fold.max_factors 覆盖


def _sleeve_sig(ic_stats: dict) -> set:
    """与核心同口径的显著因子: |ICIR|>=FOLD_ICIR_MIN 且 t 统计 >= FOLD_T_STAT_MIN。"""
    sig = set()
    for fn, st in ic_stats.items():
        n_obs = st.get("n_obs", 0)
        t_stat = abs(st["icir"]) * np.sqrt(n_obs) if n_obs > 0 else 0.0
        if t_stat >= FOLD_T_STAT_MIN and abs(st["icir"]) >= FOLD_ICIR_MIN:
            sig.add(fn)
    return sig


def sleeve_median_weights(sleeve_icirs: dict, sleeves_cfg: dict) -> dict:
    """每 sleeve 因子 5 折 ICIR 中位数 → 权重; |median|<0.02 用 fallback_weight。

    fallback 符号取正 (成长因子 p6 全样本 ICIR 均为正; 2026-08-16)。
    """
    out = {}
    for name, icirs in sleeve_icirs.items():
        scfg = (sleeves_cfg or {}).get(name, {}) or {}
        fb = float(scfg.get("fallback_weight", 0.1))
        w = {}
        for fn, arr in icirs.items():
            med = float(np.median(arr)) if arr else 0.0
            w[fn] = med if abs(med) >= 0.02 else fb
        out[name] = w
    return out


def build_extend_sleeve_weights(fold_out: dict, styles_cfg: dict | None) -> list | None:
    """extend 模拟考用 sleeve 权重列表 (中位数 ICIR + config 预算)。"""
    if not styles_cfg:
        return None
    med = fold_out.get("sleeve_median_weights") or {}
    budgets = styles_cfg.get("budgets") or {}
    out = []
    for name, w in med.items():
        budget = float(budgets.get(name, 0.0))
        if w and budget > 0:
            out.append({"name": name, "weights": w, "budget": budget})
    return out or None


def run_fold_analysis(all_data, factor_panels, close_panel, calendar, cal_idx,
                      factor_names, bt_config,
                      universe_fn=get_universe,
                      use_regime: bool = False,
                      portfolio_constraints: dict | None = None,
                      minute_layer: dict | None = None,
                      max_factors: int | None = None,
                      weight_mode: str = "equal",
                      pool_filter_cfg: dict | None = None,
                      vol_target_cfg: dict | None = None,
                      trend_timing_cfg: dict | None = None,
                      styles_cfg: dict | None = None,
                      industry_lambda: float = 0.0) -> dict:
    """
    方案C核心: 5-fold Walk-Forward。

    每个 fold:
      1. 训练期 [train] 用固定窗口计算全部因子 ICIR
      2. 验证期 [val] 用训练期固定权重回测 (无信息泄漏)
    汇总:
      - 每个因子在多少 fold 中 |ICIR| 达标 (fold_hits)
      - 保留 fold_hits >= FOLD_MIN_HITS 的稳定因子
      - 各 fold 验证期成绩汇总 → OOS 年化超额均值/IR

    minute_layer: 方案B 分钟叠加配置 {enabled, weights, lambda},
      仅 fold 4-5 验证期 (有分钟数据) 生效, fold 1-3 传 None。
    industry_lambda: 行业动量叠加 λ (2026-08-17), 透传 run_backtest。
    """
    log.info("=" * 60)
    log.info("  方案C Walk-Forward: %d folds (训练扩展, 验证固定权重)",
             len(FOLDS))
    log.info("  因子保留标准: |ICIR|>=%.2f 在 >=%d/%d folds 中达标",
             FOLD_ICIR_MIN, FOLD_MIN_HITS, len(FOLDS))
    log.info("=" * 60)

    # ── 方案B: 分钟叠加层 (fold 4-5 验证期才有分钟数据) ──
    ml_weights = None
    ml_lambda = 0.3
    if minute_layer and minute_layer.get("enabled"):
        ml_weights = minute_layer.get("weights")  # validate_minute_factors 输出
        ml_lambda = float(minute_layer.get("lambda", 0.3))
        if ml_weights:
            log.info("  分钟叠加层: %d 个因子, λ=%.2f (fold 4-5 验证期)",
                     len(ml_weights), ml_lambda)

    core_names = factor_names
    sleeves = {}
    if styles_cfg:
        core_names, sleeves = split_sleeve_factors(factor_names, styles_cfg)
        log.info("  sleeve 模式: 核心 %d 因子 + %s",
                 len(core_names),
                 ", ".join(f"{n}({len(fs)} 因子)" for n, fs in sleeves.items()))

    fold_results = {}
    factor_hits = {fn: 0 for fn in core_names}
    factor_icirs = {fn: [] for fn in core_names}
    sleeve_icirs = {}

    # 验证期首调仓日: 固定权重只需在训练期末计算一次
    for fi, fold in enumerate(FOLDS):
        ts, te = fold["train"]
        vs, ve = fold["val"]
        log.info("")
        log.info(f"── Fold {fi+1}: Train {ts}~{te} → Val {vs}~{ve} ──")

        # 训练期权重 (固定窗口, 在验证期首日估计 → 训练信息在 T 前全部实现)
        val_first = None
        for d in calendar:
            if pd.Timestamp(vs).date() <= d.date() <= pd.Timestamp(ve).date():
                val_first = d
                break
        if val_first is None:
            log.warning("  Fold %d: 验证期无交易日, 跳过", fi + 1)
            continue

        weights, ic_stats = compute_icir_weights(
            factor_panels, close_panel, calendar, cal_idx,
            val_first, core_names, train_start=ts, train_end=te,
            universe_fn=universe_fn)

        # ── sleeve 权重估计 (独立通道, 2026-08-16) ──
        fold_sleeve_weights = {}
        if sleeves:
            for sname, sfs in sleeves.items():
                s_weights, s_stats = compute_icir_weights(
                    factor_panels, close_panel, calendar, cal_idx,
                    val_first, sfs, train_start=ts, train_end=te,
                    universe_fn=universe_fn)
                min_hits = int(styles_cfg["sleeves"][sname].get("min_hits", 3))
                if min_hits > 0:
                    s_weights = {fn: w for fn, w in s_weights.items()
                                 if fn in _sleeve_sig(s_stats)}
                else:
                    # min_hits=0 (成长): 保留 |icir|>=MIN_ICIR 的因子, 符号随 ICIR
                    s_weights = {fn: w for fn, w in s_weights.items()
                                 if abs(w) >= MIN_ICIR}
                fold_sleeve_weights[sname] = s_weights
                for fn in sfs:
                    st = s_stats.get(fn)
                    sleeve_icirs.setdefault(sname, {}).setdefault(fn, []).append(
                        float(st["icir"]) if st is not None else 0.0)
                log.info("  sleeve[%s] 本折权重: %d 因子", sname, len(s_weights))

        n_sel = len(weights)
        log.info("  训练期因子: %d/%d 入选 |ICIR|>=%.2f",
                 n_sel, len(core_names), FOLD_ICIR_MIN)

        # 记录每个因子的 fold 命中 (统计显著标准: |ICIR|*sqrt(n) >= 1.645)
        sig_factors = set()
        for fn in core_names:
            st = ic_stats.get(fn)
            if st is not None:
                factor_icirs[fn].append(st["icir"])
                n_obs = st.get("n_obs", 0)
                t_stat = abs(st["icir"]) * np.sqrt(n_obs) if n_obs > 0 else 0.0
                if t_stat >= FOLD_T_STAT_MIN and abs(st["icir"]) >= FOLD_ICIR_MIN:
                    factor_hits[fn] += 1
                    sig_factors.add(fn)
            else:
                factor_icirs[fn].append(0.0)

        # 验证期回测权重: 仅统计显著因子 (过滤噪音)
        weights = {fn: w for fn, w in weights.items() if fn in sig_factors}
        # ★ 风格预算 (v29): fold 固定权重同样封顶 (与验证期调仓日一致)
        weights = _style_budget_weights(
            weights, style_cap=float(bt_config.get("style_cap", 0.4)))
        n_sel = len(weights)
        log.info("  回测权重: %d 因子 (统计显著, |ICIR|*√n>=%.2f)",
                 n_sel, FOLD_T_STAT_MIN)

        if not weights:
            log.warning("  Fold %d: 训练期无因子达标, 验证期跳过", fi + 1)
            continue

        r = run_backtest(all_data, factor_panels, close_panel, calendar,
                         cal_idx, factor_names, bt_config, vs, ve,
                         label=f"VAL{fi+1}", fixed_weights=weights,
                         sleeve_weights=([{
                             "name": sname,
                             "weights": fold_sleeve_weights.get(sname) or {},
                             "budget": float((styles_cfg.get("budgets") or {})
                                             .get(sname, 0.0)),
                         } for sname in sleeves] if sleeves else None),
                         universe_fn=universe_fn, use_regime=use_regime,
                         portfolio_constraints=portfolio_constraints,
                         minute_weights=ml_weights if fi >= 3 else None,
                         minute_lambda=ml_lambda,
                         weight_mode=weight_mode,
                         pool_filter_cfg=pool_filter_cfg,
                         vol_target_cfg=vol_target_cfg,
                         trend_timing_cfg=trend_timing_cfg,
                         industry_lambda=industry_lambda)
        if r:
            fold_results[f"fold_{fi+1}"] = {
                "train": f"{ts} ~ {te}",
                "val": f"{vs} ~ {ve}",
                "n_selected_factors": n_sel,
                **r,
            }

    # ── 稳定因子筛选 (统计显著 ≥3/5 folds + 方向一致 + 数量上限) ──
    # 方向一致性: 因子在 ≥3/5 folds 中 ICIR 符号相同 (防止正负抵消)
    cand = []
    for fn in core_names:
        if factor_hits[fn] < FOLD_MIN_HITS or not factor_icirs[fn]:
            continue
        arr = np.array(factor_icirs[fn])
        pos_cnt = int((arr > 0).sum())
        neg_cnt = int((arr < 0).sum())
        if max(pos_cnt, neg_cnt) >= FOLD_MIN_HITS:  # 方向一致
            cand.append(fn)
    cand_icir = {fn: float(np.median(factor_icirs[fn])) for fn in cand}
    # 按 |ICIR| 排序取 top FOLD_MAX_FACTORS (可由 max_factors 覆盖)
    limit = max_factors if max_factors else FOLD_MAX_FACTORS
    ranked = sorted(cand_icir.items(), key=lambda kv: -abs(kv[1]))
    stable = [fn for fn, _ in ranked[:limit]]
    stable_icir = {fn: cand_icir[fn] for fn in stable}

    log.info("")
    log.info("=" * 60)
    log.info("  稳定因子: %d/%d (≥%d folds 显著+方向一致, top%d)",
             len(stable), len(core_names), FOLD_MIN_HITS,
             limit)
    if stable:
        top = sorted(stable_icir.items(), key=lambda kv: -abs(kv[1]))[:15]
        for fn, icir in top:
            log.info("    %-30s ICIR=%+.3f (命中 %d/5)",
                     fn, icir, factor_hits[fn])
    log.info("=" * 60)

    return {
        "folds": fold_results,
        "factor_hits": factor_hits,
        "stable_factors": stable,
        "stable_factor_icir_median": {
            k: round(v, 4) for k, v in stable_icir.items()},
        "sleeve_median_weights": (
            sleeve_median_weights(sleeve_icirs, styles_cfg.get("sleeves", {}))
            if styles_cfg else {}),
    }


def run_fold_test(all_data, factor_panels, close_panel, calendar, cal_idx,
                  factor_names, bt_config, stable_factors, stable_icir,
                  test_start, test_end,
                  universe_fn=get_universe,
                  use_regime: bool = False,
                  portfolio_constraints: dict | None = None,
                  minute_layer: dict | None = None,
                  weight_mode: str = "equal",
                  pool_filter_cfg: dict | None = None,
                  vol_target_cfg: dict | None = None,
                  trend_timing_cfg: dict | None = None,
                  industry_lambda: float = 0.0) -> dict | None:
    """
    终极 Holdout: 用稳定因子的中位数 ICIR 权重, 在 TEST 期一次性回测。

    风格均衡 (方案C v4.1): 正/负 ICIR 因子权重分别归一化,
    避免单一风格 (如反转防御) 过度主导组合 → 牛市跑输基准。

    minute_layer: 方案B 分钟叠加配置 dict, 含 minute_weights + lambda,
      None 或 enabled=false 时行为与 v5 完全一致。
    """
    if not stable_factors:
        log.warning("无稳定因子, 跳过终极 TEST")
        return None
    weights = {fn: stable_icir[fn] for fn in stable_factors
               if fn in stable_icir}
    if not weights:
        log.warning("稳定因子无权重, 跳过终极 TEST")
        return None

    # ★ 风格均衡: 正负方向权重各自归一化到等和
    w_pos = sum(v for v in weights.values() if v > 0)
    w_neg = sum(-v for v in weights.values() if v < 0)
    if w_pos > 1e-9 and w_neg > 1e-9:
        for fn in weights:
            if weights[fn] > 0:
                weights[fn] = weights[fn] / w_pos
            else:
                weights[fn] = -weights[fn] / w_neg
        log.info("  风格均衡: 正方向权重合计=%+.2f, 负方向合计=%+.2f",
                 sum(v for v in weights.values() if v > 0),
                 sum(v for v in weights.values() if v < 0))

    log.info("")
    log.info("=" * 60)
    log.info("  终极 TEST (只跑一次): 稳定因子 %d 个, 权重=中位数ICIR(风格均衡)",
             len(weights))
    log.info("=" * 60)

    # ── 方案B: 分钟叠加层 ──
    ml_weights = None
    ml_lambda = 0.3
    if minute_layer and minute_layer.get("enabled"):
        ml_weights = minute_layer.get("weights")  # validate_minute_factors 输出
        ml_lambda = float(minute_layer.get("lambda", 0.3))
        if ml_weights:
            log.info("  分钟叠加层: %d 个因子, λ=%.2f", len(ml_weights), ml_lambda)

    return run_backtest(all_data, factor_panels, close_panel, calendar,
                        cal_idx, factor_names, bt_config,
                        test_start, test_end, label="TEST",
                        fixed_weights=weights, universe_fn=universe_fn,
                        use_regime=use_regime,
                        portfolio_constraints=portfolio_constraints,
                        minute_weights=ml_weights, minute_lambda=ml_lambda,
                        weight_mode=weight_mode,
                        pool_filter_cfg=pool_filter_cfg,
                        vol_target_cfg=vol_target_cfg,
                        trend_timing_cfg=trend_timing_cfg,
                        industry_lambda=industry_lambda)


# ═══════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════

def _check_data_coverage(all_data: dict, needed_end: str,
                         minute_dir: str = None) -> None:
    """结构性缺失守卫 (2026-08-16): 日线/分钟必须覆盖运行所需区间末尾。

    覆盖不足 → raise RuntimeError 硬失败 (实验可复现性优先, 不允许静默缺数据)。
    minute_dir 为分钟数据目录 (如 minute_5m); needed_end < 2022-01-01 时
    跳过分钟检查 (2022 前无分钟数据, POV 回退 VWAP/开盘属正常)。
    """
    end_dt = pd.Timestamp(needed_end)
    daily_max = None
    for df in list(all_data.values())[:20]:
        if df is not None and len(df) > 0 and "date" in df.columns:
            d = pd.to_datetime(df["date"]).max()
            if daily_max is None or d > daily_max:
                daily_max = d
    if daily_max is not None and daily_max < end_dt:
        raise RuntimeError(
            f"日线数据覆盖不足: 需要到 {end_dt.date()}, 实际最新 {daily_max.date()}")
    if minute_dir and end_dt >= pd.Timestamp("2022-01-01") \
            and os.path.isdir(minute_dir):
        from data.minute_fetcher import latest_local_minute_date
        latest = latest_local_minute_date(minute_dir)
        if latest is not None and latest < end_dt.date():
            raise RuntimeError(
                f"分钟数据覆盖不足: 需要到 {end_dt.date()}, 实际最新 {latest}")


def main():
    parser = argparse.ArgumentParser(
        description="Walk-Forward 回测: 滚动ICIR(默认) 或 方案C Fold(推荐)")
    parser.add_argument("--dev-only", action="store_true",
                        help="仅跑 Development")
    parser.add_argument("--test-only", action="store_true",
                        help="仅跑 TEST")
    parser.add_argument("--folds", action="store_true", default=None,
                        help="方案C: 5-fold Walk-Forward + 终极 TEST")
    parser.add_argument("--folds-only", action="store_true", default=None,
                        help="仅 5-fold 分析, 不执行终极 TEST (v8 新因子验证用, 不消耗 TEST 锁)")
    parser.add_argument("--force-partial-test", action="store_true",
                        help="显式确认在数据不完备时执行 TEST② (仅用于确认, 会消耗 TEST 锁)")
    parser.add_argument("--extend-val", nargs=2, metavar=("START", "END"),
                        default=None,
                        help="fold 分析后, 用稳定因子权重在扩展区间做模拟考验证 "
                             "(如 2025-01-01 2026-06-30, TEST① 毕业数据; 不消耗任何 TEST 锁)")
    parser.add_argument("--liquid", action="store_true", default=None,
                        help="使用流动性 PIT universe (全市场+过滤, 方案C推荐)")
    parser.add_argument("--unlock-test", action="store_true",
                        help="解锁终极 TEST (仅供已确认的重新验证)")
    parser.add_argument("--sample", type=int, default=None,
                        help="抽样股票数 (快速测试用)")
    parser.add_argument("--max-stocks", type=int, default=None,
                        help="最多使用股票数")
    args = parser.parse_args()

    config = load_config(os.path.join(BASE_DIR, "config.yaml"))

    # ── 回测模式参数: config.yaml backtest 段为唯一默认源 (2026-08-15) ──
    # CLI 显式传入时覆盖 config; 未传时用 config 默认 → 标准流程零参数启动,
    # 防"每次手敲参数漏掉" (如 --folds-only 缺 --folds 静默降级为普通回测)。
    _bt = config.get("backtest", {}) or {}
    if args.folds is None:
        args.folds = str(_bt.get("mode", "rolling")) == "folds"
    if args.folds_only is None:
        args.folds_only = bool(_bt.get("folds_only", True))
    if args.liquid is None:
        args.liquid = bool(_bt.get("liquid", False))
    if args.extend_val is None and _bt.get("extend_val"):
        ev = _bt["extend_val"]
        if isinstance(ev, list) and len(ev) == 2:
            args.extend_val = [str(ev[0]), str(ev[1])]
    # 防呆: --folds-only 是 --folds 的修饰符, 单独使用静默降级是历史缺陷
    if args.folds_only and not args.folds:
        log.error("🚫 --folds-only 必须搭配 --folds 使用 (它是 folds 模式的修饰符, "
                  "单独使用会静默走普通回测)。请加 --folds 或改 config backtest.mode=folds")
        sys.exit(1)

    # ── 启动防护 (2026-08-09 防低级失误) ──
    # 1. config 完整性校验 (重复键检测)
    try:
        from gate import validate_config_integrity
        with open(os.path.join(BASE_DIR, "config.yaml"), encoding="utf-8") as _f:
            _raw = _f.read()
        _errs = validate_config_integrity(config, _raw)
        if _errs:
            log.error("🚫 config.yaml 完整性校验失败: %s", _errs)
            sys.exit(1)
    except Exception as _e:
        log.warning("  config 完整性校验跳过: %s", _e)

    # 2. 结果自动备份 (带时间戳, 防覆盖丢失)
    if os.path.exists(OUTPUT_PATH):
        import shutil as _shutil
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _bak = os.path.join(IC_DIR, f"walkforward_results_bak_{_ts}.json")
        try:
            _shutil.copy2(OUTPUT_PATH, _bak)
            log.info("  📦 上一轮结果已备份: %s", _bak)
        except Exception as _e2:
            log.warning("  结果备份失败: %s", _e2)

    # 3. 关键实验状态打印 (防配置遗留)
    _pf = config.get("pool_filter", {}) or {}
    _mm = config.get("minute_factors", {}) or {}
    _dp = config.get("data_partition", {}) or {}
    log.info("── 实验状态 ──")
    log.info("  数据分区: research=%s~%s dev(TEST②)=%s~%s blind=%s",
             _dp.get("research", {}).get("start", "?"),
             _dp.get("research", {}).get("end", "?"),
             _dp.get("development", {}).get("start", "?"),
             _dp.get("development", {}).get("end", "?"),
             _dp.get("blind", {}).get("start", "?"))
    log.info("  pool_filter=%s (低波×%.1f/高波×%.1f) | minute_freq=%s | fold.max_factors=%s | industry_neutral=%s",
             "开" if _pf.get("enabled") else "关",
             _pf.get("low_vol_mult", 1.5), _pf.get("high_vol_mult", 0.5),
             _mm.get("freq", "15"), config.get("fold", {}).get("max_factors", 40),
             bool((config.get("neutralization", {}) or {}).get("industry_neutral", False)))
    log.info("────────────")

    # 前置中性化开关 (config.yaml neutralization 段)
    _neut = config.get("neutralization", {}) or {}
    neutralize_enabled = bool(_neut.get("enabled", False))
    neutralize_k = float(_neut.get("winsorize_k", 3.0))
    industry_neutral = bool(_neut.get("industry_neutral", False))
    log.info("  前置中性化: %s (MAD去极值 k=%.1f, 行业中性化=%s)",
             "开启" if neutralize_enabled else "关闭", neutralize_k,
             "开" if industry_neutral else "关")

    # 分钟因子开关 (config.yaml minute_factors 段, 方案C v5)
    _min_cfg = config.get("minute_factors", {}) or {}
    minute_enabled = bool(_min_cfg.get("enabled", False))
    minute_lookback = int(_min_cfg.get("lookback", 20))
    log.info("  分钟因子: %s (lookback=%d, 数据%s起, 仅fold模式可用)",
             "开启" if minute_enabled else "关闭",
             minute_lookback, MINUTE_DATA_START)

    # 组合后置约束 (config.yaml portfolio_constraints 段: 存在即启用,
    # None 表示禁用 → 回测走无约束路径)
    portfolio_constraints = config.get("portfolio_constraints") or None
    if portfolio_constraints:
        log.info("  组合约束: 单票≤%.0f%% 行业≤%.0f%% 换手≤%.0f%%",
                 portfolio_constraints.get("max_single_pct", 0.05) * 100,
                 portfolio_constraints.get("max_industry_pct", 0.25) * 100,
                 portfolio_constraints.get("max_turnover", 0.5) * 100)
    else:
        log.info("  组合约束: 未启用")

    # 终极 TEST 锁 (方案C: 2025-01~2026-06 只跑一次)
    TEST_LOCK_PATH = os.path.join(IC_DIR, ".test_lock_v5")
    if args.unlock_test and os.path.exists(TEST_LOCK_PATH):
        os.remove(TEST_LOCK_PATH)
        log.warning("  🔓 终极 TEST 锁已解除")
    if args.folds and not args.folds_only and not args.unlock_test and os.path.exists(TEST_LOCK_PATH):
        with open(TEST_LOCK_PATH, "r", encoding="utf-8") as f:
            lock_info = json.load(f)
        log.error("=" * 60)
        log.error("  🚫 终极 TEST 已锁定！不可重复运行。")
        log.error("  锁定时间: %s", lock_info.get("locked_at", "unknown"))
        log.error("  结果文件: %s", OUTPUT_PATH)
        log.error("  如确需重跑: --unlock-test")
        log.error("=" * 60)
        sys.exit(1)

    # universe 提供函数
    from data.pit_universe import get_liquid_universe
    universe_fn = get_liquid_universe if args.liquid else get_universe

    if args.folds:
        partitions_to_run = {"development": PARTITIONS["development"]}
    elif args.dev_only:
        partitions_to_run = {"development": PARTITIONS["development"]}
    elif args.test_only:
        partitions_to_run = {"test": PARTITIONS["test"]}
    else:
        partitions_to_run = dict(PARTITIONS)
        # v5: development 与 test 同区间时只跑一次 (避免重复回测)
        if (PARTITIONS.get("development") == PARTITIONS.get("test")
                and "development" in partitions_to_run):
            del partitions_to_run["test"]

    # 日期范围静态检查 (不触碰盲测期)
    for label, (s, e) in partitions_to_run.items():
        try:
            check_date_range(s, e, config, script_name="run_walkforward_backtest")
        except GateViolation as ex:
            log.error(str(ex))
            sys.exit(1)

    log.info("=" * 60)
    if args.folds:
        log.info("  方案C Walk-Forward Fold 分析 (训练→验证)")
    else:
        log.info("  Walk-Forward 回测 (动态IC权重, 滚动252天)")
    log.info("  分区: %s", ", ".join(partitions_to_run.keys()))
    log.info("  universe: %s", "流动性PIT(全市场+过滤)" if args.liquid else "指数成分(CSI300+ZZ500)")
    log.info("=" * 60)

    bt_config = load_bt_config()
    log.info("  top_k=%d, lot=%d, slippage=%dbps, 调仓=%d天",
             bt_config["top_k"], bt_config["lot_size"],
             bt_config["slippage_bps"], bt_config["rebalance_days"])

    # ── 1. 加载数据 (复用 run_corrected_backtest 逻辑) ──
    log.info("加载数据...")
    from data_cache import get_cached_symbols, load
    syms = get_cached_symbols()
    if args.sample and args.sample < len(syms):
        import random
        random.seed(42)
        syms = random.sample(syms, args.sample)
    elif args.max_stocks and args.max_stocks < len(syms):
        syms = syms[:args.max_stocks]
    all_data = {}
    for sym in syms:
        df = load(sym)
        if df is not None and len(df) >= 250:
            all_data[sym] = df
    log.info("  有效: %d 只", len(all_data))

    # ── 2. 交易日历 + 收盘价面板 ──
    calendar = build_calendar(all_data)
    cal_idx = {d: i for i, d in enumerate(calendar)}
    log.info("  交易日历: %d 天 (%s ~ %s)", len(calendar),
             calendar[0].date(), calendar[-1].date())

    # ── 结构性缺失守卫 (2026-08-16): 数据必须覆盖本次运行所需区间末尾 ──
    #    覆盖不足硬失败 → 实验可复现 (不允许静默缺数据)。2022 前无分钟数据
    #    属正常 (POV 回退 VWAP/开盘), 分钟检查只对 2022+ 区间生效。
    _needed_end = None
    if args.extend_val:
        _needed_end = args.extend_val[1]
    elif args.folds:
        _needed_end = "2024-12-31"  # fold 5 验证期末
    if _needed_end:
        _check_data_coverage(
            all_data, _needed_end,
            os.path.join(BASE_DIR, "data_store", "minute_5m"))

    log.info("构建收盘价面板...")
    close_panel = build_close_panel(all_data, calendar)

    # ── 3. 确定所需日期 (调仓日 + IC窗口内月度观测日) ──
    from factor_scorer import FactorScorer
    if args.folds:
        # 方案C (fold): v5 预设 = 169 价量 + 13 基本面 (fund_*) 因子
        factor_names = sorted(FactorScorer.from_preset("full_auto_v5").factor_weights.keys())
        if minute_enabled:
            # min_* 分钟因子不在 FACTOR_PRESETS 中 (数据2022起, 仅fold模式),
            # 手动并入 factor_names — 无数据期的 fold 由 IC 截面检查自动跳过
            from minute_factors import get_minute_factor_names
            factor_names = sorted(set(factor_names) | set(get_minute_factor_names()))
        # 预期差/行业动量因子 (2026-08-17): 不在 FactorScorer 预设中, styles
        # 启用时并入 factor_names (precompute 对该列无数据跳过, 面板由
        # merge_surprise_panels 补; ind_mom_60 仅行业 λ>0 时并入)
        _extra = fold_extra_factor_names(config.get("styles") or {})
        if _extra:
            factor_names = sorted(set(factor_names) | _extra)
    else:
        factor_names = sorted(FactorScorer.from_preset("full_auto").factor_weights.keys())

    needed = set()
    if args.folds:
        # 方案C: fold 训练期(全观测点) + 验证期调仓日 + 终极TEST
        for fold in FOLDS:
            ts, te = fold["train"]
            vs, ve = fold["val"]
            for d in calendar:
                d_ = d.date()
                if pd.Timestamp(ts).date() <= d_ <= pd.Timestamp(te).date():
                    needed.add(d)
                elif pd.Timestamp(vs).date() <= d_ <= pd.Timestamp(ve).date():
                    needed.add(d)
        # 终极 TEST (TEST②: 从 config data_partition.development 读取, 2026-07 起)
        _dev = config.get("data_partition", {}).get("development", {})
        test_s = _dev.get("start", "2026-07-01")
        test_e = _dev.get("end", "2026-12-31")
        for d in calendar:
            if pd.Timestamp(test_s).date() <= d.date() <= pd.Timestamp(test_e).date():
                needed.add(d)
        # 扩展模拟考 (--extend-val): 区间加入面板构建 (否则 scores 为空 → 空仓)
        if args.extend_val:
            ev_s0, ev_e0 = args.extend_val[0], args.extend_val[1]
            for d in calendar:
                if pd.Timestamp(ev_s0).date() <= d.date() <= pd.Timestamp(ev_e0).date():
                    needed.add(d)
    else:
        for label, (s, e) in partitions_to_run.items():
            part_dates = [d for d in calendar
                          if pd.Timestamp(s).date() <= d.date() <= pd.Timestamp(e).date()]
            for di, d in enumerate(part_dates):
                if di % bt_config["rebalance_days"] == 0:
                    needed.add(d)
                    i = cal_idx[d]
                    start_i = max(0, i - TRAIL_DAYS)
                    end_i = i - LABEL_HORIZON
                    for oi in range(start_i, end_i + 1, IC_STEP):
                        needed.add(calendar[oi])
    needed_dates = sorted(needed)
    log.info("  因子面板所需日期: %d 天 (%s ~ %s)",
             len(needed_dates), needed_dates[0].date(), needed_dates[-1].date())

    # ── 4. 预计算因子面板 ──
    log.info("预计算因子面板...")
    factor_panels = precompute_factor_panels(
        all_data, factor_names, needed_dates,
        include_fundamental=bool(args.folds),
        include_aux=bool(args.folds),
        include_minute=bool(args.folds) and minute_enabled,
        minute_lookback=minute_lookback,
        neutralize_enabled=neutralize_enabled,
        neutralize_k=neutralize_k,
        industry_map=_load_industry_map() if industry_neutral else None)
    # 预期差/行业动量面板合并 (2026-08-17; 无对应因子时不产生任何面板)
    log.info("  [mem] merge 前 RSS=%.1f GB", _rss_gb())
    _n_surprise = merge_surprise_panels(
        factor_panels, factor_names, pd.DatetimeIndex(needed_dates),
        sorted(all_data.keys()), all_data, _load_industry_map())
    if _n_surprise:
        log.info("  预期差/行业面板: %d 个 (sue/accel/pead/ind)", _n_surprise)
    gc.collect()
    log.info("  [mem] merge 后 RSS=%.1f GB", _rss_gb())
    log.info("  面板就绪: %d 因子", len(factor_panels))

    # ── 5. 回测 (带日期守卫) ──
    results = {}
    extra_meta = {}
    log.info("  [probe] 进入回测守卫块")
    log.info("  [mem] 守卫块前 RSS=%.1f GB", _rss_gb())
    with DateRangeGuard(config, script_name="run_walkforward_backtest") as guard:
        if args.folds:
            # 方案C: 5-fold 分析 + 终极 TEST
            guard.check_range("2015-01-01",
                             config.get("data_partition", {}).get("full_end", "2026-12-31"))
            # ── 因子正交化 (P2, 可选): fold 权重计算前 Gram-Schmidt 去冗余 ──
            # 顺序与 fold 权重一致 (sorted factor_names); 不改变第一个因子
            if config.get("factor_orthogonalize", {}).get("enabled"):
                from orthogonalize import orthogonalize_panels
                factor_panels = orthogonalize_panels(
                    factor_panels, sorted(factor_names))
                log.info("  因子正交化: %d 因子 (Gram-Schmidt, 顺序 %s)",
                         len(factor_panels), sorted(factor_names)[:3])
            # 方案B: 分钟因子独立验证 (fold 4-5 训练期)
            ml_cfg = config.get("minute_layer", {})
            ml_weights = None
            if ml_cfg.get("enabled"):
                # 训练期: fold 4 = 2015-2022, fold 5 = 2015-2023
                # 但分钟数据 2022 起 → 实际用 2022-01~2023-12 / 2022-01~2024-12
                train_folds = [
                    ("2022-01-01", "2023-12-31"),
                    ("2022-01-01", "2024-12-31"),
                ]
                log.info("  [probe] 分钟验证开始 (enabled=%s)", ml_cfg.get("enabled"))
                log.info("  [mem] 分钟验证前 RSS=%.1f GB", _rss_gb())
                ml_weights = validate_minute_factors(
                    factor_panels, close_panel, calendar, cal_idx,
                    factor_names, train_folds,
                    min_icir=float(ml_cfg.get("min_icir", 0.3)))
                log.info("  [probe] 分钟验证结束: %d 因子通过",
                         len(ml_weights or {}))
            minute_layer = {
                "enabled": ml_cfg.get("enabled", True),
                "weights": ml_weights,
                "lambda": float(ml_cfg.get("lambda", 0.3)),
            }
            # ── 多风格 sleeve (config styles 段, enabled=false → None=v27 行为) ──
            styles_cfg = load_styles_config(config)
            try:
                assert_sleeve_mode_allowed(styles_cfg, args.folds, args.folds_only)
            except RuntimeError:
                log.error("🚫 多风格 sleeve 模式仅支持 --folds-only + --extend-val; "
                          "styles.enabled=true 时禁止执行一次性 TEST (模型与验证不一致会消耗 TEST 锁)。")
                sys.exit(1)
            if styles_cfg:
                log.info("  多风格 sleeve: %s (budgets=%s)",
                         ", ".join((styles_cfg.get("sleeves") or {}).keys()),
                         styles_cfg.get("budgets"))
            fold_out = run_fold_analysis(
                all_data, factor_panels, close_panel, calendar, cal_idx,
                factor_names, bt_config, universe_fn=universe_fn,
                use_regime=True, portfolio_constraints=portfolio_constraints,
                minute_layer=minute_layer,
                max_factors=int(config.get("fold", {}).get("max_factors", 40)),
                weight_mode=str(config.get("portfolio_optimizer", "equal")),
                pool_filter_cfg=config.get("pool_filter"),
                vol_target_cfg=config.get("vol_target"),
                trend_timing_cfg=config.get("trend_timing"),
                styles_cfg=styles_cfg,
                industry_lambda=float((config.get("styles") or {}).get("industry_lambda", 0.0)))
            for k, v in fold_out.get("folds", {}).items():
                results[k] = v
            extra_meta["fold_factor_hits"] = fold_out["factor_hits"]
            extra_meta["stable_factors"] = fold_out["stable_factors"]
            extra_meta["stable_factor_icir_median"] = (
                fold_out["stable_factor_icir_median"])
            if styles_cfg and fold_out.get("sleeve_median_weights"):
                extra_meta["sleeve_median_weights"] = fold_out["sleeve_median_weights"]
            r = None
            if not args.folds_only:
                # ★ TEST② 数据完备性守卫: test_end 超过数据日历末端时拒绝
                # (避免用部分数据消耗一次性 TEST; 需 --force-partial-test 显式确认)
                _last_cal = calendar[-1] if calendar else None
                if _last_cal is not None and pd.Timestamp(test_e) > pd.Timestamp(_last_cal):
                    if not args.force_partial_test:
                        log.error("=" * 60)
                        log.error("  🚫 TEST② 区间 %s 超过数据末端 %s, 数据不完备。",
                                  test_e, _last_cal.date())
                        log.error("  请使用 --folds-only (不跑 TEST) 或 --force-partial-test 显式确认。")
                        log.error("=" * 60)
                        sys.exit(1)
                    log.warning("  ⚠️ 数据不完备, 但 --force-partial-test 已确认, 继续运行")
                r = run_fold_test(
                    all_data, factor_panels, close_panel, calendar, cal_idx,
                    factor_names, bt_config,
                    fold_out["stable_factors"],
                    fold_out["stable_factor_icir_median"],
                    test_s, test_e, universe_fn=universe_fn,
                    use_regime=True, portfolio_constraints=portfolio_constraints,
                    minute_layer=minute_layer,
                    weight_mode=str(config.get("portfolio_optimizer", "equal")),
                    pool_filter_cfg=config.get("pool_filter"),
                    vol_target_cfg=config.get("vol_target"),
                    trend_timing_cfg=config.get("trend_timing"),
                    industry_lambda=float((config.get("styles") or {}).get("industry_lambda", 0.0)))
            if r:
                results["test"] = r
                # 终极 TEST 锁 (只跑一次纪律)
                with open(TEST_LOCK_PATH, "w", encoding="utf-8") as f:
                    json.dump({
                        "locked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "period": f"{test_s} ~ {test_e}",
                        "script": "run_walkforward_backtest.py --folds",
                        "output": OUTPUT_PATH,
                    }, f, ensure_ascii=False, indent=2)
                log.info("  🔒 终极 TEST 锁已写入: %s", TEST_LOCK_PATH)

            # ★ 扩展模拟考 (--extend-val): 稳定因子固定权重在扩展区间回测
            # (TEST① 毕业数据, 权重训练期 2015-2023 未见此段 → OOS; 不消耗 TEST 锁)
            if args.extend_val:
                ev_s, ev_e = args.extend_val[0], args.extend_val[1]
                guard.check_range(ev_s, ev_e)
                log.info("=" * 60)
                log.info("  扩展模拟考: %s ~ %s (稳定因子固定权重, pool_filter=%s)",
                         ev_s, ev_e, "开" if config.get("pool_filter", {}).get("enabled") else "关")
                log.info("=" * 60)
                ev_r = run_backtest(
                    all_data, factor_panels, close_panel, calendar, cal_idx,
                    factor_names, bt_config, ev_s, ev_e, label="EXTEND",
                    fixed_weights={fn: fold_out["stable_factor_icir_median"][fn]
                                   for fn in fold_out["stable_factors"]},
                    universe_fn=universe_fn, use_regime=True,
                    portfolio_constraints=portfolio_constraints,
                    sleeve_weights=build_extend_sleeve_weights(
                        fold_out, styles_cfg),
                    minute_weights=minute_layer.get("weights")
                    if minute_layer.get("enabled") else None,
                    minute_lambda=float(minute_layer.get("lambda", 0.3)),
                    pool_filter_cfg=config.get("pool_filter"),
                    vol_target_cfg=config.get("vol_target"),
                    weight_mode=str(config.get("portfolio_optimizer", "equal")),
                    trend_timing_cfg=config.get("trend_timing"),
                    industry_lambda=float((config.get("styles") or {}).get("industry_lambda", 0.0)))
                if ev_r:
                    results["extend_val"] = ev_r
                    extra_meta["extend_val"] = {
                        "period": f"{ev_s} ~ {ev_e}",
                        "pool_filter_enabled": bool(
                            config.get("pool_filter", {}).get("enabled")),
                    }
        else:
            for label, (s, e) in partitions_to_run.items():
                log.info("")
                guard.check_range(s, e)
                r = run_backtest(all_data, factor_panels, close_panel,
                                 calendar, cal_idx, factor_names,
                                 bt_config, s, e, label=label.upper(),
                                 universe_fn=universe_fn,
                                 portfolio_constraints=portfolio_constraints,
                                 weight_mode=str(config.get("portfolio_optimizer", "equal")),
                                 pool_filter_cfg=config.get("pool_filter"),
                                 vol_target_cfg=config.get("vol_target"),
                                 trend_timing_cfg=config.get("trend_timing"))
                if r:
                    results[label] = r

    # ── 6. 保存 ──
    output = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": ("方案C Fold Walk-Forward (训练→验证固定权重)"
                            if args.folds
                            else "Walk-Forward 回测: 每个调仓日用滚动252天 ICIR 动态加权"),
            "design": {
                "trail_days": TRAIL_DAYS,
                "label_horizon": LABEL_HORIZON,
                "ic_step": IC_STEP,
                "min_icir": MIN_ICIR,
                "min_ic_obs": MIN_IC_OBS,
                "min_cross_section": MIN_CROSS_SECTION,
                "rebalance_days": REBALANCE_DAYS,
                "no_lookahead": "标签为 [T-252, T-21] 窗口内的21日前瞻收益, 在T时刻已全部实现",
                "pit_universe": ("流动性PIT(全市场+上市1年+成交额500万)"
                                 if args.liquid else "CSI300+ZZ500 月度成分"),
                "expanding_window": "早期数据不足252天时自动退化为扩展窗口",
                "regime_rotation": bool(args.folds),  # 方案C v5: fold 模式启用风格状态机
                "pool_filter": config.get("pool_filter"),  # v10: 股票池分域配置溯源
                "portfolio_constraints": (portfolio_constraints
                                          if portfolio_constraints
                                          else None),
                "minute_factors": ({"enabled": minute_enabled,
                                    "lookback": minute_lookback,
                                    "data_start": MINUTE_DATA_START,
                                    "scope": "仅fold模式; 训练期含2022+的fold(4-5)"
                                             "可用, 其余fold IC截面不足自动跳过"}
                                   if args.folds and minute_enabled else False),
                **({"fold_min_hits": FOLD_MIN_HITS,
                    "fold_icir_min": FOLD_ICIR_MIN,
                    "folds": FOLDS,
                    "ultimate_test": f"{test_s} ~ {test_e} (只跑一次, TEST②)"}
                   if args.folds else {}),
            },
            "n_stocks": len(all_data),
            "n_factors": len(factor_names),
            "bt_config": {k: v for k, v in bt_config.items()},
        },
        "results": results,
    }
    if extra_meta:
        output["meta"].update(extra_meta)

    # ── 执行质量指标 (L1, 2026-08-15): 逐笔 fill vs VWAP/到达价/完美价 ──
    try:
        from execution.exec_quality import fill_quality
        all_trades = []
        for r in results.values():
            for t in (r.get("trades") or []):
                all_trades.append(t)
        if all_trades:
            output["execution_quality"] = fill_quality(all_trades)
    except Exception as e:
        log.warning("执行质量指标计算失败 (不影响主结果): %s", e)

    os.makedirs(IC_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── 汇总 ──
    log.info("")
    log.info("=" * 60)
    log.info("  %-15s %8s %8s %8s %8s %6s %6s",
             "Period", "Excess", "IR", "MaxDD", "Sharpe", "PIT", "NFac")
    log.info("  " + "-" * 68)
    for label, r in results.items():
        log.info("  %-15s %+7.1f%% %7.2f %7.1f%% %7.2f %6d %6d",
                 label.upper(), r["excess_annual"], r["ir"],
                 r["max_drawdown"], r["sharpe"],
                 r["avg_pit_size"], r.get("avg_n_selected_factors", 0))
    if args.folds:
        log.info("  " + "-" * 68)
        log.info("  稳定因子 %d 个 (≥%d/5 folds, |ICIR|≥%.2f)",
                 len(extra_meta.get("stable_factors", [])),
                 FOLD_MIN_HITS, FOLD_ICIR_MIN)
    log.info("=" * 60)

    # ── 实验登记 (2026-08-16): 每次回测自动写入 experiments/ ──
    try:
        from experiment_tracker import log_experiment
        _styles = config.get("styles") or {}
        _partition = ("test" if "test" in results
                      else "development" + ("+extend_val" if "extend_val" in results else ""))
        log_experiment(
            script_name="run_walkforward_backtest",
            partition=_partition,
            config={"top_k": bt_config.get("top_k"),
                    "styles_enabled": bool(_styles.get("enabled")),
                    "styles_budgets": _styles.get("budgets"),
                    "styles_sleeves": {k: v.get("factors") for k, v in
                                       (_styles.get("sleeves") or {}).items()}},
            results={k: {"excess_annual": v.get("excess_annual"),
                         "total_return": v.get("total_return"),
                         "sharpe": v.get("sharpe"),
                         "max_drawdown": v.get("max_drawdown")}
                     for k, v in results.items() if isinstance(v, dict)},
            notes=(f"styles={bool(_styles.get('enabled'))} "
                   f"budgets={_styles.get('budgets')}"),
            experiments_dir=os.path.join(BASE_DIR, "experiments"),
        )
    except Exception as e:
        log.warning("experiment_tracker 登记失败 (不影响主结果): %s", e)

    log.info("  结果: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
