"""
aux_factors.py — 辅助数据因子构建 (2026-08-07)

基于 data_store/aux_* 缓存构建因子面板 (date × symbol 的 DataFrame),
供 run_walkforward_backtest.py 按 _merge_fundamental_panels 同款
searchsorted PIT 方式并入面板。

因子:
  aux_margin_balance_ratio  融资余额/流通市值            (杠杆水平)
  aux_margin_change_5d      融资余额 5 日变化率          (杠杆资金流入)
  aux_margin_buy_ratio_5d   融资买入额/成交额 5日均值     (杠杆参与度)
  (lhb/dzjy/restricted 因子在数据拉全后补充)

PIT 安全: 两融数据当日盘后披露, 面板日期取 <= date 的最新值即可 (searchsorted)。
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 因子名清单 (供 walkforward 候选池引用)
AUX_FACTOR_NAMES = [
    "aux_margin_balance_ratio",
    "aux_margin_change_5d",
    "aux_margin_buy_ratio_5d",
    "aux_lockup_pressure_30d",
    "aux_lhb_net_20d",
    "aux_dzjy_amount_20d",
    "aux_dzjy_discount_20d",
]

_MARGIN_CACHE_DIR = os.path.join(BASE_DIR, "data_store", "aux_margin")
_RESTRICTED_CACHE_DIR = os.path.join(BASE_DIR, "data_store", "aux_restricted")
_LHB_CACHE_DIR = os.path.join(BASE_DIR, "data_store", "aux_lhb")
_DZJY_CACHE_DIR = os.path.join(BASE_DIR, "data_store", "aux_dzjy")

# 模块级缓存: 原始数据一次性加载
_margin_raw: Optional[pd.DataFrame] = None


def _load_margin_raw() -> pd.DataFrame:
    """加载全部两融日文件 → 长表 (date, code, 融资余额, 融资买入额)。

    注意: 上交所源 (sse) 代码在"标的证券代码"列且含 ETF (510xxx 等),
    深交所源 (szse) 在"证券代码"列; 需合并两列并过滤非 A 股。
    """
    global _margin_raw
    if _margin_raw is not None:
        return _margin_raw
    frames = []
    for f in sorted(os.listdir(_MARGIN_CACHE_DIR)):
        if not f.endswith(".parquet"):
            continue
        try:
            df = pd.read_parquet(os.path.join(_MARGIN_CACHE_DIR, f))
            df["code"] = df["证券代码"].fillna(df["标的证券代码"])
            df["code"] = df["code"].astype(str).str.zfill(6)
            # 仅保留 A 股 (沪 60/68, 深 00/30), 过滤 ETF/基金/指数
            df = df[df["code"].str.match(r"^(60|68|00|30)")]
            frames.append(df[["date", "code", "融资余额", "融资买入额"]])
        except Exception:
            continue
    _margin_raw = pd.concat(frames, ignore_index=True)
    _margin_raw["date"] = pd.to_datetime(_margin_raw["date"])
    _margin_raw = _margin_raw.sort_values("date")
    return _margin_raw


def _float32_panel(piv: pd.DataFrame) -> pd.DataFrame:
    """透视结果转 float32 (内存友好, 与 neutralization 一致)。"""
    return piv.astype(np.float32)


def _mktcap_panel() -> pd.DataFrame:
    """流通市值面板: close × outstanding_share (date × symbol), 读日线构建。"""
    store = os.path.join(BASE_DIR, "data_store")
    rows = {}
    for f in os.listdir(store):
        if not f.endswith(".parquet") or f.startswith("index_") or "minute" in f:
            continue
        sym = f[:-8]
        try:
            df = pd.read_parquet(os.path.join(store, f), columns=["date", "close", "outstanding_share"])
        except Exception:
            try:
                df = pd.read_parquet(os.path.join(store, f), columns=["date", "close"])
            except Exception:
                continue
        if "outstanding_share" not in df.columns or df["outstanding_share"].isna().all():
            continue
        df = df[df["close"] > 0]
        if len(df) == 0:
            continue
        df["mktcap"] = df["close"] * df["outstanding_share"]
        rows[sym] = df.set_index("date")["mktcap"]
    return pd.DataFrame(rows, dtype=np.float32).sort_index()


def _amount_panel(window: int = 5) -> pd.DataFrame:
    """成交额滚动均值面板 (date × symbol)。"""
    store = os.path.join(BASE_DIR, "data_store")
    rows = {}
    for f in os.listdir(store):
        if not f.endswith(".parquet") or f.startswith("index_") or "minute" in f:
            continue
        sym = f[:-8]
        try:
            df = pd.read_parquet(os.path.join(store, f), columns=["date", "amount"])
        except Exception:
            continue
        if len(df) == 0:
            continue
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        rows[sym] = df.set_index("date")["amount"].rolling(window, min_periods=3).mean()
    return pd.DataFrame(rows, dtype=np.float32).sort_index()


def build_margin_panels() -> Dict[str, pd.DataFrame]:
    """构建两融因子面板字典 {factor_name: DataFrame(date × symbol)}。

    归一化在构建期完成 (读日线流通市值/成交额):
      - aux_margin_balance_ratio = 融资余额 / 流通市值
      - aux_margin_buy_ratio_5d  = 融资买入额5日均 / 成交额5日均
      - aux_margin_change_5d     = 融资余额 5 日变化率 (无需归一化)
    """
    raw = _load_margin_raw()
    if len(raw) == 0:
        return {}

    balance = raw.pivot_table(index="date", columns="code", values="融资余额",
                              aggfunc="last")
    buy = raw.pivot_table(index="date", columns="code", values="融资买入额",
                          aggfunc="last")

    panels = {}
    panels["aux_margin_change_5d"] = _float32_panel(balance.pct_change(5, fill_method=None))

    mktcap = _mktcap_panel()
    balance_aligned = balance.reindex(index=mktcap.index, columns=mktcap.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = balance_aligned / mktcap.where(mktcap > 0)
    panels["aux_margin_balance_ratio"] = _float32_panel(ratio)

    amount5 = _amount_panel(5)
    buy5 = buy.rolling(5, min_periods=3).mean().reindex(
        index=amount5.index, columns=amount5.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        buy_ratio = buy5 / amount5.where(amount5 > 0)
    panels["aux_margin_buy_ratio_5d"] = _float32_panel(buy_ratio)

    return panels


def build_lockup_panels(mktcap: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    """构建解禁压力因子面板:
      aux_lockup_pressure_30d = 未来30日内解禁市值 / 当前流通市值

    PIT: 解禁计划为已公开信息 (公告时点已知), 非前视。
    """
    frames = []
    for f in os.listdir(_RESTRICTED_CACHE_DIR):
        if not f.endswith(".parquet"):
            continue
        try:
            df = pd.read_parquet(os.path.join(_RESTRICTED_CACHE_DIR, f),
                                 columns=["解禁时间", "实际解禁数量市值", "code"])
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return {}
    raw = pd.concat(frames, ignore_index=True)
    raw["解禁时间"] = pd.to_datetime(raw["解禁时间"], errors="coerce")
    raw = raw.dropna(subset=["解禁时间"])
    raw["date"] = raw["解禁时间"].dt.normalize()
    raw = raw[["date", "code", "实际解禁数量市值"]]

    # 当日解禁市值面板 (date × code)
    daily = raw.pivot_table(index="date", columns="code", values="实际解禁数量市值",
                            aggfunc="sum")
    daily = daily.sort_index()

    # 未来30自然日累计解禁市值: 倒序 rolling (含当日)
    fwd = daily.iloc[::-1].rolling(31, min_periods=1).sum().iloc[::-1]

    if mktcap is None:
        mktcap = _mktcap_panel()
    fwd_aligned = fwd.reindex(index=mktcap.index, columns=mktcap.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        pressure = fwd_aligned / mktcap.where(mktcap > 0)
    panels = {"aux_lockup_pressure_30d": _float32_panel(pressure)}
    return panels


def _load_daily_panel(cache_dir: str, value_col: str, code_col: str = "code") -> pd.DataFrame:
    """通用: 读按日文件 → date × code 面板 (当日值)。"""
    frames = []
    for f in os.listdir(cache_dir):
        if not f.endswith(".parquet"):
            continue
        try:
            df = pd.read_parquet(os.path.join(cache_dir, f))
            ccol = code_col
            if ccol not in df.columns:
                for cand in ("证券代码", "代码"):
                    if cand in df.columns:
                        ccol = cand
                        break
                else:
                    continue
            df = df[[ccol, "date", value_col]]
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"])
    piv = raw.pivot_table(index="date", columns=ccol, values=value_col, aggfunc="sum")
    return piv.sort_index()


def build_lhb_panels(mktcap: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    """龙虎榜因子:
      aux_lhb_net_20d = 过去20日龙虎榜净买额累计 / 流通市值 (大资金净流入)
    """
    net = _load_daily_panel(_LHB_CACHE_DIR, "龙虎榜净买额")
    if len(net) == 0:
        return {}
    net20 = net.rolling(20, min_periods=5).sum()
    if mktcap is None:
        mktcap = _mktcap_panel()
    net_aligned = net20.reindex(index=mktcap.index, columns=mktcap.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = net_aligned / mktcap.where(mktcap > 0)
    return {"aux_lhb_net_20d": _float32_panel(ratio)}


def build_dzjy_panels(mktcap: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    """大宗交易因子:
      aux_dzjy_amount_20d    = 过去20日大宗成交额/流通市值累计 (数据源已归一化)
      aux_dzjy_discount_20d  = 过去20日折溢率均值 (负=折价成交)
    """
    amt = _load_daily_panel(_DZJY_CACHE_DIR, "成交额/流通市值")
    disc = _load_daily_panel(_DZJY_CACHE_DIR, "折溢率")
    panels = {}
    if len(amt) > 0:
        panels["aux_dzjy_amount_20d"] = _float32_panel(
            amt.rolling(20, min_periods=5).sum())
    if len(disc) > 0:
        panels["aux_dzjy_discount_20d"] = _float32_panel(
            disc.rolling(20, min_periods=5).mean())
    return panels


def get_aux_factor_names() -> List[str]:
    return list(AUX_FACTOR_NAMES)
