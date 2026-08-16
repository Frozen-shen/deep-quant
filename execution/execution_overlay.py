"""
日内执行 overlay — 用因果规则决定执行起点 bar (实验框架, 默认关闭)。

2026-08-15 (L2): 三个规则全部只用"决策时点之前"的数据, 无前视:
  - gap_wait:     开盘缺口超阈值时等回撤到缺口一半以内, 超时强制执行
  - momentum_wait: 前 30 分钟涨跌幅超阈值时等回落, 超时强制执行
  - volume_wait:  量比不足时等放量 bar, 超时强制执行

decide_start_bar 返回首个"允许开始执行"的 bar index, 供 get_pov_fills
的 start_bar 参数使用。rules=None 时返回 0 (行为与无 overlay 完全一致)。
"""
from typing import Dict, List, Optional

import pandas as pd

DEFAULT_RULES = {
    "gap_wait": {"gap_bps": 300, "timeout_bars": 12},
    "momentum_wait": {"mom_bps": 200, "timeout_bars": 12},
    "volume_wait": {"vol_min": 0.5, "timeout_bars": 6},
}


def _gap_wait(day_bars: pd.DataFrame, params: dict, prev_close: float) -> int:
    """高开缺口等回撤: 缺口 |open/prev_close-1| 超 gap_bps 时,
    等收盘回落到缺口一半以内; 超时返回 timeout_bars。"""
    gap_bps = params.get("gap_bps", 300)
    timeout = params.get("timeout_bars", 12)
    open_px = float(day_bars.iloc[0]["开盘"])
    gap = abs(open_px / prev_close - 1) * 1e4
    if gap <= gap_bps:
        return 0
    half = gap_bps / 2.0
    n = len(day_bars)
    for i in range(1, min(n, timeout + 1)):
        dev = abs(float(day_bars.iloc[i]["收盘"]) / prev_close - 1) * 1e4
        if dev <= half:
            return i
    return min(timeout, max(n - 1, 0))


def _momentum_wait(day_bars: pd.DataFrame, side: str, params: dict,
                   prev_close: float) -> int:
    """前 30 分钟急涨(买)/急跌(卖) 等回落: 前 6 根 bar 相对昨收的
    涨跌幅超 mom_bps 时, 等回到阈值一半以内; 超时返回 timeout_bars。"""
    mom_bps = params.get("mom_bps", 200)
    timeout = params.get("timeout_bars", 12)
    n = len(day_bars)
    if n < 6:
        return 0
    ret30 = (float(day_bars.iloc[5]["收盘"]) / prev_close - 1) * 1e4
    if side == "BUY":
        if ret30 <= mom_bps:
            return 0
        target = prev_close * (1 + mom_bps / 2.0 / 1e4)
        for i in range(6, min(n, timeout + 6 + 1)):
            if float(day_bars.iloc[i]["收盘"]) <= target:
                return i
    else:  # SELL
        if ret30 >= -mom_bps:
            return 0
        target = prev_close * (1 - mom_bps / 2.0 / 1e4)
        for i in range(6, min(n, timeout + 6 + 1)):
            if float(day_bars.iloc[i]["收盘"]) >= target:
                return i
    return min(timeout + 5, max(n - 1, 0))


def _volume_wait(day_bars: pd.DataFrame, params: dict) -> int:
    """量比等待: 每根 bar 的量 vs 此前 bar 量的中位数 < vol_min 时
    等放量; 超时返回 timeout_bars。bar 0 不判断 (无条件从开盘可执行)。"""
    vol_min = params.get("vol_min", 0.5)
    timeout = params.get("timeout_bars", 6)
    vols = day_bars["成交量"].astype(float).tolist()
    n = len(vols)
    for i in range(1, min(n, timeout + 1)):
        med = float(pd.Series(vols[:i]).median()) if i > 0 else 0.0
        if med > 0 and vols[i] / med >= vol_min:
            return i
    return min(timeout, max(n - 1, 0))


def decide_start_bar(day_bars: pd.DataFrame, side: str = "BUY",
                     rules: Optional[Dict] = None,
                     prev_close: Optional[float] = None) -> int:
    """返回允许开始执行的 bar index (0=立即执行)。

    Args:
      day_bars: 当日 5m bar DataFrame (时间/开盘/收盘/最高/最低/成交量)
      side: "BUY" / "SELL"
      rules: 启用的规则及参数, 如 {"gap_wait": {...}}; None/空 = 立即执行
      prev_close: 昨日收盘价 (gap/momentum 规则需要; None 时这两条跳过)
    """
    if not rules or len(day_bars) == 0:
        return 0
    starts = [0]
    if prev_close is not None and prev_close > 0:
        if "gap_wait" in rules:
            starts.append(_gap_wait(day_bars, rules["gap_wait"], prev_close))
        if "momentum_wait" in rules:
            starts.append(_momentum_wait(day_bars, side, rules["momentum_wait"],
                                         prev_close))
    if "volume_wait" in rules:
        starts.append(_volume_wait(day_bars, rules["volume_wait"]))
    return max(starts)


def replay_trades(trades: List[dict], rules: Dict, mf=None) -> Dict:
    """重放历史成交: 比较 overlay 规则后成交价 vs 首 bar 立即成交价。

    只读验证 (不跑回测): 对每笔 trade 取当日 bar, 计算
      baseline = 首 bar 收盘价 (无 overlay 的小订单近似成交价)
      overlay  = decide_start_bar 指定 bar 的收盘价
    返回两者相对当日 VWAP 的符号化偏差分布 (用 exec_quality._stats 聚合)。
    """
    from execution.exec_quality import _signed_dev, _stats

    if mf is None:
        from data.minute_fetcher import MinuteFetcher
        mf = MinuteFetcher(allow_network=False)

    base_devs, over_devs, n = [], [], 0
    for t in trades:
        action = t.get("action", "BUY")
        date_str = str(t.get("date", ""))[:10]
        try:
            df = mf.fetch(t["symbol"], days=10, end_date=date_str)
        except Exception:
            df = None
        if df is None or len(df) == 0:
            continue
        date_dt = pd.Timestamp(date_str)
        day = df[df["时间"].dt.date == date_dt.date()]
        if len(day) == 0:
            continue
        day = day.reset_index(drop=True)
        vol = day["成交量"].astype(float)
        closes = day["收盘"].astype(float)
        vwap = float((closes * vol).sum() / vol.sum()) if vol.sum() > 0 \
            else float(closes.mean())
        prev = df[df["时间"].dt.date < date_dt.date()]
        prev_close = float(prev["收盘"].iloc[-1]) if len(prev) > 0 else None

        start = decide_start_bar(day, action, rules, prev_close)
        base_px = float(closes.iloc[0])
        over_px = float(closes.iloc[min(start, len(day) - 1)])
        base_devs.append(_signed_dev(base_px, vwap, action))
        over_devs.append(_signed_dev(over_px, vwap, action))
        n += 1

    return {
        "n": n,
        "baseline_first_bar": {"vwap_dev_bps": _stats(base_devs)},
        "overlay": {"vwap_dev_bps": _stats(over_devs)},
    }
