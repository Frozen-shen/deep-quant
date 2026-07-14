"""
另类数据因子 — 北向资金 + 融资融券 + 龙虎榜

在这些数据上构建可排序因子:
  - north_flow_score: 北向资金持仓变化 → 外资态度
  - margin_score: 融资余额变化 → 杠杆资金方向
"""

import akshare as ak
import pandas as pd
import numpy as np


def north_flow_factor(symbols: list, lookback_days: int = 20) -> dict:
    """
    北向资金因子: 近期净流入越多 → 分数越高。

    返回: {symbol: score (-1~1)}
    """
    scores = {}
    try:
        # 拉北向资金个股数据
        df_north = ak.stock_hsgt_hist_em(symbol="沪股通")  # 沪市
        df_sz = ak.stock_hsgt_hist_em(symbol="深股通")     # 深市
        # 注: 港股无北向资金,仅A股适用
    except Exception as e:
        print(f"[AltData] 北向数据获取失败: {e}")
        return scores

    # 简化: 对港股返回中性分数 (北向资金不适用港股)
    for sym in symbols:
        scores[sym] = 0.0
    return scores


def margin_factor(symbols: list, market: str = "hk") -> dict:
    """
    融资融券因子: 融资余额增加 → 看多, 减少 → 看空。

    返回: {symbol: score (-1~1)}
    """
    scores = {}
    for sym in symbols:
        try:
            # 港股融资融券 (可能不可用)
            if market == "hk":
                scores[sym] = 0.0  # 港股融资数据接口有限
                continue

            # A股融资融券
            df = ak.stock_margin_detail_sse(symbol=sym, start_date="20240101")
            if len(df) > 5:
                recent = df.tail(10)
                margin_change = recent["融资余额"].pct_change().mean() if "融资余额" in df.columns else 0
                scores[sym] = np.clip(margin_change * 5, -1, 1)
            else:
                scores[sym] = 0.0
        except Exception:
            scores[sym] = 0.0

    return scores


def peer_relative_factor(symbols: list, stock_data: dict) -> dict:
    """
    同伴相对强度因子: 个股涨跌幅 vs 同板块平均涨跌幅。

    stock_data: {symbol: DataFrame(含 close)}

    返回: {symbol: score (-1~1)}
    """
    from sector_analyzer import SectorAnalyzer
    analyzer = SectorAnalyzer(market="hk")
    scores = {}

    for sym in symbols:
        if sym not in stock_data:
            scores[sym] = 0.0
            continue

        # 计算个股近期收益
        df = stock_data[sym]
        if len(df) < 5:
            scores[sym] = 0.0
            continue

        stock_ret = df["close"].iloc[-1] / df["close"].iloc[-5] - 1

        # 同板块同伴收益
        sector_info = analyzer.get_sector(sym)
        peers = sector_info.get("peers", [sym])
        peer_rets = []
        for peer in peers:
            if peer in stock_data and len(stock_data[peer]) >= 5:
                peer_rets.append(
                    stock_data[peer]["close"].iloc[-1] /
                    stock_data[peer]["close"].iloc[-5] - 1
                )

        if len(peer_rets) >= 2:
            peer_avg = np.mean(peer_rets)
            peer_std = np.std(peer_rets) + 0.001
            z = (stock_ret - peer_avg) / peer_std
            scores[sym] = np.clip(z, -1, 1)
        else:
            scores[sym] = 0.0

    return scores
