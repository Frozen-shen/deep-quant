"""
板块分析层 — 手工板块映射 + 同伴比较

akshare 的港股板块接口不可用 → 用手工板块分类 + 同板块股票比较
"""

import pandas as pd
import numpy as np


# ================================================================
#  港股板块映射 (手工维护)
# ================================================================
HK_SECTOR_MAP = {
    "01810": {"sector": "消费电子", "peers": ["01810", "02018", "01415"]},
    "00700": {"sector": "互联网平台", "peers": ["00700", "09988", "03690", "09888"]},
    "09988": {"sector": "互联网平台", "peers": ["09988", "00700", "03690", "09618"]},
    "09618": {"sector": "互联网平台", "peers": ["09618", "09988", "03690"]},
    "03690": {"sector": "互联网平台", "peers": ["03690", "00700", "09988"]},
    "09999": {"sector": "游戏", "peers": ["09999", "00700", "09888"]},
    "02020": {"sector": "体育用品", "peers": ["02020", "02331", "01368"]},
    "02318": {"sector": "保险", "peers": ["02318", "02628", "01339"]},
    "01211": {"sector": "新能源车", "peers": ["01211", "00175", "02015"]},
    "00981": {"sector": "半导体", "peers": ["00981", "01347", "02166"]},
    "02269": {"sector": "医药", "peers": ["02269", "01177", "01801"]},
    "09888": {"sector": "互联网平台", "peers": ["09888", "00700", "09988"]},
}


class SectorAnalyzer:
    """
    板块分析器 — 基于手工映射 + 同伴比较。

    同伴比较: 计算同板块股票的平均收益 → 判断个股是否跑赢板块
    """

    def __init__(self, market: str = "hk"):
        self.market = market
        self.sector_map = HK_SECTOR_MAP if market == "hk" else {}

    def get_sector(self, symbol: str) -> dict:
        """获取股票所属板块信息。"""
        return self.sector_map.get(symbol, {"sector": "未知", "peers": [symbol]})

    def score_at(self, symbol: str, stock_prices: dict) -> float:
        """
        计算板块相对强度评分。

        stock_prices: {symbol: pct_change_today}
        返回: sector_score (-1 = 严重跑输板块, +1 = 大幅跑赢板块)
        """
        sector_info = self.get_sector(symbol)
        peers = sector_info["peers"]
        
        # 收集同伴涨跌幅
        peer_changes = []
        for p in peers:
            if p in stock_prices and stock_prices[p] is not None:
                peer_changes.append(stock_prices[p])
        
        if len(peer_changes) < 2:
            return 0.0
        
        peer_avg = np.mean(peer_changes)
        peer_std = np.std(peer_changes) if len(peer_changes) > 1 else 0.01
        
        stock_change = stock_prices.get(symbol, 0)
        
        # Z-score: 个股涨跌 vs 板块平均
        if peer_std > 0:
            z_score = (stock_change - peer_avg) / peer_std
        else:
            z_score = 0
        
        # 映射到 -1~1
        score = np.clip(z_score / 2, -1, 1)
        
        return float(score)

    def analyze(self, symbol: str, stock_prices: dict) -> dict:
        """完整板块分析。"""
        sector_info = self.get_sector(symbol)
        score = self.score_at(symbol, stock_prices)
        
        return {
            "sector": sector_info["sector"],
            "sector_score": score,
            "vs_peers": "跑赢" if score > 0.2 else ("跑输" if score < -0.2 else "同步"),
        }
