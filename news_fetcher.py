"""
另类数据获取模块 — 财经新闻 & 公告文本采集

数据源:
- CCTV 新闻联播头条（宏观政策信号）
- 财新头条（市场情绪）
- 搜索式个股新闻（基于关键词过滤）
- Mock 模式：生成模拟新闻数据用于离线测试
"""

import os
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Optional, List

import akshare as ak
import pandas as pd
import requests

# 禁用代理
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")
requests.Session.trust_env = False


class NewsFetcher:
    """
    财经新闻采集器。

    支持三种模式:
    - "live": 从 akshare 各接口拉取真实新闻
    - "mock":  生成模拟数据，供离线测试 LLM 因子管线
    - "hybrid": 优先拉真实数据，失败时 fallback 到 mock
    """

    # 知名公司关键词映射（股票名/简称/产品/高管名）
    STOCK_KEYWORDS = {
        "600519": ["茅台", "贵州茅台", "飞天茅台", "白酒", "酱香", "丁雄军", "i茅台", "生肖茅台"],
        "000858": ["五粮液", "普五", "浓香", "白酒"],
        "300750": ["宁德时代", "CATL", "麒麟电池", "动力电池", "曾毓群", "钠电池", "储能"],
        "000001": ["平安银行", "零售银行"],
        "600036": ["招商银行", "招行", "零售之王"],
        "002594": ["比亚迪", "BYD", "仰望", "方程豹", "刀片电池", "王传福", "新能源汽车"],
    }

    @staticmethod
    def fetch_cctv_headlines(date_str: str = None) -> pd.DataFrame:
        """
        获取 CCTV 新闻联播头条（宏观政策风向）。

        参数
        ----
        date_str : str
            日期，YYYYMMDD 格式，默认昨天

        返回
        ----
        pd.DataFrame [date, title, content, source]
        """
        if date_str is None:
            date_str = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

        try:
            raw = ak.news_cctv(date=date_str)
            df = pd.DataFrame(raw)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["source"] = "cctv"
            df = df.rename(columns={"content": "content_raw"})
            # 统一列名
            df = df[["date", "title", "content_raw", "source"]]
            return df
        except Exception as e:
            print(f"[NewsFetcher] CCTV 新闻获取失败: {e}")
            return pd.DataFrame()

    @staticmethod
    def fetch_caixin_headlines() -> pd.DataFrame:
        """
        获取财新头条新闻列表。

        返回
        ----
        pd.DataFrame [date, title, content_raw, source, url]
        """
        try:
            raw = ak.stock_news_main_cx()
            df = pd.DataFrame(raw)
            df["date"] = datetime.now().strftime("%Y-%m-%d")
            df["source"] = "caixin"
            df = df.rename(columns={"summary": "content_raw", "tag": "title"})
            keep_cols = [c for c in ["date", "title", "content_raw", "source", "url"] if c in df.columns]
            return df[keep_cols]
        except Exception as e:
            print(f"[NewsFetcher] 财新头条获取失败: {e}")
            return pd.DataFrame()

    @classmethod
    def fetch_stock_news(
        cls,
        symbol: str = "600519",
        lookback_days: int = 30,
        mode: str = "hybrid",
    ) -> pd.DataFrame:
        """
        获取与指定股票相关的新闻。

        实现方式：拉取宏观新闻 + 财新头条 → 用关键词匹配过滤 →
        如果结果为空且是 hybrid/mock 模式，生成模拟新闻。

        参数
        ----
        symbol : str
            股票代码
        lookback_days : int
            回溯天数
        mode : str
            "live" / "mock" / "hybrid"

        返回
        ----
        pd.DataFrame [symbol, date, title, content, source]
        """
        # mock 模式：直接返回模拟数据，跳过所有网络请求
        if mode == "mock":
            return cls._generate_mock_news(symbol, lookback_days)

        all_news = []

        # 1. 拉 CCTV 头条
        for i in range(lookback_days):
            date_str = (datetime.now() - timedelta(days=i + 1)).strftime("%Y%m%d")
            cctv = cls.fetch_cctv_headlines(date_str)
            if not cctv.empty:
                all_news.append(cctv)
            time.sleep(0.3)  # 礼貌限速

        # 2. 拉财新头条
        caixin = cls.fetch_caixin_headlines()
        if not caixin.empty:
            all_news.append(caixin)

        if not all_news:
            if mode in ("mock", "hybrid"):
                print("[NewsFetcher] 无真实新闻数据，使用模拟数据")
                return cls._generate_mock_news(symbol, lookback_days)
            return pd.DataFrame()

        combined = pd.concat(all_news, ignore_index=True)

        # 3. 用关键词过滤个股相关新闻
        keywords = cls.STOCK_KEYWORDS.get(symbol, [symbol])
        if keywords:
            pattern = "|".join(keywords)
            mask = combined["title"].str.contains(pattern, case=False, na=False)
            if "content_raw" in combined.columns:
                mask |= combined["content_raw"].str.contains(pattern, case=False, na=False)
            filtered = combined[mask].copy()
        else:
            filtered = combined.copy()

        # 4. 标准化输出
        if filtered.empty:
            if mode in ("mock", "hybrid"):
                print("[NewsFetcher] 关键词过滤后无结果，使用模拟数据")
                return cls._generate_mock_news(symbol, lookback_days)
            return pd.DataFrame()

        filtered["symbol"] = symbol
        filtered["content"] = filtered.get("content_raw", filtered.get("title", ""))
        out_cols = [c for c in ["symbol", "date", "title", "content", "source"] if c in filtered.columns]
        result = filtered[out_cols].reset_index(drop=True)

        print(f"[NewsFetcher] 获取到 {len(result)} 条 {symbol} 相关新闻")
        return result

    @classmethod
    def _generate_mock_news(cls, symbol: str, lookback_days: int = 30) -> pd.DataFrame:
        """
        生成模拟个股新闻数据，用于在没有网络或 API 时测试全管线。

        注意：此数据仅用于测试系统连通性，不能用于真实回测。
        """
        stock_name = {
            "600519": "贵州茅台",
            "000858": "五粮液",
            "300750": "宁德时代",
            "002594": "比亚迪",
        }.get(symbol, f"股票{symbol}")

        # 模拟新闻模板
        templates = [
            {
                "title": f"{stock_name}一季度业绩超预期，净利润同比增长25%",
                "content": f"{stock_name}发布2026年一季报，实现营业收入同比增长18%，净利润同比增长25%，超出市场普遍预期的20%增速。公司表示，产品结构优化和渠道改革是主要驱动力。多家券商上调目标价。",
                "sentiment": 0.85,
                "event": "earnings_beat",
            },
            {
                "title": f"{stock_name}遭遇北向资金大幅减持",
                "content": f"今日北向资金净卖出{stock_name}超5亿元，为近三个月来最大单日净卖出。分析人士指出，这与海外市场波动加剧、部分外资降低新兴市场配置有关。",
                "sentiment": -0.60,
                "event": "capital_flow",
            },
            {
                "title": f"{stock_name}宣布启动新一轮回购计划",
                "content": f"{stock_name}公告称，拟以自有资金不低于10亿元、不超过20亿元回购公司股份，回购价格不超过2000元/股，回购股份将用于员工持股计划。",
                "sentiment": 0.65,
                "event": "buyback",
            },
            {
                "title": f"行业利好：消费税改革预期升温，{stock_name}有望受益",
                "content": f"据接近政策制定层的人士透露，新一轮消费税改革方案正在酝酿中，可能将部分高端消费品消费税后移至零售环节。业内分析认为，{stock_name}等高端白酒企业将从中受益，渠道利润有望增厚。",
                "sentiment": 0.55,
                "event": "policy_positive",
            },
            {
                "title": f"{stock_name}董事长在股东大会上释放乐观信号",
                "content": f"{stock_name}董事长在年度股东大会上表示，公司对全年业绩目标充满信心，产品需求旺盛，渠道库存处于健康水平。同时透露下半年将有新产品线推出。",
                "sentiment": 0.70,
                "event": "management_guidance",
            },
            {
                "title": f"机构研报：上调{stock_name}评级至'强烈推荐'",
                "content": f"某头部券商发布研报，将{stock_name}评级从'推荐'上调至'强烈推荐'，目标价上调15%。报告认为公司品牌壁垒深厚，盈利能力持续提升，当前估值具备吸引力。",
                "sentiment": 0.75,
                "event": "analyst_upgrade",
            },
            {
                "title": f"{stock_name}遭遇产品质量投诉风波",
                "content": f"有消费者在社交平台投诉{stock_name}产品存在质量问题，相关话题引发广泛讨论。公司在回应中表示已启动调查程序，将及时公布结果。市场情绪短期内可能承压。",
                "sentiment": -0.70,
                "event": "negative_rumor",
            },
            {
                "title": f"宏观经济数据不及预期，消费板块承压",
                "content": f"国家统计局公布最新PMI数据为49.2，连续第二个月低于荣枯线。消费板块整体走弱，{stock_name}作为权重股首当其冲。不过分析师认为，防御性消费龙头长期配置价值不变。",
                "sentiment": -0.35,
                "event": "macro_headwind",
            },
        ]

        records = []
        today = datetime.now()
        for i in range(min(lookback_days, len(templates) * 3)):
            # 循环使用模板，略微修改标题避免完全相同
            tmpl = templates[i % len(templates)]
            news_date = today - timedelta(days=(i % lookback_days) + 1)
            variation = chr(ord("A") + (i // len(templates)))
            title = tmpl["title"].replace("A", variation) if i >= len(templates) else tmpl["title"]

            records.append({
                "symbol": symbol,
                "date": news_date.strftime("%Y-%m-%d"),
                "title": title,
                "content": tmpl["content"],
                "source": "mock",
                "_mock_sentiment": tmpl["sentiment"],
                "_mock_event": tmpl["event"],
            })

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        print(f"[NewsFetcher] 生成 {len(df)} 条模拟新闻 (symbol={symbol})")
        return df
