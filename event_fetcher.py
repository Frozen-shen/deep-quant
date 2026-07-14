"""
事件数据获取模块 — 公司公告 + 研报 + 财报事件

LLM 事件驱动策略的核心数据源:
- stock_individual_notice_report: 个股公告（回购、分红、重大合同、股权变动等）
- stock_research_report_em: 券商研报（评级调整、盈利预测、目标价变化）
- _generate_event_mock: 模拟事件数据（离线测试用）

与 news_fetcher.py 的区别:
- news_fetcher: 拉宏观新闻/头条 → LLM 作为技术信号过滤器（配角）
- event_fetcher: 拉公司公告/研报 → LLM 直接生成交易信号（主角）
"""

import os
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional, List

import akshare as ak
import pandas as pd
import numpy as np
import requests

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")
requests.Session.trust_env = False


class EventFetcher:
    """
    公司事件采集器 — 公告 + 研报。

    三种模式:
    - "live":  从 akshare 拉取真实公告和研报
    - "mock":  生成模拟公司事件（用于离线测试全管线）
    - "hybrid": 优先拉真实数据，空则 fallback 到 mock
    """

    # 公告类型 → 重要性评级
    NOTICE_TYPE_IMPORTANCE = {
        "回购进展情况": "high",
        "股东增持": "high",
        "股东减持": "high",
        "业绩预告": "critical",
        "业绩快报": "critical",
        "年度报告": "critical",
        "半年度报告": "critical",
        "季度报告": "critical",
        "重大合同": "high",
        "对外投资": "high",
        "资产重组": "critical",
        "分红派息": "high",
        "股权激励": "high",
        "非公开发行": "high",
        "配股": "high",
        "可转债": "medium",
        "担保": "low",
        "人事变动": "medium",
        "诉讼": "high",
        "行政处罚": "critical",
        "退市风险": "critical",
    }

    @classmethod
    def fetch_notices(
        cls,
        symbol: str = "600519",
        begin_date: str = "2025-01-01",
        end_date: str = "2025-07-10",
        mode: str = "hybrid",
    ) -> pd.DataFrame:
        """
        获取个股公告列表。

        返回
        ----
        pd.DataFrame [symbol, notice_date, title, notice_type, url, importance, source]
        """
        if mode == "mock":
            return cls._generate_event_mock(symbol, begin_date, end_date, mode="notice")

        try:
            raw = ak.stock_individual_notice_report(
                security=symbol,
                begin_date=begin_date,
                end_date=end_date,
            )
            if raw.empty:
                if mode == "hybrid":
                    return cls._generate_event_mock(symbol, begin_date, end_date, mode="notice")
                return pd.DataFrame()

            df = pd.DataFrame(raw)
            df = df.rename(columns={
                "代码": "symbol",
                "公告标题": "title",
                "公告类型": "notice_type",
                "公告日期": "notice_date",
                "网址": "url",
            })

            # 添加重要性
            df["importance"] = df["notice_type"].map(
                cls.NOTICE_TYPE_IMPORTANCE
            ).fillna("low")
            df["source"] = "notice"

            # 日期标准化
            df["notice_date"] = pd.to_datetime(df["notice_date"], errors="coerce")

            keep_cols = ["symbol", "notice_date", "title", "notice_type", "url", "importance", "source"]
            return df[[c for c in keep_cols if c in df.columns]].reset_index(drop=True)

        except Exception as e:
            print(f"[EventFetcher] 公告获取失败: {e}")
            if mode == "hybrid":
                return cls._generate_event_mock(symbol, begin_date, end_date, mode="notice")
            return pd.DataFrame()

    @classmethod
    def fetch_research_reports(
        cls,
        symbol: str = "600519",
        mode: str = "hybrid",
    ) -> pd.DataFrame:
        """
        获取个股研报列表（券商评级报告）。

        返回
        ----
        pd.DataFrame [symbol, report_date, title, rating, institution, earnings_forecast, source]
        """
        if mode == "mock":
            # 研报 mock 也用 generate_event_mock 但标记 source="report"
            return cls._generate_event_mock(symbol, "2025-01-01", "2025-07-10", mode="report")

        try:
            raw = ak.stock_research_report_em(symbol=symbol)
            if raw.empty:
                if mode == "hybrid":
                    return cls._generate_event_mock(symbol, "2025-01-01", "2025-07-10", mode="report")
                return pd.DataFrame()

            df = pd.DataFrame(raw)
            df = df.rename(columns={
                "股票代码": "symbol",
                "报告名称": "title",
                "东财评级": "rating",
                "机构": "institution",
            })

            # 日期列（研报数据可能没有日期列，用当前时间）
            if "报告日期" in df.columns:
                df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
            else:
                df["report_date"] = datetime.now()

            df["source"] = "research_report"
            df["importance"] = "high"  # 研报默认重要

            keep_cols = ["symbol", "report_date", "title", "rating", "institution",
                         "source", "importance"]
            return df[[c for c in keep_cols if c in df.columns]].reset_index(drop=True)

        except Exception as e:
            print(f"[EventFetcher] 研报获取失败: {e}")
            if mode == "hybrid":
                return cls._generate_event_mock(symbol, "2025-01-01", "2025-07-10", mode="report")
            return pd.DataFrame()

    @classmethod
    def fetch_all_events(
        cls,
        symbol: str = "600519",
        begin_date: str = "2025-01-01",
        end_date: str = "2025-07-10",
        mode: str = "hybrid",
    ) -> pd.DataFrame:
        """
        获取所有公司事件（公告 + 研报），合并返回。

        这是 LLM 事件驱动策略的主入口。
        """
        if mode == "mock":
            return cls._generate_event_mock(symbol, begin_date, end_date, mode="all")

        notices = cls.fetch_notices(symbol, begin_date, end_date, mode)
        reports = cls.fetch_research_reports(symbol, mode)

        frames = []
        if not notices.empty:
            frames.append(notices)
        if not reports.empty:
            frames.append(reports)

        if not frames:
            if mode == "hybrid":
                return cls._generate_event_mock(symbol, begin_date, end_date, mode="all")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)

        # 统一事件日期列
        if "notice_date" in combined.columns and "report_date" in combined.columns:
            combined["event_date"] = combined["notice_date"].fillna(combined["report_date"])
        elif "notice_date" in combined.columns:
            combined["event_date"] = combined["notice_date"]
        elif "report_date" in combined.columns:
            combined["event_date"] = combined["report_date"]
        else:
            combined["event_date"] = datetime.now()

        combined["event_date"] = pd.to_datetime(combined["event_date"])

        print(f"[EventFetcher] 获取 {symbol} 事件: "
              f"{len(notices)} 条公告 + {len(reports)} 条研报 = {len(combined)} 条")

        return combined.sort_values("event_date").reset_index(drop=True)

    # ================================================================
    #  Mock 事件生成
    # ================================================================
    @classmethod
    def _generate_event_mock(
        cls, symbol: str, begin_date: str, end_date: str, mode: str = "all"
    ) -> pd.DataFrame:
        """
        生成模拟公司事件数据。每条事件模拟真实公告/研报的标题和类型。
        """
        stock_name = {
            "600519": "贵州茅台", "000858": "五粮液",
            "300750": "宁德时代", "002594": "比亚迪",
        }.get(symbol, f"股票{symbol}")

        # 模拟公告模板
        notice_templates = [
            {
                "title": f"{stock_name}：关于回购股份实施进展的公告",
                "notice_type": "回购进展情况",
                "importance": "high",
                "_llm_action": "buy",
                "_llm_confidence": 0.75,
                "_llm_reason": "公司持续回购，彰显管理层对当前估值的信心",
            },
            {
                "title": f"{stock_name}：2026年第一季度报告",
                "notice_type": "季度报告",
                "importance": "critical",
                "_llm_action": "buy",
                "_llm_confidence": 0.85,
                "_llm_reason": "Q1营收同比增长18%，净利润增长25%，超市场预期",
            },
            {
                "title": f"{stock_name}：关于控股股东减持股份计划的预披露公告",
                "notice_type": "股东减持",
                "importance": "high",
                "_llm_action": "sell",
                "_llm_confidence": 0.80,
                "_llm_reason": "控股股东计划减持2%股份，信号极为负面",
            },
            {
                "title": f"{stock_name}：关于签订重大经营合同的公告",
                "notice_type": "重大合同",
                "importance": "high",
                "_llm_action": "buy",
                "_llm_confidence": 0.70,
                "_llm_reason": "签订大额战略合同，未来2年营收有保障",
            },
            {
                "title": f"{stock_name}：关于收到中国证监会立案调查通知的公告",
                "notice_type": "行政处罚",
                "importance": "critical",
                "_llm_action": "sell",
                "_llm_confidence": 0.95,
                "_llm_reason": "被证监会立案调查，重大不确定性，应立即离场",
            },
            {
                "title": f"{stock_name}：2025年度权益分派实施公告",
                "notice_type": "分红派息",
                "importance": "high",
                "_llm_action": "buy",
                "_llm_confidence": 0.60,
                "_llm_reason": "高分红方案落地，股息率有吸引力",
            },
            {
                "title": f"{stock_name}：关于董事及高级管理人员变动的公告",
                "notice_type": "人事变动",
                "importance": "medium",
                "_llm_action": "hold",
                "_llm_confidence": 0.30,
                "_llm_reason": "常规人事调整，对经营影响有限",
            },
            {
                "title": f"{stock_name}：关于为子公司提供担保的公告",
                "notice_type": "担保",
                "importance": "low",
                "_llm_action": "hold",
                "_llm_confidence": 0.10,
                "_llm_reason": "常规担保事项，不影响基本面",
            },
        ]

        # 模拟研报模板
        report_templates = [
            {
                "title": f"{stock_name}：Q1超预期，上调盈利预测和目标价",
                "notice_type": "研报",
                "importance": "high",
                "_llm_action": "buy",
                "_llm_confidence": 0.80,
                "_llm_reason": "券商上调盈利预测和目标价，基本面持续改善",
            },
            {
                "title": f"{stock_name}：行业竞争加剧，下调评级至中性",
                "notice_type": "研报",
                "importance": "high",
                "_llm_action": "sell",
                "_llm_confidence": 0.65,
                "_llm_reason": "券商下调评级，行业格局恶化",
            },
        ]

        if mode == "notice":
            templates = notice_templates
        elif mode == "report":
            templates = report_templates
        else:
            templates = notice_templates + report_templates

        # 在日期范围内均匀分布事件
        start_dt = pd.to_datetime(begin_date)
        end_dt = pd.to_datetime(end_date)
        total_days = max((end_dt - start_dt).days, 1)
        interval = max(total_days // len(templates), 1)

        records = []
        for i, tmpl in enumerate(templates):
            event_date = start_dt + timedelta(days=min(i * interval, total_days))
            records.append({
                "symbol": symbol,
                "event_date": event_date,
                "title": tmpl["title"],
                "notice_type": tmpl["notice_type"],
                "importance": tmpl["importance"],
                "source": "mock",
                "_mock_action": tmpl["_llm_action"],
                "_mock_confidence": tmpl["_llm_confidence"],
                "_mock_reason": tmpl["_llm_reason"],
            })

        df = pd.DataFrame(records)
        df["event_date"] = pd.to_datetime(df["event_date"])
        df = df.sort_values("event_date").reset_index(drop=True)

        print(f"[EventFetcher] 生成 {len(df)} 条模拟公司事件 (symbol={symbol}, "
              f"{begin_date[:7]}~{end_date[:7]})")
        return df
