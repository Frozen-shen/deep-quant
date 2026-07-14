"""
LLM 因子引擎 — 将非结构化财经文本转为结构化量化因子

架构:
    ┌──────────────────────────────────────────────┐
    │  LLMFactorEngine                              │
    │  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
    │  │ OpenAI   │  │ Ollama   │  │ RuleBased  │  │
    │  │ Backend  │  │ Backend  │  │ Fallback   │  │
    │  └────┬─────┘  └────┬─────┘  └─────┬──────┘  │
    │       └──────────────┴─────────────┘         │
    │                      │                        │
    │              ┌───────▼────────┐               │
    │              │  Cache Layer   │               │
    │              │  (文本哈希)     │               │
    │              └───────┬────────┘               │
    │                      │                        │
    │  输入: 新闻文本  →  输出: {sentiment, event, …}│
    └──────────────────────────────────────────────┘

用法:
    engine = LLMFactorEngine(backend="mock")       # 离线测试
    engine = LLMFactorEngine(backend="openai",     # 真实 API
                             api_key="sk-...")
    engine = LLMFactorEngine(backend="ollama",     # 本地模型
                             model="qwen2.5:7b")

    scores = engine.batch_score(news_df)
"""

import os
import json
import hashlib
import time
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
from functools import lru_cache

import pandas as pd
import numpy as np


# ============================================================================
# 提示词模板
# ============================================================================

SYSTEM_PROMPT = """你是一个专业的金融分析师，专注于评估财经新闻对A股个股的短期（1-5个交易日）市场影响。

对于每条新闻，请严格按以下JSON格式输出分析结果（不要输出任何其他内容）:
{
  "sentiment": 0.0,       // 情感评分，-1.0(非常利空) 到 1.0(非常利好)，0为中性
  "impact_score": 0.0,    // 影响力评分，0.0(无影响) 到 1.0(重大影响)
  "event_type": "",       // 事件类型: earnings_beat/earnings_miss/buyback/policy_positive/
                          //   policy_negative/management_guidance/analyst_upgrade/
                          //   analyst_downgrade/negative_rumor/capital_flow/
                          //   macro_headwind/macro_tailwind/industry_news/other
  "urgency": "low",       // 紧迫度: low/medium/high
  "summary": ""           // 10字以内的一句话摘要
}

评分原则:
- sentiment > 0.5: 明确利好（如业绩超预期、回购、政策支持）
- sentiment < -0.5: 明确利空（如业绩暴雷、监管处罚、负面舆情）
- sentiment ≈ 0: 中性或影响不明显
- impact_score: 考虑信息的新颖性、权威性、与公司基本面的关联度
- urgency "high": 信息有即时交易价值（如突发利空、重大合同公告）
"""

USER_PROMPT_TEMPLATE = """请分析以下关于股票 {symbol}（{stock_name}）的财经新闻：

标题: {title}

内容: {content}

请输出JSON格式的分析结果。"""


# ============================================================================
#  事件→交易动作 提示词模板 (LLM 作为主信号源)
# ============================================================================

EVENT_ACTION_SYSTEM_PROMPT = """你是一个专业的量化交易策略分析师，你的任务是根据公司公告/研报事件直接生成交易信号。

对于每个公司事件，严格按以下JSON格式输出:

{
  "action": "buy",          // 交易动作: "buy"(买入), "sell"(卖出), "hold"(观望)
  "confidence": 0.0,       // 对交易动作的信心度 0.0~1.0
  "horizon_days": 5,       // 建议持有天数 (1~20)
  "reason": ""             // 15字以内的决策理由
}

决策原则:
- action="buy" + confidence > 0.7: 明确利好事件(如业绩超预期、大额回购、重大利好合同)
- action="sell" + confidence > 0.7: 明确利空事件(如监管处罚、大股东减持、业绩暴雷)
- action="hold": 常规公告、影响不明确的事件
- confidence > 0.9: 极端事件，几乎确定会引发大幅波动
- confidence 0.5~0.7: 有影响但不确定性较高
- confidence < 0.5: 轻微影响或噪声
- horizon_days: 利好/利空的预期影响时长
  * 1~3天: 短期情绪冲击(如研报评级调整)
  * 5~10天: 基本面事件(如财报超预期、重大合同)
  * 10~20天: 重大结构性变化(如资产重组、控制权变更)
"""

EVENT_ACTION_USER_TEMPLATE = """请分析以下 {stock_name}（{symbol}）的公司公告/事件，判断是否应该买入或卖出:

公告标题: {title}
公告类型: {notice_type}
重要性评级: {importance}

请输出JSON格式的交易决策。"""


# ============================================================================
# 后端抽象
# ============================================================================

class LLMBackend(ABC):
    """LLM 后端抽象基类"""

    @abstractmethod
    def query(self, system_prompt: str, user_prompt: str) -> str:
        """发送请求，返回原始文本响应"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class OpenAIBackend(LLMBackend):
    """OpenAI 兼容 API 后端（支持任何兼容接口，如 DeepSeek、Qwen API 等）"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini",
                 base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def query(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai 库: pip install openai")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # 低温度保证输出一致性
            max_tokens=300,
        )
        return resp.choices[0].message.content


class OllamaBackend(LLMBackend):
    """Ollama 本地模型后端"""

    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def query(self, system_prompt: str, user_prompt: str) -> str:
        import requests as req

        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        resp = req.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["response"]


class RuleBasedBackend(LLMBackend):
    """基于词典的规则引擎后端 — 零依赖、零延迟的兜底方案"""

    # 积极词汇
    POSITIVE_WORDS = [
        "增长", "超预期", "突破", "回购", "利好", "分红", "创新高",
        "业绩预增", "扭亏", "中标", "签订", "战略合作", "补贴",
        "升级", "获批", "上市", "扩张", "签约", "领跑", "夺冠",
        "强烈推荐", "上调", "买入", "增持", "展望积极", "信心",
        "新业务", "量产", "交付", "提价", "供不应求", "景气",
        "政策支持", "减税", "降准", "宽松", "刺激", "扶持",
    ]

    # 消极词汇
    NEGATIVE_WORDS = [
        "下降", "亏损", "暴雷", "违约", "处罚", "调查", "诉讼",
        "减持", "套现", "跌停", "暴跌", "预警", "预亏", "下滑",
        "召回", "停产", "限产", "裁员", "缩减", "关店", "失败",
        "下调", "卖出", "减持", "看空", "警告", "风险提示",
        "投诉", "曝光", "造假", "违规", "处罚", "整改",
        "贸易摩擦", "制裁", "加税", "收紧", "收紧", "监管",
    ]

    # 事件关键词 → 事件类型
    EVENT_PATTERNS = {
        "earnings_beat": ["业绩超预期", "利润增长", "营收增长", "扭亏为盈", "业绩预增"],
        "earnings_miss": ["业绩下滑", "亏损", "预亏", "暴雷", "业绩不及预期"],
        "buyback": ["回购", "增持"],
        "policy_positive": ["政策支持", "减税", "补贴", "产业规划", "利好政策"],
        "policy_negative": ["监管", "处罚", "整改", "收紧", "加税", "制裁"],
        "management_guidance": ["董事长", "CEO", "总裁", "管理层", "股东会"],
        "analyst_upgrade": ["上调评级", "强烈推荐", "买入评级", "目标价上调"],
        "analyst_downgrade": ["下调评级", "卖出评级", "目标价下调"],
        "negative_rumor": ["投诉", "曝光", "谣言", "质疑", "维权"],
        "capital_flow": ["北向资金", "减持", "增持", "主力资金"],
        "macro_headwind": ["PMI", "经济下行", "不景气", "消费疲软"],
        "macro_tailwind": ["经济复苏", "消费回暖", "GDP", "景气回升"],
    }

    @property
    def name(self) -> str:
        return "rule_based"

    def query(self, system_prompt: str, user_prompt: str) -> str:
        # 从 user_prompt 中提取标题和内容
        title_match = re.search(r"标题:\s*(.+?)(?:\n|$)", user_prompt)
        content_match = re.search(r"内容:\s*(.+?)(?:\n|$)", user_prompt, re.DOTALL)

        title = title_match.group(1).strip() if title_match else ""
        content = content_match.group(1).strip() if content_match else ""
        text = title + " " + content

        # 情感计算
        pos_count = sum(1 for w in self.POSITIVE_WORDS if w in text)
        neg_count = sum(1 for w in self.NEGATIVE_WORDS if w in text)

        if pos_count + neg_count == 0:
            sentiment = 0.0
        else:
            sentiment = (pos_count - neg_count) / (pos_count + neg_count + 2)
            sentiment = max(-1.0, min(1.0, sentiment * 1.5))

        # 影响力
        total_matches = pos_count + neg_count
        impact_score = min(1.0, total_matches / 8.0 + 0.2)

        # 事件类型识别
        event_type = "other"
        for evt_type, keywords in self.EVENT_PATTERNS.items():
            if any(kw in text for kw in keywords):
                event_type = evt_type
                break

        # 紧迫度
        urgent_words = ["突发", "紧急", "立即", "重大", "刚刚", "快讯"]
        if any(w in text for w in urgent_words):
            urgency = "high"
        elif abs(sentiment) > 0.5:
            urgency = "medium"
        else:
            urgency = "low"

        result = {
            "sentiment": round(sentiment, 2),
            "impact_score": round(impact_score, 2),
            "event_type": event_type,
            "urgency": urgency,
            "summary": title[:20] if title else "无标题",
        }

        return json.dumps(result, ensure_ascii=False)


class MockBackend(LLMBackend):
    """
    Mock 后端 — 返回预设的模拟分数，用于离线测试全管线。

    注意：此模式下的回测结果无实际参考价值，仅用于验证代码流程。
    """

    @property
    def name(self) -> str:
        return "mock"

    def query(self, system_prompt: str, user_prompt: str) -> str:
        # 基于文本哈希生成确定性但看起来合理的分数
        text_hash = int(hashlib.md5(user_prompt.encode()).hexdigest()[:8], 16)
        np.random.seed(text_hash)

        # 检测是否为事件→动作模式
        if "交易动作" in system_prompt or "action" in system_prompt.lower():
            actions = ["buy", "buy", "sell", "hold", "hold", "hold", "buy", "sell"]
            action = actions[text_hash % len(actions)]
            confidence = round(np.random.uniform(0.4, 0.95), 2) if action != "hold" else round(np.random.uniform(0.1, 0.5), 2)
            horizon = np.random.choice([3, 5, 7, 10, 15])
            result = {
                "action": action,
                "confidence": confidence,
                "horizon_days": int(horizon),
                "reason": f"模拟决策 (hash={text_hash})",
            }
            return json.dumps(result, ensure_ascii=False)

        # 默认：情感评分模式
        sentiment = round(np.random.uniform(-0.8, 0.9), 2)
        impact = round(np.random.uniform(0.1, 0.9), 2)
        events = ["earnings_beat", "buyback", "policy_positive", "analyst_upgrade",
                  "negative_rumor", "macro_headwind", "capital_flow", "other"]
        event = events[text_hash % len(events)]
        urgency = ["low", "medium", "high"][text_hash % 3]

        result = {
            "sentiment": sentiment,
            "impact_score": impact,
            "event_type": event,
            "urgency": urgency,
            "summary": "模拟评分",
        }
        return json.dumps(result, ensure_ascii=False)


# ============================================================================
# 主引擎
# ============================================================================

class LLMFactorEngine:
    """
    LLM 因子引擎主类。

    参数
    ----
    backend : str
        "openai" / "ollama" / "rule_based" / "mock"
    api_key : str, optional
        OpenAI API Key (backend="openai" 时必填)
    model : str, optional
        模型名 (默认 gpt-4o-mini / qwen2.5:7b)
    base_url : str, optional
        API 基础URL (支持自定义兼容端点)
    cache_dir : str, optional
        缓存目录路径
    stock_names : dict, optional
        股票代码 → 中文名称映射
    """

    STOCK_NAMES = {
        "600519": "贵州茅台",
        "000858": "五粮液",
        "300750": "宁德时代",
        "002594": "比亚迪",
        "000001": "平安银行",
        "600036": "招商银行",
        "01810": "小米集团",
        "00700": "腾讯控股",
        "09988": "阿里巴巴",
        "09618": "京东集团",
        "09888": "百度集团",
    }

    def __init__(
        self,
        backend: str = "mock",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_dir: Optional[str] = None,
        stock_names: Optional[Dict[str, str]] = None,
    ):
        self.stock_names = stock_names or self.STOCK_NAMES

        # 初始化后端
        if backend == "openai":
            if api_key is None:
                api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OpenAI 后端需要 api_key，或设置 OPENAI_API_KEY 环境变量")
            model = model or "gpt-4o-mini"
            base_url = base_url or "https://api.openai.com/v1"
            self._backend = OpenAIBackend(api_key, model, base_url)

        elif backend == "ollama":
            model = model or "qwen2.5:7b"
            self._backend = OllamaBackend(model)

        elif backend == "rule_based":
            self._backend = RuleBasedBackend()

        elif backend == "mock":
            self._backend = MockBackend()

        else:
            raise ValueError(f"不支持的后端: {backend}，可选: openai/ollama/rule_based/mock")

        # 缓存
        self._cache: Dict[str, Dict] = {}
        self._cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._load_cache()

        self._request_count = 0
        print(f"[LLMFactor] 初始化完成, 后端: {self._backend.name}")

    # ----- 缓存 -----
    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def _load_cache(self):
        if not self._cache_dir:
            return
        cache_file = os.path.join(self._cache_dir, "llm_factor_cache.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                self._cache = json.load(f)

    def _save_cache(self):
        if not self._cache_dir:
            return
        cache_file = os.path.join(self._cache_dir, "llm_factor_cache.json")
        # 将缓存中的 Timestamp 等对象转成可序列化类型
        serializable = {}
        for k, v in self._cache.items():
            clean = {}
            for fk, fv in v.items():
                if hasattr(fv, "isoformat"):  # Timestamp / datetime
                    clean[fk] = fv.isoformat()
                elif isinstance(fv, (np.integer,)):
                    clean[fk] = int(fv)
                elif isinstance(fv, (np.floating,)):
                    clean[fk] = float(fv)
                else:
                    clean[fk] = fv
            serializable[k] = clean
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    # ----- 核心方法 -----
    def score_single(
        self, symbol: str, title: str, content: str
    ) -> Dict[str, Any]:
        """
        对单条新闻进行情感/事件评分。

        返回
        ----
        dict: {sentiment, impact_score, event_type, urgency, summary}
        """
        text_key = title + content[:200]  # 用前200字做键
        cache_key = self._text_hash(text_key)

        # 查缓存
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 构造 prompt
        stock_name = self.stock_names.get(symbol, symbol)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            symbol=symbol,
            stock_name=stock_name,
            title=title,
            content=content[:800],  # 截断过长的内容
        )

        # 调用后端
        try:
            raw_resp = self._backend.query(SYSTEM_PROMPT, user_prompt)
            self._request_count += 1
        except Exception as e:
            print(f"[LLMFactor] 后端调用失败: {e}，使用规则引擎兜底")
            fallback = RuleBasedBackend()
            raw_resp = fallback.query(SYSTEM_PROMPT, user_prompt)

        # 解析 JSON
        result = self._parse_response(raw_resp)

        # 缓存 & 返回
        self._cache[cache_key] = result
        if self._request_count % 10 == 0:
            self._save_cache()

        return result

    def batch_score(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        批量评分一批新闻。

        参数
        ----
        news_df : pd.DataFrame
            必须包含: symbol, title, content

        返回
        ----
        pd.DataFrame
            原 DataFrame + 新增列:
            - llm_sentiment : float
            - llm_impact : float
            - llm_event : str
            - llm_urgency : str
            - llm_summary : str
        """
        results = []
        total = len(news_df)

        for i, row in news_df.iterrows():
            symbol = row.get("symbol", "")
            title = row.get("title", "")
            content = row.get("content", "")

            score = self.score_single(symbol, title, content)
            score["_symbol"] = symbol
            score["_date"] = row.get("date", None)
            results.append(score)

            if (i + 1) % 10 == 0:
                print(f"[LLMFactor] 进度: {i + 1}/{total}")

        # 存缓存
        self._save_cache()

        # 合并
        score_df = pd.DataFrame(results)
        news_df = news_df.reset_index(drop=True)

        news_df["llm_sentiment"] = score_df["sentiment"].values
        news_df["llm_impact"] = score_df["impact_score"].values
        news_df["llm_event"] = score_df["event_type"].values
        news_df["llm_urgency"] = score_df["urgency"].values
        news_df["llm_summary"] = score_df["summary"].values
        news_df["_date"] = score_df["_date"].values  # 用于日聚合的日期列

        print(f"[LLMFactor] 批量评分完成, 共 {total} 条, "
              f"API调用 {self._request_count} 次, "
              f"缓存命中 {total - self._request_count} 次")

        return news_df

    def aggregate_to_daily(
        self, scored_news: pd.DataFrame, date_col: str = "_date"
    ) -> pd.DataFrame:
        """
        将逐条新闻评分为每日聚合因子。

        返回
        ----
        pd.DataFrame [date, llm_sentiment_daily, llm_impact_daily,
                       llm_news_count, llm_event_dominant]
        """
        if scored_news.empty:
            return pd.DataFrame()

        df = scored_news.copy()
        if date_col not in df.columns:
            return pd.DataFrame()

        df["_dt"] = pd.to_datetime(df[date_col])

        daily = df.groupby("_dt").agg(
            llm_sentiment_daily=("llm_sentiment", "mean"),
            llm_impact_daily=("llm_impact", "mean"),
            llm_news_count=("llm_sentiment", "count"),
            llm_sentiment_max=("llm_sentiment", "max"),
            llm_sentiment_min=("llm_sentiment", "min"),
            llm_sentiment_std=("llm_sentiment", "std"),
        ).reset_index()

        daily = daily.rename(columns={"_dt": "date"})
        daily["llm_sentiment_std"] = daily["llm_sentiment_std"].fillna(0)

        return daily

    @staticmethod
    def _parse_response(raw: str) -> Dict[str, Any]:
        """解析 LLM 返回的 JSON 字符串，做容错处理。"""
        defaults = {
            "sentiment": 0.0,
            "impact_score": 0.0,
            "event_type": "other",
            "urgency": "low",
            "summary": "",
        }

        # 尝试提取 JSON 块
        json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                # 类型校验
                parsed["sentiment"] = float(parsed.get("sentiment", 0))
                parsed["impact_score"] = float(parsed.get("impact_score", 0))
                defaults.update(parsed)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # 钳制范围
        defaults["sentiment"] = max(-1.0, min(1.0, defaults["sentiment"]))
        defaults["impact_score"] = max(0.0, min(1.0, defaults["impact_score"]))

        return defaults

    # ================================================================
    #  事件→交易动作 评分 (LLM 作为主信号源)
    # ================================================================
    def score_event_to_action(
        self, symbol: str, title: str, notice_type: str = "",
        importance: str = "medium",
    ) -> Dict[str, Any]:
        """
        对单条公司事件判断交易动作。

        返回
        ----
        dict: {action: "buy"/"sell"/"hold", confidence: 0.0~1.0,
               horizon_days: int, reason: str}
        """
        text_key = title + notice_type
        cache_key = self._text_hash("event_" + text_key)

        # 查缓存
        if cache_key in self._cache:
            return self._cache[cache_key]

        stock_name = self.stock_names.get(symbol, symbol)
        user_prompt = EVENT_ACTION_USER_TEMPLATE.format(
            stock_name=stock_name,
            symbol=symbol,
            title=title,
            notice_type=notice_type,
            importance=importance,
        )

        # 调用后端
        try:
            raw_resp = self._backend.query(EVENT_ACTION_SYSTEM_PROMPT, user_prompt)
            self._request_count += 1
        except Exception as e:
            print(f"[LLMFactor] 事件评分后端调用失败: {e}，使用规则兜底")
            raw_resp = self._rule_based_event_action(title, notice_type)

        # 解析
        result = self._parse_event_response(raw_resp)

        self._cache[cache_key] = result
        if self._request_count % 10 == 0:
            self._save_cache()

        return result

    def batch_score_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        批量对公司事件评分，生成交易信号。

        参数
        ----
        events_df : pd.DataFrame
            必须包含: symbol, title, notice_type, importance, event_date

        返回
        ----
        pd.DataFrame
            原 DataFrame + 新增:
            - llm_action: "buy"/"sell"/"hold"
            - llm_confidence: float
            - llm_horizon_days: int
            - llm_reason: str
        """
        results = []
        total = len(events_df)

        for i, row in events_df.iterrows():
            score = self.score_event_to_action(
                symbol=row.get("symbol", ""),
                title=row.get("title", ""),
                notice_type=row.get("notice_type", ""),
                importance=row.get("importance", "medium"),
            )
            score["_event_date"] = row.get("event_date", None)
            results.append(score)

            if (i + 1) % 10 == 0:
                print(f"[LLMFactor] 事件评分进度: {i + 1}/{total}")

        self._save_cache()

        score_df = pd.DataFrame(results)
        events_df = events_df.reset_index(drop=True)

        events_df["llm_action"] = score_df["action"].values
        events_df["llm_confidence"] = score_df["confidence"].values
        events_df["llm_horizon_days"] = score_df["horizon_days"].values
        events_df["llm_reason"] = score_df["reason"].values

        buy_count = (events_df["llm_action"] == "buy").sum()
        sell_count = (events_df["llm_action"] == "sell").sum()
        hold_count = (events_df["llm_action"] == "hold").sum()
        print(f"[LLMFactor] 事件评分完成, 共 {total} 条, "
              f"买入={buy_count} 卖出={sell_count} 观望={hold_count}, "
              f"API调用={self._request_count} 次")

        return events_df

    @staticmethod
    def _parse_event_response(raw: str) -> Dict[str, Any]:
        """解析 LLM 事件→动作的 JSON 响应。"""
        defaults = {
            "action": "hold",
            "confidence": 0.0,
            "horizon_days": 5,
            "reason": "",
        }

        json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                action = parsed.get("action", "hold")
                if action not in ("buy", "sell", "hold"):
                    action = "hold"
                defaults["action"] = action
                defaults["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence", 0))))
                defaults["horizon_days"] = max(1, min(20, int(parsed.get("horizon_days", 5))))
                defaults["reason"] = str(parsed.get("reason", ""))[:30]
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        return defaults

    @staticmethod
    def _rule_based_event_action(title: str, notice_type: str) -> str:
        """规则兜底：基于公告类型的关键词判断交易动作。"""
        buy_types = ["回购", "增持", "重大合同", "分红派息", "业绩预告.*增",
                     "股权激励", "战略合作"]
        sell_types = ["减持", "行政处罚", "立案调查", "退市风险", "诉讼",
                      "业绩预告.*亏", "业绩快报.*降"]

        import re as _re
        for pat in buy_types:
            if _re.search(pat, notice_type + title):
                return json.dumps({"action": "buy", "confidence": 0.6,
                                   "horizon_days": 5, "reason": f"规则匹配:{notice_type}"},
                                  ensure_ascii=False)
        for pat in sell_types:
            if _re.search(pat, notice_type + title):
                return json.dumps({"action": "sell", "confidence": 0.6,
                                   "horizon_days": 5, "reason": f"规则匹配:{notice_type}"},
                                  ensure_ascii=False)

        return json.dumps({"action": "hold", "confidence": 0.1,
                           "horizon_days": 1, "reason": "无明确信号"},
                          ensure_ascii=False)
