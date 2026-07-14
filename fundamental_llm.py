"""
LLM 基本面评分 — 财报数据 → LLM → fundamental_score

将财务指标喂给 LLM，让 LLM 判断公司基本面质量。
fundamental_score 注入因子打分器: factor_score += fundamental_score * 0.3

用法:
    from fundamental_llm import FundamentalLLM
    f_llm = FundamentalLLM(backend="openai")
    score = f_llm.evaluate("01810")  # → 0.65 (基本面良好)
"""

import os
import json
import pandas as pd
import akshare as ak


FUNDAMENTAL_PROMPT = """你是一个基本面分析专家。根据以下港股公司的财务数据，给出基本面评分。

评分标准:
  +1.0: 极优质 (利润高增长+低估值+高ROE)
  +0.5: 良好 (利润增长+合理估值)
   0.0: 中性 (无明显亮点或风险)
  -0.5: 较差 (利润下滑+高估值)
  -1.0: 极差 (亏损+高负债+估值虚高)

只输出JSON: {"score": 0.0, "summary": "一句话理由"}"""


class FundamentalLLM:
    """LLM 基本面评估器。"""

    def __init__(self, backend: str = "mock", api_key: str = None,
                 model: str = None, base_url: str = None):
        self.backend = backend
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("LLM_MODEL", "deepseek-chat")
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
        self._cache = {}

    def fetch_financials(self, symbol: str) -> dict:
        """拉取港股财务数据。"""
        try:
            df = ak.stock_hk_financial_indicator_em(symbol=symbol)
            if df.empty:
                return {}
            row = df.iloc[-1]
            return {
                "eps": float(row.get("基本每股收益(元)", 0) or 0),
                "book_per_share": float(row.get("每股净资产(元)", 0) or 0),
                "pe": float(row.get("市盈率", 0) or 0),
                "pb": float(row.get("市净率", 0) or 0),
                "roe": float(row.get("股东权益回报率(%)", 0) or 0),
                "revenue": float(row.get("营业总收入", 0) or 0),
                "net_profit": float(row.get("净利润", 0) or 0),
                "revenue_qoq": float(row.get("营业总收入滚动环比增长(%)", 0) or 0),
                "profit_qoq": float(row.get("净利润滚动环比增长(%)", 0) or 0),
                "net_margin": float(row.get("销售净利率(%)", 0) or 0),
            }
        except Exception as e:
            print(f"[FundamentalLLM] 财务数据获取失败: {e}")
            return {}

    def evaluate(self, symbol: str) -> float:
        """
        评估公司基本面。

        返回: fundamental_score (-1.0 ~ 1.0)
        """
        if symbol in self._cache:
            return self._cache[symbol]

        financials = self.fetch_financials(symbol)
        if not financials:
            self._cache[symbol] = 0.0
            return 0.0

        # Mock 模式: 基于数据规则打分
        if self.backend == "mock":
            score = self._rule_based_score(financials)
            self._cache[symbol] = score
            print(f"[FundamentalLLM] {symbol}: mock_score={score:+.2f} "
                  f"(PE={financials.get('pe',0):.1f}, ROE={financials.get('roe',0):.1f}%)")
            return score

        # LLM 模式
        try:
            from llm_factor import OpenAIBackend
            llm = OpenAIBackend(self.api_key, self.model, self.base_url)
            user_prompt = f"公司代码: {symbol}\n财务数据:\n{json.dumps(financials, ensure_ascii=False, indent=2)}"
            resp = llm.query(FUNDAMENTAL_PROMPT, user_prompt)

            # 解析
            import re
            json_match = re.search(r'\{[^{}]*\}', resp)
            if json_match:
                result = json.loads(json_match.group())
                score = float(result.get("score", 0))
                summary = result.get("summary", "")
                print(f"[FundamentalLLM] {symbol}: LLM_score={score:+.2f} ({summary})")
                self._cache[symbol] = max(-1.0, min(1.0, score))
                return self._cache[symbol]
        except Exception as e:
            print(f"[FundamentalLLM] LLM调用失败,用规则引擎: {e}")

        score = self._rule_based_score(financials)
        self._cache[symbol] = score
        return score

    @staticmethod
    def _rule_based_score(f: dict) -> float:
        """规则引擎后备。"""
        score = 0.0
        pe = f.get("pe", 0)
        roe = f.get("roe", 0)
        revenue_qoq = f.get("revenue_qoq", 0)
        profit_qoq = f.get("profit_qoq", 0)

        if 5 < pe < 30:
            score += 0.2
        elif pe > 100 or pe < 0:
            score -= 0.2

        if roe > 15:
            score += 0.3
        elif roe < 5:
            score -= 0.2

        if revenue_qoq > 5:
            score += 0.2
        elif revenue_qoq < -10:
            score -= 0.2

        if profit_qoq > 10:
            score += 0.2
        elif profit_qoq < -20:
            score -= 0.3

        return max(-1.0, min(1.0, score))
