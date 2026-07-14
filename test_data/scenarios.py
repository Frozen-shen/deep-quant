"""
测试场景配置

定义用于量化策略验证的各种数据集和场景。
"""

from dataclasses import dataclass, field
from typing import Dict, List

# ================================================================
#  基准股票池
# ================================================================

BENCHMARK_STOCKS = {
    "600519": {
        "name": "贵州茅台", "market": "a", "sector": "白酒消费",
        "style": "大盘蓝筹", "volatility": "中低",
        "start": "20180101", "end": "20260710",
        "description": "A股标杆，长期单边牛市，适合测试趋势策略是否跑输买入持有",
    },
    "300750": {
        "name": "宁德时代", "market": "a", "sector": "新能源电池",
        "style": "成长型", "volatility": "高",
        "start": "20180611", "end": "20260710",
        "description": "高波动成长股，适合测试策略在剧烈波动中的表现",
    },
    "01810": {
        "name": "小米集团", "market": "hk", "sector": "消费电子+汽车",
        "style": "波段型", "volatility": "中高",
        "start": "20180709", "end": "20260710",
        "description": "港股科技股，波段特征明显，测试T+0环境下的策略",
    },
    "00700": {
        "name": "腾讯控股", "market": "hk", "sector": "互联网",
        "style": "大市值", "volatility": "中",
        "start": "20180101", "end": "20260710",
        "description": "港股最大市值，流动性极好，测试大市值策略表现",
    },
    "000001": {
        "name": "平安银行", "market": "a", "sector": "银行",
        "style": "低波动", "volatility": "低",
        "start": "20180101", "end": "20260710",
        "description": "低波动金融股，测试策略在低波动环境是否值得交易",
    },
}

# ================================================================
#  场景定义 (时间段)
# ================================================================

SCENARIOS = {
    "bull_2019_2021": {
        "name": "单边牛市",
        "start": "20190102", "end": "20211231",
        "symbol": "600519",
        "description": "茅台从600涨到2200，测试趋势策略是否跑输买入持有",
        "expected": {"策略应跑输买入持有", "交易次数<全周期平均"},
    },
    "bear_2018": {
        "name": "单边熊市",
        "start": "20180102", "end": "20181228",
        "symbol": "600519",
        "description": "2018全年阴跌，测试策略的防御能力",
        "expected": {"策略应减少回撤", "可能空仓避险"},
    },
    "crash_2020_covid": {
        "name": "新冠崩盘",
        "start": "20200102", "end": "20200630",
        "symbol": "01810",
        "description": "2020年3月全球流动性危机+V型反弹",
        "expected": {"极端波动下交易信号密集", "V型反弹中可能踏空"},
    },
    "sideways_2023": {
        "name": "震荡整理",
        "start": "20230101", "end": "20231231",
        "symbol": "600519",
        "description": "2023年茅台横盘整理，测试策略在无趋势环境的表现",
        "expected": {"频繁假信号", "手续费磨损"},
    },
    "multi_stock_2024": {
        "name": "多股票2024",
        "start": "20240101", "end": "20241231",
        "symbols": ["600519", "300750", "01810", "00700", "000001"],
        "description": "5只股票在2024年的横截面对比",
    },
}

# ================================================================
#  已知结果 (用于回归测试)
# ================================================================

KNOWN_RESULTS = {
    "600519_ma5x20": {
        "symbol": "600519", "market": "a",
        "strategy": "MA5×MA20",
        "start": "20180101", "end": "20260710",
        # 这些值会在生成时自动计算，填入 manifest
        "expected_trades_min": 50,
        "expected_trades_max": 200,
        "expected_max_dd_max": -0.80,  # 最大回撤不超过80%
        "expected_sharpe_range": (-1.0, 2.0),
    },
    "01810_ma5x20": {
        "symbol": "01810", "market": "hk",
        "strategy": "MA5×MA20",
        "start": "20180709", "end": "20260710",
        "expected_trades_min": 40,
        "expected_trades_max": 200,
        "expected_max_dd_max": -0.80,
        "expected_sharpe_range": (-1.0, 2.0),
    },
}

# ================================================================
#  模拟事件 (用于LLM测试)
# ================================================================

MOCK_EVENTS = [
    # 利好事件
    {"symbol": "600519", "date": "2019-04-01", "title": "贵州茅台：2019年一季度净利润同比增长30%",
     "notice_type": "季度报告", "importance": "critical",
     "expected_llm_action": "buy", "expected_confidence_min": 0.6},
    {"symbol": "01810", "date": "2024-03-28", "title": "小米集团：SU7正式发布，24小时订单超8万台",
     "notice_type": "重大合同", "importance": "critical",
     "expected_llm_action": "buy", "expected_confidence_min": 0.7},
    {"symbol": "600519", "date": "2022-03-08", "title": "贵州茅台：关于控股股东增持公司股份计划的公告",
     "notice_type": "股东增持", "importance": "high",
     "expected_llm_action": "buy", "expected_confidence_min": 0.5},
    # 利空事件
    {"symbol": "01810", "date": "2022-01-27", "title": "小米集团：印度税务部门冻结公司资产",
     "notice_type": "诉讼", "importance": "high",
     "expected_llm_action": "sell", "expected_confidence_min": 0.6},
    {"symbol": "600519", "date": "2021-02-18", "title": "贵州茅台：市场监督总局约谈白酒企业",
     "notice_type": "行政处罚", "importance": "critical",
     "expected_llm_action": "sell", "expected_confidence_min": 0.5},
    {"symbol": "300750", "date": "2023-12-01", "title": "宁德时代：关于股东减持股份预披露公告",
     "notice_type": "股东减持", "importance": "high",
     "expected_llm_action": "sell", "expected_confidence_min": 0.6},
    # 中性事件
    {"symbol": "000001", "date": "2022-05-15", "title": "平安银行：关于董事辞职的公告",
     "notice_type": "人事变动", "importance": "medium",
     "expected_llm_action": "hold", "expected_confidence_max": 0.4},
    {"symbol": "600519", "date": "2020-06-01", "title": "贵州茅台：关于为子公司提供担保的公告",
     "notice_type": "担保", "importance": "low",
     "expected_llm_action": "hold", "expected_confidence_max": 0.3},
]
