"""
A股交易规则模块 — 涨跌停/停牌/ST/费率约束

用法:
  from trading_rules import TradingRules
  rules = TradingRules()
  
  # 预检: 哪些股票今天可以交易?
  tradeable = rules.filter_tradeable(symbols, price_data)
  
  # 费率: 计算真实A股手续费
  commission = rules.calc_commission(qty, price, "SELL")
"""

import os
import numpy as np
import pandas as pd

# ════════════════════════════════════════
#  费率配置 (A股 2025年标准)
# ════════════════════════════════════════

A_SHARE_COMMISSION = {
    "broker_rate": 0.00025,      # 万2.5 (双边)
    "stamp_duty_sell": 0.0005,   # 千0.5 印花税 (仅卖出)
    "transfer_fee": 0.00001,     # 万0.1 过户费 (双边)
    "min_commission": 5.0,       # 最低5元/笔
}

# 板块涨跌停幅度
BOARD_LIMITS = {
    "main_sh": 0.10,    # 主板上海 (60xxxx)
    "main_sz": 0.10,    # 主板深圳 (00xxxx)
    "gem": 0.20,        # 创业板 (30xxxx)
    "star": 0.20,       # 科创板 (688xxx)
    "st": 0.05,         # ST/*ST
    "bj": 0.30,         # 北交所 (8xxxxx/4xxxxx)
}


def get_board_type(symbol: str) -> str:
    """根据股票代码判断板块类型。"""
    code = str(symbol)
    if code.startswith("60"):
        return "main_sh"
    elif code.startswith("00"):
        return "main_sz"
    elif code.startswith("30"):
        return "gem"
    elif code.startswith("688"):
        return "star"
    elif code.startswith(("83", "87", "43", "8")):
        return "bj"
    return "main_sz"  # default


def get_limit_pct(symbol: str, is_st: bool = False) -> float:
    """获取涨跌停幅度。"""
    if is_st:
        return BOARD_LIMITS["st"]
    return BOARD_LIMITS.get(get_board_type(symbol), 0.10)


def calc_commission(qty: int, price: float, side: str) -> float:
    """
    计算A股真实手续费。
    
    Args:
      qty: 股数
      price: 成交价
      side: "BUY" or "SELL"
    
    Returns:
      手续费 (CNY)
    """
    cfg = A_SHARE_COMMISSION
    amount = qty * price
    
    # 券商佣金 + 过户费
    fee = amount * (cfg["broker_rate"] + cfg["transfer_fee"])
    
    # 卖出附加印花税
    if side.upper() == "SELL":
        fee += amount * cfg["stamp_duty_sell"]
    
    # 最低5元
    if 0 < fee < cfg["min_commission"]:
        fee = cfg["min_commission"]
    
    return round(fee, 2)


def calc_buy_commission(qty: int, price: float) -> float:
    """买入手续费。"""
    return calc_commission(qty, price, "BUY")


def calc_sell_commission(qty: int, price: float) -> float:
    """卖出手续费 (含印花税)。"""
    return calc_commission(qty, price, "SELL")


# ════════════════════════════════════════
#  交易约束检测
# ════════════════════════════════════════

class TradingRules:
    """A股交易规则检查器。"""

    def __init__(self, stock_status: dict = None):
        """
        Args:
          stock_status: {symbol: {"is_st": bool, "suspended_dates": [date, ...]}}
                       如果为 None, 使用默认空映射
        """
        self.stock_status = stock_status or {}

    def is_suspended(self, symbol: str, df_today: pd.DataFrame) -> bool:
        """
        检测停牌 (volume == 0 或 close 连续相等)。
        
        Args:
          symbol: 股票代码
          df_today: 该股票截至今日的K线DataFrame (tail 120)
        """
        if df_today is None or len(df_today) == 0:
            return True
        vol = df_today["volume"].values
        if len(vol) == 0 or vol[-1] == 0:
            return True
        # 也检查一字横盘 (连续N天close完全相同)
        if len(df_today) >= 3:
            closes = df_today["close"].values[-3:]
            if len(set(round(c, 2) for c in closes)) == 1 and vol[-1] < 100:
                return True
        return False

    def is_limit_hit(self, symbol: str, df_today: pd.DataFrame) -> tuple:
        """
        检测涨停/跌停。
        
        Returns:
          (is_limit_up, is_limit_down, is_one_word_board)
        """
        if df_today is None or len(df_today) < 2:
            return False, False, False
        
        close = df_today["close"].values[-1]
        open_p = df_today["open"].values[-1]
        high = df_today["high"].values[-1]
        low = df_today["low"].values[-1]
        prev_close = df_today["close"].values[-2]
        
        if prev_close <= 0:
            return False, False, False
        
        is_st = self.stock_status.get(str(symbol), {}).get("is_st", False)
        limit_pct = get_limit_pct(symbol, is_st)
        
        limit_up_price = round(prev_close * (1 + limit_pct), 2)
        limit_down_price = round(prev_close * (1 - limit_pct), 2)
        
        is_limit_up = abs(close - limit_up_price) < 0.01
        is_limit_down = abs(close - limit_down_price) < 0.01
        
        # 一字板: open == high == low == close == limit price
        is_one_word = (is_limit_up or is_limit_down) and \
                      abs(open_p - close) < 0.01 and \
                      abs(high - close) < 0.01 and \
                      abs(low - close) < 0.01
        
        return is_limit_up, is_limit_down, is_one_word

    def filter_tradeable(self, sd: dict, cp_today: dict) -> tuple:
        """
        过滤掉不可交易的股票。
        
        Args:
          sd: {symbol: DataFrame} — 当日可用股票
          cp_today: {symbol: close_price}
        
        Returns:
          (filtered_sd, filtered_cp)
        """
        untradeable = []
        for sym in list(sd.keys()):
            df = sd[sym]
            
            # 停牌检测
            if self.is_suspended(sym, df):
                untradeable.append((sym, "suspended"))
                continue
            
            # 涨跌停检测
            is_up, is_down, is_word = self.is_limit_hit(sym, df)
            if is_word:
                untradeable.append((sym, "one_word_board"))
                continue
            
            # 涨停不能买 (但我们仍可持有/卖出)
            # 跌停不能卖 (但我们仍可持有)
            # 在具体买卖时再判断
        
        # 移除不可交易股票
        for sym, reason in untradeable:
            sd.pop(sym, None)
            cp_today.pop(sym, None)
        
        return sd, cp_today

    def can_buy(self, symbol: str, df_today: pd.DataFrame) -> bool:
        """检查是否可以买入。"""
        is_up, is_down, is_word = self.is_limit_hit(symbol, df_today)
        if is_word or is_up:
            return False  # 一字板或涨停不能买
        if self.is_suspended(symbol, df_today):
            return False
        if self.stock_status.get(str(symbol), {}).get("is_st", False):
            return False  # 不买ST
        return True

    def can_sell(self, symbol: str, df_today: pd.DataFrame) -> bool:
        """检查是否可以卖出。"""
        is_up, is_down, is_word = self.is_limit_hit(symbol, df_today)
        if is_word and is_down:
            return False  # 一字跌停不能卖
        if self.is_suspended(symbol, df_today):
            return False
        return True


# ════════════════════════════════════════
#  快速测试
# ════════════════════════════════════════

def demo():
    """演示交易规则模块。"""
    print("TradingRules 演示...")
    print(f"  买入600519 100股@1800: 手续费={calc_buy_commission(100, 1800):.2f}")
    print(f"  卖出600519 100股@1800: 手续费={calc_sell_commission(100, 1800):.2f}")
    print(f"  买入600519 100股@10: 手续费={calc_buy_commission(100, 10):.2f} (最低5元)")
    
    print(f"  板块: 600519={get_board_type('600519')} limit={get_limit_pct('600519'):.0%}")
    print(f"  板块: 300750={get_board_type('300750')} limit={get_limit_pct('300750'):.0%}")
    print(f"  板块: 688981={get_board_type('688981')} limit={get_limit_pct('688981'):.0%}")
    print("✅ TradingRules 正常")


if __name__ == "__main__":
    demo()
