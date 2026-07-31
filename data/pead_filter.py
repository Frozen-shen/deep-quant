"""
PEAD 防守过滤器 — 负面预告→强制卖出

用法:
  from data.pead_filter import load_pead_alerts
  alerts = load_pead_alerts()
  if alerts.has_bad_news(symbol, today):
      # 强制卖出
"""
import os, json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "pead_cache")

BAD_TYPES = {'预减', '首亏', '增亏', '略减', '续亏'}


class PEADAlerts:
    """PEAD负面预告数据库"""

    def __init__(self):
        self._alerts = {}  # {symbol: [ann_date, ...]}

    def load(self):
        """加载所有缓存的预告数据"""
        if not os.path.exists(CACHE_DIR):
            return

        for fname in os.listdir(CACHE_DIR):
            if not fname.endswith('.parquet'):
                continue
            path = os.path.join(CACHE_DIR, fname)
            try:
                df = pd.read_parquet(path)
                bad = df[df['预告类型'].isin(BAD_TYPES)]
                for _, row in bad.iterrows():
                    sym = str(row.get('股票代码', '')).zfill(6)
                    ann_date = pd.to_datetime(row['公告日期'])
                    if sym not in self._alerts:
                        self._alerts[sym] = []
                    self._alerts[sym].append(ann_date)
            except:
                pass

        print(f"  [PEAD] 加载负面预警: {len(self._alerts)}只股票")

    def has_bad_news(self, symbol: str, today, lookback_days: int = 30):
        """
        检查某只股票在最近 lookback_days 天是否有负面预告。

        Returns:
          True: 有负面预告, 应卖出
          False: 无
        """
        if symbol not in self._alerts:
            return False

        if isinstance(today, str):
            today = pd.Timestamp(today)

        cutoff = today - pd.Timedelta(days=lookback_days)
        for ann_date in self._alerts[symbol]:
            if cutoff <= ann_date <= today:
                return True
        return False


# 全局单例
_alerts_instance = None


def load_pead_alerts() -> PEADAlerts:
    global _alerts_instance
    if _alerts_instance is None:
        _alerts_instance = PEADAlerts()
        _alerts_instance.load()
    return _alerts_instance
