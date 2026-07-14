"""
测试数据加载器 — 便捷加载预生成的测试数据集

用法:
    from test_data import loader
    df = loader.load_ohlcv("600519")        # 加载茅台日线
    df = loader.load_multi()                 # 加载多股票合并
    df = loader.load_scenario("bear_2018")   # 加载2018熊市场景
    df = loader.load_known_signals()         # 加载已知信号(回归测试)
    df = loader.load_mock_events()           # 加载模拟事件
"""

import os
import json
import pandas as pd

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json")


def _read(path: str) -> pd.DataFrame:
    """读取 CSV，自动解析日期，强制 symbol 为字符串。"""
    full = os.path.join(DATASETS_DIR, path)
    if not os.path.exists(full):
        raise FileNotFoundError(
            f"{full} 不存在。请先运行: python test_data/generate.py"
        )
    df = pd.read_csv(full, encoding="utf-8-sig")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str)
    return df


def load_ohlcv(symbol: str) -> pd.DataFrame:
    """加载单只股票的日线数据。"""
    return _read(f"ohlcv_{symbol}.csv")


def load_multi() -> pd.DataFrame:
    """加载多股票合并数据。"""
    return _read("ohlcv_multi.csv")


def load_scenario(scenario_id: str, symbol: str = None) -> pd.DataFrame:
    """加载场景数据。"""
    if symbol:
        return _read(f"scenario_{scenario_id}_{symbol}.csv")
    # 尝试找到第一个匹配
    for f in os.listdir(DATASETS_DIR):
        if f.startswith(f"scenario_{scenario_id}_") and f.endswith(".csv"):
            return _read(f)
    raise FileNotFoundError(f"场景 {scenario_id} 未找到")


def load_known_signals() -> pd.DataFrame:
    """加载已知信号集。"""
    return _read("known_signals.csv")


def load_mock_events() -> pd.DataFrame:
    """加载模拟事件。"""
    return _read("mock_events.csv")


def load_manifest() -> dict:
    """加载数据集清单。"""
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError("manifest.json 不存在，请先运行 generate.py")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def list_datasets() -> list:
    """列出所有已生成的数据集。"""
    files = os.listdir(DATASETS_DIR) if os.path.exists(DATASETS_DIR) else []
    result = []
    for f in sorted(files):
        path = os.path.join(DATASETS_DIR, f)
        result.append({
            "file": f,
            "size_kb": round(os.path.getsize(path) / 1024, 1),
            "modified": os.path.getmtime(path),
        })
    return result


def is_available() -> bool:
    """检查测试数据集是否已生成。"""
    return os.path.exists(os.path.join(DATASETS_DIR, "ohlcv_600519.csv"))
