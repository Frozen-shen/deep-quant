"""
临时验证脚本 (验证后删除, 不属于 scripts/active)
验证 efinance 数据能否落盘为 aux 面板格式:
  data_store/aux_flow/{code}.parquet  — 按股, 历史资金流 (最近~120交易日)
  data_store/aux_realtime/{date}.parquet — 按日, 全市场市值/换手率快照
"""
import os, sys, time, random, glob
os.environ.pop("HTTP_PROXY", None); os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)

from efinance.shared.tickflow_prompt import session
session.trust_env = False  # 关键: 避免读 Windows 系统代理
import efinance as ef
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

# 资金流字段: 中文 -> 英文 (与 aux 面板英文列风格一致)
FLOW_COLS = {
    "日期": "date", "主力净流入": "main_net", "小单净流入": "small_net",
    "中单净流入": "medium_net", "大单净流入": "large_net",
    "超大单净流入": "xlarge_net", "主力净流入占比": "main_net_pct",
    "小单流入净占比": "small_net_pct", "中单流入净占比": "medium_net_pct",
    "大单流入净占比": "large_net_pct", "超大单流入净占比": "xlarge_net_pct",
    "收盘价": "close", "涨跌幅": "pct_change",
}


def retry_call(fn, tries=5, sleep=1.5):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            time.sleep(sleep + random.uniform(0, 1))
    return None


def fetch_flow(code: str) -> pd.DataFrame:
    df = retry_call(lambda: ef.stock.get_history_bill(code))
    if df is None or len(df) == 0:
        return None
    df = df.rename(columns=FLOW_COLS)[list(FLOW_COLS.values())]
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = code
    return df[["code", "date", "main_net", "small_net", "medium_net",
               "large_net", "xlarge_net", "main_net_pct", "small_net_pct",
               "medium_net_pct", "large_net_pct", "xlarge_net_pct",
               "close", "pct_change"]]


def main():
    out_dir = os.path.join(BASE, "data_store", "aux_flow")
    os.makedirs(out_dir, exist_ok=True)

    # 从 data_store 抽样 20 只 (含一只 688 科创板)
    codes = [os.path.basename(f).replace(".parquet", "")
             for f in sorted(glob.glob(os.path.join(BASE, "data_store", "*.parquet")))]
    random.seed(42)
    sample = random.sample(codes, 19) + ["688981"]

    ok, fail = 0, []
    for code in sample:
        path = os.path.join(out_dir, f"{code}.parquet")
        df = fetch_flow(code)
        if df is None:
            fail.append(code)
            print(f"  {code} FAIL")
            continue
        df.to_parquet(path, index=False)
        ok += 1
        print(f"  {code} OK {len(df)}条 {df['date'].min().date()}~{df['date'].max().date()}")
        time.sleep(0.3)

    print(f"\n验证完成: {ok}/{len(sample)} 成功, 失败 {len(fail)}")
    # 抽查一只
    probe = os.path.join(out_dir, "688981.parquet")
    if os.path.exists(probe):
        df = pd.read_parquet(probe)
        print("抽查 688981:", list(df.columns))
        print(df.tail(2).to_string())


if __name__ == "__main__":
    main()
