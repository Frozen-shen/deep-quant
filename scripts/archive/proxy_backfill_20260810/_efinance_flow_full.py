"""
临时全量脚本 (验证后删除)
全量拉取 data_store 所有股票的历史资金流 -> data_store/aux_flow/{code}.parquet
断点续传: 已存在文件跳过。失败列表写 aux_flow/_failed.json
"""
# ⚠️ 插件引入必须放在最顶部, 在 efinance 之前!
import akshare_proxy_patch
akshare_proxy_patch.install_patch(
    "101.201.173.125",
    auth_token="20260809YL16REJA",
    retry=30,
    hook_domains=[
        "fund.eastmoney.com",
        "push2.eastmoney.com",
        "push2his.eastmoney.com",
        "emweb.securities.eastmoney.com",
        "searchapi.eastmoney.com/api/suggest/get",
    ],
    fast=True,
)

import os, sys, time, random, glob, json
os.environ.pop("HTTP_PROXY", None); os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)

from efinance.shared.tickflow_prompt import session
session.trust_env = False
import efinance as ef
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "data_store", "aux_flow")
os.makedirs(OUT_DIR, exist_ok=True)

FLOW_COLS = {
    "日期": "date", "主力净流入": "main_net", "小单净流入": "small_net",
    "中单净流入": "medium_net", "大单净流入": "large_net",
    "超大单净流入": "xlarge_net", "主力净流入占比": "main_net_pct",
    "小单流入净占比": "small_net_pct", "中单流入净占比": "medium_net_pct",
    "大单流入净占比": "large_net_pct", "超大单流入净占比": "xlarge_net_pct",
    "收盘价": "close", "涨跌幅": "pct_change",
}
FLOW_COLS_OUT = ["code", "date", "main_net", "small_net", "medium_net",
                 "large_net", "xlarge_net", "main_net_pct", "small_net_pct",
                 "medium_net_pct", "large_net_pct", "xlarge_net_pct",
                 "close", "pct_change"]


def retry_call(fn, tries=2, sleep=1.0):
    """快速失败策略: 网络断连时尽快跳过, 靠断点续传重跑补漏"""
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
    return df[FLOW_COLS_OUT]


def main():
    codes = sorted(os.path.basename(f).replace(".parquet", "")
                   for f in glob.glob(os.path.join(BASE, "data_store", "*.parquet")))

    # 智能循环: 网络恢复前每 60s 探测一次, 恢复后全量断点续传
    while True:
        probe = retry_call(lambda: ef.stock.get_history_bill("600519"), tries=1)
        if probe is None:
            print(f"[{time.strftime('%H:%M:%S')}] 网络不可达, 60s 后重试...", flush=True)
            time.sleep(60)
            continue
        print(f"[{time.strftime('%H:%M:%S')}] 网络恢复, 开始全量拉取", flush=True)
        run_once(codes)
        # 一轮结束后检查是否全部完成
        done = set(os.path.basename(f).replace(".parquet", "")
                   for f in glob.glob(os.path.join(OUT_DIR, "*.parquet")))
        todo = [c for c in codes if c not in done]
        if not todo:
            print("全部完成!", flush=True)
            return
        print(f"本轮后剩余 {len(todo)} 只, 60s 后继续下一轮", flush=True)
        time.sleep(60)


def run_once(codes):
    done = set(os.path.basename(f).replace(".parquet", "")
               for f in glob.glob(os.path.join(OUT_DIR, "*.parquet")))
    todo = [c for c in codes if c not in done]
    if not todo:
        return
    print(f"总计 {len(codes)} 只, 已完成 {len(done)}, 待拉 {len(todo)}", flush=True)

    ok, fail = 0, []
    t0 = time.time()
    for i, code in enumerate(todo, 1):
        df = fetch_flow(code)
        if df is None:
            fail.append(code)
        else:
            df.to_parquet(os.path.join(OUT_DIR, f"{code}.parquet"), index=False)
            ok += 1
        if i % 100 == 0 or i == len(todo):
            el = (time.time() - t0) / 60
            print(f"  [{i}/{len(todo)}] ok={ok} fail={len(fail)} elapsed={el:.1f}min", flush=True)
        time.sleep(0.25)

    with open(os.path.join(OUT_DIR, "_failed.json"), "w", encoding="utf-8") as f:
        json.dump(fail, f, ensure_ascii=False, indent=1)
    print(f"本轮完成: 成功 {ok}, 失败 {len(fail)} -> {OUT_DIR}/_failed.json", flush=True)


if __name__ == "__main__":
    main()
