"""
量化交易看板 — Streamlit 4页仪表盘

启动:
    streamlit run dashboard.py
    MARKET=hk streamlit run dashboard.py

页面:
    1. 概览 — 权益曲线、持仓汇总、今日信号
    2. 信号历史 — 策略信号列表、准确率统计
    3. 交易记录 — 成交明细、手续费汇总
    4. 策略对比 — 多策略收益曲线叠加
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

import storage
from data_fetcher import DataFetcher, MARKET_CONFIG
from portfolio import PortfolioManager

# 页面配置
st.set_page_config(
    page_title="量化交易看板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 初始化数据库
storage.init_db()

# 市场选择
MARKET = os.environ.get("MARKET", "hk")
cfg = MARKET_CONFIG[MARKET]
SYMBOL = os.environ.get("SYMBOL", "01810")

st.sidebar.title("⚙️ 设置")
st.sidebar.markdown(f"**市场**: {cfg['name']} ({cfg['currency']})")
st.sidebar.markdown(f"**标的**: {SYMBOL}")
st.sidebar.markdown(f"**手续费**: {cfg['commission_default']*10000:.0f}bp")
st.sidebar.markdown(f"**制度**: T+{cfg['t_plus']}")

page = st.sidebar.radio("页面", ["📈 概览", "📡 信号历史", "📋 交易记录", "🆚 策略对比"])

# ================================================================
#  页面 1: 概览
# ================================================================
if page == "📈 概览":
    st.title("📈 概览")

    pm = PortfolioManager(market=MARKET)

    # 拉最新价格
    try:
        fetcher = DataFetcher()
        df_price = fetcher.fetch(SYMBOL, "20260701", "20260712", "qfq", market=MARKET)
        last_price = df_price["close"].iloc[-1]
        last_date = df_price["date"].iloc[-1].date()
    except Exception:
        last_price = 0
        last_date = "N/A"

    summary = pm.get_summary({SYMBOL: last_price})

    # 核心指标
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总权益", f"{cfg['currency']} {summary['total_equity']:,.0f}",
                  delta=f"{summary['total_return_pct']*100:+.2f}%")
    with col2:
        st.metric("现金", f"{cfg['currency']} {summary['cash']:,.0f}")
    with col3:
        st.metric("持仓市值", f"{cfg['currency']} {summary['holdings_value']:,.0f}")
    with col4:
        st.metric(f"{SYMBOL} 最新价", f"{cfg['currency']} {last_price:.2f}")

    # 权益曲线
    st.subheader("权益曲线")
    equity_data = storage.get_equity_log(limit=252)
    if equity_data:
        df_eq = pd.DataFrame(equity_data)
        df_eq["date"] = pd.to_datetime(df_eq["date"])
        df_eq = df_eq.sort_values("date")
        st.line_chart(df_eq.set_index("date")["total_equity"])
    else:
        st.info("暂无权益数据，请先运行 scheduler.py")

    # 持仓明细
    st.subheader("持仓明细")
    if summary["positions"]:
        df_pos = pd.DataFrame(summary["positions"])
        st.dataframe(df_pos, use_container_width=True)
    else:
        st.info("当前无持仓")

# ================================================================
#  页面 2: 信号历史
# ================================================================
elif page == "📡 信号历史":
    st.title("📡 信号历史")

    signals = storage.get_pending_signals()
    # 也获取已执行的
    conn = storage.get_db()
    all_sigs = conn.execute(
        "SELECT * FROM signals ORDER BY date DESC LIMIT 500"
    ).fetchall()
    conn.close()
    all_sigs = [dict(r) for r in all_sigs]

    if all_sigs:
        df_sig = pd.DataFrame(all_sigs)
        df_sig["action"] = df_sig["signal"].map({1: "BUY", -1: "SELL", 0: "HOLD"})

        # 统计
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总信号数", len(df_sig))
        with col2:
            st.metric("未执行", df_sig["executed"].value_counts().get(0, 0))
        with col3:
            buy_pct = (df_sig["signal"] == 1).mean() * 100
            st.metric("BUY占比", f"{buy_pct:.0f}%")

        st.dataframe(df_sig[["date", "strategy", "action", "confidence", "reason", "executed"]],
                     use_container_width=True)
    else:
        st.info("暂无信号数据")

# ================================================================
#  页面 3: 交易记录
# ================================================================
elif page == "📋 交易记录":
    st.title("📋 交易记录")

    trades = storage.get_trades(limit=500)

    if trades:
        df_tr = pd.DataFrame(trades)
        total_buy = df_tr[df_tr["action"] == "BUY"]["qty"].sum()
        total_sell = df_tr[df_tr["action"] == "SELL"]["qty"].sum()
        total_comm = df_tr["commission"].sum()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总交易数", len(df_tr))
        with col2:
            st.metric("总买入(股)", f"{total_buy:,}")
        with col3:
            st.metric("总手续费", f"{cfg['currency']} {total_comm:,.2f}")

        st.dataframe(df_tr, use_container_width=True)
    else:
        st.info("暂无交易记录")

# ================================================================
#  页面 4: 策略对比
# ================================================================
elif page == "🆚 策略对比":
    st.title("🆚 策略对比 & 回测历史")

    # 从数据库读取回测历史
    try:
        backtests = storage.get_backtests(limit=50)
        if backtests:
            df_bt = pd.DataFrame(backtests)
            st.subheader(f"回测历史 ({len(df_bt)} 次)")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总回测次数", len(df_bt))
            with col2:
                avg_sharpe = df_bt["sharpe_ratio"].dropna().mean()
                st.metric("平均 Sharpe", f"{avg_sharpe:.3f}")
            with col3:
                pos = (df_bt["excess_return"].dropna() > 0).sum()
                st.metric("正超额占比", f"{pos}/{len(df_bt)}")

            # 回测对比表
            st.dataframe(
                df_bt[["run_at", "symbol", "market", "strategy",
                       "sharpe_ratio", "total_return", "max_drawdown",
                       "total_trades", "excess_return"]].sort_values("run_at", ascending=False),
                use_container_width=True,
            )

            # 最近回测的趋势图
            st.subheader("最近回测 Sharpe 趋势")
            recent = df_bt.sort_values("run_at").tail(20)
            st.line_chart(recent.set_index("run_at")["sharpe_ratio"])
        else:
            st.info("暂无回测历史。运行 main.py 会自动保存。")
    except Exception as e:
        st.warning(f"数据库读取失败: {e}")

    st.markdown("---")
    st.markdown("已有回测图表:")

    # 尝试加载已有的回测图表
    png_files = [
        f for f in os.listdir(".") if f.startswith("backtest_") and f.endswith(".png")
    ]
    if png_files:
        st.subheader("已有回测图表")
        for f in sorted(png_files)[-6:]:
            st.image(f, caption=f, use_container_width=True)
    else:
        st.info("暂无回测图表，运行 main.py 或 main_test.py 生成")
