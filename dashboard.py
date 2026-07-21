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
MARKET = os.environ.get("MARKET", "a")   # 默认A股
cfg = MARKET_CONFIG[MARKET]
SYMBOL = os.environ.get("SYMBOL", "01810")

st.sidebar.title("⚙️ 设置")
st.sidebar.markdown(f"**市场**: {cfg['name']} ({cfg['currency']})")
st.sidebar.markdown(f"**标的**: {SYMBOL}")
st.sidebar.markdown(f"**手续费**: {cfg['commission_default']*10000:.0f}bp")
st.sidebar.markdown(f"**制度**: T+{cfg['t_plus']}")
st.sidebar.markdown(f"**初始资金**: {cfg['currency']} 100,000")

page = st.sidebar.radio("页面", ["📈 概览", "📡 信号历史", "📋 交易记录", "🆚 策略对比", "🧪 测试结果", "📍 买卖标记"])

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
    equity_data = storage.get_equity_log(limit=9999)
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
        # ★ 加股票名 (20只A股)
        A_NAMES = {"688981":"中芯","002371":"北华创","603986":"兆易","002049":"紫光",
            "300033":"同花顺","002230":"讯飞","688111":"金山","300750":"宁德",
            "002594":"比亚迪","601012":"隆基","600519":"茅台","000858":"五粮液",
            "601318":"平安","600036":"招行","300760":"迈瑞","600276":"恒瑞",
            "600760":"沈飞","000625":"长安","601668":"中建","601899":"紫金",
            "688012":"中微","300782":"卓胜微","688396":"华润微","300454":"深信服",
            "688561":"奇安信","300274":"阳光","688005":"容百","000568":"泸州",
            "002714":"牧原","000001":"平安银行","300122":"智飞","688180":"君实"}
        df_tr["name"] = df_tr["symbol"].astype(str).map(A_NAMES).fillna(df_tr["symbol"])
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

# ================================================================
#  页面 5: 测试结果
# ================================================================
elif page == "🧪 测试结果":
    st.title("🧪 A/B 测试结果")

    # 查找最近的测试CSV
    import glob
    csv_files = sorted(glob.glob("test_results_*.csv"), reverse=True)
    if not csv_files:
        # 也从DB读
        try:
            bts = storage.get_backtests(limit=20)
            if bts:
                df_bt = pd.DataFrame(bts)
                st.subheader(f"数据库测试记录 ({len(df_bt)} 条)")
                st.dataframe(df_bt[["run_at","strategy","total_return","excess_return","total_trades"]],
                           use_container_width=True)
            else:
                st.info("暂无测试记录。运行 python test_a_share.py 生成。")
        except:
            st.info("暂无测试记录")
    else:
        latest = csv_files[0]
        df = pd.read_csv(latest)
        st.subheader(f"最新测试: {latest}")

        # 柱状图
        st.bar_chart(df.set_index("name")[["total_return", "benchmark"]], use_container_width=True)

        # 超额对比
        st.subheader("超额收益对比")
        cols = st.columns(len(df))
        for i, (_, r) in enumerate(df.iterrows()):
            with cols[i]:
                color = "green" if r['excess'] > 0 else "red"
                label = r['name'].split(":")[1].strip() if ":" in r['name'] else r['name']
                st.metric(label, f"{r['excess']:+.1f}%",
                         delta=f"{r['trades']:.0f}笔")

        # 详细表格
        st.subheader("详细数据")
        st.dataframe(df[["name","total_return","benchmark","excess","trades"]],
                    use_container_width=True)

        # 历史测试记录
        if len(csv_files) > 1:
            st.subheader("历史测试记录")
            history = []
            for f in csv_files[:10]:
                t = os.path.getmtime(f)
                history.append({"文件": f, "时间": datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")})
            st.dataframe(pd.DataFrame(history), use_container_width=True)

# ================================================================
#  页面 6: 买卖标记
# ================================================================
elif page == "📍 买卖标记":
    st.title("📍 买卖操作标记")

    st.markdown("展示策略在哪些日期买入了哪些股票,在哪些日期卖出了。")

    # 从DB读交易
    trades = storage.get_trades(limit=9999)
    if not trades:
        st.info("暂无交易记录")
    else:
        df_tr = pd.DataFrame(trades)
        df_tr["date"] = pd.to_datetime(df_tr["date"])

        # 股票名映射
        names = {"688981":"中芯","002371":"北华创","603986":"兆易","002049":"紫光",
            "300033":"同花顺","002230":"讯飞","688111":"金山","300750":"宁德",
            "002594":"比亚迪","601012":"隆基","600519":"茅台","000858":"五粮液",
            "601318":"平安","600036":"招行","300760":"迈瑞","600276":"恒瑞",
            "600760":"沈飞","000625":"长安","601668":"中建","601899":"紫金"}
        df_tr["股票"] = df_tr["symbol"].astype(str).map(names).fillna(df_tr["symbol"])
        df_tr["操作"] = df_tr["action"].map({"BUY": "🔴 买入", "SELL": "🟢 卖出"})

        # 按股票筛选
        syms = sorted(df_tr["symbol"].unique())
        selected = st.selectbox("选择股票", syms, format_func=lambda s: names.get(s, s))

        df_sym = df_tr[df_tr["symbol"] == selected]

        # 拉价格曲线
        try:
            df_px = DataFetcher().fetch(selected, "20240101", "20260710", "qfq", market="a")
            df_px["date"] = pd.to_datetime(df_px["date"])

            st.subheader(f"{names.get(selected,selected)} 价格与操作")
            # Matplotlib overlay
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df_px["date"], df_px["close"], color="gray", alpha=0.6, linewidth=0.8, label="收盘价")

            buys = df_sym[df_sym["action"] == "BUY"]
            sells = df_sym[df_sym["action"] == "SELL"]
            ax.scatter(buys["date"], buys["price"], marker="^", color="red", s=100, zorder=5, label=f"买入({len(buys)}次)")
            ax.scatter(sells["date"], sells["price"], marker="v", color="green", s=100, zorder=5, label=f"卖出({len(sells)}次)")

            ax.set_title(f"{names.get(selected,selected)} 买卖操作标记")
            ax.legend(); ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.xticks(rotation=45)

            st.pyplot(fig)

            # 交易明细
            st.dataframe(df_sym[["date","操作","qty","price","reason"]].sort_values("date"),
                        use_container_width=True)
        except Exception as e:
            st.warning(f"价格数据获取失败: {e}")

        # 总览表
        st.subheader("全部交易")
        df_all = df_tr[["date","股票","操作","qty","price"]].sort_values("date", ascending=False)
        st.dataframe(df_all, use_container_width=True)
