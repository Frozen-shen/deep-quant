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

page = st.sidebar.radio("页面", ["🧪 模型评测", "🧪 测试结果", "📈 概览", "📡 信号历史", "📋 交易记录", "🆚 策略对比", "📍 买卖标记"])

# ================================================================
#  页面 0: 模型评测 (默认首页)
# ================================================================
if page == "🧪 模型评测":
    st.title("🧪 模型评测")
    
    import json
    import glob as glob_mod
    
    # 读取最新评测报告
    report_files = sorted(glob_mod.glob("test_results/report_card_*.json"), reverse=True)
    detail_files = sorted(glob_mod.glob("test_results/rolling_v3_*.csv"), reverse=True)
    
    if report_files:
        with open(report_files[0]) as f:
            report = json.load(f)
        grade = report["grade"]
        cross = report["cross_window"]
        
        # ── 核心评级 ──
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            grade_color = {"A": "green", "B": "blue", "C": "orange", "D": "red"}
            gc = grade_color.get(grade["grade"][0], "gray")
            st.markdown(f"### 综合评级")
            st.markdown(f"<h1 style='color:{gc};font-size:72px;'>{grade['grade']}</h1>", unsafe_allow_html=True)
        with col2:
            st.metric("得分", f"{grade['score']}/100", delta="合格" if grade["pass"] else "不合格")
        with col3:
            st.metric("多窗口IR", f"{cross.get('cross_window_ir', 0):.2f}")
        with col4:
            n_w = cross.get("n_windows", 0)
            p_w = cross.get("pos_windows", 0)
            st.metric("正窗口", f"{p_w}/{n_w}", delta=f"{p_w/n_w*100:.0f}%" if n_w > 0 else None)

        # ── 维度得分柱状图 ──
        st.subheader("📊 各维度得分")
        details = grade.get("details", {})
        if details:
            dims = list(details.keys())
            scores = [details[d]["score"] * 100 for d in dims]
            grades = [details[d]["grade"] for d in dims]
            
            # 用 streamlit 原生柱状图
            chart_data = pd.DataFrame({"得分": scores, "评级": grades}, index=dims)
            st.bar_chart(chart_data[["得分"]], use_container_width=True)
            
            # 详细表
            st.subheader("📋 指标明细")
            detail_rows = []
            for metric, d in details.items():
                val = d["value"]
                if abs(val) > 100:
                    val_str = f"{val:.0f}"
                elif abs(val) > 1:
                    val_str = f"{val:.2f}"
                else:
                    val_str = f"{val:.4f}"
                detail_rows.append({
                    "指标": metric, "数值": val_str,
                    "得分": f"{d['score']*100:.0f}",
                    "评级": d["grade"],
                    "权重": d["weight"],
                })
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

        # ── 历史对比 ──
        if len(report_files) > 1:
            st.subheader("📁 历史评测")
            history = []
            for f in report_files[:10]:
                t = f.replace("test_results/report_card_", "").replace(".json", "")
                t_str = f"{t[:4]}-{t[4:6]}-{t[6:8]} {t[9:11]}:{t[11:13]}"
                try:
                    with open(f) as fh:
                        r = json.load(fh)
                    history.append({"时间": t_str, "评级": r["grade"]["grade"],
                                   "得分": r["grade"]["score"],
                                   "合格": "✅" if r["grade"]["pass"] else "❌"})
                except: pass
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)

    elif detail_files:
        # Fallback: no report card yet, show basic results
        st.info("📝 评测报告尚未生成。运行 python test_rolling_v3.py 生成。当前显示最新测试结果:")
        df = pd.read_csv(detail_files[0])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("暂无测试数据。运行 python test_rolling_v3.py 生成评测报告。")

# ================================================================
#  页面 1: 测试结果
# ================================================================
elif page == "🧪 测试结果":
    st.title("🧪 滚动重训练 — 测试结果")

    # 优先读取 v3 测试结果
    import glob
    v3_files = sorted(glob.glob("test_results/rolling_v3_*.csv"), reverse=True)
    
    if v3_files:
        latest_v3 = v3_files[0]
        df = pd.read_csv(latest_v3)
        test_time = latest_v3.replace("test_results/rolling_v3_", "").replace(".csv", "")
        test_time_str = f"{test_time[:4]}-{test_time[4:6]}-{test_time[6:8]} {test_time[9:11]}:{test_time[11:13]}:{test_time[13:15]}"

        st.subheader(f"📊 最新滚动重训练: {test_time_str}")
        
        avg_excess = df["excess"].mean()
        pos_windows = (df["excess"] > 0).sum()
        total_windows = len(df)
        median_ex = df["excess"].median()
        std_ex = df["excess"].std()
        ir = avg_excess / std_ex if std_ex > 0 else 0
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("平均超额", f"{avg_excess:+.1f}%", delta="🎯≥10%" if avg_excess >= 10 else None)
        with col2:
            st.metric("信息比率", f"{ir:.2f}")
        with col3:
            st.metric("正窗口", f"{pos_windows}/{total_windows}")
        with col4:
            st.metric("中位数超额", f"{median_ex:+.1f}%")
        with col5:
            st.metric("超额标准差", f"{std_ex:.1f}%")

        if avg_excess >= 10:
            st.success(f"🎉 目标达成! 平均超额 {avg_excess:+.1f}% ≥ 10%")
        elif avg_excess > 0:
            st.info(f"📈 超额为正, 距10%目标差 {10-avg_excess:+.1f}%")
        else:
            st.warning("⚠️ 超额为负, 需要继续优化")

        st.subheader("📊 各窗口超额收益")
        st.bar_chart(df.set_index("window")[["excess"]], use_container_width=True)

        st.subheader("📋 窗口详情")
        display_df = df[["window","train","test","strategy","benchmark","excess","trades"]].copy()
        display_df.columns = ["窗口","训练期","测试期","策略%","基准%","超额%","交易数"]
        st.dataframe(display_df.style.applymap(
            lambda v: "color: green" if v > 0 else "color: red", subset=["超额%"]), 
            use_container_width=True)

        if len(v3_files) > 1:
            st.subheader("📁 历史测试记录")
            history = []
            for f in v3_files[:10]:
                t = f.replace("test_results/rolling_v3_","").replace(".csv","")
                t_str = f"{t[:4]}-{t[4:6]}-{t[6:8]} {t[9:11]}:{t[11:13]}"
                try:
                    df_h = pd.read_csv(f)
                    history.append({"时间": t_str, "平均超额": f"{df_h['excess'].mean():+.1f}%",
                                   "正窗口": f"{(df_h['excess']>0).sum()}/{len(df_h)}"})
                except: pass
            st.dataframe(pd.DataFrame(history), use_container_width=True)
    else:
        st.info("暂无测试记录。运行 python test_rolling_v3.py 生成。")

# ================================================================
#  页面 2: 概览
# ================================================================
elif page == "📈 概览":
    st.title("📈 概览")

    pm = PortfolioManager(market=MARKET)
    try:
        fetcher = DataFetcher()
        df_price = fetcher.fetch(SYMBOL, "20260701", "20260712", "qfq", market=MARKET)
        last_price = df_price["close"].iloc[-1]
        last_date = df_price["date"].iloc[-1].date()
    except Exception:
        last_price = 0
        last_date = "N/A"

    summary = pm.get_summary({SYMBOL: last_price})
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

    st.subheader("权益曲线")
    equity_data = storage.get_equity_log(limit=9999)
    if equity_data:
        df_eq = pd.DataFrame(equity_data)
        df_eq["date"] = pd.to_datetime(df_eq["date"])
        df_eq = df_eq.sort_values("date")
        st.line_chart(df_eq.set_index("date")["total_equity"])
    else:
        st.info("暂无权益数据，请先运行 test_rolling_v3.py")

    st.subheader("持仓明细")
    if summary["positions"]:
        st.dataframe(pd.DataFrame(summary["positions"]), use_container_width=True)
    else:
        st.info("当前无持仓")

# ================================================================
#  页面 3: 信号历史
# ================================================================
elif page == "📡 信号历史":
    st.title("📡 信号历史")

    signals = storage.get_pending_signals()
    conn = storage.get_db()
    all_sigs = conn.execute(
        "SELECT * FROM signals ORDER BY date DESC LIMIT 500"
    ).fetchall()
    conn.close()
    all_sigs = [dict(r) for r in all_sigs]

    if all_sigs:
        df_sig = pd.DataFrame(all_sigs)
        df_sig["action"] = df_sig["signal"].map({1: "BUY", -1: "SELL", 0: "HOLD"})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总信号数", len(df_sig))
        with col2:
            st.metric("未执行", df_sig["executed"].value_counts().get(0, 0))
        with col3:
            buy_pct = (df_sig["signal"] == 1).mean() * 100
            st.metric("BUY占比", f"{buy_pct:.0f}%")
        st.dataframe(df_sig[["date","strategy","action","confidence","reason","executed"]],
                     use_container_width=True)
    else:
        st.info("暂无信号数据")

# ================================================================
#  页面 4: 交易记录
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
