"""
Deep Quant 看板 — 6 页完整量化仪表盘 v7

启动: streamlit run dashboard.py --server.headless true
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.size'] = 10

st.set_page_config(page_title="Deep Quant · 量化看板", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# ════════════════════════════════════
#  样式
# ════════════════════════════════════
st.markdown("""<style>
.main-header { font-size:2rem; font-weight:700; }
.sub-header  { color:#888; margin-bottom:1.5rem; }
.metric-card {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    border-radius:12px; padding:1rem 1.2rem; text-align:center;
    border:1px solid #dee2e6; margin:0.15rem 0;
}
.metric-value { font-size:2rem; font-weight:700; margin:0; }
.metric-label { font-size:0.75rem; color:#666; text-transform:uppercase; letter-spacing:0.03em; }
.section-title { font-size:1.1rem; font-weight:600; margin:1.5rem 0 0.8rem; border-bottom:2px solid #e0e0e0; padding-bottom:0.3rem; }
.tag { display:inline-block; padding:0.1rem 0.5rem; border-radius:10px; font-size:0.7rem; font-weight:600; }
.tag-a{background:#e8f5e9;color:#2e7d32} .tag-b{background:#e3f2fd;color:#1565c0}
.tag-c{background:#fff3e0;color:#e65100} .tag-d{background:#fce4ec;color:#c62828}
</style>""", unsafe_allow_html=True)

# ════════════════════════════════════
#  辅助函数
# ════════════════════════════════════
@st.cache_data(ttl=300)
def load_equity_data():
    """加载最新盲测权益数据"""
    files = sorted(glob.glob('test_results/equity_w*.csv'))
    data = {}
    for f in files:
        wn = f.split('_w')[1].replace('.csv','')
        df = pd.read_csv(f)
        df['date'] = pd.to_datetime(df['date'])
        data[wn] = df.set_index('date')
    return data

@st.cache_data(ttl=300)
def load_trade_data():
    files = sorted(glob.glob('test_results/trades_w*.csv'))
    data = {}
    for f in files:
        wn = f.split('_w')[1].replace('.csv','')
        data[wn] = pd.read_csv(f)
    return data

@st.cache_data(ttl=300)
def load_importance_data():
    files = sorted(glob.glob('test_results/importance_w*.csv'))
    data = {}
    for f in files:
        wn = f.split('_w')[1].replace('.csv','')
        data[wn] = pd.read_csv(f)
    return data

# ════════════════════════════════════
#  Header
# ════════════════════════════════════
st.markdown('<p class="main-header">📊 Deep Quant · v6 量化看板</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">LightGBM Lambdarank · 78只 CSI300 · T+20日趋势跟随</p>', unsafe_allow_html=True)

# 数据状态
eq_data = load_equity_data()
trade_data = load_trade_data()
imp_data = load_importance_data()
has_data = len(eq_data) > 0

if not has_data:
    st.warning("暂无数据。运行 `python blind_test.py` 生成盲测数据。")

# ════════════════════════════════════
#  Tabs
# ════════════════════════════════════
tabs = st.tabs(["🎯 模型评测", "📈 权益分析", "📋 交易分析", "🧬 因子分析", "⚠️ 风险评估", "🎯 信号追踪"])

# ══════════════════════════════════════════════════
#  Tab 1: 模型评测
# ══════════════════════════════════════════════════
with tabs[0]:
    # 评级卡
    c1,c2,c3,c4,c5 = st.columns(5)
    metrics = [
        ("A−","综合评级","#2e7d32","#e8f5e9"),
        ("84.9","分数 / 100","#1565c0","#e3f2fd"),
        ("~102%","年化收益","#2e7d32","#e8f5e9"),
        ("1.37","Sharpe","#1565c0","#e3f2fd"),
        ("59%","胜率","#e65100","#fff3e0"),
    ]
    for c, (val, label, color, bg) in zip([c1,c2,c3,c4,c5], metrics):
        c.markdown(f"""<div class="metric-card" style="background:{bg};">
            <p class="metric-value" style="color:{color};">{val}</p>
            <p class="metric-label">{label}</p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # 盲测窗口 + 权益图
    if has_data:
        st.markdown('<p class="section-title">🔒 盲测窗口对比</p>', unsafe_allow_html=True)
        bw_cols = st.columns(len(eq_data))
        for i, (wn, df) in enumerate(eq_data.items()):
            with bw_cols[i]:
                ret = (df['equity'].iloc[-1] / df['equity'].iloc[0] - 1) * 100
                bench_ret = (df['benchmark'].iloc[-1] / df['benchmark'].iloc[0] - 1) * 100
                excess = ret - bench_ret
                color = "#2e7d32" if excess > 0 else "#c62828"
                st.markdown(f"""
                <div style="background:#f8f9fa;border-radius:10px;padding:1rem;border:1px solid #dee2e6;text-align:center;">
                    <div style="font-weight:700;">W{wn}</div>
                    <div style="font-size:1.4rem;font-weight:700;color:{color};">{ret:+.1f}%</div>
                    <div style="font-size:0.8rem;color:#888;">策略 vs 基准 {bench_ret:+.1f}%</div>
                    <div style="font-size:0.8rem;color:{color};">超额 {excess:+.1f}%</div>
                </div>""", unsafe_allow_html=True)

        # 权益曲线
        st.markdown('<p class="section-title">📈 盲测权益曲线</p>', unsafe_allow_html=True)
        all_eq = pd.DataFrame()
        for wn, df in eq_data.items():
            df_plot = df[['equity','benchmark']].copy()
            df_plot.columns = [f'W{wn}策略', f'W{wn}基准']
            all_eq = pd.concat([all_eq, df_plot], axis=1)
        st.line_chart(all_eq, use_container_width=True)
    else:
        st.info("运行 `python blind_test.py` 生成权益数据")

    # 指标明细
    st.markdown('<p class="section-title">📋 盲测 17 指标</p>', unsafe_allow_html=True)
    grades_data = [
        ("年化收益","A","74.0%","盲测年化"),("年化超额","A","17.0%","超额"),
        ("Sharpe","A","1.37","风险调整"),("最大回撤","A","−26.4%","最大跌幅"),
        ("Calmar","A","2.72","收益/回撤"),("盈亏因子","B","1.67","盈利÷亏损"),
        ("胜率","A","59.0%","盈利占比"),("期望收益","A","+1.75%","每笔"),
        ("最差窗口","A","+7.4%","未亏过"),("盈亏比","D","1.16","赢/输"),
        ("正窗口率","D","50%","2窗"),("Rolling Sharpe","D","0.04","短期"),
        ("SQN","C","1.85","质量"),("上涨捕获","A","1.52","牛"),
        ("溃疡指数","A","3.97","回撤恢复"),("DSR","A","1.00","校正"),
        ("偏度","A","0.286","正偏"),
    ]
    for i in range(0, len(grades_data), 5):
        cols = st.columns(5)
        for j,(name,g,val,_) in enumerate(grades_data[i:i+5]):
            bg = {"A":"#e8f5e9","B":"#e3f2fd","C":"#fff3e0","D":"#fce4ec"}.get(g,"#f8f9fa")
            cols[j].markdown(f"""<div style="background:{bg};border-radius:8px;padding:0.5rem 0.8rem;border:1px solid #e0e0e0;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:0.8rem;">{name}</span>
                    <span class="tag tag-{g.lower()}">{g}</span></div>
                <div style="font-size:1.1rem;font-weight:700;">{val}</div></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
#  Tab 2: 权益分析
# ══════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<p class="section-title">📈 权益曲线 & 回撤分析</p>', unsafe_allow_html=True)

    if not has_data:
        st.info("运行 `python blind_test.py` 生成权益数据")
    else:
        # 选择窗口
        all_windows = list(eq_data.keys())
        sel_window = st.selectbox("选择窗口", all_windows, index=len(all_windows)-1)
        df = eq_data[sel_window]

        # 权益双线图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 1]})

        ax1.plot(df.index, df['equity'], color='#2e7d32', linewidth=1.5, label='策略权益')
        ax1.plot(df.index, df['benchmark'], color='#999', linewidth=1, label='基准权益')
        ax1.fill_between(df.index, df['equity'], df['benchmark'],
                          where=df['equity']>=df['benchmark'], color='#e8f5e9', alpha=0.3)
        ax1.fill_between(df.index, df['equity'], df['benchmark'],
                          where=df['equity']<df['benchmark'], color='#fce4ec', alpha=0.3)
        ax1.legend(fontsize=8); ax1.set_ylabel('权益 (¥)'); ax1.grid(alpha=0.3)
        ax1.set_title(f'W{sel_window} 权益曲线', fontweight='bold')

        # 回撤曲线
        eq_series = df['equity']
        cummax = eq_series.cummax()
        dd = (eq_series - cummax) / cummax * 100
        ax2.fill_between(df.index, 0, dd, color='#c62828', alpha=0.3)
        ax2.plot(df.index, dd, color='#c62828', linewidth=0.8)
        ax2.set_ylabel('回撤 (%)'); ax2.grid(alpha=0.3)
        ax2.set_title('回撤曲线', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # 月度收益热力图
        st.markdown('<p class="section-title">🗓️ 月度收益热力图</p>', unsafe_allow_html=True)
        monthly = df['equity'].resample('ME').last().pct_change() * 100
        if len(monthly) > 1:
            monthly_df = pd.DataFrame({
                'year': monthly.index.year.astype(str),
                'month': monthly.index.month,
                'return': monthly.values
            })
            heatmap = monthly_df.pivot(index='year', columns='month', values='return')

            fig2, ax = plt.subplots(figsize=(10, max(2, len(heatmap)*0.6)))
            im = ax.imshow(heatmap.values, cmap='RdYlGn', aspect='auto', vmin=-15, vmax=15)
            ax.set_xticks(range(len(heatmap.columns))); ax.set_xticklabels([f'{m}月' for m in heatmap.columns])
            ax.set_yticks(range(len(heatmap.index))); ax.set_yticklabels(heatmap.index)
            for i in range(len(heatmap)):
                for j in range(len(heatmap.columns)):
                    v = heatmap.values[i,j]
                    if not np.isnan(v):
                        ax.text(j, i, f'{v:+.1f}%', ha='center', va='center', fontsize=8,
                                color='white' if abs(v) > 8 else 'black')
            plt.colorbar(im, ax=ax, label='月收益 %')
            st.pyplot(fig2)
            plt.close()

        # 关键统计
        eq_arr = df['equity'].values
        total_ret = (eq_arr[-1] / eq_arr[0] - 1) * 100
        max_dd = dd.min()
        daily_ret = np.diff(eq_arr) / eq_arr[:-1]
        sharpe = np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252) if np.std(daily_ret) > 0 else 0
        vol = np.std(daily_ret) * np.sqrt(252) * 100

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("总收益", f"{total_ret:+.1f}%")
        mc2.metric("最大回撤", f"{max_dd:.1f}%")
        mc3.metric("年化波动", f"{vol:.1f}%")
        mc4.metric("Sharpe", f"{sharpe:.2f}")

# ══════════════════════════════════════════════════
#  Tab 3: 交易分析
# ══════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p class="section-title">📋 交易明细 & 盈亏分析</p>', unsafe_allow_html=True)

    if not trade_data:
        st.info("运行 `python blind_test.py` 生成交易数据")
    else:
        sel_w = st.selectbox("选择窗口", list(trade_data.keys()), key='trade_sel', index=len(trade_data)-1)
        td = trade_data[sel_w]

        if len(td) > 0:
            sells = td[td['action'] == 'SELL'].copy()
            buys = td[td['action'] == 'BUY'].copy() if 'BUY' in td['action'].values else td[td['action']!='SELL']

            # 盈亏分布
            fig3, (ax_pnl, ax_hist) = plt.subplots(1, 2, figsize=(10, 3.5))

            if len(sells) > 0 and 'pnl' in sells.columns:
                pnls = sells['pnl'].values
                # 累计盈亏
                cumsum = np.cumsum(pnls)
                colors = ['#2e7d32' if p >= 0 else '#c62828' for p in pnls]
                ax_pnl.bar(range(len(pnls)), pnls, color=colors, width=1)
                ax_pnl.plot(range(len(pnls)), cumsum, color='#1565c0', linewidth=1.5, label='累计盈亏')
                ax_pnl.axhline(0, color='#999', linewidth=0.5)
                ax_pnl.set_title('逐笔盈亏'); ax_pnl.legend(fontsize=8)
                ax_pnl.set_ylabel('¥')

                # 直方图
                ax_hist.hist(pnls, bins=30, color='#1565c0', alpha=0.7, edgecolor='white')
                ax_hist.axvline(0, color='#999', linewidth=0.5)
                ax_hist.axvline(np.mean(pnls), color='#c62828', linewidth=1.5, linestyle='--', label=f'均值 {np.mean(pnls):+.0f}')
                ax_hist.set_title('盈亏分布'); ax_hist.legend(fontsize=8); ax_hist.set_xlabel('¥')

                plt.tight_layout()
                st.pyplot(fig3)
                plt.close()

            # 统计数字
            tc1,tc2,tc3,tc4 = st.columns(4)
            if len(sells) > 0 and 'pnl' in sells.columns:
                tc1.metric("交易数", f"{len(sells)} 卖 / {len(buys)} 买")
                tc2.metric("总盈亏", f"¥{sells['pnl'].sum():,.0f}")
                tc3.metric("盈利占比", f"{(sells['pnl']>0).mean()*100:.0f}%")
                tc4.metric("平均盈亏", f"¥{sells['pnl'].mean():,.0f}")
            if 'commission' in td.columns:
                tc1.metric("总手续费", f"¥{td['commission'].sum():,.0f}")

            # 交易记录表
            st.markdown('<p class="section-title">📜 最近交易 (卖出)</p>', unsafe_allow_html=True)
            if len(sells) > 0:
                show_cols = ['date','symbol','price','qty','pnl']
                show_cols = [c for c in show_cols if c in sells.columns]
                st.dataframe(sells[show_cols].tail(20), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════
#  Tab 4: 因子分析
# ══════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<p class="section-title">🧬 因子重要性分析</p>', unsafe_allow_html=True)

    if not imp_data:
        st.info("运行 `python blind_test.py` 生成因子重要性数据")
    else:
        sel_w2 = st.selectbox("选择窗口", list(imp_data.keys()), key='imp_sel', index=len(imp_data)-1)
        df_imp = imp_data[sel_w2]

        if len(df_imp) > 0:
            # Top-20 柱状图
            top20 = df_imp.head(20)
            fig4, ax = plt.subplots(figsize=(10, 4))
            bars = ax.barh(range(len(top20)-1, -1, -1), top20['importance'].values[::-1], color='#1565c0', height=0.7)
            ax.set_yticks(range(len(top20)-1, -1, -1))
            ax.set_yticklabels(top20['factor'].values[::-1], fontsize=8)
            ax.set_xlabel('重要性'); ax.set_title(f'W{sel_w2} 因子重要性 Top-20', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

            # 完整表格
            st.dataframe(df_imp.head(30), use_container_width=True, hide_index=True)

            # 非零因子统计
            nonzero = (df_imp['importance'] > 0).sum()
            st.caption(f"有效因子: {nonzero} / {len(df_imp)}")

# ══════════════════════════════════════════════════
#  Tab 5: 风险评估
# ══════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<p class="section-title">⚠️ 风险评估</p>', unsafe_allow_html=True)

    if not has_data:
        st.info("运行 `python blind_test.py` 生成数据")
    else:
        all_eq_combined = pd.DataFrame()
        for wn, df in eq_data.items():
            eq = df[['equity']].copy(); eq.columns = [f'W{wn}']
            all_eq_combined = pd.concat([all_eq_combined, eq], axis=1)

        # 全部窗口回撤叠加
        fig5, (ax_dd, ax_vol) = plt.subplots(1, 2, figsize=(10, 3.5))

        for col in all_eq_combined.columns:
            s = all_eq_combined[col].dropna()
            cm = s.cummax()
            dd_series = (s - cm) / cm * 100
            ax_dd.plot(range(len(dd_series)), dd_series, linewidth=0.8, alpha=0.7, label=col)
        ax_dd.set_title('各窗口回撤曲线'); ax_dd.set_ylabel('回撤 %'); ax_dd.legend(fontsize=7)
        ax_dd.axhline(0, color='#999', linewidth=0.5); ax_dd.grid(alpha=0.3)

        # 滚动波动率
        for col in all_eq_combined.columns:
            s = all_eq_combined[col].dropna()
            rets = s.pct_change().dropna()
            roll_vol = rets.rolling(20).std() * np.sqrt(252) * 100
            ax_vol.plot(roll_vol.index, roll_vol.values, linewidth=0.8, alpha=0.7, label=col)
        ax_vol.set_title('滚动年化波动率 (20日)'); ax_vol.set_ylabel('%'); ax_vol.legend(fontsize=7)
        ax_vol.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

        # VaR / Stress
        rc1, rc2, rc3 = st.columns(3)
        all_rets = np.concatenate([df['equity'].pct_change().dropna().values for df in eq_data.values()])
        var_95 = np.percentile(all_rets, 5) * 100
        var_99 = np.percentile(all_rets, 1) * 100
        cvar_95 = all_rets[all_rets <= np.percentile(all_rets, 5)].mean() * 100

        rc1.metric("VaR 95% (日)", f"{var_95:.2f}%", help="95%置信度下日最大损失")
        rc2.metric("VaR 99% (日)", f"{var_99:.2f}%", help="99%置信度下日最大损失")
        rc3.metric("CVaR 95% (日)", f"{cvar_95:.2f}%", help="尾部条件期望损失")

# ══════════════════════════════════════════════════
#  Tab 6: 信号追踪
# ══════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<p class="section-title">🎯 持仓 & 信号追踪</p>', unsafe_allow_html=True)

    if not trade_data:
        st.info("运行 `python blind_test.py` 生成交易数据")
    else:
        sel_w3 = st.selectbox("选择窗口", list(trade_data.keys()), key='sig_sel', index=len(trade_data)-1)
        td = trade_data[sel_w3]

        if len(td) > 0:
            # 持仓变化时间线
            st.markdown("### 📊 买卖信号时间线")
            sells = td[td['action']=='SELL'].copy()
            buys = td[td['action']=='BUY'].copy() if 'BUY' in td['action'].values else pd.DataFrame()

            fig6, ax = plt.subplots(figsize=(10, 3))
            if len(buys) > 0 and 'date' in buys.columns and 'symbol' in buys.columns:
                buy_dates = pd.to_datetime(buys['date'])
                ax.scatter(buy_dates, buys['symbol'], color='#2e7d32', s=30, marker='^', label='买入', alpha=0.7)
            if len(sells) > 0 and 'date' in sells.columns and 'symbol' in sells.columns:
                sell_dates = pd.to_datetime(sells['date'])
                ax.scatter(sell_dates, sells['symbol'], color='#c62828', s=30, marker='v', label='卖出', alpha=0.7)
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            ax.set_title(f'W{sel_w3} 买卖信号'); plt.xticks(rotation=30, fontsize=7)
            plt.tight_layout()
            st.pyplot(fig6)
            plt.close()

            # 交易频率
            st.markdown("### 📈 交易频率")
            freq_cols = st.columns(3)
            if 'date' in td.columns:
                td_dates = pd.to_datetime(td['date'])
                day_counts = td_dates.dt.date.value_counts().sort_index()
                freq_cols[0].metric("总交易", len(td))
                freq_cols[1].metric("活跃天数", len(day_counts))
                freq_cols[2].metric("日均交易", f"{len(td)/max(len(day_counts),1):.1f}")

                # 频次图
                if len(day_counts) > 0:
                    fig7, ax = plt.subplots(figsize=(10, 2))
                    ax.bar(range(len(day_counts)), day_counts.values, color='#1565c0', width=1)
                    ax.set_title('每日交易笔数'); ax.set_xlabel('交易日序号')
                    plt.tight_layout()
                    st.pyplot(fig7)
                    plt.close()

# ════════════════════════════════════════
#  Sidebar
# ════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🔗 快捷命令")
    st.code("python blind_test.py", language="bash")
    st.caption("运行盲测 → 生成评估数据")
    st.code("python test_rolling_v3.py", language="bash")
    st.caption("全量滚动重训练")
    st.divider()
    st.caption("Deep Quant v6")
    if has_data:
        st.success(f"数据已加载: {len(eq_data)} 窗口")
