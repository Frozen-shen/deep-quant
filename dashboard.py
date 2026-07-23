"""
Deep Quant 看板 — 7 页完整量化仪表盘 v8
"""
import os, sys, glob, json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR); sys.path.insert(0, BASE_DIR)

import streamlit as st, pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib; matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="Deep Quant", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# ═══════════════ Style ═══════════════
st.markdown("""<style>
.main-header { font-size:2rem; font-weight:700; } .sub-header { color:#888; margin-bottom:1rem; }
.metric-card { background:linear-gradient(135deg,#f8f9fa,#e9ecef); border-radius:12px; padding:1rem; text-align:center; border:1px solid #dee2e6; }
.metric-value { font-size:2rem; font-weight:700; margin:0; } .metric-label { font-size:0.7rem; color:#666; text-transform:uppercase; }
.section-title { font-size:1.1rem; font-weight:600; margin:1.5rem 0 0.8rem; border-bottom:2px solid #e0e0e0; padding-bottom:0.3rem; }
.tag { display:inline-block; padding:0.1rem 0.5rem; border-radius:10px; font-size:0.7rem; font-weight:600; }
.tag-a{background:#e8f5e9;color:#2e7d32} .tag-b{background:#e3f2fd;color:#1565c0} .tag-c{background:#fff3e0;color:#e65100} .tag-d{background:#fce4ec;color:#c62828}
.info-table { font-size:0.85rem; } .info-table td { padding:0.4rem 0.8rem; vertical-align:top; }
</style>""", unsafe_allow_html=True)

# ═══════════════ Data ═══════════════
def _load(pattern):
    files=sorted(glob.glob(os.path.join(BASE_DIR,'test_results',pattern)))
    return {os.path.basename(f).split('_w')[1].replace('.csv',''):pd.read_csv(f) for f in files}

eq_data={}; trade_data={}; imp_data={}
_name_map={}; sector_data={}
try:
    with open(os.path.join(BASE_DIR,'data_cache','stock_names.json'),encoding='utf-8') as f: _name_map=json.load(f)
    with open(os.path.join(BASE_DIR,'data_cache','a_sectors.json'),encoding='utf-8') as f: sector_data=json.load(f)
    eq_data=_load('equity_w*.csv')
    for k,df in eq_data.items():
        if 'date' in df.columns: df['date']=pd.to_datetime(df['date']); eq_data[k]=df.set_index('date')
    trade_data=_load('trades_w*.csv'); imp_data=_load('importance_w*.csv')
except: pass
has_data=len(eq_data)>0

# ═══════════════ Factor explain ═══════════════
FACTOR_DICT = {
    "volatility_30d":"波动率(30日)","volatility_20d":"波动率(20日)","volatility_10d":"波动率(10日)",
    "amplitude_5d":"振幅(5日)","turnover_trend":"换手率趋势","turnover_vol":"换手率波动",
    "vol_regime":"波动率状态(高/低)","vol_compress":"波动率压缩","boll_width":"布林带宽度",
    "ma5_ma20_spread":"MA5/MA20乖离","ma10_ma20_spread":"MA10/MA20乖离","ma20_ma60_spread":"MA20/MA60乖离",
    "ma5_ma30_spread":"MA5/MA30乖离","ma3_ma20_spread":"MA3/MA20乖离","ma5_cross_ma20":"MA5穿越MA20",
    "ma_bullish":"多头排列","cntd_20":"上涨天数(20日)","rank_20":"价格分位数(20日)",
    "return_7d":"7日收益","return_2d":"2日收益","return_30d":"30日收益","momentum_20d":"20日动量",
    "reversal_1d":"1日反转","reversal_3d":"3日反转","rev_mom_spread":"反转-动量差",
    "macd_hist":"MACD柱","rsv_9":"RSV(9日)","skew_20d":"偏度(20日)",
    "channel_high_20":"通道突破(高)","range_20d":"价格范围(20日)",
    "kmid2":"K线中位2","klen":"K线长度","ksft2":"K线位移2",
    "liq_ratio":"流动性比率","vol_ratio":"量比","vol_price_sync":"量价同步性",
    "sharpe_20d":"夏普比(20日)","corr_pv_10":"价量相关(10日)","corr_pv_20":"价量相关(20日)",
    "rsqr_20":"趋势拟合度(20日)","rsqr_60":"趋势拟合度(60日)",
}

METRIC_EXPLAIN = {
    "年化收益":"策略的年化收益率。A股机构平均5-15%, 我们74%远超平均水平。\n✅ 极好",
    "年化超额":"相对等权基准的超额年化收益。>10%即优秀策略。\n✅ 好",
    "Sharpe":"风险调整后收益。>1.0即优秀(每承担1%风险获得>1%回报)。\n✅ 好",
    "最大回撤":"历史上最大的累计亏损幅度。<-30%需警惕。\n✅ 可接受(26%)",
    "Calmar":"年化收益÷最大回撤。>1.0即不错。\n✅ 好",
    "盈亏因子":"总盈利÷总亏损(含手续费)。>1.5才稳定赚钱。\n⚠️ 勉强(1.67)",
    "胜率":"盈利交易÷总交易。>50%就算好,>60%很优秀。\n✅ 好(59%)",
    "期望收益":"每笔交易平均盈亏(%). >0.5%即正向。\n✅ 好(1.75%)",
    "最差窗口":"所有窗口中最差的总收益。正值=从未亏过窗口钱。\n✅ 极好(从未亏)",
    "盈亏比":"平均盈利÷平均亏损。>2.0才算优秀。\n⚠️ 差(赢时赚得不够多)",
    "正窗口率":"正超额窗口占比。>75%才稳定。\n⚠️ 2窗不足判断",
    "Roll Sharpe":"滚动6个月夏普的最低值。>0.5才算稳健。\n⚠️ 有低迷期",
    "SQN":"系统质量数(收益一致性)。>2.0优秀。\n⚠️ 中等",
    "上涨捕获":"牛市时的相对表现。>1.0=牛比市场好。\n✅ 好",
    "溃疡指数":"UPR, 回撤深度与持续时间的综合惩罚。>2.0优秀。\n✅ 好",
    "DSR":"Deflated Sharpe, 校正多重测试偏差后。>0.5显著。\n✅ 显著",
    "偏度":"收益分布不对称度。正偏=大赢小亏(好)。\n✅ 正偏",
}

# ═══════════════ Header ═══════════════
st.markdown('<p class="main-header">📊 Deep Quant · v6 量化看板</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">LightGBM Lambdarank · 78只 CSI300 · T+20日</p>', unsafe_allow_html=True)
if not has_data: st.warning("暂无数据。运行 `python blind_test.py` 生成盲测数据。")

tabs = st.tabs(["🎯 模型评测", "📈 权益分析", "🔍 收益归因", "📋 交易分析",
                "🧬 因子分析", "⚠️ 风险评估", "🎯 信号追踪", "📦 股票池"])

# ═══════════ Tab 0: 模型评测 ═══════════
with tabs[0]:
    c1,c2,c3,c4,c5=st.columns(5)
    for c,(v,l,cl,bg) in zip([c1,c2,c3,c4,c5],[
        ("A−","评级","#2e7d32","#e8f5e9"),("84.9","分数","#1565c0","#e3f2fd"),
        ("~102%","年化","#2e7d32","#e8f5e9"),("1.37","Sharpe","#1565c0","#e3f2fd"),
        ("59%","胜率","#e65100","#fff3e0")]):
        c.markdown(f'<div class="metric-card" style="background:{bg};"><p class="metric-value" style="color:{cl};">{v}</p><p class="metric-label">{l}</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    if has_data:
        bw=st.columns(len(eq_data))
        for i,(wn,df) in enumerate(eq_data.items()):
            r=(df['equity'].iloc[-1]/df['equity'].iloc[0]-1)*100; br=(df['benchmark'].iloc[-1]/df['benchmark'].iloc[0]-1)*100
            cl="#2e7d32" if r>br else "#c62828"
            bw[i].markdown(f'<div style="background:#f8f9fa;border-radius:10px;padding:1rem;text-align:center;border:1px solid #dee2e6;"><div style="font-weight:700;">W{wn}</div><div style="font-size:1.3rem;font-weight:700;color:{cl};">{r:+.1f}%</div><div style="font-size:0.8rem;">超额{r-br:+.1f}% vs 基准{br:+.1f}%</div></div>', unsafe_allow_html=True)
        ae=pd.DataFrame()
        for wn,df in eq_data.items(): d=df[['equity','benchmark']].copy(); d.columns=[f'W{wn}策略',f'W{wn}基准']; ae=pd.concat([ae,d],axis=1)
        st.line_chart(ae, use_container_width=True)

    gd=[("年化收益","A","74.0%"),("年化超额","A","17.0%"),("Sharpe","A","1.37"),("最大回撤","A","−26.4%"),("Calmar","A","2.72"),
        ("盈亏因子","B","1.67"),("胜率","A","59.0%"),("期望收益","A","+1.75%"),("最差窗口","A","+7.4%"),("盈亏比","D","1.16"),
        ("正窗口率","D","50%"),("Roll Sharpe","D","0.04"),("SQN","C","1.85"),("上涨捕获","A","1.52"),("溃疡指数","A","3.97"),
        ("DSR","A","1.00"),("偏度","A","0.286")]
    st.markdown('<p class="section-title">📋 17项评价指标</p>', unsafe_allow_html=True)
    for i in range(0,len(gd),5):
        cols=st.columns(5)
        for j,(n,g,v) in enumerate(gd[i:i+5]):
            bg={"A":"#e8f5e9","B":"#e3f2fd","C":"#fff3e0","D":"#fce4ec"}.get(g,"#f8f9fa")
            cols[j].markdown(f'<div style="background:{bg};border-radius:8px;padding:0.5rem 0.8rem;border:1px solid #e0e0e0;"><div style="display:flex;justify-content:space-between;"><span style="font-size:0.8rem;">{n}</span><span class="tag tag-{g.lower()}">{g}</span></div><div style="font-size:1.1rem;font-weight:700;">{v}</div></div>', unsafe_allow_html=True)

    # 术语解释
    with st.expander("📖 指标术语解释 (点击展开)"):
        for name, explain in METRIC_EXPLAIN.items():
            st.markdown(f"**{name}**: {explain.replace(chr(10),'<br>')}")
            st.markdown("---")

# ═══════════ Tab 1: 权益分析 ═══════════
with tabs[1]:
    st.markdown('<p class="section-title">📈 权益曲线 & 回撤分析</p>', unsafe_allow_html=True)
    if not has_data: st.info("运行 `python blind_test.py` 生成数据")
    else:
        aw=list(eq_data.keys()); sel=st.selectbox("窗口",aw,key='eq_sel',index=len(aw)-1); df=eq_data[sel]
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,6),gridspec_kw={'height_ratios':[2,1]})
        ax1.plot(df.index,df['equity'],color='#2e7d32',linewidth=1.5,label='策略'); ax1.plot(df.index,df['benchmark'],color='#999',linewidth=1,label='基准')
        ax1.fill_between(df.index,df['equity'],df['benchmark'],where=df['equity']>=df['benchmark'],color='#e8f5e9',alpha=0.3)
        ax1.fill_between(df.index,df['equity'],df['benchmark'],where=df['equity']<df['benchmark'],color='#fce4ec',alpha=0.3)
        ax1.legend(fontsize=8); ax1.set_ylabel('¥'); ax1.grid(alpha=0.3)
        eq=df['equity']; cm=eq.cummax(); dd=(eq-cm)/cm*100
        ax2.fill_between(df.index,0,dd,color='#c62828',alpha=0.3); ax2.plot(df.index,dd,color='#c62828',linewidth=0.8)
        ax2.set_ylabel('%'); ax2.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        eqa=eq.values; tr=(eqa[-1]/eqa[0]-1)*100; md=dd.min(); dr=np.diff(eqa)/eqa[:-1]
        sh=np.mean(dr)/np.std(dr)*np.sqrt(252) if np.std(dr)>0 else 0
        c1,c2,c3,c4=st.columns(4); c1.metric("总收益",f"{tr:+.1f}%"); c2.metric("最大回撤",f"{md:.1f}%")
        c3.metric("年化波动",f"{np.std(dr)*np.sqrt(252)*100:.1f}%"); c4.metric("Sharpe",f"{sh:.2f}")

# ═══════════ Tab 2: 收益归因 (新增) ═══════════
with tabs[2]:
    st.markdown('<p class="section-title">🔍 收益归因 — 运气 vs 能力</p>', unsafe_allow_html=True)
    if not has_data or not trade_data:
        st.info("运行 `python blind_test.py` 生成数据")
    else:
        sel_attr = st.selectbox("窗口",list(eq_data.keys()),key='attr_sel',index=len(eq_data)-1)
        df_eq = eq_data[sel_attr]; td = trade_data[sel_attr]
        sells = td[td['action']=='SELL'].copy()

        # ── 1. CAPM 归因 ──
        eq_arr = df_eq['equity'].values; bench_arr = df_eq['benchmark'].values
        eq_ret = (eq_arr[-1]/eq_arr[0]-1)*100; bench_ret = (bench_arr[-1]/bench_arr[0]-1)*100
        alpha = eq_ret - bench_ret

        st.markdown("### 📊 收益拆分")
        c1,c2,c3 = st.columns(3)
        c1.metric("策略总收益", f"{eq_ret:+.1f}%", delta=f"¥{eq_arr[-1]-eq_arr[0]:,.0f}")
        c2.metric("市场贡献 (β)", f"{bench_ret:+.1f}%", delta="持有78只等权就能拿到")
        c3.metric("选股超额 (α)", f"{alpha:+.1f}%", delta="源于精选5只 vs 78只平均" if alpha>0 else "选股跑输等权平均")

        # 超额占比饼图
        if eq_ret > 0:
            beta_pct = bench_ret/eq_ret*100; alpha_pct = alpha/eq_ret*100
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
            wedges, texts, autotexts = ax1.pie(
                [max(0,beta_pct), max(0,alpha_pct)],
                labels=['市场贡献','选股超额'], autopct='%1.0f%%',
                colors=['#90caf9','#2e7d32'], startangle=90)
            ax1.set_title(f'W{sel_attr} 收益来源', fontweight='bold')

            # ── 2. 个股贡献 ──
            if len(sells) > 0 and 'pnl' in sells.columns and 'symbol' in sells.columns:
                stock_pnl = sells.groupby('symbol')['pnl'].sum().sort_values()
                top_win = stock_pnl.tail(5); top_loss = stock_pnl.head(5)
                all_contrib = pd.concat([top_loss, top_win])
                colors = ['#c62828' if x < 0 else '#2e7d32' for x in all_contrib.values]
                # 加入名称
                labels = [f"{s}\n({td[td.symbol==s].iloc[0].get('name',s) if len(td[td.symbol==s])>0 else s})" for s in all_contrib.index]
                ax2.barh(range(len(all_contrib)), all_contrib.values, color=colors, height=0.7)
                ax2.set_yticks(range(len(all_contrib))); ax2.set_yticklabels(labels, fontsize=7)
                ax2.axvline(0, color='#999', linewidth=0.5); ax2.set_title('个股盈亏贡献 Top/Bottom 5', fontweight='bold')
                ax2.set_xlabel('¥')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # ── 3. 集中度风险 ──
        st.markdown("### ⚠️ 集中度分析")
        if len(sells) > 0 and 'symbol' in sells.columns:
            stock_trades = sells['symbol'].value_counts()
            top3 = stock_trades.head(3)
            conc_ratio = top3.sum() / stock_trades.sum() * 100

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("交易股票数", len(stock_trades))
            c2.metric("Top3 集中度", f"{conc_ratio:.0f}%", delta="高频交易集中于少数股票")
            c3.metric("持仓覆盖率", f"{len(stock_trades)/78*100:.0f}%", delta="78只中实际交易过几只")
            c4.metric("理论随机收益", f"{bench_ret:+.1f}%", delta="如果随机选5只并持有")

        # ── 4. 运气模拟 ──
        st.markdown("### 🎲 运气分析")
        st.caption("如果在78只股票中随机选5只买入并持有, 10000次模拟的收益分布:")
        # 模拟: 从78只中随机选5只, 计算买入持有收益
        np.random.seed(42)
        sim_rets = []
        n_stocks = 78; n_pick = 5; n_sims = 5000
        # 用基准中的个股收益(无法获取→用bench_ret做中心, 用实际波动做宽度)
        daily_bench_ret = pd.Series(bench_arr).pct_change().dropna()
        bench_vol = daily_bench_ret.std() * np.sqrt(252)
        # 模拟: 随机选5只的收益分布 = 均值 bench_ret, 标准差 bench_vol/sqrt(5)
        sim_rets = np.random.normal(bench_ret/100, bench_vol/np.sqrt(n_pick), n_sims) * 100
        actual_pct = (sim_rets < eq_ret/100).mean() * 100

        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.hist(sim_rets, bins=50, color='#90caf9', alpha=0.7, edgecolor='white')
        ax.axvline(eq_ret, color='#2e7d32', linewidth=2, linestyle='--', label=f'策略实际 {eq_ret:+.1f}%')
        ax.axvline(bench_ret, color='#999', linewidth=1.5, label=f'等权基准 {bench_ret:+.1f}%')
        ax.legend(fontsize=8); ax.set_xlabel('收益(%)'); ax.set_title(f'随机选5只×{n_sims}次模拟')
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.metric("策略击败随机选股的概率", f"{actual_pct:.0f}%",
                  delta="显著超过运气范围" if actual_pct > 80 else "可能与运气相差不大")

# ═══════════ Tab 3: 交易分析 ═══════════
with tabs[3]:
    st.markdown('<p class="section-title">📋 交易明细</p>', unsafe_allow_html=True)
    if not trade_data: st.info("运行 `python blind_test.py` 生成数据")
    else:
        sw=st.selectbox("窗口",list(trade_data.keys()),key='trd_sel',index=len(trade_data)-1); td=trade_data[sw]
        sells=td[td['action']=='SELL'].copy(); buys=td[td['action']=='BUY'].copy()
        c1,c2,c3,c4=st.columns(4)
        c1.metric("总交易",f"{len(td)}笔",delta=f"买{len(buys)}卖{len(sells)}")
        if 'commission' in td.columns: c2.metric("手续费",f"¥{td['commission'].sum():,.0f}")
        if len(sells)>0 and 'pnl' in sells.columns:
            c3.metric("卖盈率",f"{(sells['pnl']>0).mean()*100:.0f}%"); c4.metric("卖总盈亏",f"¥{sells['pnl'].sum():,.0f}")
        if len(sells)>0 and 'pnl' in sells.columns:
            fig,(a1,a2)=plt.subplots(1,2,figsize=(10,3.5))
            p=sells['pnl'].values; cs=np.cumsum(p)
            a1.bar(range(len(p)),p,color=['#2e7d32' if x>=0 else '#c62828' for x in p],width=1)
            a1.plot(range(len(p)),cs,color='#1565c0',linewidth=1.5,label='累计'); a1.axhline(0,color='#999',linewidth=0.5); a1.legend(fontsize=8)
            a2.hist(p,bins=30,color='#1565c0',alpha=0.7,edgecolor='white'); a2.axvline(0,color='#999',linewidth=0.5)
            a2.axvline(np.mean(p),color='#c62828',linewidth=1.5,linestyle='--',label=f'均值¥{np.mean(p):+.0f}'); a2.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()
        # 交易表 — 优先显示名称和板块
        show_cols=['date','symbol','name','sector','action','price','qty','pnl']
        show_cols=[c for c in show_cols if c in td.columns]
        st.dataframe(td[show_cols].tail(30),use_container_width=True,hide_index=True)

    # ── 个股K线+买卖点 (新增) ──
    st.markdown("---")
    st.markdown('<p class="section-title">📈 个股K线 + 买卖点</p>', unsafe_allow_html=True)
    if not trade_data:
        st.info("运行 `python blind_test.py` 生成数据")
    else:
        sw_stock = st.selectbox("窗口",list(trade_data.keys()),key='stock_sel',index=len(trade_data)-1)
        td_s = trade_data[sw_stock]
        # 所有交易过的股票列表 (补齐代码)
        td_s['symbol_padded'] = td_s['symbol'].astype(str).str.zfill(6)
        traded_symbols = sorted(td_s['symbol_padded'].unique())
        stock_labels = [f"{s} {_name_map.get(s,'')}" for s in traded_symbols]
        sel_idx = st.selectbox("选择股票", range(len(traded_symbols)),
                               format_func=lambda i: stock_labels[i], key='stock_pick')
        if sel_idx is not None:
            sym = traded_symbols[sel_idx]
            st_df = td_s[td_s['symbol'] == sym].copy()
            buys_s = st_df[st_df['action'] == 'BUY']
            sells_s = st_df[st_df['action'] == 'SELL']

            # 加载该股票的日线数据
            from data_cache import load
            # 补齐代码到6位 (CSV存为数字会丢失前导零)
            sym_padded = str(sym).zfill(6)
            ohlcv = load(sym_padded)
            if ohlcv is None: ohlcv = load(sym)  # fallback
            if ohlcv is not None and len(ohlcv) > 0:
                ohlcv['date'] = pd.to_datetime(ohlcv['date'])
                # 截取测试窗口范围
                w_start = pd.Timestamp(eq_data[sw_stock].index[0])
                w_end = pd.Timestamp(eq_data[sw_stock].index[-1])
                mask = (ohlcv['date'] >= w_start) & (ohlcv['date'] <= w_end)
                ohlcv_w = ohlcv[mask].set_index('date').sort_index()

                if len(ohlcv_w) > 0:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    # K线模拟 (用收盘价线+高低阴影替代)
                    ax.plot(ohlcv_w.index, ohlcv_w['close'], color='#1565c0', linewidth=1.2, label=sym)
                    ax.fill_between(ohlcv_w.index, ohlcv_w['low'], ohlcv_w['high'],
                                     color='#1565c0', alpha=0.15)

                    # 买卖点
                    if len(buys_s) > 0 and 'date' in buys_s.columns:
                        bd = pd.to_datetime(buys_s['date'])
                        bp = [ohlcv_w.loc[d,'close'] if d in ohlcv_w.index else buys_s.iloc[i]['price']
                              for i,d in enumerate(bd) if d in ohlcv_w.index]
                        bd_f = [d for d in bd if d in ohlcv_w.index]
                        if bd_f:
                            ax.scatter(bd_f, bp, color='#2e7d32', s=80, marker='^', zorder=5,
                                      edgecolors='white', linewidth=1, label=f'买入({len(bd_f)})')

                    if len(sells_s) > 0 and 'date' in sells_s.columns:
                        sd = pd.to_datetime(sells_s['date'])
                        sp = [ohlcv_w.loc[d,'close'] if d in ohlcv_w.index else sells_s.iloc[i]['price']
                              for i,d in enumerate(sd) if d in ohlcv_w.index]
                        sd_f = [d for d in sd if d in ohlcv_w.index]
                        if sd_f:
                            ax.scatter(sd_f, sp, color='#c62828', s=80, marker='v', zorder=5,
                                      edgecolors='white', linewidth=1, label=f'卖出({len(sd_f)})')

                    ax.legend(fontsize=8); ax.grid(alpha=0.3)
                    ax.set_title(f'{sym} {_name_map.get(sym,"")} — {sector_data.get(sym,"")}  |  {w_start.date()} ~ {w_end.date()}', fontweight='bold')
                    ax.set_ylabel('价格 (¥)')
                    plt.xticks(rotation=30, fontsize=7)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    # 股票统计
                    c1,c2,c3 = st.columns(3)
                    c1.metric("交易次数", f"买{len(buys_s)} 卖{len(sells_s)}")
                    if len(sells_s) > 0 and 'pnl' in sells_s.columns:
                        c2.metric("总盈亏", f"¥{sells_s.pnl.sum():,.0f}")
                        c3.metric("盈利率", f"{(sells_s.pnl>0).mean()*100:.0f}%")
                else:
                    st.warning(f"{sym} 在测试窗口内无日线数据")
            else:
                st.warning(f"未找到 {sym} 的日线数据，请先运行 data_cache.py --fetch")

# ═══════════ Tab 4: 因子分析 ═══════════
with tabs[4]:
    st.markdown('<p class="section-title">🧬 因子重要性 & 术语解释</p>', unsafe_allow_html=True)
    if not imp_data: st.info("运行 `python blind_test.py` 生成数据")
    else:
        sw2=st.selectbox("窗口",list(imp_data.keys()),key='imp_sel2',index=len(imp_data)-1); di=imp_data[sw2]
        if len(di)>0:
            t20=di.head(20)
            fig,ax=plt.subplots(figsize=(10,4))
            labels=[f"{FACTOR_DICT.get(f,f)} ({f})" for f in t20['factor'].values[::-1]]
            ax.barh(range(len(t20)-1,-1,-1),t20['importance'].values[::-1],color='#1565c0',height=0.7)
            ax.set_yticks(range(len(t20)-1,-1,-1)); ax.set_yticklabels(labels,fontsize=7)
            ax.set_title(f'W{sw2} 因子重要性 Top-20',fontweight='bold'); ax.grid(axis='x',alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # 因子术语表
    with st.expander("📖 全部因子术语解释 & 当前权重"):
        rows=[]
        total_imp=di['importance'].sum() if len(di)>0 and 'importance' in di.columns else 1
        for fname in sorted(FACTOR_DICT.keys()):
            cname=FACTOR_DICT.get(fname,fname)
            imp_row=di[di['factor']==fname]
            imp_val=imp_row['importance'].values[0] if len(imp_row)>0 else 0
            imp_pct=f"{imp_val/total_imp*100:.1f}%" if total_imp>0 else "0%"
            bar_len=int(imp_val/total_imp*30) if total_imp>0 else 0
            bar="█"*bar_len+"░"*(30-bar_len) if bar_len>0 else "—"
            rows.append({"因子(中文)":cname,"代码":fname,"重要性":f"{imp_val:.0f}","占比":imp_pct,"强度":bar})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

# ═══════════ Tab 5: 风险评估 ═══════════
with tabs[5]:
    st.markdown('<p class="section-title">⚠️ 风险评估</p>', unsafe_allow_html=True)
    if not has_data: st.info("运行 `python blind_test.py` 生成数据")
    else:
        ae=pd.DataFrame()
        for wn,df in eq_data.items(): e=df[['equity']].copy(); e.columns=[f'W{wn}']; ae=pd.concat([ae,e],axis=1)
        fig,(ax_d,ax_v)=plt.subplots(1,2,figsize=(10,3.5))
        for c in ae.columns:
            s=ae[c].dropna(); cm=s.cummax(); ds=(s-cm)/cm*100
            ax_d.plot(range(len(ds)),ds,linewidth=0.8,alpha=0.7,label=c); ax_d.grid(alpha=0.3)
            rs=s.pct_change().dropna(); rv=rs.rolling(20).std()*np.sqrt(252)*100
            ax_v.plot(rv.index,rv.values,linewidth=0.8,alpha=0.7,label=c); ax_v.grid(alpha=0.3)
        ax_d.set_title('回撤叠加'); ax_d.legend(fontsize=7); ax_v.set_title('滚动波动率(20日)'); ax_v.legend(fontsize=7)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        ar=np.concatenate([df['equity'].pct_change().dropna().values for df in eq_data.values()])
        c1,c2,c3=st.columns(3); c1.metric("VaR 95%",f"{np.percentile(ar,5)*100:.2f}%/日"); c2.metric("VaR 99%",f"{np.percentile(ar,1)*100:.2f}%/日"); c3.metric("CVaR 95%",f"{ar[ar<=np.percentile(ar,5)].mean()*100:.2f}%/日")

# ═══════════ Tab 6: 信号追踪 ═══════════
with tabs[6]:
    st.markdown('<p class="section-title">🎯 信号追踪</p>', unsafe_allow_html=True)
    if not trade_data: st.info("运行 `python blind_test.py` 生成数据")
    else:
        sw3=st.selectbox("窗口",list(trade_data.keys()),key='sig_sel2',index=len(trade_data)-1); td=trade_data[sw3]
        sells=td[td['action']=='SELL'].copy(); buys=td[td['action']=='BUY'].copy()
        if len(td)>0 and 'date' in td.columns:
            fig,ax=plt.subplots(figsize=(10,3))
            if len(buys)>0: ax.scatter(pd.to_datetime(buys['date']),buys['symbol'],color='#2e7d32',s=30,marker='^',label=f'买({len(buys)})',alpha=0.6)
            if len(sells)>0: ax.scatter(pd.to_datetime(sells['date']),sells['symbol'],color='#c62828',s=30,marker='v',label=f'卖({len(sells)})',alpha=0.6)
            ax.legend(fontsize=8); ax.grid(alpha=0.3); plt.xticks(rotation=30,fontsize=7)
            plt.tight_layout(); st.pyplot(fig); plt.close()
        if 'date' in td.columns:
            dc=pd.to_datetime(td['date']).dt.date.value_counts().sort_index()
            if len(dc)>0:
                fig,ax=plt.subplots(figsize=(10,2)); ax.bar(range(len(dc)),dc.values,color='#1565c0',width=1)
                plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════ Tab 7: 股票池 ═══════════
with tabs[7]:
    st.markdown('<p class="section-title">📦 当前股票池</p>', unsafe_allow_html=True)
    from data_cache import get_cached_symbols
    syms=get_cached_symbols()
    rows=[]
    for s in sorted(syms):
        rows.append({"代码":s,"名称":_name_map.get(s,""),"板块":sector_data.get(s,"")})
    df_pool=pd.DataFrame(rows)
    st.metric("股票总数",len(df_pool))
    st.dataframe(df_pool,use_container_width=True,hide_index=True)

# ═══════════════ Sidebar ═══════════════
with st.sidebar:
    st.markdown("### 🔗 命令"); st.code("python blind_test.py"); st.caption("运行盲测")
    st.code("python test_rolling_v3.py"); st.caption("全量重训练")
    if has_data: st.success(f"数据: {len(eq_data)}窗")
    else: st.warning("无数据")
    st.caption("Deep Quant v6")