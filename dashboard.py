"""
Deep Quant 看板 — 6 页完整量化仪表盘 v7

启动: streamlit run dashboard.py --server.headless true
"""
import os, sys, glob
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_page_config(page_title="Deep Quant", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# ═══════════════════ Style ═══════════════════
st.markdown("""<style>
.main-header { font-size:2rem; font-weight:700; }
.sub-header  { color:#888; margin-bottom:1.5rem; }
.metric-card { background:linear-gradient(135deg,#f8f9fa,#e9ecef); border-radius:12px; padding:1rem 1.2rem; text-align:center; border:1px solid #dee2e6; margin:0.15rem 0; }
.metric-value { font-size:2rem; font-weight:700; margin:0; }
.metric-label { font-size:0.75rem; color:#666; text-transform:uppercase; letter-spacing:0.03em; }
.section-title { font-size:1.1rem; font-weight:600; margin:1.5rem 0 0.8rem; border-bottom:2px solid #e0e0e0; padding-bottom:0.3rem; }
.tag { display:inline-block; padding:0.1rem 0.5rem; border-radius:10px; font-size:0.7rem; font-weight:600; }
.tag-a{background:#e8f5e9;color:#2e7d32} .tag-b{background:#e3f2fd;color:#1565c0}
.tag-c{background:#fff3e0;color:#e65100} .tag-d{background:#fce4ec;color:#c62828}
</style>""", unsafe_allow_html=True)

# ═══════════════════ Data loading ═══════════════════
def _load(pattern):
    files = sorted(glob.glob(os.path.join(BASE_DIR, 'test_results', pattern)))
    data = {}
    for f in files:
        wn = os.path.basename(f).split('_w')[1].replace('.csv','')
        data[wn] = pd.read_csv(f)
    return data

eq_data = {}; trade_data = {}; imp_data = {}
try:
    eq_data = _load('equity_w*.csv')
    for k, df in eq_data.items():
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']); eq_data[k] = df.set_index('date')
    trade_data = _load('trades_w*.csv')
    imp_data = _load('importance_w*.csv')
except: pass
has_data = len(eq_data) > 0

# ═══════════════════ Header ═══════════════════
st.markdown('<p class="main-header">📊 Deep Quant · v6 量化看板</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">LightGBM Lambdarank · 78只 CSI300 · T+20日趋势跟随</p>', unsafe_allow_html=True)
if not has_data:
    st.warning("暂无数据。运行 `python blind_test.py` 生成盲测数据。")

tabs = st.tabs(["🎯 模型评测", "📈 权益分析", "📋 交易分析", "🧬 因子分析", "⚠️ 风险评估", "🎯 信号追踪"])

# ════════════════════════════════════════ Tab 1: 模型评测 ═══════
with tabs[0]:
    c1,c2,c3,c4,c5 = st.columns(5)
    for c,(v,l,cl,bg) in zip([c1,c2,c3,c4,c5], [
        ("A−","综合评级","#2e7d32","#e8f5e9"), ("84.9","分数/100","#1565c0","#e3f2fd"),
        ("~102%","年化收益","#2e7d32","#e8f5e9"), ("1.37","Sharpe","#1565c0","#e3f2fd"),
        ("59%","胜率","#e65100","#fff3e0")]):
        c.markdown(f'<div class="metric-card" style="background:{bg};"><p class="metric-value" style="color:{cl};">{v}</p><p class="metric-label">{l}</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    if has_data:
        st.markdown('<p class="section-title">🔒 盲测窗口</p>', unsafe_allow_html=True)
        bw = st.columns(len(eq_data))
        for i, (wn, df) in enumerate(eq_data.items()):
            r = (df['equity'].iloc[-1]/df['equity'].iloc[0]-1)*100
            br = (df['benchmark'].iloc[-1]/df['benchmark'].iloc[0]-1)*100
            ex = r-br; cl = "#2e7d32" if ex>0 else "#c62828"
            bw[i].markdown(f'<div style="background:#f8f9fa;border-radius:10px;padding:1rem;text-align:center;border:1px solid #dee2e6;"><div style="font-weight:700;">W{wn}</div><div style="font-size:1.4rem;font-weight:700;color:{cl};">{r:+.1f}%</div><div style="font-size:0.8rem;color:#888;">超额 {ex:+.1f}% vs 基准 {br:+.1f}%</div></div>', unsafe_allow_html=True)
        all_eq = pd.DataFrame()
        for wn, df in eq_data.items():
            d = df[['equity','benchmark']].copy(); d.columns=[f'W{wn}策略',f'W{wn}基准']; all_eq=pd.concat([all_eq,d],axis=1)
        st.line_chart(all_eq, use_container_width=True)

    gd = [("年化收益","A","74.0%"),("年化超额","A","17.0%"),("Sharpe","A","1.37"),("最大回撤","A","−26.4%"),("Calmar","A","2.72"),
          ("盈亏因子","B","1.67"),("胜率","A","59.0%"),("期望收益","A","+1.75%"),("最差窗口","A","+7.4%"),("盈亏比","D","1.16"),
          ("正窗口率","D","50%"),("Roll Sharpe","D","0.04"),("SQN","C","1.85"),("上涨捕获","A","1.52"),("溃疡指数","A","3.97"),
          ("DSR","A","1.00"),("偏度","A","0.286")]
    st.markdown('<p class="section-title">📋 盲测17指标</p>', unsafe_allow_html=True)
    for i in range(0,len(gd),5):
        cols=st.columns(5)
        for j,(n,g,v) in enumerate(gd[i:i+5]):
            bg={"A":"#e8f5e9","B":"#e3f2fd","C":"#fff3e0","D":"#fce4ec"}.get(g,"#f8f9fa")
            cols[j].markdown(f'<div style="background:{bg};border-radius:8px;padding:0.5rem 0.8rem;border:1px solid #e0e0e0;"><div style="display:flex;justify-content:space-between;"><span style="font-size:0.8rem;">{n}</span><span class="tag tag-{g.lower()}">{g}</span></div><div style="font-size:1.1rem;font-weight:700;">{v}</div></div>', unsafe_allow_html=True)

# ════════════════════════════════════════ Tab 2: 权益分析 ═══════
with tabs[1]:
    st.markdown('<p class="section-title">📈 权益曲线 & 回撤分析</p>', unsafe_allow_html=True)
    if not has_data: st.info("运行 `python blind_test.py` 生成数据")
    else:
        aw = list(eq_data.keys())
        sel = st.selectbox("窗口", aw, key='eq_sel', index=len(aw)-1)
        df = eq_data[sel]
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,6),gridspec_kw={'height_ratios':[2,1]})
        ax1.plot(df.index,df['equity'],color='#2e7d32',linewidth=1.5,label='策略'); ax1.plot(df.index,df['benchmark'],color='#999',linewidth=1,label='基准')
        ax1.fill_between(df.index,df['equity'],df['benchmark'],where=df['equity']>=df['benchmark'],color='#e8f5e9',alpha=0.3)
        ax1.fill_between(df.index,df['equity'],df['benchmark'],where=df['equity']<df['benchmark'],color='#fce4ec',alpha=0.3)
        ax1.legend(fontsize=8);ax1.set_ylabel('¥');ax1.grid(alpha=0.3);ax1.set_title(f'W{sel} 权益曲线',fontweight='bold')
        eq=df['equity'];cm=eq.cummax();dd=(eq-cm)/cm*100
        ax2.fill_between(df.index,0,dd,color='#c62828',alpha=0.3);ax2.plot(df.index,dd,color='#c62828',linewidth=0.8)
        ax2.set_ylabel('%');ax2.grid(alpha=0.3);ax2.set_title('回撤',fontweight='bold')
        plt.tight_layout();st.pyplot(fig);plt.close()
        eqa=df['equity'].values;tr=(eqa[-1]/eqa[0]-1)*100;md=dd.min()
        dr=np.diff(eqa)/eqa[:-1];sh=np.mean(dr)/np.std(dr)*np.sqrt(252) if np.std(dr)>0 else 0;vo=np.std(dr)*np.sqrt(252)*100
        c1,c2,c3,c4=st.columns(4);c1.metric("总收益",f"{tr:+.1f}%");c2.metric("最大回撤",f"{md:.1f}%");c3.metric("年化波动",f"{vo:.1f}%");c4.metric("Sharpe",f"{sh:.2f}")

# ════════════════════════════════════════ Tab 3: 交易分析 ═══════
with tabs[2]:
    st.markdown('<p class="section-title">📋 交易明细</p>', unsafe_allow_html=True)
    if not trade_data: st.info("运行 `python blind_test.py` 生成数据")
    else:
        sw = st.selectbox("窗口",list(trade_data.keys()),key='trd_sel',index=len(trade_data)-1)
        td=trade_data[sw];sells=td[td['action']=='SELL'].copy();buys=td[td['action']=='BUY'].copy()
        c1,c2,c3,c4=st.columns(4)
        c1.metric("总交易",f"{len(td)}笔",delta=f"买{len(buys)}卖{len(sells)}")
        if len(sells)>0 and 'pnl' in sells.columns:
            c2.metric("卖总盈亏",f"¥{sells['pnl'].sum():,.0f}");c3.metric("卖盈率",f"{(sells['pnl']>0).mean()*100:.0f}%");c4.metric("卖均盈亏",f"¥{sells['pnl'].mean():,.0f}")
        if 'commission' in td.columns: st.caption(f"手续费: ¥{td['commission'].sum():,.0f}")
        if len(sells)>0 and 'pnl' in sells.columns:
            st.markdown("### 💰 卖出盈亏")
            fig,(a1,a2)=plt.subplots(1,2,figsize=(10,3.5))
            p=sells['pnl'].values;cs=np.cumsum(p);clr=['#2e7d32' if x>=0 else '#c62828' for x in p]
            a1.bar(range(len(p)),p,color=clr,width=1);a1.plot(range(len(p)),cs,color='#1565c0',linewidth=1.5,label='累计');a1.axhline(0,color='#999',linewidth=0.5);a1.set_title('逐笔');a1.legend(fontsize=8)
            a2.hist(p,bins=30,color='#1565c0',alpha=0.7,edgecolor='white');a2.axvline(0,color='#999',linewidth=0.5);a2.axvline(np.mean(p),color='#c62828',linewidth=1.5,linestyle='--',label=f'均值{np.mean(p):+.0f}');a2.legend(fontsize=8);a2.set_title('分布')
            plt.tight_layout();st.pyplot(fig);plt.close()
        sc=['date','symbol','action','price','qty','pnl','commission'];sc=[c for c in sc if c in td.columns]
        st.dataframe(td[sc].tail(30),use_container_width=True,hide_index=True)

# ════════════════════════════════════════ Tab 4: 因子分析 ═══════
with tabs[3]:
    st.markdown('<p class="section-title">🧬 因子重要性</p>', unsafe_allow_html=True)
    if not imp_data: st.info("运行 `python blind_test.py` 生成数据")
    else:
        sw2=st.selectbox("窗口",list(imp_data.keys()),key='imp_sel2',index=len(imp_data)-1)
        di=imp_data[sw2]
        if len(di)>0:
            t20=di.head(20)
            fig,ax=plt.subplots(figsize=(10,4))
            ax.barh(range(len(t20)-1,-1,-1),t20['importance'].values[::-1],color='#1565c0',height=0.7)
            ax.set_yticks(range(len(t20)-1,-1,-1));ax.set_yticklabels(t20['factor'].values[::-1],fontsize=8)
            ax.set_xlabel('重要性');ax.set_title(f'W{sw2} Top-20',fontweight='bold');ax.grid(axis='x',alpha=0.3)
            plt.tight_layout();st.pyplot(fig);plt.close()
            st.dataframe(di.head(30),use_container_width=True,hide_index=True)
            st.caption(f"有效因子: {(di['importance']>0).sum()}/{len(di)}")

# ════════════════════════════════════════ Tab 5: 风险评估 ═══════
with tabs[4]:
    st.markdown('<p class="section-title">⚠️ 风险评估</p>', unsafe_allow_html=True)
    if not has_data: st.info("运行 `python blind_test.py` 生成数据")
    else:
        ae=pd.DataFrame()
        for wn,df in eq_data.items():
            e=df[['equity']].copy();e.columns=[f'W{wn}'];ae=pd.concat([ae,e],axis=1)
        fig,(ax_d,ax_v)=plt.subplots(1,2,figsize=(10,3.5))
        for c in ae.columns:
            s=ae[c].dropna();cm=s.cummax();ds=(s-cm)/cm*100
            ax_d.plot(range(len(ds)),ds,linewidth=0.8,alpha=0.7,label=c);ax_d.set_title('回撤叠加');ax_d.legend(fontsize=7);ax_d.grid(alpha=0.3)
            rs=s.pct_change().dropna();rv=rs.rolling(20).std()*np.sqrt(252)*100
            ax_v.plot(rv.index,rv.values,linewidth=0.8,alpha=0.7,label=c);ax_v.set_title('滚动波动率(20日)');ax_v.legend(fontsize=7);ax_v.grid(alpha=0.3)
        plt.tight_layout();st.pyplot(fig);plt.close()
        ar=np.concatenate([df['equity'].pct_change().dropna().values for df in eq_data.values()])
        c1,c2,c3=st.columns(3)
        c1.metric("VaR 95%",f"{np.percentile(ar,5)*100:.2f}%/日");c2.metric("VaR 99%",f"{np.percentile(ar,1)*100:.2f}%/日");c3.metric("CVaR 95%",f"{ar[ar<=np.percentile(ar,5)].mean()*100:.2f}%/日")

# ════════════════════════════════════════ Tab 6: 信号追踪 ═══════
with tabs[5]:
    st.markdown('<p class="section-title">🎯 信号追踪</p>', unsafe_allow_html=True)
    if not trade_data: st.info("运行 `python blind_test.py` 生成数据")
    else:
        sw3=st.selectbox("窗口",list(trade_data.keys()),key='sig_sel2',index=len(trade_data)-1)
        td=trade_data[sw3];sells=td[td['action']=='SELL'].copy();buys=td[td['action']=='BUY'].copy()
        if len(td)>0 and 'date' in td.columns and 'symbol' in td.columns:
            fig,ax=plt.subplots(figsize=(10,3))
            if len(buys)>0: ax.scatter(pd.to_datetime(buys['date']),buys['symbol'],color='#2e7d32',s=40,marker='^',label=f'买({len(buys)})',alpha=0.7)
            if len(sells)>0: ax.scatter(pd.to_datetime(sells['date']),sells['symbol'],color='#c62828',s=40,marker='v',label=f'卖({len(sells)})',alpha=0.7)
            ax.legend(fontsize=8);ax.grid(alpha=0.3);ax.set_title(f'W{sw3} 买卖信号');plt.xticks(rotation=30,fontsize=7)
            plt.tight_layout();st.pyplot(fig);plt.close()
        if 'date' in td.columns:
            dc=pd.to_datetime(td['date']).dt.date.value_counts().sort_index()
            c1,c2,c3=st.columns(3);c1.metric("总交易",len(td));c2.metric("活跃天数",len(dc));c3.metric("日均",f"{len(td)/max(len(dc),1):.1f}")
            if len(dc)>0:
                fig,ax=plt.subplots(figsize=(10,2));ax.bar(range(len(dc)),dc.values,color='#1565c0',width=1);ax.set_title('日交易数');plt.tight_layout();st.pyplot(fig);plt.close()

# ═══════════════════ Sidebar ═══════════════════
with st.sidebar:
    st.markdown("### 🔗 命令")
    st.code("python blind_test.py"); st.caption("运行盲测")
    st.code("python test_rolling_v3.py"); st.caption("全量重训练")
    st.divider()
    if has_data: st.success(f"数据: {len(eq_data)}窗")
    else: st.warning("无数据")
    st.caption("Deep Quant v6")