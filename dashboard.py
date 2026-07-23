"""
Deep Quant 看板 v11 — 完整重写, 稳健数据加载, 8个Tab
"""
import os, sys, glob, json, traceback
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR); sys.path.insert(0, BASE_DIR)

import streamlit as st, pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib; matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="Deep Quant", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# ═══════════════ CSS ═══════════════
st.markdown("""<style>
.main-header { font-size:2rem; font-weight:700; } .sub-header { color:#888; margin-bottom:1rem; }
.metric-card { background:linear-gradient(135deg,#f8f9fa,#e9ecef); border-radius:12px; padding:1rem; text-align:center; border:1px solid #dee2e6; }
.metric-value { font-size:2rem; font-weight:700; margin:0; } .metric-label { font-size:0.7rem; color:#666; text-transform:uppercase; }
.section-title { font-size:1.1rem; font-weight:600; margin:1.5rem 0 0.8rem; border-bottom:2px solid #e0e0e0; padding-bottom:0.3rem; }
.tag { display:inline-block; padding:0.1rem 0.5rem; border-radius:10px; font-size:0.7rem; font-weight:600; }
.tag-a{background:#e8f5e9;color:#2e7d32} .tag-b{background:#e3f2fd;color:#1565c0} .tag-c{background:#fff3e0;color:#e65100} .tag-d{background:#fce4ec;color:#c62828}
.err-box { background:#fff3e0; border:1px solid #ffcc80; border-radius:8px; padding:0.5rem 1rem; font-size:0.8rem; color:#e65100; margin:0.5rem 0; }
</style>""", unsafe_allow_html=True)

# ═══════════════ Data (with error tracking) ═══════════════
_data_errors = []

def _safe_load_json(path, default=None):
    try:
        with open(path, encoding='utf-8') as f: return json.load(f)
    except Exception as e:
        _data_errors.append(f"加载{os.path.basename(path)}失败: {e}")
        return default or {}

def _safe_load_csv(pattern):
    files = sorted(glob.glob(os.path.join(BASE_DIR, 'test_results', pattern)))
    data = {}
    for f in files:
        try:
            wn = os.path.basename(f).split('_w')[1].replace('.csv','')
            df = pd.read_csv(f)
            # 补齐代码列
            if 'symbol' in df.columns:
                df['symbol'] = df['symbol'].astype(str).str.zfill(6)
                df['symbol_display'] = df['symbol'].astype(str)
            data[wn] = df
        except Exception as e:
            _data_errors.append(f"加载{os.path.basename(f)}失败: {e}")
    return data

# 加载所有数据
_name_map = _safe_load_json(os.path.join(BASE_DIR, 'data_cache', 'stock_names.json'))
_sector_map = _safe_load_json(os.path.join(BASE_DIR, 'data_cache', 'a_sectors.json'))

eq_data_raw = _safe_load_csv('equity_w*.csv')
trade_data = _safe_load_csv('trades_w*.csv')
imp_data = _safe_load_csv('importance_w*.csv')

# 处理权益数据 (date设为index)
eq_data = {}
for k, df in eq_data_raw.items():
    try:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']); eq_data[k] = df.set_index('date')
        else:
            eq_data[k] = df
    except Exception as e:
        _data_errors.append(f"处理equity_w{k}失败: {e}")

has_data = len(eq_data) > 0

# 辅助: 获取名称
def stock_name(sym):
    s = str(sym).zfill(6)
    return _name_map.get(s, '')

def stock_sector(sym):
    s = str(sym).zfill(6)
    return _sector_map.get(s, '')

# 从缓存加载OHLCV
from data_cache import load as load_ohlcv

# ═══════════════ Term tables ═══════════════
FACTOR_DICT = {
    "volatility_30d":"波动率(30日)","volatility_20d":"波动率(20日)","volatility_10d":"波动率(10日)",
    "amplitude_5d":"振幅(5日)","turnover_trend":"换手率趋势","turnover_vol":"换手率波动",
    "vol_regime":"波动率状态","vol_compress":"波动率压缩","boll_width":"布林带宽度",
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
METRIC_HELP = {
    "年化收益":"策略年化收益率。A股机构平均5-15%。","年化超额":"相对基准的超额年化。>10%优秀。",
    "Sharpe":"风险调整收益。>1.0优秀。","最大回撤":"最大累计跌幅。<-30%需警惕。",
    "Calmar":"年化收益÷最大回撤。>1.0不错。","盈亏因子":"总盈利÷总亏损。>1.5稳定。",
    "胜率":"盈利交易占比。>50%好。","期望收益":"每笔平均盈亏%。>0.5%正向。",
    "最差窗口":"所有窗口最差总收益。正值=未亏过。","盈亏比":"平均盈利÷平均亏损。>2.0优秀。",
    "正窗口率":"正超额窗口占比。>75%稳定。","Roll Sharpe":"滚动6月夏普最低值。>0.5稳健。",
    "SQN":"系统质量数。>2.0优秀。","上涨捕获":"牛市相对表现。>1.0=跑赢。",
    "溃疡指数":"UPR回撤综合惩罚。>2.0优秀。","DSR":"Deflated Sharpe。>0.5显著。",
    "偏度":"收益分布不对称。正偏=大赢小亏。",
}

# ═══════════════ Header ═══════════════
st.markdown('<p class="main-header">📊 Deep Quant · v6 量化看板</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">LightGBM Lambdarank · 78只 CSI300 · T+20日趋势跟随</p>', unsafe_allow_html=True)

if _data_errors:
    with st.expander(f"⚠️ {len(_data_errors)}个数据加载问题", expanded=False):
        for e in _data_errors: st.markdown(f'- {e}')
if not has_data:
    st.warning("无数据。运行 `python blind_test.py` 生成盲测数据。")

# ═══════════════ Tabs ═══════════════
tabs = st.tabs(["🎯 模型评测", "📈 权益分析", "🔍 收益归因", "📋 交易分析",
                "🧬 因子分析", "⚠️ 风险评估", "🎯 信号追踪", "📦 股票池"])

# ========== Tab 0: 模型评测 ==========
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
            bw[i].markdown(f'<div style="background:#f8f9fa;border-radius:10px;padding:1rem;text-align:center;border:1px solid #dee2e6;"><div style="font-weight:700;">W{wn}</div><div style="font-size:1.3rem;font-weight:700;color:{cl};">{r:+.1f}%</div><div style="font-size:0.8rem;">超额{r-br:+.1f}% | 基准{br:+.1f}%</div></div>', unsafe_allow_html=True)
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
    with st.expander("📖 指标解释"):
        for k,v in METRIC_HELP.items(): st.caption(f"**{k}**: {v}")

# ========== Tab 1: 权益分析 ==========
with tabs[1]:
    if not has_data: st.info("运行 `python blind_test.py`")
    else:
        aw=list(eq_data.keys()); sel=st.selectbox("窗口",aw,key='eq_sel',index=len(aw)-1); df=eq_data[sel]
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,6),gridspec_kw={'height_ratios':[2,1]})
        ax1.plot(df.index,df['equity'],color='#2e7d32',linewidth=1.5,label='Strategy'); ax1.plot(df.index,df['benchmark'],color='#999',linewidth=1,label='Benchmark')
        ax1.fill_between(df.index,df['equity'],df['benchmark'],where=df['equity']>=df['benchmark'],color='#e8f5e9',alpha=0.3)
        ax1.fill_between(df.index,df['equity'],df['benchmark'],where=df['equity']<df['benchmark'],color='#fce4ec',alpha=0.3)
        ax1.legend(fontsize=8);ax1.set_ylabel('Equity');ax1.grid(alpha=0.3);ax1.set_title(f'W{sel} Equity Curve',fontweight='bold')
        eq=df['equity'];cm=eq.cummax();dd=(eq-cm)/cm*100
        ax2.fill_between(df.index,0,dd,color='#c62828',alpha=0.3);ax2.plot(df.index,dd,color='#c62828',linewidth=0.8)
        ax2.set_ylabel('%');ax2.grid(alpha=0.3);ax2.set_title('Drawdown',fontweight='bold')
        plt.tight_layout();st.pyplot(fig);plt.close()
        eqa=eq.values;tr=(eqa[-1]/eqa[0]-1)*100;md=dd.min();dr=np.diff(eqa)/eqa[:-1]
        sh=np.mean(dr)/np.std(dr)*np.sqrt(252) if np.std(dr)>0 else 0
        c1,c2,c3,c4=st.columns(4);c1.metric("Total Return",f"{tr:+.1f}%");c2.metric("Max DD",f"{md:.1f}%")
        c3.metric("Ann.Vol",f"{np.std(dr)*np.sqrt(252)*100:.1f}%");c4.metric("Sharpe",f"{sh:.2f}")

# ========== Tab 2: 收益归因 ==========
with tabs[2]:
    st.markdown('<p class="section-title">🔍 收益归因</p>', unsafe_allow_html=True)
    if not has_data or not trade_data: st.info("运行 `python blind_test.py`")
    else:
        sa=st.selectbox("窗口",list(eq_data.keys()),key='attr_sel',index=len(eq_data)-1)
        dfe=eq_data[sa]; td=trade_data[sa]; sells=td[td['action']=='SELL'].copy()
        eqr=(dfe['equity'].values[-1]/dfe['equity'].values[0]-1)*100
        bmr=(dfe['benchmark'].values[-1]/dfe['benchmark'].values[0]-1)*100; al=eqr-bmr
        c1,c2,c3=st.columns(3);c1.metric("Strategy",f"{eqr:+.1f}%");c2.metric("Beta (market)",f"{bmr:+.1f}%");c3.metric("Alpha (skill)",f"{al:+.1f}%")
        if eqr>0:
            bp=max(0,bmr/eqr*100);ap=max(0,al/eqr*100)
            fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,3.5))
            ax1.pie([bp,ap],labels=['Market','Alpha'],autopct='%1.0f%%',colors=['#90caf9','#2e7d32'],startangle=90);ax1.set_title('Return Source')
            if len(sells)>0 and 'pnl' in sells.columns:
                sp=sells.groupby('symbol')['pnl'].sum().sort_values()
                tw=sp.tail(5);tl=sp.head(5);ac=pd.concat([tl,tw])
                colors=['#c62828' if x<0 else '#2e7d32' for x in ac.values]
                labels=[f"{s}\n{stock_name(s)}" for s in ac.index]
                ax2.barh(range(len(ac)),ac.values,color=colors,height=0.7);ax2.set_yticks(range(len(ac)));ax2.set_yticklabels(labels,fontsize=7);ax2.set_title('Top/Bottom 5 Stocks');ax2.axvline(0,color='#999',linewidth=0.5)
            plt.tight_layout();st.pyplot(fig);plt.close()
        if len(sells)>0:
            sc=sells['symbol'].value_counts()
            c1,c2,c3=st.columns(3);c1.metric("Traded stocks",len(sc));c2.metric("Top3 conc.",f"{sc.head(3).sum()/sc.sum()*100:.0f}%");c3.metric("Coverage",f"{len(sc)/78*100:.0f}% of 78")

# ========== Tab 3: 交易分析 ==========
with tabs[3]:
    st.markdown('<p class="section-title">📋 交易明细</p>', unsafe_allow_html=True)
    if not trade_data: st.info("运行 `python blind_test.py`")
    else:
        sw=st.selectbox("窗口",list(trade_data.keys()),key='trd_sel',index=len(trade_data)-1); td=trade_data[sw]
        sells=td[td['action']=='SELL']; buys=td[td['action']=='BUY']
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Trades",f"{len(td)}",delta=f"Buy{len(buys)} Sell{len(sells)}")
        c2.metric("Commission",f"¥{td['commission'].sum():,.0f}" if 'commission' in td.columns else "N/A")
        if len(sells)>0 and 'pnl' in sells.columns:
            c3.metric("Win rate",f"{(sells['pnl']>0).mean()*100:.0f}%");c4.metric("Total PnL",f"¥{sells['pnl'].sum():,.0f}")
        if len(sells)>0 and 'pnl' in sells.columns:
            fig,(a1,a2)=plt.subplots(1,2,figsize=(10,3.5))
            p=sells['pnl'].values;cs=np.cumsum(p)
            a1.bar(range(len(p)),p,color=['#2e7d32' if x>=0 else '#c62828' for x in p],width=1)
            a1.plot(range(len(p)),cs,color='#1565c0',linewidth=1.5,label='Cum.');a1.axhline(0,color='#999',linewidth=0.5);a1.legend(fontsize=8)
            a2.hist(p,bins=30,color='#1565c0',alpha=0.7,edgecolor='white');a2.axvline(0,color='#999',linewidth=0.5)
            a2.axvline(np.mean(p),color='#c62828',linewidth=1.5,linestyle='--',label=f'Mean ¥{np.mean(p):+.0f}');a2.legend(fontsize=8)
            plt.tight_layout();st.pyplot(fig);plt.close()
        # 交易表 - 加名称列
        show_cols=['date','symbol','action','price','qty','pnl','commission']
        show_cols=[c for c in show_cols if c in td.columns]
        display_df = td[show_cols].tail(30).copy()
        if 'symbol' in display_df.columns:
            display_df.insert(1,'name',display_df['symbol'].apply(stock_name))
            display_df.insert(2,'sector',display_df['symbol'].apply(stock_sector))
        st.dataframe(display_df,use_container_width=True,hide_index=True)

    # ── 个股K线+买卖点 ──
    st.markdown("---")
    st.markdown('<p class="section-title">📈 个股K线 + 买卖点</p>', unsafe_allow_html=True)
    if not trade_data:
        st.info("运行 `python blind_test.py`")
    else:
        sw2=st.selectbox("窗口",list(trade_data.keys()),key='stockw_sel',index=len(trade_data)-1)
        td2=trade_data[sw2]
        syms=sorted(td2['symbol'].unique())
        labels=[f"{s} {stock_name(s)}" for s in syms]
        sel_idx=st.selectbox("股票",range(len(syms)),format_func=lambda i:labels[i],key='stock_pick')
        if sel_idx is not None:
            sym=syms[sel_idx]; sym_padded=str(sym).zfill(6)
            sdf=td2[td2['symbol']==sym]; bs=sdf[sdf['action']=='BUY']; ss=sdf[sdf['action']=='SELL']
            ohlcv=load_ohlcv(sym_padded)
            if ohlcv is None:
                st.error(f"未找到 {sym_padded} 的日线数据。请先运行 data_cache.py --fetch")
            elif len(ohlcv)>0:
                ohlcv['date']=pd.to_datetime(ohlcv['date'])
                ws=pd.Timestamp(eq_data[sw2].index[0]); we=pd.Timestamp(eq_data[sw2].index[-1])
                mask=(ohlcv['date']>=ws)&(ohlcv['date']<=we); ohlcv_w=ohlcv[mask].set_index('date').sort_index()
                if len(ohlcv_w)>0:
                    fig,ax=plt.subplots(figsize=(12,5))
                    ax.plot(ohlcv_w.index,ohlcv_w['close'],color='#1565c0',linewidth=1.2,label=sym_padded)
                    ax.fill_between(ohlcv_w.index,ohlcv_w['low'],ohlcv_w['high'],color='#1565c0',alpha=0.15)
                    if len(bs)>0:
                        bd=pd.to_datetime(bs['date']); bp=[ohlcv_w.loc[d,'close'] for d in bd if d in ohlcv_w.index]; bdf=[d for d in bd if d in ohlcv_w.index]
                        if bdf: ax.scatter(bdf,bp,color='#2e7d32',s=80,marker='^',zorder=5,edgecolors='white',linewidth=1,label=f'Buy({len(bdf)})')
                    if len(ss)>0:
                        sd_=pd.to_datetime(ss['date']); sp=[ohlcv_w.loc[d,'close'] for d in sd_ if d in ohlcv_w.index]; sdf_=[d for d in sd_ if d in ohlcv_w.index]
                        if sdf_: ax.scatter(sdf_,sp,color='#c62828',s=80,marker='v',zorder=5,edgecolors='white',linewidth=1,label=f'Sell({len(sdf_)})')
                    ax.legend(fontsize=8);ax.grid(alpha=0.3);ax.set_title(f'{sym_padded} {stock_name(sym)} — {stock_sector(sym)}',fontweight='bold');ax.set_ylabel('Price')
                    plt.xticks(rotation=30,fontsize=7);plt.tight_layout();st.pyplot(fig);plt.close()
                    c1,c2,c3=st.columns(3);c1.metric("Trades",f"Buy{len(bs)} Sell{len(ss)}")
                    if len(ss)>0 and 'pnl' in ss.columns: c2.metric("Total PnL",f"¥{ss['pnl'].sum():,.0f}");c3.metric("Win%",f"{(ss['pnl']>0).mean()*100:.0f}%")

# ========== Tab 4: 因子分析 ==========
with tabs[4]:
    st.markdown('<p class="section-title">🧬 因子重要性</p>', unsafe_allow_html=True)
    if not imp_data: st.info("运行 `python blind_test.py`")
    else:
        si=st.selectbox("窗口",list(imp_data.keys()),key='imp_sel',index=len(imp_data)-1); di=imp_data[si]
        if len(di)>0:
            t20=di.head(20)
            fig,ax=plt.subplots(figsize=(10,4))
            labels=[f"{FACTOR_DICT.get(f,f)} ({f})" for f in t20['factor'].values[::-1]]
            ax.barh(range(len(labels)),t20['importance'].values[::-1],color='#1565c0',height=0.7)
            ax.set_yticks(range(len(labels)));ax.set_yticklabels(labels,fontsize=7);ax.set_title(f'W{si} Factor Importance Top-20');ax.grid(axis='x',alpha=0.3)
            plt.tight_layout();st.pyplot(fig);plt.close()
    with st.expander("📖 因子术语表"):
        rows=[]; ti=di['importance'].sum() if len(di)>0 else 1
        for f in sorted(FACTOR_DICT.keys()):
            r=di[di['factor']==f]; iv=r['importance'].values[0] if len(r)>0 else 0
            ip=f"{iv/ti*100:.1f}%" if ti>0 else "0%"; bl=int(iv/ti*30) if ti>0 else 0
            rows.append({"中文名":FACTOR_DICT.get(f,f),"代码":f,"重要性":f"{iv:.0f}","占比":ip,"强度":"█"*bl+"░"*(30-bl) if bl>0 else "—"})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

# ========== Tab 5: 风险评估 ==========
with tabs[5]:
    if not has_data: st.info("运行 `python blind_test.py`")
    else:
        ae=pd.DataFrame()
        for wn,df in eq_data.items(): e=df[['equity']].copy();e.columns=[f'W{wn}'];ae=pd.concat([ae,e],axis=1)
        fig,(ax_d,ax_v)=plt.subplots(1,2,figsize=(10,3))
        for c in ae.columns:
            s=ae[c].dropna();cm=s.cummax();ds=(s-cm)/cm*100;ax_d.plot(range(len(ds)),ds,linewidth=0.8,alpha=0.7,label=c);ax_d.grid(alpha=0.3)
            rs=s.pct_change().dropna();rv=rs.rolling(20).std()*np.sqrt(252)*100;ax_v.plot(rv.index,rv.values,linewidth=0.8,alpha=0.7,label=c);ax_v.grid(alpha=0.3)
        ax_d.set_title('Drawdown overlay');ax_d.legend(fontsize=7);ax_v.set_title('Rolling vol (20d)');ax_v.legend(fontsize=7)
        plt.tight_layout();st.pyplot(fig);plt.close()
        ar=np.concatenate([df['equity'].pct_change().dropna().values for df in eq_data.values()])
        c1,c2,c3=st.columns(3);c1.metric("VaR 95%",f"{np.percentile(ar,5)*100:.2f}%/day");c2.metric("VaR 99%",f"{np.percentile(ar,1)*100:.2f}%/day");c3.metric("CVaR 95%",f"{ar[ar<=np.percentile(ar,5)].mean()*100:.2f}%/day")

# ========== Tab 6: 信号追踪 ==========
with tabs[6]:
    if not trade_data: st.info("运行 `python blind_test.py`")
    else:
        sw3=st.selectbox("窗口",list(trade_data.keys()),key='sig_sel',index=len(trade_data)-1); td=trade_data[sw3]
        sells=td[td['action']=='SELL']; buys=td[td['action']=='BUY']
        if len(td)>0 and 'date' in td.columns:
            fig,ax=plt.subplots(figsize=(10,3))
            if len(buys)>0:ax.scatter(pd.to_datetime(buys['date']),buys['symbol'],color='#2e7d32',s=30,marker='^',label=f'Buy({len(buys)})',alpha=0.6)
            if len(sells)>0:ax.scatter(pd.to_datetime(sells['date']),sells['symbol'],color='#c62828',s=30,marker='v',label=f'Sell({len(sells)})',alpha=0.6)
            ax.legend(fontsize=8);ax.grid(alpha=0.3);plt.xticks(rotation=30,fontsize=7)
            plt.tight_layout();st.pyplot(fig);plt.close()
            dc=pd.to_datetime(td['date']).dt.date.value_counts().sort_index()
            c1,c2,c3=st.columns(3);c1.metric("Total",len(td));c2.metric("Active days",len(dc));c3.metric("Daily avg",f"{len(td)/max(len(dc),1):.1f}")
            if len(dc)>0:
                fig,ax=plt.subplots(figsize=(10,2));ax.bar(range(len(dc)),dc.values,color='#1565c0',width=1);plt.tight_layout();st.pyplot(fig);plt.close()

# ========== Tab 7: 股票池 ==========
with tabs[7]:
    from data_cache import get_cached_symbols
    syms=get_cached_symbols()
    rows=[{"代码":s,"名称":stock_name(s),"板块":stock_sector(s)} for s in sorted(syms)]
    st.metric("股票总数",len(rows))
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

# ═══════════════ Sidebar ═══════════════
with st.sidebar:
    st.markdown("### 🔗 Commands"); st.code("python blind_test.py"); st.caption("Run blind test")
    st.code("python test_rolling_v3.py"); st.caption("Full rolling train")
    if has_data:st.success(f"Data: {len(eq_data)} windows")
    if _data_errors:st.warning(f"{len(_data_errors)} errors")
    st.caption("Deep Quant v11")
