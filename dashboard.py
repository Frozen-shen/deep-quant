"""
量化交易看板 — Streamlit 仪表盘 v6

启动: streamlit run dashboard.py --server.headless true
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Deep Quant · 量化看板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ════════════════════════════════════════
#  自定义样式
# ════════════════════════════════════════
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0; }
    .sub-header  { font-size: 1.0rem; color: #888; margin-top: -0.5rem; margin-bottom: 1.5rem; }
    .metric-card { 
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
        border: 1px solid #dee2e6;
    }
    .metric-card.green  { background: linear-gradient(135deg, #e8f5e9, #c8e6c9); border-color: #a5d6a7; }
    .metric-card.blue   { background: linear-gradient(135deg, #e3f2fd, #bbdefb); border-color: #90caf9; }
    .metric-card.orange { background: linear-gradient(135deg, #fff3e0, #ffe0b2); border-color: #ffcc80; }
    .metric-card.red    { background: linear-gradient(135deg, #fce4ec, #f8bbd0); border-color: #f48fb1; }
    .metric-value  { font-size: 2.4rem; font-weight: 700; margin: 0; }
    .metric-label  { font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
    .section-title { font-size: 1.2rem; font-weight: 600; margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #e0e0e0; }
    .tag { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 10px; font-size: 0.75rem; font-weight: 600; }
    .tag-a { background: #e8f5e9; color: #2e7d32; }
    .tag-b { background: #e3f2fd; color: #1565c0; }
    .tag-c { background: #fff3e0; color: #e65100; }
    .tag-d { background: #fce4ec; color: #c62828; }
    .footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e0e0e0; color: #999; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════
#  Header
# ════════════════════════════════════════
st.markdown('<p class="main-header">📊 Deep Quant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">LightGBM Lambdarank · 78只CSI300 · T+20日趋势跟随 · v6</p>', unsafe_allow_html=True)

# ════════════════════════════════════════
#  Tab 导航
# ════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🎯 模型评测", "📈 开发历程", "⚙️ 架构配置"])

# ════════════════════════════════════════
#  Tab 1: 模型评测
# ════════════════════════════════════════
with tab1:
    # ── 评级卡片行 ──
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown("""
        <div class="metric-card green">
            <p class="metric-value" style="color:#2e7d32;">A−</p>
            <p class="metric-label">综合评级</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card blue">
            <p class="metric-value" style="color:#1565c0;">84.9</p>
            <p class="metric-label">分数 / 100</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card green">
            <p class="metric-value" style="color:#2e7d32;">~102%</p>
            <p class="metric-label">年化收益</p>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card blue">
            <p class="metric-value" style="color:#1565c0;">1.37</p>
            <p class="metric-label">Sharpe Ratio</p>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown("""
        <div class="metric-card orange">
            <p class="metric-value" style="color:#e65100;">59%</p>
            <p class="metric-label">胜率</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 盲测窗口 ──
    st.markdown('<p class="section-title">🔒 盲测窗口 (W7–W8) · 参数冻结 · 仅运行一次</p>', unsafe_allow_html=True)

    bw1, bw2 = st.columns(2)
    with bw1:
        st.markdown("""
        <div style="background:#f8f9fa;border-radius:12px;padding:1.2rem;border:1px solid #dee2e6;">
            <p style="font-size:1.1rem;font-weight:600;">W7 · <span style="color:#888;">2025-07 → 2026-04</span></p>
            <table style="width:100%;font-size:0.9rem;">
                <tr><td style="color:#888;">策略收益</td><td style="text-align:right;font-weight:700;color:#2e7d32;">+83.5%</td></tr>
                <tr><td style="color:#888;">基准收益</td><td style="text-align:right;">+48.1%</td></tr>
                <tr><td style="color:#888;">超额收益</td><td style="text-align:right;font-weight:700;color:#2e7d32;">+35.5%</td></tr>
                <tr><td style="color:#888;">交易笔数</td><td style="text-align:right;">113 笔</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    with bw2:
        st.markdown("""
        <div style="background:#f8f9fa;border-radius:12px;padding:1.2rem;border:1px solid #dee2e6;">
            <p style="font-size:1.1rem;font-weight:600;">W8 · <span style="color:#888;">2026-04 → 2026-07</span></p>
            <table style="width:100%;font-size:0.9rem;">
                <tr><td style="color:#888;">策略收益</td><td style="text-align:right;font-weight:700;color:#2e7d32;">+7.4%</td></tr>
                <tr><td style="color:#888;">基准收益</td><td style="text-align:right;">+10.5%</td></tr>
                <tr><td style="color:#888;">超额收益</td><td style="text-align:right;font-weight:700;color:#c62828;">−3.1%</td></tr>
                <tr><td style="color:#888;">交易笔数</td><td style="text-align:right;">26 笔</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # ── 17 指标评分表 ──
    st.markdown('<p class="section-title">📋 盲测 17 指标明细</p>', unsafe_allow_html=True)

    grades_data = [
        ("年化收益",     "A", "74.0%",    "盲测窗口年化收益率"),
        ("年化超额",     "A", "17.0%",    "相对基准的超额"),
        ("Sharpe Ratio","A", "1.37",     "风险调整收益, >1.0 即优秀"),
        ("最大回撤",     "A", "−26.4%",   "盲测期内最大跌幅"),
        ("Calmar",      "A", "2.72",     "收益回撤比"),
        ("盈亏因子",     "B", "1.67",     "总盈利 ÷ 总亏损"),
        ("胜率",        "A", "59.0%",    "盈利交易占比"),
        ("期望收益",     "A", "+1.75%",   "每笔交易平均收益"),
        ("最差窗口",     "A", "+7.4%",    "盲测从未亏过"),
        ("盈亏比",       "D", "1.16",     "盈利仅比亏损大 16%"),
        ("正窗口占比",   "D", "50%",      "仅 2 窗, 统计不足"),
        ("Rolling Sharpe","D","0.04",    "短期有低迷期"),
        ("SQN",         "C", "1.85",     "系统质量中等"),
        ("上涨捕获率",   "A", "1.52",     "牛市跟涨能力强"),
        ("溃疡指数",     "A", "3.97",     "回撤恢复快"),
        ("DSR",         "A", "1.00",     "多重测试校正后仍显著"),
        ("偏度",         "A", "0.286",    "正偏, 大赢小亏"),
    ]

    # 渲染为卡片网格 (每行 4 个)
    for i in range(0, len(grades_data), 4):
        cols = st.columns(4)
        for j, (name, grade, val, desc) in enumerate(grades_data[i:i+4]):
            tag_class = f"tag-{grade.lower()}"
            bg = {"A":"#e8f5e9","B":"#e3f2fd","C":"#fff3e0","D":"#fce4ec"}.get(grade,"#f8f9fa")
            with cols[j]:
                st.markdown(f"""
                <div style="background:{bg};border-radius:10px;padding:0.8rem 1rem;margin:0.3rem 0;border:1px solid #e0e0e0;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-size:0.85rem;font-weight:500;">{name}</span>
                        <span class="tag {tag_class}">{grade}</span>
                    </div>
                    <div style="font-size:1.3rem;font-weight:700;margin:0.3rem 0;">{val}</div>
                    <div style="font-size:0.7rem;color:#888;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── 结论 ──
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.markdown("""
        <div style="background:#e8f5e9;border-radius:12px;padding:1.2rem;border:1px solid #a5d6a7;">
            <p style="font-weight:700;color:#2e7d32;margin:0;">✅ 优势</p>
            <ul style="margin:0.5rem 0 0 1.2rem;font-size:0.9rem;color:#333;">
                <li>盲测绝对收益为正, 年化约 102%</li>
                <li>Sharpe 1.37, 风险调整后优秀</li>
                <li>59% 胜率, 最差窗口 +7.4%</li>
                <li>12 / 17 指标 A/B 水平</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("""
        <div style="background:#fff3e0;border-radius:12px;padding:1.2rem;border:1px solid #ffcc80;">
            <p style="font-weight:700;color:#e65100;margin:0;">⚠️ 短板</p>
            <ul style="margin:0.5rem 0 0 1.2rem;font-size:0.9rem;color:#333;">
                <li>仅有 2 个盲测窗口, 统计意义不足</li>
                <li>盈亏比 1.16, 赢得不够多</li>
                <li>需积累 6+ 盲测窗口后方可更有信心</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown(f'<p class="footer">Deep Quant v6 · 可信度: 中 (251天 / 2窗) · 评估时间: 2026-07-23</p>', unsafe_allow_html=True)

# ════════════════════════════════════════
#  Tab 2: 开发历程
# ════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">🔄 版本演进</p>', unsafe_allow_html=True)

    ver_data = [
        {"version": "v1–v3", "milestone": "Lambdarank + 前瞻标签", "excess": "+1~13%", "date": "2026-07 早期"},
        {"version": "v4",     "milestone": "Bug 修复 + Phase 1–3 优化", "excess": "+13.7%", "date": "2026-07-23"},
        {"version": "v5",     "milestone": "★ 20 日标签 + 78 股 + 1 年训练", "excess": "+15.9%", "date": "2026-07-23"},
        {"version": "v6",     "milestone": "★ 评估体系重构 (dev / blind 分离)", "excess": "+16.2% 盲测", "date": "当前"},
    ]

    for v in ver_data:
        is_current = v["version"] == "v6"
        bg = "#e8f5e9" if is_current else "#f8f9fa"
        border = "#a5d6a7" if is_current else "#dee2e6"
        badge = '<span style="background:#2e7d32;color:white;padding:2px 8px;border-radius:8px;font-size:0.7rem;margin-left:8px;">当前</span>' if is_current else ""
        st.markdown(f"""
        <div style="background:{bg};border-radius:10px;padding:1rem 1.2rem;margin:0.5rem 0;border:1px solid {border};display:flex;align-items:center;gap:1rem;">
            <div style="font-weight:700;font-size:1rem;min-width:60px;">{v["version"]}{badge}</div>
            <div style="flex:1;font-size:0.9rem;">{v["milestone"]}</div>
            <div style="font-weight:700;color:#2e7d32;min-width:100px;text-align:right;">超额 {v["excess"]}</div>
            <div style="color:#999;font-size:0.8rem;min-width:90px;text-align:right;">{v["date"]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<p class="section-title">🔧 核心技术决策</p>', unsafe_allow_html=True)

    decisions = [
        ("🎯", "5日 → 20日标签", "根本性修复。5 日标签导致模型学成均值回归 (熊市好牛市差)。20 日标签让模型转为趋势跟随。"),
        ("📦", "18只 → 78只股票", "截面排名的统计意义取决于股票数量。18 只跨 6–7 行业使排名退化为行业选择。"),
        ("⏱️", "3年 → 1年训练窗口", "市场结构快速变化。1 年训练 + 0.5 年半衰期确保模型学到最近的市场规律。"),
        ("🔒", "dev / blind 分离", "最关键的评估改进。开发集调参, 盲测集锁定参数只跑一次。"),
    ]

    for icon, title, desc in decisions:
        st.markdown(f"""
        <div style="display:flex;gap:1rem;align-items:flex-start;margin:0.8rem 0;">
            <div style="font-size:1.5rem;">{icon}</div>
            <div>
                <div style="font-weight:600;">{title}</div>
                <div style="color:#666;font-size:0.9rem;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════
#  Tab 3: 架构配置
# ════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">📐 模型架构</p>', unsafe_allow_html=True)

    arch_cols = st.columns(3)
    arch_items = [
        ("🧠", "LightGBM Lambdarank", "depth=5 · 200轮 · L1=0.8\n截面排序优化, 非回归预测"),
        ("📊", "39 个技术因子", "趋势 · 波动 · 动量\n量价 · K线 · 风险调整"),
        ("🎯", "Top-5 选股", "持有 ≥ 10天 · 成本门槛 8%\n等权重分配 · 每日再平衡"),
    ]
    for col, (icon, title, desc) in zip(arch_cols, arch_items):
        col.markdown(f"""
        <div style="background:#f8f9fa;border-radius:12px;padding:1.2rem;text-align:center;border:1px solid #dee2e6;">
            <div style="font-size:2rem;">{icon}</div>
            <div style="font-weight:600;margin:0.5rem 0;">{title}</div>
            <div style="color:#666;font-size:0.85rem;white-space:pre-line;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<p class="section-title">📋 完整参数表</p>', unsafe_allow_html=True)
    param_data = pd.DataFrame([
        ("股票池", "78 只", "CSI 300 成分股, 自动缓存"),
        ("训练标签", "T+20 日", "前瞻收益, 无数据泄露"),
        ("训练窗口", "12 个月", "滚动重训练, 每 9 个月推进"),
        ("时间衰减", "半衰期 0.5 年", "近期样本权重更高"),
        ("因子标准化", "截面 z-score", "每日同股票池内标准化"),
        ("持仓数量", "5 只", "等权重, 单票 20%"),
        ("持有期", "≥ 10 天", "卖出缓冲 2 格"),
        ("入场上限", "每天 ≤ 3 只", "避免单日大换仓"),
        ("买入费率", "0.026%", "佣金 2.5bp + 过户费 0.1bp"),
        ("卖出费率", "0.076%", "佣金 2.5bp + 印花税 5bp + 过户费 0.1bp"),
        ("LightGBM", "200 轮 / depth 5", "L1=0.8 / leaves ≥ 60"),
        ("冒烟测试", "dev 6 窗 + blind 2 窗", "盲测可信度: 中"),
    ], columns=["参数", "值", "说明"])
    st.dataframe(param_data, use_container_width=True, hide_index=True)

# ════════════════════════════════════════
#  Sidebar (collapsed by default)
# ════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🔗 快捷操作")
    st.code("python blind_test.py", language="bash")
    st.caption("运行盲测 → 生成最新评估")
    st.code("python test_rolling_v3.py", language="bash")
    st.caption("全量滚动重训练 (所有窗口)")
    st.code("python data_cache.py --fetch-index 000300", language="bash")
    st.caption("拉取 CSI 300 成分股数据")
    st.divider()
    st.caption("Deep Quant v6 · LightGBM Lambdarank")
