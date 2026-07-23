"""
量化交易看板 — Streamlit 仪表盘 (v6)

启动:
    streamlit run dashboard.py --server.headless true
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="量化交易看板", page_icon="📊", layout="wide")

st.sidebar.title("⚙️ 模型配置")
st.sidebar.markdown("""
**模型**: LightGBM Lambdarank v6
**股票池**: 78只 (CSI 300)
**因子**: 39个
**标签**: T+20日前瞻收益
**训练**: 12月滚动窗口
**持仓**: Top-5, 持有≥10天
**费率**: 买0.026% / 卖0.076%
""")

page = st.sidebar.radio("页面", ["🧪 模型评测", "📊 开发历史", "📈 权益曲线"])

# ════════════════════════════════════════
#  页面 0: 模型评测
# ════════════════════════════════════════
if page == "🧪 模型评测":
    st.title("🧪 量化模型 v6 — 最终评估")

    st.markdown("""
    > **评估原则**: 开发集(W1-W6)用于调参, 仅供参考。盲测集(W7-W8)参数冻结, 只看一次, 是真实水平。
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<h1 style='color:#2e7d32;font-size:80px;text-align:center;'>A-</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#888;'>综合评级</p>", unsafe_allow_html=True)
    with col2:
        st.markdown("<h1 style='color:#1565c0;font-size:48px;text-align:center;'>84.9</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#888;'>分数 / 100</p>", unsafe_allow_html=True)
    with col3:
        st.markdown("<h1 style='color:#2e7d32;font-size:48px;text-align:center;'>✅</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#888;'>合格判定</p>", unsafe_allow_html=True)
    with col4:
        st.markdown("<h1 style='color:#e65100;font-size:48px;text-align:center;'>中</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#888;'>可信度 (251天/2窗)</p>", unsafe_allow_html=True)

    st.divider()

    # ── 盲测窗口 ──
    st.subheader("🔒 盲测窗口 (W7-W8)")
    st.caption("参数冻结, 未曾用于调参。结果即为真实水平。")
    blind_df = pd.DataFrame([
        {"窗口": "W7", "测试期": "2025-07~2026-04", "策略": "+83.5%", "基准": "+48.1%", "超额": "+35.5%", "笔数": 113},
        {"窗口": "W8", "测试期": "2026-04~2026-07", "策略": "+7.4%", "基准": "+10.5%", "超额": "-3.1%", "笔数": 26},
    ])
    st.dataframe(blind_df, use_container_width=True, hide_index=True)
    st.metric("年化策略收益 (时间加权)", "~102%", border=True)

    # ── 各维度得分 ──
    st.subheader("📋 盲测 17 指标明细 (12/17 为 A/B)")
    metrics_data = pd.DataFrame([
        {"指标": "年化收益", "评级": "A", "数值": "74.0%", "说明": "盲测窗口年化收益率"},
        {"指标": "年化超额", "评级": "A", "数值": "17.0%", "说明": "相对基准的超额"},
        {"指标": "Sharpe", "评级": "A", "数值": "1.37", "说明": "风险调整收益, >1.0即优秀"},
        {"指标": "最大回撤", "评级": "A", "数值": "-26.4%", "说明": "盲测期内最大跌幅"},
        {"指标": "Calmar", "评级": "A", "数值": "2.72", "说明": "收益/回撤比"},
        {"指标": "盈亏因子", "评级": "B", "数值": "1.67", "说明": "总盈利/总亏损"},
        {"指标": "胜率", "评级": "A", "数值": "59.0%", "说明": "盈利交易占比"},
        {"指标": "期望收益", "评级": "A", "数值": "+1.75%", "说明": "每笔交易平均收益"},
        {"指标": "最差窗口", "评级": "A", "数值": "+7.4%", "说明": "盲测从未亏过"},
        {"指标": "盈亏比", "评级": "D", "数值": "1.16", "说明": "⚠️ 盈利仅比亏损大16%"},
        {"指标": "正窗口占比", "评级": "D", "数值": "50%", "说明": "⚠️ 仅2窗, 统计不足"},
        {"指标": "Rolling Sharpe", "评级": "D", "数值": "0.04", "说明": "⚠️ 短期有低迷期"},
        {"指标": "SQN", "评级": "C", "数值": "1.85", "说明": "系统质量"},
        {"指标": "上涨捕获率", "评级": "A", "数值": "1.52", "说明": "牛市跟涨能力强"},
        {"指标": "溃疡指数", "评级": "A", "数值": "3.97", "说明": "回撤恢复快"},
        {"指标": "DSR", "评级": "A", "数值": "1.00", "说明": "多重测试校正后仍显著"},
        {"指标": "偏度", "评级": "A", "数值": "0.286", "说明": "正偏, 大赢小亏"},
    ])
    
    # 颜色标注
    def color_grade(val):
        if val == 'A': return 'background-color: #e8f5e9; color: #2e7d32; font-weight: bold'
        if val == 'B': return 'background-color: #e3f2fd; color: #1565c0'
        if val == 'C': return 'background-color: #fff3e0; color: #e65100'
        if val == 'D': return 'background-color: #fce4ec; color: #c62828'
        return ''
    
    st.dataframe(
        metrics_data.style.applymap(color_grade, subset=['评级']),
        use_container_width=True, hide_index=True
    )

    st.divider()

    # ── 结论 ──
    st.subheader("📝 评估结论")
    col_a, col_b = st.columns(2)
    with col_a:
        st.success("**优势**\n\n• 盲测绝对收益为正, 年化约102%\n• Sharpe 1.37, 风险调整后优秀\n• 59% 胜率, 最差窗口 +7.4%\n• 12/17 指标 A/B 水平")
    with col_b:
        st.warning("**短板**\n\n• 仅有 2 个盲测窗口, 统计意义不足\n• 盈亏比 1.16, 赢得不够多\n• 需积累 6+ 盲测窗口后方可更有信心")

# ════════════════════════════════════════
#  页面 1: 开发历史
# ════════════════════════════════════════
elif page == "📊 开发历史":
    st.title("📊 版本演进与开发记录")
    
    st.subheader("🔄 版本演进")
    ver_df = pd.DataFrame([
        {"版本": "v1-v3", "改动": "Regression→Lambdarank, 前瞻标签, FactorCache", "均值超额": "+1~13%", "状态": "✅"},
        {"版本": "v4", "改动": "P0-P3 Bug修复, Phase1-3 优化", "均值超额": "+13.7%", "状态": "✅"},
        {"版本": "v5", "改动": "★ 20日标签, 78只CSI300, 1年训练", "均值超额": "+15.9%", "状态": "✅"},
        {"版本": "v6", "改动": "★ 评估体系重构 (dev/blind分离)", "盲测超额": "+16.2%", "状态": "✅ 当前"},
    ])
    st.dataframe(ver_df, use_container_width=True, hide_index=True)

    st.subheader("🔧 核心技术决策")
    st.markdown("""
    1. **5日标签→20日标签**: 根本性修复。5日标签导致模型学成均值回归(熊市好牛市差)。
       20日标签让模型转为趋势跟随, 牛市也能赚钱。
    
    2. **18只→78只股票**: 截面排名的统计意义取决于股票数量。
       18只跨6-7行业使排名退化为行业选择。78只CSI300提供丰富的比较基准。
    
    3. **3年→1年训练窗口**: 市场结构快速变化。
       1年训练+0.5年半衰期确保模型学到的是最近的市场规律。
    
    4. **dev/blind分离**: 最关键的评估改进。
       开发集调参, 盲测集锁定参数只跑一次。不再用测试集反馈调参。
    """)

    st.subheader("📐 当前模型架构")
    st.markdown("""
    ```
    78只CSI300日线 → 39因子预计算 → 截面z-score标准化
        → LightGBM Lambdarank (depth=5, 200轮)
        → 时间衰减权重 (半衰期0.5年)
        → 前瞻20日收益排序 → Top-5选股
        → 持有≥10天 + 成本门槛8%
    ```
    """)

# ════════════════════════════════════════
#  页面 2: 权益曲线
# ════════════════════════════════════════
elif page == "📈 权益曲线":
    st.title("📈 策略权益曲线")

    st.info("""
    运行 `python blind_test.py` 生成最新的盲测权益数据。
    
    当前盲测结果:
    - W7 (2025-07~2026-04, 9个月): 策略 +83.5%, 基准 +48.1%
    - W8 (2026-04~2026-07, 3个月): 策略 +7.4%, 基准 +10.5%
    
    要查看详细权益曲线, 运行 blind_test.py 后刷新此页面。
    """)

    # 简易权益模拟 (基于已知结果)
    st.subheader("W7 权益曲线 (示意)")
    days = 183
    np.random.seed(42)
    eq = 100000 * (1 + np.cumsum(np.random.normal(0.003, 0.015, days)))
    bench = 100000 * (1 + np.cumsum(np.random.normal(0.002, 0.012, days)))
    chart_df = pd.DataFrame({"策略权益": eq, "基准权益": bench})
    st.line_chart(chart_df)
    st.caption("⚠️ 示意数据。运行 blind_test.py 获取真实权益曲线。")
