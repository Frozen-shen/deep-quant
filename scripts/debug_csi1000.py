"""
诊断 CSI1000 零交易问题 — 单窗口逐步追踪信号生成流程
"""
import sys, os, copy
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from model.pipeline import load_config, QuantPipeline
from trading_rules import TradingRules

config = load_config()
config["model"]["type"] = "lgb"

pipeline = QuantPipeline(config, mode="dev")
pipeline._load_universe()
pipeline._load_data()

print(f"\n=== 诊断: CSI1000 信号流 ===")
print(f"  Universe all_symbols: {len(pipeline._universe.all_symbols)}")
print(f"  Loaded stocks: {len(pipeline._all_data)}")

# 检查数据覆盖
min_dates = {}
for sym, df in pipeline._all_data.items():
    min_dates[sym] = (df["date"].min(), df["date"].max(), len(df))

# 每个窗口有多少有效股票
pipeline._precompute_factors()
windows = pipeline._generate_windows()
print(f"  Windows: {len(windows)}")

# 只跑 W1 诊断
w = windows[0]
print(f"\n  诊断 W1: {w['test_start'].date()} ~ {w['test_end'].date()}")

# 看第一个 test day 的信号
all_days = sorted(set().union(*[set(df["date"].tolist()) for df in pipeline._all_data.values()]))
test_days = [d for d in all_days if w["test_start"] <= d <= w["test_end"]]
print(f"  Test days: {len(test_days)}")

rules = TradingRules()

# 取第一个 test_day
for today in test_days[:3]:
    print(f"\n  --- {today.date()} ---")
    
    # 有多少股票有今天之前的数据
    sd, cpt = {}, {}
    for sym in pipeline._all_data:
        dt = pipeline._all_data[sym][pipeline._all_data[sym]["date"] <= today].tail(120)
        if len(dt) >= 60:
            sd[sym] = dt
            cpt[sym] = dt["close"].iloc[-1]
    
    print(f"    有足够数据的股票: {len(sd)}")
    
    # filter_tradeable
    sd2, cpt2 = rules.filter_tradeable(sd, cpt)
    print(f"    可交易股票: {len(sd2)}")
    
    # 因子特征
    sym_feats, swd = [], []
    nan_count = 0
    for sym in sd2:
        feats = pipeline._factor_cache.get_features(sym, today)
        if feats is not None:
            if any(np.isnan(f) for f in feats):
                nan_count += 1
            else:
                sym_feats.append(feats)
                swd.append(sym)
    
    print(f"    有效因子: {len(sym_feats)} (含NaN: {nan_count})")
    
    if len(sym_feats) >= pipeline.top_k:
        fa = np.array(sym_feats)
        m, s = fa.mean(axis=0), fa.std(axis=0)
        s[s == 0] = 1.0
        fn = (fa - m) / s
        
        # 训练模型（简化：用当天数据快速训练）
        from ml_ranker import MLRanker
        mc = config["model"]
        model = MLRanker(n_estimators=50, max_depth=mc["max_depth"],
                         learning_rate=mc["learning_rate"], lambda_l1=mc["lambda_l1"],
                         min_data_in_leaf=mc["min_data_in_leaf"])
        
        # 简单标签
        rets = np.array([float(pipeline._all_data[s][pipeline._all_data[s]["date"] == today]["close"].iloc[0]) for s in swd])
        labels = np.floor(np.argsort(np.argsort(rets)) / len(rets) * 30).astype(int)
        
        # 快速测试预测
        preds = model.predict(fn) if hasattr(model, 'predict') else np.zeros(len(fn))
        print(f"    预测分数: min={preds.min():.4f} max={preds.max():.4f} std={preds.std():.4f}")
        print(f"    Top-10 分数字: {sorted(preds, reverse=True)[:10]}")
        
        # 检查 score 是否有足够变化
        from portfolio_ranker import PortfolioRanker
        ranker = PortfolioRanker(top_k=pipeline.top_k, n_drop=pipeline.n_drop,
                                 hold_thresh=pipeline.hold_thresh,
                                 sell_rank_buffer=pipeline.sell_rank_buffer,
                                 buy_confirm_days=pipeline.buy_confirm_days,
                                 cost_threshold=pipeline.cost_threshold)
        scores = {swd[i]: float(preds[i]) for i in range(len(swd))}
        decision = ranker.rank(scores, {})
        print(f"    空仓信号: buy={len(decision.get('buy',[]))} sell={len(decision.get('sell',[]))}")
    else:
        print(f"    ❌ 有效因子不够 top_k({pipeline.top_k})")
