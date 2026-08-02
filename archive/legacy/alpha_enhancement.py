"""
Alpha 增强层 — 行业/市值中性化 + LightGBM 非线性因子组合

解决当前因子体系的三个核心问题:
  1. 因子未做行业/市值中性化 → 收益来自风格 beta 而非纯 alpha
  2. 60% 因子负 IC (防御/反转) → 上涨市失效
  3. 线性 IC 加权无法捕获因子间交互 → 用 LightGBM lambdarank 替代

用法:
    from alpha_enhancement import neutralize_factors, load_industry_map, load_market_cap
    from alpha_enhancement import train_lgb_ranker, predict_composite

    # 1) 中性化
    ind_map = load_industry_map()
    mcap    = load_market_cap(all_data)
    neutral = neutralize_factors(factor_values, ind_map, mcap)

    # 2) 非线性组合
    model  = train_lgb_ranker(panel, "fwd_ret", feature_cols, train_period, valid_period)
    scores = predict_composite(model, neutral, symbols)
"""

import os
import json
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================================
#  1. 行业 + 市值中性化
# ============================================================================


def neutralize_factors(
    factor_values: Dict[str, Dict[str, float]],
    industry_map: Dict[str, str],
    market_cap: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """
    对每个因子做行业哑变量 + log(市值) 的截面回归, 取残差作为中性化因子值。

    原理:
        factor_i = beta_0 + sum(beta_k * Industry_k) + beta_m * log(MktCap_i) + epsilon_i
        中性化后因子 = epsilon_i (残差), 剥离了行业和市值的影响。

    缺失值处理:
        - 缺少行业信息的股票 → 归入 "未知" 行业
        - 缺少市值的股票 → 用截面中位数填充
        - 因子值为 NaN 的股票 → 跳过回归, 结果保持 NaN

    参数:
        factor_values: {factor_name: {symbol: value}} 原始因子截面值
        industry_map:  {symbol: industry_name} 行业分类映射
        market_cap:    {symbol: float} 市值 (或市值代理变量)

    返回:
        {factor_name: {symbol: neutralized_value}} 中性化后的因子值
    """
    if not factor_values:
        return {}

    # 收集所有出现过的 symbols 的并集
    all_symbols = set()
    for fv in factor_values.values():
        all_symbols.update(fv.keys())
    all_symbols = sorted(all_symbols)

    if not all_symbols:
        return {k: {} for k in factor_values}

    # ── 构建设计矩阵 (对所有因子共用) ──
    # 行业哑变量
    industries = []
    for sym in all_symbols:
        industries.append(industry_map.get(sym, "未知"))

    unique_industries = sorted(set(industries))
    n_ind = len(unique_industries)
    ind_to_idx = {ind: i for i, ind in enumerate(unique_industries)}

    # 设计矩阵: [intercept, industry_dummies (drop first), log_mcap]
    # drop first 避免多重共线性
    n_cols = 1 + max(n_ind - 1, 0) + 1  # intercept + (n_ind-1) dummies + log_mcap
    X_full = np.zeros((len(all_symbols), n_cols))
    X_full[:, 0] = 1.0  # intercept

    # log 市值
    mcap_vals = np.array([market_cap.get(sym, np.nan) for sym in all_symbols])
    # 用中位数填充缺失市值
    valid_mcap = mcap_vals[~np.isnan(mcap_vals)]
    if len(valid_mcap) > 0:
        median_mcap = np.nanmedian(valid_mcap)
    else:
        median_mcap = 1.0
    mcap_filled = np.where(np.isnan(mcap_vals) | (mcap_vals <= 0), median_mcap, mcap_vals)
    log_mcap = np.log(mcap_filled + 1.0)  # +1 防止 log(0)
    X_full[:, -1] = log_mcap

    # 行业哑变量 (跳过第一个行业作为基准)
    for i, ind in enumerate(industries):
        idx = ind_to_idx[ind]
        if idx > 0:  # 第一个行业是基准, 不加入哑变量
            X_full[i, idx] = 1.0  # 列 1..n_ind-1 对应行业 1..n_ind-1

    result: Dict[str, Dict[str, float]] = {}

    for factor_name, fv in factor_values.items():
        y = np.array([fv.get(sym, np.nan) for sym in all_symbols])
        valid_mask = ~np.isnan(y)

        neutralized: Dict[str, float] = {}

        if valid_mask.sum() < n_cols + 1:
            # 有效样本太少, 无法回归, 原样返回 (跳过 NaN)
            for sym, val in fv.items():
                if not np.isnan(val):
                    neutralized[sym] = val
            result[factor_name] = neutralized
            continue

        X_valid = X_full[valid_mask]
        y_valid = y[valid_mask]

        # OLS: beta = (X'X)^{-1} X'y, 用 lstsq 更稳定
        try:
            beta, _, _, _ = np.linalg.lstsq(X_valid, y_valid, rcond=None)
            residuals = y_valid - X_valid @ beta
        except np.linalg.LinAlgError:
            # 回归失败, 原样返回 (跳过 NaN)
            for sym, val in fv.items():
                if not np.isnan(val):
                    neutralized[sym] = val
            result[factor_name] = neutralized
            continue

        # 填回结果
        valid_indices = np.where(valid_mask)[0]
        for j, idx in enumerate(valid_indices):
            sym = all_symbols[idx]
            neutralized[sym] = float(residuals[j])

        # 原始值中 NaN 的 symbol 保持 NaN (不放入结果)
        result[factor_name] = neutralized

    return result


def load_industry_map() -> Dict[str, str]:
    """
    获取申万行业分类映射 {symbol: industry_name}。

    策略:
        1. 优先从本地缓存 data_cache/sw_industry.json 读取
        2. 缓存不存在时, 通过 akshare 获取:
           - ak.stock_board_industry_name_em() 获取行业列表
           - ak.stock_board_industry_cons_em(symbol=行业名) 获取成分股
        3. akshare 失败时, 回退到 sector_analyzer 的前缀分类

    返回:
        {symbol: industry_name} 行业映射
    """
    cache_path = os.path.join(os.path.dirname(__file__), "data_cache", "sw_industry.json")

    # 1) 尝试缓存
    if os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                cached = json.load(f)
            if cached:
                return cached
        except (json.JSONDecodeError, IOError):
            pass

    # 2) 尝试 akshare
    industry_map: Dict[str, str] = {}
    try:
        import akshare as ak

        # 获取东方财富行业板块列表
        board_df = ak.stock_board_industry_name_em()
        if board_df is not None and len(board_df) > 0:
            # 列名可能是 "板块名称" 或 "industry_name"
            name_col = None
            for col in board_df.columns:
                if "名称" in str(col) or "name" in str(col).lower():
                    name_col = col
                    break
            if name_col is None:
                name_col = board_df.columns[0]

            industry_names = board_df[name_col].tolist()

            for ind_name in industry_names:
                try:
                    cons_df = ak.stock_board_industry_cons_em(symbol=str(ind_name))
                    if cons_df is not None and len(cons_df) > 0:
                        code_col = None
                        for col in cons_df.columns:
                            if "代码" in str(col) or "code" in str(col).lower():
                                code_col = col
                                break
                        if code_col is None:
                            code_col = cons_df.columns[0]

                        for code in cons_df[code_col].tolist():
                            industry_map[str(code)] = str(ind_name)
                except Exception:
                    continue

    except Exception:
        pass

    # 3) 回退: sector_analyzer 前缀分类
    if not industry_map:
        try:
            from sector_analyzer import build_a_share_sector_map
            from data_cache import get_cached_symbols

            symbols = get_cached_symbols()
            if symbols:
                industry_map = build_a_share_sector_map(symbols)
        except Exception:
            pass

    # 保存缓存
    if industry_map:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(industry_map, f, ensure_ascii=False)
        except IOError:
            pass

    return industry_map


def load_market_cap(all_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    从行情数据估算市值代理变量。

    方法: close * volume 的滚动均值 (即日均成交额) 作为市值的代理变量。
    真正的市值需要总股本数据, 但成交额与市值高度正相关 (相关系数 > 0.8),
    在中性化回归中作为 size 控制变量足够。

    参数:
        all_data: {symbol: DataFrame} 行情数据, 需包含 'close' 和 'volume' 列

    返回:
        {symbol: float} 市值代理值 (最近 60 个交易日的日均成交额)
    """
    mcap: Dict[str, float] = {}
    window = 60  # 滚动窗口

    for sym, df in all_data.items():
        if df is None or len(df) == 0:
            continue
        try:
            close = df["close"].astype(float)
            volume = df["volume"].astype(float)
            turnover = close * volume  # 成交额代理

            # 取最近 window 天的均值
            recent = turnover.tail(window)
            val = recent.mean()

            if np.isfinite(val) and val > 0:
                mcap[sym] = float(val)
        except (KeyError, TypeError):
            continue

    return mcap


# ============================================================================
#  2. LightGBM 非线性因子组合
# ============================================================================


def train_lgb_ranker(
    factor_panel: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    train_period: Tuple[str, str],
    valid_period: Tuple[str, str],
) -> "lgb.Booster":
    """
    用 LightGBM lambdarank 训练截面排序模型。

    与 ml_ranker.MLRanker 的区别:
        - 极端正则化 (n_estimators=50, max_depth=3, min_data_in_leaf=200)
          → 防止在因子数量多、样本有限时过拟合
        - 面板数据输入 (date × symbol × factors), 自动按日期切分
        - 返回原生 lgb.Booster, 方便后续集成

    参数:
        factor_panel: 面板 DataFrame, 需包含:
            - 'date' 列 (datetime 或 str)
            - 'symbol' 列
            - feature_cols 中的各因子列
            - label_col 列 (前瞻收益率, 连续值)
        label_col: 标签列名 (如 'fwd_ret_20d')
        feature_cols: 特征列名列表
        train_period: (start_date, end_date) 训练区间, 格式 'YYYY-MM-DD'
        valid_period: (start_date, end_date) 验证区间, 格式 'YYYY-MM-DD'

    返回:
        训练好的 lgb.Booster 模型
    """
    import lightgbm as lgb
    from scipy.stats import rankdata

    # ── 数据准备 ──
    df = factor_panel.copy()
    df["date"] = pd.to_datetime(df["date"])

    train_start, train_end = pd.Timestamp(train_period[0]), pd.Timestamp(train_period[1])
    valid_start, valid_end = pd.Timestamp(valid_period[0]), pd.Timestamp(valid_period[1])

    train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
    valid_df = df[(df["date"] >= valid_start) & (df["date"] <= valid_end)].copy()

    if len(train_df) == 0:
        raise ValueError(
            f"训练集为空: {train_period[0]} ~ {train_period[1]} 无数据"
        )
    if len(valid_df) == 0:
        warnings.warn(
            f"验证集为空: {valid_period[0]} ~ {valid_period[1]} 无数据, "
            "将使用训练集尾部 20% 作为验证集"
        )
        # 回退: 从训练集尾部切出 20%
        dates = sorted(train_df["date"].unique())
        split_idx = int(len(dates) * 0.8)
        valid_dates = dates[split_idx:]
        valid_df = train_df[train_df["date"].isin(valid_dates)].copy()
        train_df = train_df[~train_df["date"].isin(valid_dates)].copy()

    # ── 截面排名标签 + 分组 ──
    def _prepare_group(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """将面板数据转换为 lambdarank 所需的 X, y(rank), group_counts。"""
        X_list, y_list, group_sizes = [], [], []

        for date, group in data.groupby("date"):
            # 去掉特征全 NaN 的行
            valid_rows = group.dropna(subset=feature_cols, how="all")
            if len(valid_rows) < 5:
                continue

            feats = valid_rows[feature_cols].values.astype(float)
            labels_raw = valid_rows[label_col].values.astype(float)

            # 跳过标签全 NaN 的组
            valid_label_mask = ~np.isnan(labels_raw)
            if valid_label_mask.sum() < 5:
                continue

            feats = feats[valid_label_mask]
            labels_raw = labels_raw[valid_label_mask]

            # NaN 特征用截面均值填充
            col_means = np.nanmean(feats, axis=0)
            for c in range(feats.shape[1]):
                nan_mask = np.isnan(feats[:, c])
                if nan_mask.any():
                    feats[nan_mask, c] = col_means[c] if np.isfinite(col_means[c]) else 0.0

            # 截面排名 → 整数标签 (0 = 最差, N-1 = 最好)
            rank_labels = rankdata(labels_raw) - 1

            X_list.append(feats)
            y_list.append(rank_labels.astype(int))
            group_sizes.append(len(feats))

        if not X_list:
            raise ValueError("无有效训练样本 (每个截面至少需要 5 只股票)")

        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        return X, y, np.array(group_sizes)

    X_train, y_train, g_train = _prepare_group(train_df)
    X_valid, y_valid, g_valid = _prepare_group(valid_df)

    # ── 训练 ──
    train_data = lgb.Dataset(
        X_train, label=y_train, group=g_train, feature_name=feature_cols
    )
    valid_data = lgb.Dataset(
        X_valid, label=y_valid, group=g_valid, feature_name=feature_cols,
        reference=train_data,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5, 10],
        "boosting_type": "gbdt",
        "num_leaves": 2 ** 3,           # max_depth=3 → 最多 8 叶子
        "max_depth": 3,
        "learning_rate": 0.05,
        "lambda_l1": 1.0,               # 极端 L1 正则化
        "lambda_l2": 1.0,               # 极端 L2 正则化
        "min_data_in_leaf": 200,        # 极端叶节点最小样本数
        "feature_fraction": 0.8,        # 每棵树随机选 80% 特征
        "bagging_fraction": 0.8,        # 每棵树随机选 80% 样本
        "bagging_freq": 1,
        "max_position": 100,
        "lambdarank_truncation_level": 100,
        "verbose": -1,
        "seed": 42,
        "deterministic": True,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=10, verbose=False),
        lgb.log_evaluation(period=0),  # 静默
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=50,             # n_estimators=50
        valid_sets=[valid_data],
        callbacks=callbacks,
    )

    return model


def predict_composite(
    model: "lgb.Booster",
    factor_values: Dict[str, Dict[str, float]],
    symbols: List[str],
) -> Dict[str, float]:
    """
    用训练好的 LightGBM 模型预测复合因子得分。

    参数:
        model: 训练好的 lgb.Booster (来自 train_lgb_ranker)
        factor_values: {factor_name: {symbol: value}} 因子值 (建议先中性化)
        symbols: 要预测的股票代码列表

    返回:
        {symbol: composite_score} 复合得分, 分数越高 → 排名越靠前 → 优先买入
    """
    if not symbols:
        return {}

    # 按模型的特征顺序构建矩阵
    feature_names = model.feature_name()
    n_features = len(feature_names)

    # 构建特征矩阵
    rows = []
    valid_symbols = []

    for sym in symbols:
        feat_row = []
        for fname in feature_names:
            val = factor_values.get(fname, {}).get(sym, np.nan)
            feat_row.append(val)
        rows.append(feat_row)
        valid_symbols.append(sym)

    X = np.array(rows, dtype=float)

    # NaN 用列均值填充
    for c in range(X.shape[1]):
        col = X[:, c]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            col_mean = np.nanmean(col)
            fill_val = col_mean if np.isfinite(col_mean) else 0.0
            col[nan_mask] = fill_val

    # 预测
    scores = model.predict(X)

    return {sym: float(score) for sym, score in zip(valid_symbols, scores)}


# ============================================================================
#  快速自检
# ============================================================================

def _self_test():
    """内部自检: 验证核心逻辑正确性。"""
    print("[alpha_enhancement] 自检...")

    # 1) 中性化测试
    factor_values = {
        "momentum": {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0,
                      "F": 1.5, "G": 2.5, "H": 3.5},
    }
    industry_map = {"A": "银行", "B": "银行", "C": "银行", "D": "科技",
                    "E": "科技", "F": "科技", "G": "医药", "H": "医药"}
    market_cap = {"A": 100, "B": 200, "C": 300, "D": 150,
                  "E": 250, "F": 350, "G": 180, "H": 280}

    neutral = neutralize_factors(factor_values, industry_map, market_cap)
    assert "momentum" in neutral, "中性化结果缺少因子"
    assert len(neutral["momentum"]) == 8, f"中性化结果数量错误: {len(neutral['momentum'])}"

    # 残差均值应接近 0
    residuals = list(neutral["momentum"].values())
    mean_r = np.mean(residuals)
    assert abs(mean_r) < 0.5, f"残差均值偏离过大: {mean_r}"
    print(f"  中性化 OK: 残差均值={mean_r:.4f}, 残差={residuals}")

    # 缺失值处理
    factor_with_nan = {
        "test": {"A": 1.0, "B": np.nan, "C": 3.0},
    }
    neutral_nan = neutralize_factors(factor_with_nan, {"A": "X", "C": "X"}, {"A": 100, "C": 300})
    assert "B" not in neutral_nan["test"], "NaN 因子值应被跳过"
    print("  缺失值处理 OK")

    # 2) LightGBM 测试
    import lightgbm as lgb

    np.random.seed(42)
    n_days, n_stocks = 60, 20
    records = []
    for d in range(n_days):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for s in range(n_stocks):
            f1 = np.random.randn()
            f2 = np.random.randn()
            fwd_ret = f1 * 0.3 + f2 * 0.1 + np.random.randn() * 0.5
            records.append({
                "date": date, "symbol": f"S{s:03d}",
                "f1": f1, "f2": f2, "fwd_ret": fwd_ret,
            })
    panel = pd.DataFrame(records)

    model = train_lgb_ranker(
        panel, "fwd_ret", ["f1", "f2"],
        train_period=("2024-01-01", "2024-02-15"),
        valid_period=("2024-02-16", "2024-03-01"),
    )
    assert isinstance(model, lgb.Booster), "模型类型错误"

    # 预测
    fv = {
        "f1": {f"S{s:03d}": np.random.randn() for s in range(n_stocks)},
        "f2": {f"S{s:03d}": np.random.randn() for s in range(n_stocks)},
    }
    scores = predict_composite(model, fv, [f"S{s:03d}" for s in range(n_stocks)])
    assert len(scores) == n_stocks, f"预测数量错误: {len(scores)}"
    print(f"  LightGBM OK: {len(scores)} 个预测, 分数范围=[{min(scores.values()):.3f}, {max(scores.values()):.3f}]")

    print("[alpha_enhancement] 自检通过!")


if __name__ == "__main__":
    _self_test()
