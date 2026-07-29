"""
无泄露训练管道 — 替代 test_rolling_v3.py

核心改进:
  1. embargo: 训练截止到 train_end - embargo_days，标签不穿越测试期
  2. T+1 成交: 今天收盘确认信号 → 明天开盘执行
  3. 涨跌停接线: can_buy/can_sell 被实际调用
  4. 从 config.yaml 读取所有参数
  5. Point-in-time 成分股宇宙

用法:
  python model/pipeline.py --mode dev    # 开发期 walk-forward 验证
  python model/pipeline.py --mode blind  # 盲测 (参数冻结, 只跑一次)
"""

import os, sys, yaml, json, argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


def load_config() -> dict:
    path = os.path.join(BASE_DIR, "config.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.yaml not found at {path}")
    with open(path) as f:
        return yaml.safe_load(f)


class QuantPipeline:
    """
    完整的量化管道: 数据加载 → 因子预计算 → 模型训练 → 回测 → 评估。
    """

    def __init__(self, config: dict, mode: str = "dev"):
        self.cfg = config
        self.mode = mode  # "dev" or "blind"

        # 解包配置
        dp = config["data_partition"]
        self.research_period = (pd.Timestamp(dp["research"]["start"]),
                                pd.Timestamp(dp["research"]["end"]))
        self.dev_period = (pd.Timestamp(dp["development"]["start"]),
                           pd.Timestamp(dp["development"]["end"]))
        self.blind_period = (pd.Timestamp(dp["blind_test"]["start"]),
                             pd.Timestamp(dp["blind_test"]["end"]))

        r = config["rolling"]
        self.train_months = r["train_months"]
        self.test_months = r["test_months"]
        self.day_step = r["day_step"]
        self.embargo_days = r["embargo_days"]

        lb = config["label"]
        self.label_horizon = lb["horizon_days"]

        ex = config["execution"]
        self.top_k = ex["top_k"]
        self.initial_capital = ex["initial_capital"]
        self.lot_size = ex["lot_size"]

        pf = config["portfolio"]
        self.hold_thresh = pf["hold_thresh"]
        self.sell_rank_buffer = pf["sell_rank_buffer"]
        self.buy_confirm_days = pf["buy_confirm_days"]
        self.cost_threshold = pf["cost_threshold"]
        self.n_drop = pf["n_drop"]

        self._all_data: Dict[str, pd.DataFrame] = {}
        self._factor_cache = None
        self._factor_names: List[str] = []
        self._universe = None

    def run(self):
        """主入口。"""
        print("=" * 60)
        print(f"  Deep Quant v2 Pipeline — mode: {self.mode}")
        print("=" * 60)

        # 1. 加载 Universe
        self._load_universe()

        # 2. 加载股票数据
        self._load_data()

        # 3. 预计算因子
        self._precompute_factors()

        # 4. 生成训练窗口
        windows = self._generate_windows()
        print(f"  窗口数: {len(windows)}")

        # 5. 逐个窗口训练+测试
        results = []
        for wi, w in enumerate(windows):
            result = self._run_window(wi, w)
            if result:
                results.append(result)

        # 6. 汇总
        self._summarize(results)

    # ════════════════════════════════════════
    #  步骤 1-3: 数据准备
    # ════════════════════════════════════════

    def _load_universe(self):
        from data.universe import StockUniverse
        self._universe = StockUniverse(self.cfg["universe"])
        if not self._universe.load():
            self._universe.build_from_akshare(
                self.cfg["data_partition"]["full_start"],
                self.cfg["data_partition"]["full_end"])

    def _load_data(self):
        """加载所有宇宙内股票的数据。"""
        from data_cache import load as load_single
        symbols = self._universe.all_symbols
        if not symbols:
            # 回退: 从缓存加载
            from data_cache import get_cached_symbols
            symbols = get_cached_symbols()
        print(f"  股票池: {len(symbols)} 只")
        for sym in symbols:
            df = load_single(sym)
            if df is not None and len(df) >= 100:
                df["date"] = pd.to_datetime(df["date"])
                self._all_data[sym] = df
        print(f"  加载完成: {len(self._all_data)} 只有效数据")

    def _precompute_factors(self):
        from factor_scorer import FactorScorer
        from factor_cache import FactorCache
        scorer = FactorScorer.from_preset(self.cfg["factors"]["preset"])
        self._factor_names = sorted(scorer.factor_weights.keys())
        self._factor_cache = FactorCache(scorer, self._factor_names)
        self._factor_cache.precompute(self._all_data)
        print(f"  因子: {len(self._factor_names)} 个, 预计算完成")

    # ════════════════════════════════════════
    #  步骤 4: 窗口生成
    # ════════════════════════════════════════

    def _generate_windows(self) -> List[dict]:
        """生成滚动训练/测试窗口。"""
        period = self.dev_period if self.mode == "dev" else self.blind_period
        test_start = period[0]
        test_end = period[1]
        windows = []

        current = test_start
        while current < test_end:
            test_end_dt = min(current + pd.DateOffset(months=self.test_months), test_end)
            train_start = current - pd.DateOffset(months=self.train_months)
            windows.append({
                "train_start": train_start,
                "train_end": current - timedelta(days=1),
                "test_start": current,
                "test_end": test_end_dt,
            })
            current = test_end_dt

        return windows

    # ════════════════════════════════════════
    #  步骤 5: 单窗口训练+测试
    # ════════════════════════════════════════

    def _run_window(self, wi: int, w: dict) -> Optional[dict]:
        """执行单个窗口的完整训练+回测流程。"""
        train_end_clean = w["train_end"] - timedelta(days=self.embargo_days)

        print(f"\n  W{wi+1}: train {w['train_start'].date()}~{train_end_clean.date()} "
              f"→ test {w['test_start'].date()}~{w['test_end'].date()} "
              f"(embargo: {self.embargo_days}d)")

        # ── 训练 ──
        all_days = sorted(set().union(
            *[set(df["date"].tolist()) for df in self._all_data.values()]))
        train_days = [d for d in all_days
                      if w["train_start"] <= d <= train_end_clean][::self.day_step]

        if len(train_days) < 30:
            print(f"    ⚠️ 训练日不足, 跳过")
            return None

        X_list, y_list, group_list = [], [], []
        for today in train_days:
            feats_norm, labels, _ = self._build_cross_section(today)
            if feats_norm is None: continue
            n = len(labels)
            X_list.extend(feats_norm.tolist())
            y_list.extend(labels.tolist())
            group_list.extend([str(today)] * n)

        if len(X_list) < 100:
            print(f"    ⚠️ 训练样本不足 ({len(X_list)}), 跳过")
            return None

        # 训练模型
        train_result = self._train_model(
            np.array(X_list), np.array(y_list, dtype=int), group_list,
            train_end_clean)

        # ── 回测 ──
        test_result = self._backtest(w, train_result["model"])
        test_result["window"] = wi + 1
        test_result["train_start"] = w["train_start"].strftime("%Y-%m-%d")
        test_result["train_end"] = train_end_clean.strftime("%Y-%m-%d")
        test_result["test_start"] = w["test_start"].strftime("%Y-%m-%d")
        test_result["test_end"] = w["test_end"].strftime("%Y-%m-%d")
        test_result["embargo_applied"] = True
        test_result["train_samples"] = len(X_list)

        return test_result

    def _build_cross_section(self, today):
        """构建截面样本（补上 embargo 约束）。"""
        from factor_cache import FactorCache
        day_feats, day_rets = {}, {}

        for sym in self._all_data:
            feats = self._factor_cache.get_features(sym, today)
            if feats is None: continue
            full_df = self._all_data[sym]
            try:
                dm = full_df["date"] == today
                if not dm.any(): continue
                tp = full_df.index[dm][0]
                ip = full_df.index.get_loc(tp)
                # ★ 标签前瞻 (已在 train_end_clean 保证不穿越)
                if ip + self.label_horizon >= len(full_df): continue
                fwd_close = full_df.iloc[ip + self.label_horizon]["close"]
                today_close = full_df.iloc[ip]["close"]
                fwd = fwd_close / today_close - 1
            except (IndexError, KeyError):
                continue
            day_feats[sym] = feats
            day_rets[sym] = fwd

        if len(day_feats) < self.top_k:
            return None, None, None

        syms = list(day_feats.keys())
        fa = np.array([day_feats[s] for s in syms])
        m, s = fa.mean(axis=0, keepdims=True), fa.std(axis=0, keepdims=True)
        s[s == 0] = 1.0
        fn = (fa - m) / s
        rets = np.array([day_rets[s] for s in syms])
        labels = np.floor(rankdata(rets) / len(rets) * 30).astype(int)
        return fn, labels, syms

    def _train_model(self, X, y, group_list, train_end):
        """训练 LightGBM Lambdarank 模型（含时间衰减权重）。"""
        from ml_ranker import MLRanker

        mc = self.cfg["model"]
        model = MLRanker(
            n_estimators=mc["n_estimators"],
            max_depth=mc["max_depth"],
            learning_rate=mc["learning_rate"],
            lambda_l1=mc["lambda_l1"],
            min_data_in_leaf=mc["min_data_in_leaf"],
        )
        model.feature_names = self._factor_names

        # 时间衰减权重
        td = self.cfg["time_decay"]
        decay_lambda = np.log(2) / td["half_life_years"]
        dw = np.array([np.exp(-decay_lambda * max(0,
                        (train_end - pd.Timestamp(str(g))).days / 365.0))
                       for g in group_list])

        groups = pd.Series(group_list).astype(str).factorize()[0]
        model.fit(X, y, groups, val_ratio=mc["val_ratio"], sample_weight=dw)

        return {"model": model}

    def _backtest(self, w: dict, model) -> dict:
        """执行回测 (T+1 开盘成交, 涨跌停约束)。"""
        from portfolio import PortfolioManager
        from portfolio_ranker import PortfolioRanker
        from trading_rules import TradingRules, calc_buy_commission, calc_sell_commission

        pm = PortfolioManager(market="a", initial_capital=self.initial_capital)
        ranker = PortfolioRanker(
            top_k=self.top_k, n_drop=self.n_drop,
            hold_thresh=self.hold_thresh,
            sell_rank_buffer=self.sell_rank_buffer,
            buy_confirm_days=self.buy_confirm_days,
            cost_threshold=self.cost_threshold,
        )
        rules = TradingRules()

        # ★ 用前一天的信号
        all_days = sorted(set().union(
            *[set(df["date"].tolist()) for df in self._all_data.values()]))
        test_days = [d for d in all_days
                     if w["test_start"] <= d <= w["test_end"]]

        trades = 0
        equity_curve = []
        prev_decision = None  # ★ 存储前一天的决定

        for i, today in enumerate(test_days):
            ts = today.strftime("%Y-%m-%d")

            # ★ T+1: 今天的成交基于昨天的信号
            if prev_decision:
                b, s = self._execute_decision(pm, prev_decision, today, rules, self._all_data)
                trades += b + s

            # ★ 生成今天的信号（明天执行）
            prev_decision = self._generate_signal(model, today, rules, pm, ranker)

            # 按收盘价标记权益
            state = pm.load()

            # 计算收盘市值
            cp_today = self._get_close_prices(today)
            holdings_val = sum(
                cp_today.get(s, 0) * p.get("qty", 0)
                for s, p in state.positions.items())
            total_eq = state.cash + holdings_val
            equity_curve.append({"date": ts, "equity": total_eq})

        # ── 绩效 ──
        state = pm.load()
        cp = self._get_close_prices(test_days[-1])
        holdings_val = sum(cp.get(s, 0) * p.get("qty", 0)
                           for s, p in state.positions.items())
        total_eq = state.cash + holdings_val

        ret = (total_eq / self.initial_capital - 1) * 100

        return {
            "total_return": ret,
            "trades": trades,
            "n_test_days": len(test_days),
            "equity_curve": equity_curve,
        }

    def _generate_signal(self, model, today, rules, pm, ranker):
        """基于模型预测生成买卖信号（含涨跌停过滤）。"""
        sd, cpt = {}, {}
        for sym in self._all_data:
            dt = self._all_data[sym][self._all_data[sym]["date"] <= today].tail(120)
            if len(dt) >= 60:
                sd[sym] = dt
                cpt[sym] = dt["close"].iloc[-1]

        if len(sd) < self.top_k:
            return None

        # 过滤停牌
        sd, cpt = rules.filter_tradeable(sd, cpt)
        if len(sd) < self.top_k:
            return None

        # 特征 + 预测
        sym_feats, swd = [], []
        for sym in sd:
            feats = self._factor_cache.get_features(sym, today)
            if feats is not None:
                sym_feats.append(feats)
                swd.append(sym)

        if len(sym_feats) < self.top_k:
            return None

        fa = np.array(sym_feats)
        m, s = fa.mean(axis=0), fa.std(axis=0)
        s[s == 0] = 1.0
        fn = (fa - m) / s
        preds = model.predict(fn)
        scores = {swd[i]: float(preds[i]) for i in range(len(swd))}

        state = pm.load()
        holdings = [s for s, p in state.positions.items() if p["qty"] > 0]

        decision = ranker.rank(scores, holdings)

        # ★ 接线涨跌停约束 (只对仍在sd中的股票)
        decision["buy"] = [s for s in decision["buy"] if s in sd and rules.can_buy(s, sd[s])]
        decision["sell"] = [s for s in decision["sell"] if s in sd and rules.can_sell(s, sd[s])]

        return decision

    def _execute_decision(self, pm, decision, today, rules, all_data):
        """执行买卖决定 (T+1 开盘价成交)。返回 (buy_count, sell_count)"""
        buys, sells = 0, 0
        from trading_rules import calc_buy_commission, calc_sell_commission

        for s in decision.get("sell", []):
            pos = pm.load().positions.get(s, {})
            qty = pos.get("qty", 0)
            if qty <= 0 or s not in all_data:
                continue
            dt = all_data[s][all_data[s]["date"] <= today].tail(1)
            if len(dt) == 0: continue
            px = dt["open"].iloc[-1] if "open" in dt.columns else dt["close"].iloc[-1]
            comm = calc_sell_commission(qty, px)
            pm.apply_sell(s, qty, px, trade_date=today.strftime("%Y-%m-%d"), commission=comm)
            sells += 1

        for s in decision.get("buy", []):
            if s not in all_data: continue
            dt = all_data[s][all_data[s]["date"] <= today].tail(1)
            if len(dt) == 0: continue
            px = dt["open"].iloc[-1] if "open" in dt.columns else dt["close"].iloc[-1]
            if not rules.can_buy(s, dt): continue
            cash_per = pm.load().cash * 0.9 / max(1, len(decision.get("buy", [])))
            qty = int(cash_per / px / self.lot_size) * self.lot_size
            if qty >= self.lot_size:
                comm = calc_buy_commission(qty, px)
                pm.apply_buy(s, qty, px, trade_date=today.strftime("%Y-%m-%d"), commission=comm)
                buys += 1
        return buys, sells

    def _get_close_prices(self, today):
        """获取某天的收盘价。"""
        cp = {}
        for sym in self._all_data:
            dt = self._all_data[sym][self._all_data[sym]["date"] <= today].tail(1)
            if len(dt) > 0:
                cp[sym] = dt["close"].iloc[-1]
        return cp

    def _summarize(self, results: list):
        """汇总所有窗口结果。"""
        if not results:
            print("\n  ⚠️ 无有效结果")
            return
        print(f"\n{'=' * 65}")
        print(f"  最终结果 ({self.mode})")
        print(f"{'=' * 65}")
        for r in results:
            mark = "✅" if r.get("total_return", 0) > 0 else "❌"
            print(f"  W{r['window']}: {r['test_start'][:7]}~{r['test_end'][:7]}  "
                  f"ret:{r['total_return']:+.1f}%  trades:{r['trades']}  {mark}")
        mean_ret = np.mean([r["total_return"] for r in results])
        pos = sum(1 for r in results if r.get("total_return", 0) > 0)
        print(f"\n  均值: {mean_ret:+.1f}%  正窗口: {pos}/{len(results)}")
        print(f"  embargo: {self.embargo_days}d  T+1 开盘成交  "
              f"涨跌停已接线  universe={self.cfg['universe']['index']}")
        print(f"⚠️ 模式: {self.mode} — "
              f"{'参数可调' if self.mode == 'dev' else '★ 参数冻结, 此为最终结果'}")
        print(f"{'=' * 65}")


# ════════════════════════════════════════
#  CLI entry
# ════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dev", "blind"], default="dev",
                        help="dev: 开发期可调参 | blind: 盲测期参数冻结")
    args = parser.parse_args()

    config = load_config()

    # 盲测模式检查
    if args.mode == "blind":
        bt = config.get("blind_test", {})
        bt["trial_count"] = bt.get("trial_count", 0) + 1
        bt["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config["blind_test"] = bt
        with open(os.path.join(BASE_DIR, "config.yaml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"★ 盲测 Trial #{bt['trial_count']} — 结果即为最终答案，绝不调参")

    pipeline = QuantPipeline(config, mode=args.mode)
    pipeline.run()
