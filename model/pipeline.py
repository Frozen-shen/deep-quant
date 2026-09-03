"""
无泄露训练管道 v3 — 简单透明回测引擎, 无数据库依赖
"""
import os, sys, yaml, argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np, pandas as pd
from data.calendar import get_trading_days

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


def load_config():
    with open(os.path.join(BASE_DIR, "config.yaml")) as f:
        return yaml.safe_load(f)


class QuantPipeline:
    def __init__(self, config: dict, mode: str = "dev"):
        self.cfg = config; self.mode = mode
        dp = config["data_partition"]
        self.research_period = (pd.Timestamp(dp["research"]["start"]), pd.Timestamp(dp["research"]["end"]))
        self.dev_period = (pd.Timestamp(dp["development"]["start"]), pd.Timestamp(dp["development"]["end"]))
        # blind 窗口: 兼容历史 data_partition.blind_test 布局, 当前 config.yaml 的
        # 窗口在 data_partition.blind (blind_test 顶层段只是状态标志, 无 end)。
        # (2026-09-03 配置结构漂移修复: 旧引用 dp["blind_test"] 会 KeyError)
        bd = dp.get("blind_test") or dp.get("blind") or {}
        self.blind_period = (pd.Timestamp(bd.get("start", "2027-01-01")),
                             pd.Timestamp(bd.get("end", "2027-12-31")))
        r = config["rolling"]
        self.train_months = r["train_months"]; self.test_months = r["test_months"]
        self.day_step = r["day_step"]; self.embargo_days = r["embargo_days"]
        self.label_horizon = config["label"]["horizon_days"]
        ex = config["execution"]
        self.top_k = ex["top_k"]; self.initial_capital = ex["initial_capital"]
        self.lot_size = ex["lot_size"]
        self.market_filter = ex.get("market_filter", False); self.market_ma = ex.get("market_ma", 60)
        pf = config["portfolio"]
        self.hold_thresh = pf["hold_thresh"]; self.sell_rank_buffer = pf["sell_rank_buffer"]
        self.buy_confirm_days = pf["buy_confirm_days"]; self.n_drop = pf["n_drop"]
        # cost_threshold: 2026-08-16 组合层参数 config 化后顶层键改为 bt_cost_threshold
        # (旧扁平键随历史 config 迁移, 此处兼容读取, 2026-09-03 结构漂移修复)
        self.cost_threshold = pf.get("cost_threshold", pf.get("bt_cost_threshold", 0.05))
        self._all_data: Dict[str, pd.DataFrame] = {}
        self._factor_cache = None; self._factor_names = []
        self._universe = None; self._index_data = None; self._index_ma = None

    def run(self):
        print("=" * 60)
        print(f"  Deep Quant v3 Pipeline — mode: {self.mode}")
        print("=" * 60)
        self._load_universe(); self._load_data(); self._precompute_factors()
        windows = self._generate_windows()
        print(f"  窗口数: {len(windows)}")
        results = []
        for wi, w in enumerate(windows):
            r = self._run_window(wi, w)
            if r: results.append(r)
        summary = self._summarize(results)
        return {"results": results, "summary": summary}

    def _load_universe(self):
        # ★ 宽度改造: all_cached 直接使用所有缓存股票
        if self.cfg["universe"].get("index") == "all_cached":
            from data_cache import get_cached_symbols
            class AllCachedUniverse:
                all_symbols = set(get_cached_symbols())
                def load(self): return True
            self._universe = AllCachedUniverse()
            return

        from data.universe import StockUniverse
        self._universe = StockUniverse(self.cfg["universe"])
        if not self._universe.load():
            self._universe.build_from_akshare(
                self.cfg["data_partition"]["full_start"], self.cfg["data_partition"]["full_end"])

    def _load_data(self):
        from data_cache import load as load_single
        symbols = self._universe.all_symbols
        if not symbols:
            from data_cache import get_cached_symbols
            symbols = get_cached_symbols()
        print(f"  股票池: {len(symbols)} 只")
        for sym in symbols:
            df = load_single(sym)
            if df is not None and len(df) >= 100:
                df["date"] = pd.to_datetime(df["date"]); self._all_data[sym] = df
        print(f"  加载完成: {len(self._all_data)} 只有效数据")

        # ★ 基本面因子
        self._fund_cache = {}
        if self.cfg["factors"].get("use_fundamental", False):
            from data.fundamental_cache_builder import precompute_fundamental_factors, FUND_FACTOR_NAMES
            self._fund_cache = precompute_fundamental_factors(self._all_data)
            self._fund_factor_names = FUND_FACTOR_NAMES
        else:
            self._fund_factor_names = []

        # ★ 加载未复权数据
        self._unadj_data = {}
        unadj_dir = os.path.join(BASE_DIR, "data_cache", "unadjusted")
        if os.path.isdir(unadj_dir):
            for sym in self._all_data:
                path = os.path.join(unadj_dir, f"{sym}.parquet")
                if os.path.exists(path):
                    df = pd.read_parquet(path)
                    df["date"] = pd.to_datetime(df["date"])
                    self._unadj_data[sym] = df
            print(f"  未复权数据: {len(self._unadj_data)} 只")

        # ★ 加载未复权数据 (用于涨跌停判断)
        self._unadj_data = {}
        unadj_dir = os.path.join(BASE_DIR, "data_cache", "unadjusted")
        if os.path.isdir(unadj_dir):
            for sym in self._all_data:
                path = os.path.join(unadj_dir, f"{sym}.parquet")
                if os.path.exists(path):
                    df = pd.read_parquet(path)
                    df["date"] = pd.to_datetime(df["date"])
                    self._unadj_data[sym] = df
            print(f"  未复权数据: {len(self._unadj_data)} 只 (用于涨跌停判断)")

    def _load_index_data(self):
        from data_fetcher import DataFetcher
        try:
            idx = DataFetcher.fetch("sh000001", start_date="20180101", end_date="20260710")
            if idx is not None and len(idx) > self.market_ma:
                idx["date"] = pd.to_datetime(idx["date"])
                self._index_data = idx.set_index("date")["close"]
                self._index_ma = self._index_data.rolling(self.market_ma).mean()
                print(f"  市场过滤: MA{self.market_ma}已计算 (sh000001 上证指数)")
        except Exception as e:
            print(f"  市场过滤: 加载失败({e}), 禁用")

    def _is_market_bullish(self, today):
        if self._index_ma is None: return True
        try:
            mask = self._index_ma.index <= today
            if not mask.any(): return True
            return self._index_data[mask].iloc[-1] > self._index_ma[mask].iloc[-1]
        except: return True

    def _precompute_factors(self):
        from factor_scorer import FactorScorer
        from factor_cache import FactorCache
        scorer = FactorScorer.from_preset(self.cfg["factors"]["preset"])
        self._factor_names = sorted(scorer.factor_weights.keys())
        self._factor_cache = FactorCache(scorer, self._factor_names)
        self._factor_cache.precompute(self._all_data)
        # ★ 基本面因子名附加到总因子列表
        self._all_factor_names = self._factor_names + self._fund_factor_names
        print(f"  因子: {len(self._factor_names)} 价量 + {len(self._fund_factor_names)} 基本面 = {len(self._all_factor_names)} 个, 预计算完成")

    def _generate_windows(self):
        period = self.dev_period if self.mode == "dev" else self.blind_period
        test_start, test_end = period[0], period[1]
        windows = []
        current = test_start
        while current < test_end:
            te = min(current + pd.DateOffset(months=self.test_months), test_end)
            windows.append({"train_start": current - pd.DateOffset(months=self.train_months),
                           "train_end": current - timedelta(days=1),
                           "test_start": current, "test_end": te})
            current = te
        return windows

    def _run_window(self, wi, w):
        train_end_clean = w["train_end"] - timedelta(days=self.embargo_days)
        print(f"\n  W{wi+1}: train {w['train_start'].date()}~{train_end_clean.date()} → test {w['test_start'].date()}~{w['test_end'].date()} (embargo:{self.embargo_days}d)")

        all_days = get_trading_days(self.cfg["data_partition"]["full_start"], self.cfg["data_partition"]["full_end"])
        if not all_days:
            all_days = sorted(set().union(*[set(df["date"].tolist()) for df in self._all_data.values()]))
        train_days = [d for d in all_days if w["train_start"] <= d <= train_end_clean][::self.day_step]
        if len(train_days) < 30: return None

        Xl, yl, gl = [], [], []
        for today in train_days:
            fn_, lbls, _ = self._build_cs(today)
            if fn_ is None: continue
            n = len(lbls); Xl.extend(fn_.tolist()); yl.extend(lbls.tolist()); gl.extend([str(today)] * n)
        if len(Xl) < 100: return None

        model = self._train_model(np.array(Xl), np.array(yl, dtype=int), gl, train_end_clean)
        result = self._backtest(w, model)
        result["window"] = wi + 1
        result.update({k: w[k].strftime("%Y-%m-%d") if hasattr(w[k], 'strftime') else w[k]
                       for k in ["train_start","test_start","test_end"]})
        result["train_end"] = train_end_clean.strftime("%Y-%m-%d")
        result["embargo_applied"] = True; result["train_samples"] = len(Xl)
        return result

    def _build_cs(self, today):
        day_feats, day_rets = {}, {}
        for sym in self._all_data:
            feats = self._factor_cache.get_features(sym, today)
            if feats is None: continue
            # ★ 合并基本面因子
            if self._fund_cache:
                from data.fundamental_cache_builder import merge_fundamental_to_features
                feats = merge_fundamental_to_features(sym, today, self._fund_cache, feats)
            fdf = self._all_data[sym]
            try:
                dm = fdf["date"] == today
                if not dm.any(): continue
                tp = fdf.index[dm][0]; ip = fdf.index.get_loc(tp)
                if ip + self.label_horizon >= len(fdf): continue
                fwd = fdf.iloc[ip + self.label_horizon]["close"] / fdf.iloc[ip]["close"] - 1
            except: continue
            day_feats[sym] = feats; day_rets[sym] = fwd
        if len(day_feats) < self.top_k: return None, None, None
        syms = list(day_feats.keys())
        fa = np.array([day_feats[s] for s in syms])
        m, s = fa.mean(axis=0, keepdims=True), fa.std(axis=0, keepdims=True); s[s == 0] = 1.0
        fn = (fa - m) / s
        rets = np.array([day_rets[s] for s in syms])
        # 标签: 30档粗排 (唯一口径, 与 ml_ranker.coarse_rank_labels 共用; 2026-09-03 统一)
        from ml_ranker import coarse_rank_labels
        labels = coarse_rank_labels(rets)
        return fn, labels, syms

    def _train_model(self, X, y, group_list, train_end):
        model_type = self.cfg["model"].get("type", "lgb")

        if model_type == "l0":
            from model.baselines import L0EqualWeight
            model = L0EqualWeight()
        elif model_type == "l1":
            from model.baselines import L1SingleFactor
            model = L1SingleFactor()
        elif model_type == "l2":
            from model.baselines import L2LinearRanker
            alpha = self.cfg["model"].get("ridge_alpha", 1.0)
            model = L2LinearRanker(alpha=alpha)
        elif model_type == "ensemble":
            from model.ensemble import EnsembleRanker
            members = self.cfg["model"].get("ensemble_members", ["lgb", "lgb", "lgb"])
            blend = self.cfg["model"].get("blend_method", "equal")
            model = EnsembleRanker(members=members, blend_method=blend)
        elif model_type == "linear":
            # linear 是生产路径类型 (IC加权线性, 见 config.yaml model.type), 本管道
            # 只用于 ML 研究对比 — 不做任何静默降级, 直接报错, 避免落进 else 分支
            # 去读已不存在的顶层扁平 ML 键而抛 KeyError (2026-09-03 配置漂移修复)。
            raise ValueError(
                "model.type='linear' 是生产路径类型, QuantPipeline 是 ML 研究对比管道; "
                "请显式指定 model.type ∈ {l0, l1, l2, ensemble, lgb} 之一 (ML 参数从 "
                "config.yaml model.research_lgb 读取)。"
            )
        else:
            # lgb 等 → LightGBM Lambdarank 研究分支。
            # 参数读嵌套的 model.research_lgb (config.yaml 2026-08-01 起的结构),
            # 不再假设存在于 model 顶层 (旧扁平键布局已于 b210bfe 迁移, 此处为遗留 bug)。
            from ml_ranker import MLRanker
            rl = self.cfg["model"].get("research_lgb") or {}
            model = MLRanker(n_estimators=rl.get("n_estimators", 200),
                             max_depth=rl.get("max_depth", 6),
                             learning_rate=rl.get("learning_rate", 0.05),
                             lambda_l1=rl.get("lambda_l1", 0.5),
                             min_data_in_leaf=rl.get("min_data_in_leaf", 30))

        model.feature_names = getattr(self, '_all_factor_names', self._factor_names)
        td = self.cfg["time_decay"]
        dl = np.log(2) / td["half_life_years"]
        dw = np.array([np.exp(-dl * max(0, (train_end - pd.Timestamp(str(g))).days / 365.0)) for g in group_list])
        groups = pd.Series(group_list).astype(str).factorize()[0]
        model.fit(X, y, groups, val_ratio=self.cfg["model"].get("val_ratio", 0.15), sample_weight=dw)
        return model

    def _backtest(self, w, model):
        from model.engine import SimpleBacktest
        from portfolio_ranker import PortfolioRanker
        from trading_rules import TradingRules

        bt = SimpleBacktest(initial_capital=self.initial_capital, top_k=self.top_k, lot_size=self.lot_size,
                           slippage_bps=self.cfg["execution"].get("slippage_bps", 0),
                           turnover_limit_pct=self.cfg["execution"].get("turnover_limit_pct", 1.0))
        # L0等权: 一次性建仓, n_drop=top_k避免多日分批买入
        nd = self.top_k if self.cfg["model"].get("type") == "l0" else self.n_drop
        sn = self.cfg["portfolio"].get("sector_neutral", False)
        ranker = PortfolioRanker(top_k=self.top_k, n_drop=nd, hold_thresh=self.hold_thresh,
                                 sell_rank_buffer=self.sell_rank_buffer, buy_confirm_days=self.buy_confirm_days,
                                 cost_threshold=self.cost_threshold, sector_neutral=sn)
        rules = TradingRules()

        all_days = get_trading_days(self.cfg["data_partition"]["full_start"], self.cfg["data_partition"]["full_end"])
        if not all_days:
            all_days = sorted(set().union(*[set(df["date"].tolist()) for df in self._all_data.values()]))
        test_days = [d for d in all_days if w["test_start"] <= d <= w["test_end"]]

        total_trades = 0; equity_curve = []; prev_decision = None

        for today in test_days:
            if prev_decision:
                b, s, _ = bt.execute(prev_decision, today, self._all_data, rules,
                                    unadjusted_data=self._unadj_data)
                total_trades += b + s
            prev_decision = self._generate_signal(model, today, rules, bt, ranker)
            cp = self._get_close_prices(today)
            eq = bt.mark_to_market(cp)
            equity_curve.append({"date": today.strftime("%Y-%m-%d"), "equity": eq})

        cp_final = self._get_close_prices(test_days[-1])
        final_eq = bt.mark_to_market(cp_final)
        ret = (final_eq / self.initial_capital - 1) * 100
        bench_ret = self._calc_benchmark(w)

        print(f"    策略:{ret:+.1f}%  基准:{bench_ret:+.1f}%  超额:{ret-bench_ret:+.1f}%  交易:{total_trades}笔")
        return {"total_return": ret, "benchmark_return": bench_ret, "excess": ret - bench_ret,
                "trades": total_trades, "n_test_days": len(test_days), "equity_curve": equity_curve}

    def _calc_benchmark(self, w):
        rets = []
        for sym in self._all_data:
            df = self._all_data[sym]
            bdf = df[(df["date"] >= w["test_start"]) & (df["date"] <= w["test_end"])]
            if len(bdf) > 1: rets.append(bdf["close"].iloc[-1] / bdf["close"].iloc[0] - 1)
        return np.mean(rets) * 100 if rets else 0.0

    def _generate_signal(self, model, today, rules, bt, ranker):
        sd, cpt = {}, {}
        for sym in self._all_data:
            dt = self._all_data[sym][self._all_data[sym]["date"] <= today].tail(120)
            if len(dt) >= 60: sd[sym] = dt; cpt[sym] = dt["close"].iloc[-1]
        if len(sd) < self.top_k: return None
        sd, cpt = rules.filter_tradeable(sd, cpt)
        if len(sd) < self.top_k: return None

        sym_feats, swd = [], []
        for sym in sd:
            feats = self._factor_cache.get_features(sym, today)
            if feats is not None:
                # ★ 合并基本面因子
                if self._fund_cache:
                    from data.fundamental_cache_builder import merge_fundamental_to_features
                    feats = merge_fundamental_to_features(sym, today, self._fund_cache, feats)
                sym_feats.append(feats); swd.append(sym)
        if len(sym_feats) < self.top_k: return None

        fa = np.array(sym_feats); m, s = fa.mean(axis=0), fa.std(axis=0); s[s == 0] = 1.0
        fn = (fa - m) / s; preds = model.predict(fn)
        scores = {swd[i]: float(preds[i]) for i in range(len(swd))}
        if len(scores) < self.top_k: return None

        # ★ 回购事件增强
        bb_w = self.cfg["factors"].get("buyback_weight", 0.0)
        if bb_w > 0:
            if not hasattr(self, '_buyback_factor'):
                from factors.event_factors import BuybackFactor
                decay = self.cfg["factors"].get("buyback_decay_days", 20)
                self._buyback_factor = BuybackFactor(decay_days=decay)
            bb_scores = self._buyback_factor.compute_scores(today)
            if bb_scores:
                scores = self._buyback_factor.enhance_scores(scores, bb_scores, weight=bb_w)

        holdings = list(bt.positions.keys())
        decision = ranker.rank(scores, holdings)

        # ★ 涨跌停判断: 优先用未复权价, 无则退回后复权
        limit_data = dict(sd)  # 复制后复权数据
        if self._unadj_data:
            limit_data.update({s: self._unadj_data[s] for s in self._unadj_data if s in sd})
        decision["buy"] = [s for s in decision["buy"] if s in limit_data and rules.can_buy(s, limit_data[s])]
        decision["sell"] = [s for s in decision["sell"] if s in limit_data and rules.can_sell(s, limit_data[s])]

        # ★ PEAD防守: 持仓股有负面预告→强制卖出
        if holdings and self.cfg["execution"].get("pead_defense", False):
            from data.pead_filter import load_pead_alerts
            alerts = load_pead_alerts()
            for s in holdings:
                if alerts.has_bad_news(s, today) and s not in decision["sell"]:
                    decision["sell"].append(s)
        return decision

    def _get_close_prices(self, today):
        cp = {}
        for sym in self._all_data:
            dt = self._all_data[sym][self._all_data[sym]["date"] <= today].tail(1)
            if len(dt) > 0: cp[sym] = float(dt["close"].iloc[-1])
        return cp

    def _summarize(self, results):
        if not results: return {}
        print(f"\n{'='*65}\n  最终结果 ({self.mode})\n{'='*65}")
        for r in results:
            m = "✅" if r.get("excess", 0) > 0 else "❌"
            print(f"  W{r['window']}: {r['test_start'][:7]}~{r['test_end'][:7]}  "
                  f"策略:{r['total_return']:+.1f}%  基准:{r.get('benchmark_return',0):+.1f}%  "
                  f"超额:{r.get('excess',0):+.1f}%  trades:{r['trades']}  {m}")
        mr = float(np.mean([r["total_return"] for r in results]))
        me = float(np.mean([r.get("excess", 0) for r in results]))
        pr = sum(1 for r in results if r.get("total_return", 0) > 0)
        pe = sum(1 for r in results if r.get("excess", 0) > 0)
        print(f"\n  策略均值:{mr:+.1f}%  超额均值:{me:+.1f}%  正窗口:{pr}/{len(results)}  正超额:{pe}/{len(results)}")
        print(f"{'='*65}")

        # ★ 实验记账
        from model.experiment import log_experiment
        log_experiment(self.mode, self.cfg, results)

        return {"mean_return": mr, "mean_excess": me, "n_windows": len(results),
                "pos_windows": pr, "pos_excess": pe, "per_window": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dev", "blind"], default="dev")
    args = parser.parse_args()
    config = load_config()
    if args.mode == "blind":
        bt = config.get("blind_test", {})
        bt["trial_count"] = bt.get("trial_count", 0) + 1
        bt["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config["blind_test"] = bt
        with open(os.path.join(BASE_DIR, "config.yaml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"★ 盲测 Trial #{bt['trial_count']}")
    QuantPipeline(config, mode=args.mode).run()
