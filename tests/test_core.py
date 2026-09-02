"""
交易规则 + 因子计算 单元测试
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import numpy as np


class TestTradingRules(unittest.TestCase):
    """A股真实交易规则测试。"""

    def test_buy_commission(self):
        from trading_rules import calc_buy_commission
        # 1000股 × 10元 = 10000元, 万2.5=2.5, 过户费0.1, 合计2.6 < 最低5元
        fee = calc_buy_commission(1000, 10.0)
        self.assertAlmostEqual(fee, 5.0, places=1)  # 最低5元生效

    def test_buy_commission_no_min(self):
        from trading_rules import calc_buy_commission
        # 10000股 × 10元 = 100000元, 万2.5=25, 过户费1.0, 合计26 > 最低5元
        fee = calc_buy_commission(10000, 10.0)
        self.assertAlmostEqual(fee, 26.0, places=1)

    def test_sell_commission_with_stamp(self):
        from trading_rules import calc_sell_commission
        # 1000股 × 10元 = 10000元
        # 万2.5佣金 + 千0.5印花税 + 过户费 = 2.5 + 5.0 + 0.1 = 7.6
        fee = calc_sell_commission(1000, 10.0)
        self.assertAlmostEqual(fee, 7.6, places=1)

    def test_board_type_main(self):
        from trading_rules import get_board_type
        self.assertEqual(get_board_type("600519"), "main_sh")

    def test_board_type_gem(self):
        from trading_rules import get_board_type
        self.assertEqual(get_board_type("300750"), "gem")


class TestFactorEngine(unittest.TestCase):
    """因子DSL解析测试。"""

    def test_operator_precedence(self):
        from factor_engine import parse_factor
        import pandas as pd
        df = pd.DataFrame({"close": [1.0], "volume": [1.0]})
        # 2 + 3 * 4 = 14 (不是 20)
        r = parse_factor("2 + 3 * 4").evaluate(df).iloc[0]
        self.assertEqual(r, 14.0)

    def test_unary_minus(self):
        from factor_engine import parse_factor
        import pandas as pd
        # -(5) = -5
        df = pd.DataFrame({"close": [5.0], "volume": [1.0]})
        r = parse_factor("-$close").evaluate(df).iloc[0]
        self.assertEqual(r, -5.0)

    def test_ma_spread(self):
        from factor_engine import parse_factor
        import pandas as pd
        import numpy as np
        close = np.linspace(10, 20, 60)
        df = pd.DataFrame({"close": close, "volume": np.ones(60) * 1e6})
        r = parse_factor("Mean($close, 5) / Mean($close, 20) - 1").evaluate(df).iloc[-1]
        self.assertGreater(r, 0)  # 上升趋势中MA5 > MA20

    def test_ma_bullish_fixed(self):
        from factor_engine import parse_factor
        import pandas as pd
        import numpy as np
        # 上升趋势中, MA5 > MA10 > MA20 应该成立
        close = np.linspace(10, 20, 60)
        df = pd.DataFrame({"close": close, "volume": np.ones(60) * 1e6})
        r = parse_factor("(Mean($close, 5) > Mean($close, 10)) * (Mean($close, 10) > Mean($close, 20))").evaluate(df).iloc[-1]
        self.assertEqual(r, 1.0)


class TestEvaluator(unittest.TestCase):
    """评分系统测试。"""

    def test_grade_blind_separation(self):
        from evaluator import ModelEvaluator
        e = ModelEvaluator()
        dev_metrics = [{"total_return": 0.1, "excess_vs_benchmark": 0.05}]
        blind_metrics = [{"total_return": 0.05, "excess_vs_benchmark": 0.02}]
        report = e.report(dev_metrics, blind_metrics)
        self.assertIn("dev", report)
        self.assertIn("blind", report)
        self.assertIn("oos_pct", report)
        self.assertIn("trust", report)

    def test_deflated_sharpe_uses_real_trial_count(self):
        # 2026-09-02: n_trials 默认不再硬编码 6, 自动读 experiments/ 累计登记数
        from evaluator import _resolve_n_trials
        import os
        exp_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "experiments")
        n_files = sum(1 for f in os.listdir(exp_dir)
                      if f.startswith("exp_") and f.endswith(".json"))
        auto = _resolve_n_trials(None)
        self.assertGreaterEqual(auto, 1)
        self.assertEqual(auto, max(1, n_files))
        self.assertEqual(_resolve_n_trials(0), 1)    # 下限 1
        self.assertEqual(_resolve_n_trials(30), 30)  # 显式覆盖

    def test_deflated_sharpe_discounts_with_more_trials(self):
        # 试验次数越多, DSR 越保守 (单调不增); n_trials=1 无多重测试惩罚
        from evaluator import _deflated_sharpe
        d1 = _deflated_sharpe(0.4, 60, 0.0, 3.0, n_trials=1)
        d6 = _deflated_sharpe(0.4, 60, 0.0, 3.0, n_trials=6)
        d48 = _deflated_sharpe(0.4, 60, 0.0, 3.0, n_trials=48)
        self.assertGreaterEqual(d1, d6)
        self.assertGreaterEqual(d6, d48)
        self.assertGreater(d6, d48)  # 短窗口+中等SR 下折扣应可观测
        self.assertEqual(_deflated_sharpe(-0.1, 60, 0.0, 3.0), 0.0)

    def test_analyze_window_reports_n_trials(self):
        # 评测输出带 deflated_sharpe_n_trials, 显式 n_trials 原样透传
        import numpy as np
        from evaluator import ModelEvaluator
        rng = np.random.RandomState(0)
        rets = rng.normal(0.001, 0.01, 300)
        bench = rng.normal(0.0002, 0.008, 300)
        eq = np.cumprod(1 + rets) * 100000
        wm = ModelEvaluator().analyze_window(
            eq, rets, benchmark_returns=bench, n_trials=10)
        self.assertEqual(wm["deflated_sharpe_n_trials"], 10)
        self.assertIn("deflated_sharpe", wm)

    def test_score_metric(self):
        from evaluator import _score_metric
        # 值 >= great → 1.0
        self.assertEqual(_score_metric(0.25, {"min": 0.05, "good": 0.12, "great": 0.20}), 1.0)
        # 值 >= good → 0.7~1.0
        s = _score_metric(0.12, {"min": 0.05, "good": 0.12, "great": 0.20})
        self.assertAlmostEqual(s, 0.7, places=1)


class TestConfig(unittest.TestCase):
    """配置文件测试。"""

    def test_config_loadable(self):
        import yaml
        with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        self.assertIn("data_partition", cfg)
        self.assertIn("universe", cfg)
        self.assertIn("rolling", cfg)
        self.assertIn("execution", cfg)
        self.assertIn("model", cfg)
        # 关键参数存在
        self.assertGreater(cfg["rolling"]["embargo_days"], 0)
        self.assertEqual(cfg["execution"]["signal_delay"], 1)
        # 执行价: pov 进生产 (v24e 2026-08-12: 成交量比例拆单, 2022+分钟数据;
        # 之前 vwap 进生产 2026-08-12, open 更早)
        self.assertIn(cfg["execution"]["execution_price"], ["open", "vwap", "pov"])
        self.assertIn(cfg["execution"]["vwap_residual_bps"], [0, 10])


if __name__ == "__main__":
    unittest.main()
