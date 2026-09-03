"""model/pipeline.py 研究管道单测 — 配置键路径修复 + 30档粗排标签统一 (2026-09-03)

覆盖本方案 P1 三件事:
1. `_train_model()` 在当前真实 config.yaml 结构下, l0/l1/l2/ensemble/lgb
   各分支都能正确构造模型 (不实际训练; lgb 分支参数须来自 model.research_lgb
   嵌套段, 而非已不存在的顶层扁平键)。
2. `model.type == "linear"`(生产类型)时给出信息清晰的 ValueError, 而非落入
   else 分支读缺失键抛 KeyError。
3. 标签编码统一为 30 档粗排 [0, 29]: pipeline `_build_cs()` 与
   `ml_ranker.coarse_rank_labels()` 同一实现; 最大秩不再越界产出第 30 档
   (旧实现 np.floor(rank/N*30) 在最大秩处会得到 30)。
"""
import copy
import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import baselines  # noqa: E402
from model import ensemble as ensemble_mod  # noqa: E402
from ml_ranker import MLRanker, coarse_rank_labels  # noqa: E402

from model.pipeline import QuantPipeline  # noqa: E402


@pytest.fixture()
def real_config():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture()
def sample_xyg():
    """60 样本 × 4 因子 + 30档标签 + 5 个日期组 (不触发真实训练)。"""
    rng = np.random.RandomState(0)
    X = rng.randn(60, 4)
    y = coarse_rank_labels(rng.randn(60))
    g = [f"2025-01-{(i % 5) + 1:02d}" for i in range(60)]  # 5 个交易日分组
    return X, y, g, pd.Timestamp("2025-06-30")


def _noop_fit(self, X, y, groups=None, val_ratio=None, sample_weight=None):
    self._fit_called = True
    return self


def test_train_model_l0_l1_l2(real_config, sample_xyg):
    X, y, g, train_end = sample_xyg
    for t, expected in [("l0", baselines.L0EqualWeight),
                        ("l1", baselines.L1SingleFactor),
                        ("l2", baselines.L2LinearRanker)]:
        cfg = copy.deepcopy(real_config)
        cfg["model"]["type"] = t
        p = QuantPipeline(cfg, mode="dev")
        m = p._train_model(X, y, g, train_end)  # l0/l1/l2 真 fit 也是微秒级
        assert isinstance(m, expected), f"{t}: {type(m)}"
    # l2 ridge alpha 默认 1.0 (config 无 ridge_alpha 键)
    cfg = copy.deepcopy(real_config)
    cfg["model"]["type"] = "l2"
    p = QuantPipeline(cfg, mode="dev")
    m = p._train_model(X, y, g, train_end)
    assert m.model.alpha == 1.0


def test_train_model_lgb_reads_nested_research_lgb(real_config, sample_xyg, monkeypatch):
    """else 分支 (lgb) 必须读 model.research_lgb 嵌套键 — 修复 KeyError 崩溃。"""
    X, y, g, train_end = sample_xyg
    monkeypatch.setattr(MLRanker, "fit", _noop_fit)
    cfg = copy.deepcopy(real_config)
    cfg["model"]["type"] = "lgb"
    p = QuantPipeline(cfg, mode="dev")
    m = p._train_model(X, y, g, train_end)
    assert isinstance(m, MLRanker)
    rl = cfg["model"]["research_lgb"]
    assert (m.n_estimators, m.max_depth, m.learning_rate,
            m.lambda_l1, m.min_data_in_leaf) == (
        rl["n_estimators"], rl["max_depth"], rl["learning_rate"],
        rl["lambda_l1"], rl["min_data_in_leaf"])


def test_train_model_ensemble(real_config, sample_xyg, monkeypatch):
    X, y, g, train_end = sample_xyg
    monkeypatch.setattr(ensemble_mod.EnsembleRanker, "fit", _noop_fit)
    cfg = copy.deepcopy(real_config)
    cfg["model"]["type"] = "ensemble"
    p = QuantPipeline(cfg, mode="dev")
    m = p._train_model(X, y, g, train_end)
    assert isinstance(m, ensemble_mod.EnsembleRanker)
    assert m.members_config == ["lgb", "lgb", "lgb"]
    assert m.blend_method == "equal"
    assert m._fit_called


def test_train_model_linear_raises_clear_error(real_config, sample_xyg):
    """linear 是生产路径类型 → 明确报错, 绝不静默落进 else 分支抛 KeyError。"""
    X, y, g, train_end = sample_xyg
    cfg = copy.deepcopy(real_config)
    assert cfg["model"]["type"] == "linear"  # 当前 config.yaml 的真实值
    p = QuantPipeline(cfg, mode="dev")
    with pytest.raises(ValueError, match="生产路径"):
        p._train_model(X, y, g, train_end)


# ── 标签口径: coarse_rank_labels 边界 ──

def test_coarse_rank_labels_range_and_boundary():
    rng = np.random.RandomState(1)
    v = rng.randn(1000)
    lab = coarse_rank_labels(v)
    assert lab.min() == 0 and lab.max() == 29  # 最大秩不越界为 30
    assert len(lab) == 1000

    # 并列值 → 同档 (平均秩); 档位随值严格不降
    t = coarse_rank_labels(np.array([1.0, 1.0, 2.0]))
    assert t[0] == t[1] and t[2] > t[0]
    assert t.min() >= 0 and t.max() <= 29

    # 单元素截面边界: 不抛错, 档位仍在 [0, 29]
    assert int(coarse_rank_labels(np.array([5.0]))[0]) in range(30)
    assert coarse_rank_labels(np.array([])).size == 0


def test_build_cs_labels_are_30_buckets(real_config):
    """_build_cs 产出的标签与 ml_ranker.coarse_rank_labels 同口径 (0~29 档)。"""
    cfg = copy.deepcopy(real_config)
    cfg["model"]["type"] = "lgb"
    p = QuantPipeline(cfg, mode="dev")

    dates = pd.date_range("2025-01-01", periods=25, freq="B")

    def mk_df(closes):
        return pd.DataFrame({"date": dates, "close": np.asarray(closes, dtype=float)})

    # 22 只股票 (>top_k=20 才过截面门槛), 未来 20 日收益随序号严格递增
    base = np.linspace(10.0, 20.0, 25)
    fwd = {}
    for i in range(22):
        slope = 1.0 + 0.10 * (i - 10) / 10.0  # 0.9 ~ 1.1, 决定 20 日涨幅排序
        closes = 10.0 * slope ** np.linspace(0, 1, 25)
        p._all_data[f"S{i:02d}"] = mk_df(closes)
        fwd[f"S{i:02d}"] = closes[-1] / closes[4] - 1.0  # idx4 + 20 日
    p._fund_cache = {}

    class StubCache:
        def get_features(self, sym, today):
            i = int(sym[1:])
            return np.array([i / 22.0, -i / 22.0])

    p._factor_cache = StubCache()
    today = dates[4]  # idx 4, ip+horizon(20) = 24 < 25 ✓

    fn, labels, syms = p._build_cs(today)
    assert len(syms) == 22
    assert labels.min() >= 0 and labels.max() <= 29  # 旧实现此处可能出 30
    rank = {s: fwd[s] for s in syms}
    order = sorted(syms, key=lambda s: rank[s])
    # 未来收益越高的股票, 档位标签不更低
    assert labels[list(syms).index(order[-1])] >= labels[list(syms).index(order[0])]
