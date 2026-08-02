"""
Model layer for quant-starter.

The PRIMARY model is the IC-weighted linear factor combiner, which has no
learned parameters and therefore cannot overfit. Machine-learning models
(here, an extremely regularized LightGBM ranker) are retained only as a
fallback / comparison, because backtesting showed they underperform equal
weight once transaction costs are accounted for.

Recommended starting point::

    from quant.model import ICWeightedLinear, WalkForwardTrainer

    model = ICWeightedLinear(ic_lookback=120, min_ic=0.02, decay_halflife=60)
    trainer = WalkForwardTrainer(model, {"train_days": 504, "test_days": 63})
    result = trainer.run(factor_panels, returns, symbols)
    print(result.metrics)

To compare against the ML fallback::

    from quant.model import LightGBMRanker, SimpleEnsemble

    ranker = LightGBMRanker()                       # extremely regularized
    ensemble = SimpleEnsemble([model, ranker], weights=[0.8, 0.2])
"""

from quant.model.linear import ICWeightedLinear, RankICCalculator
from quant.model.ranker import LightGBMRanker, DEFAULT_PARAMS
from quant.model.ensemble import SimpleEnsemble
from quant.model.trainer import WalkForwardTrainer, WFOResult

__all__ = [
    "ICWeightedLinear",
    "RankICCalculator",
    "LightGBMRanker",
    "DEFAULT_PARAMS",
    "SimpleEnsemble",
    "WalkForwardTrainer",
    "WFOResult",
]
