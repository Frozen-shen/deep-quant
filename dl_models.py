"""
深度学习模型接口 — LSTM/Transformer 预留 (参考 Qlib contrib/model/)

当前状态: 接口定义 + 随机数据验证
后续: 接入PyTorch + 真实因子数据训练

用法:
  from dl_models import LSTMPredictor
  model = LSTMPredictor(input_dim=14, hidden_dim=64)
  # model.fit(dataset)  # TODO
  # pred = model.predict(dataset)  # TODO
"""

import numpy as np


class BaseDLModel:
    """深度学习模型基类 (参考 Qlib model/base.py)。"""

    def fit(self, dataset):
        raise NotImplementedError("子类实现")

    def predict(self, dataset):
        raise NotImplementedError("子类实现")

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError


class LSTMPredictor(BaseDLModel):
    """
    LSTM 时序预测器。

    输入: (n_samples, seq_len, n_features) 因子序列
    输出: (n_samples,) 预测分数

    TODO: 接入 PyTorch
    """

    def __init__(self, input_dim: int = 14, hidden_dim: int = 64,
                 num_layers: int = 2, seq_len: int = 20):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.model = None  # torch.nn.Module

    def fit(self, dataset):
        """训练 (TODO: 实现)。"""
        print("[LSTM] fit() 待实现 — 需要 PyTorch")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测 (TODO: 实现)。"""
        print("[LSTM] predict() 待实现 — 返回随机分数")
        return np.random.randn(len(X))

    def save(self, path: str):
        print(f"[LSTM] save() 待实现 — {path}")

    def load(self, path: str):
        print(f"[LSTM] load() 待实现 — {path}")


class TransformerPredictor(BaseDLModel):
    """
    Transformer 预测器 (参考 Qlib HIST/Transformer)。

    TODO: 接入 PyTorch
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, seq_len: int = 20):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.model = None

    def fit(self, dataset):
        print("[Transformer] fit() 待实现 — 需要 PyTorch")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        print("[Transformer] predict() 待实现 — 返回随机分数")
        return np.random.randn(len(X))

    def save(self, path: str):
        print(f"[Transformer] save() 待实现 — {path}")

    def load(self, path: str):
        print(f"[Transformer] load() 待实现 — {path}")


def demo():
    """演示: 接口可用性验证。"""
    print("DL Models 演示...")

    lstm = LSTMPredictor(input_dim=14, hidden_dim=64)
    X = np.random.randn(100, 20, 14)
    pred = lstm.predict(X)
    print(f"  LSTM pred: {pred[:5].round(3)}")

    trans = TransformerPredictor()
    pred2 = trans.predict(X[:, :, :14])
    print(f"  Transformer pred: {pred2[:5].round(3)}")

    print("✅ DL Models 接口正常 (待PyTorch实现)")


if __name__ == "__main__":
    demo()
