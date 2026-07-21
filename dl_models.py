"""
深度学习模型 — LSTM/Transformer 实装 (PyTorch)

参考 Qlib contrib/model/ 的 PyTorch 模型模式:
  - fit(dataset): 训练
  - predict(dataset): 预测 → pd.Series

用法:
  from dl_models import LSTMPredictor
  model = LSTMPredictor(input_dim=14, hidden_dim=64)
  model.fit(X_train, y_train)
  pred = model.predict(X_test)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ================================================================
#  LSTM 预测器
# ================================================================

class LSTMModel(nn.Module):
    """单层LSTM + 全连接输出。"""

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, (h, c) = self.lstm(x)
        return self.fc(out[:, -1, :])  # 最后一步的输出


class LSTMPredictor:
    """
    LSTM 时序预测器。

    参数:
      input_dim: 因子数量
      hidden_dim: 隐藏层维度
      num_layers: LSTM层数
      seq_len: 回看天数
      lr: 学习率
    """

    def __init__(self, input_dim: int = 14, hidden_dim: int = 64,
                 num_layers: int = 2, seq_len: int = 20,
                 lr: float = 1e-3, epochs: int = 50):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.lr = lr
        self.epochs = epochs
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        将 (n_samples, n_features) 转为 (n_sequences, seq_len, n_features)。

        默认每seq_len条归为一组。
        """
        n = len(X)
        n_seq = max(1, n // self.seq_len)
        seqs = []
        for i in range(n_seq):
            start = i * self.seq_len
            end = start + self.seq_len
            seqs.append(X[start:end])
        return np.array(seqs)

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """
        训练 LSTM。

        X: (n_samples, n_features) 因子矩阵
        y: (n_samples,) 标签 (未来收益)
        """
        self.input_dim = X.shape[1]
        self.model = LSTMModel(
            self.input_dim, self.hidden_dim, self.num_layers
        ).to(self.device)

        # 转为序列
        X_seq = self._to_sequences(X)
        y_seq = np.array([y[i*self.seq_len:(i+1)*self.seq_len][-1]
                          for i in range(len(X_seq))])

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = loss_fn(pred, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if verbose and epoch % 10 == 0:
                print(f"  LSTM epoch {epoch}/{self.epochs}, loss={total_loss/len(loader):.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测。"""
        if self.model is None:
            raise ValueError("模型未训练")
        self.model.eval()
        # 用最后一个序列预测
        seq = X[-self.seq_len:] if len(X) >= self.seq_len else X
        if len(seq) < self.seq_len:
            pad = np.zeros((self.seq_len - len(seq), X.shape[1]))
            seq = np.vstack([pad, seq])
        X_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(X_t).cpu().numpy()
        return np.full(len(X), pred[0, 0])  # 广播到所有样本

    def save(self, path: str):
        if self.model:
            torch.save({"model": self.model.state_dict(), "config": {
                "input_dim": self.input_dim, "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers, "seq_len": self.seq_len,
            }}, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        cfg = data["config"]
        self.__init__(**cfg)
        self.model = LSTMModel(
            self.input_dim, self.hidden_dim, self.num_layers
        ).to(self.device)
        self.model.load_state_dict(data["model"])


# ================================================================
#  Transformer 预测器
# ================================================================

class TransformerModel(nn.Module):
    """简化版 Transformer 编码器。"""

    def __init__(self, input_dim: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


class TransformerPredictor(LSTMPredictor):
    """Transformer 预测器 (复用LSTM的fit/predict接口)。"""

    def __init__(self, input_dim: int = 14, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2,
                 seq_len: int = 20, lr: float = 1e-3, epochs: int = 50):
        # 调用父类初始化(设置 device)
        LSTMPredictor.__init__(self, input_dim, d_model, n_layers,
                               seq_len, lr, epochs)
        self.d_model = d_model
        self.n_heads = n_heads

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        self.input_dim = X.shape[1]
        self.model = TransformerModel(
            self.input_dim, self.d_model, self.n_heads, self.num_layers
        ).to(self.device)

        X_seq = self._to_sequences(X)
        y_seq = np.array([y[i*self.seq_len:(i+1)*self.seq_len][-1]
                          for i in range(len(X_seq))])

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1).to(self.device)

        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for bx, by in loader:
                opt.zero_grad()
                loss = loss_fn(self.model(bx), by)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            if verbose and epoch % 10 == 0:
                print(f"  Transformer epoch {epoch}/{self.epochs}, loss={total_loss/len(loader):.4f}")
        return self


# ================================================================
#  演示
# ================================================================

def demo():
    print("PyTorch DL Models 演示...")
    np.random.seed(42)

    # 生成模拟因子数据: 500天, 10个因子
    X = np.random.randn(500, 10) * 0.1
    y = X[:, 0] * 0.3 + X[:, 1] * 0.2 + np.random.randn(500) * 0.05

    # LSTM
    lstm = LSTMPredictor(input_dim=10, hidden_dim=32, seq_len=20, epochs=30)
    lstm.fit(X, y, verbose=False)
    pred_lstm = lstm.predict(X[:50])
    print(f"  LSTM pred (前5): {pred_lstm[:5].round(4)}")

    # Transformer
    trans = TransformerPredictor(input_dim=10, d_model=32, seq_len=20, epochs=30)
    trans.fit(X, y, verbose=False)
    pred_trans = trans.predict(X[:50])
    print(f"  Transformer pred (前5): {pred_trans[:5].round(4)}")

    print("✅ PyTorch DL Models 正常")


if __name__ == "__main__":
    demo()
