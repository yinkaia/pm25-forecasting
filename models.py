import torch
from torch import nn

from dataset import get_dataloaders


class MLPRegressor(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim=128, dropout=0.2):
        """
        MLP 回归模型

        输入:
            x: [batch_size, seq_len, input_dim]

        处理方式:
            先展平为 [batch_size, seq_len * input_dim]
            再经过全连接层预测下一小时 PM2.5
        """
        super().__init__()

        in_features = seq_len * input_dim

        self.net = nn.Sequential(
            nn.Flatten(),

            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class RNNRegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=1,
        dropout=0.0,
    ):
        """
        RNN 回归模型

        参数:
            input_dim: 每个时间步的特征数，比如 14
            hidden_dim: RNN 隐藏状态维度
            num_layers: RNN 层数
            dropout: 多层 RNN 之间的 dropout

        输入:
            x: [batch_size, seq_len, input_dim]

        输出:
            out: [batch_size, 1]
        """
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x.shape:
            [batch_size, seq_len, input_dim]

        rnn_output.shape:
            [batch_size, seq_len, hidden_dim]

        h_n.shape:
            [num_layers, batch_size, hidden_dim]
        """
        rnn_output, h_n = self.rnn(x)

        # 取最后一层 RNN 的最后隐藏状态
        last_hidden = h_n[-1]

        # 用最后隐藏状态预测下一小时 PM2.5
        out = self.fc(last_hidden)

        return out

class GRURegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=1,
        dropout=0.0,
    ):
        """
        GRU 回归模型

        输入:
            x: [batch_size, seq_len, input_dim]

        输出:
            out: [batch_size, 1]
        """
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x.shape:
            [batch_size, seq_len, input_dim]

        gru_output.shape:
            [batch_size, seq_len, hidden_dim]

        h_n.shape:
            [num_layers, batch_size, hidden_dim]
        """
        gru_output, h_n = self.gru(x)

        # 取最后一层 GRU 的最后隐藏状态
        last_hidden = h_n[-1]

        # 用最后隐藏状态预测下一小时 PM2.5
        out = self.fc(last_hidden)

        return out


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=1,
        dropout=0.0,
    ):
        """
        LSTM 回归模型

        输入:
            x: [batch_size, seq_len, input_dim]

        输出:
            out: [batch_size, 1]
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x.shape:
            [batch_size, seq_len, input_dim]

        lstm_output.shape:
            [batch_size, seq_len, hidden_dim]

        h_n.shape:
            [num_layers, batch_size, hidden_dim]

        c_n.shape:
            [num_layers, batch_size, hidden_dim]
        """
        lstm_output, (h_n, c_n) = self.lstm(x)

        # 取最后一层 LSTM 的最后隐藏状态
        last_hidden = h_n[-1]

        # 用最后隐藏状态预测下一小时 PM2.5
        out = self.fc(last_hidden)

        return out



def count_parameters(model):
    """
    统计模型中可训练参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    seq_len = 24
    horizon = 1
    batch_size = 64

    train_loader, _, _ = get_dataloaders(
        seq_len=seq_len,
        horizon=horizon,
        batch_size=batch_size,
    )

    x_batch, y_batch = next(iter(train_loader))
    input_dim = x_batch.shape[-1]

    print("=" * 80)
    print("1. 输入数据形状")
    print("=" * 80)
    print("x_batch.shape:", x_batch.shape)
    print("y_batch.shape:", y_batch.shape)
    print("input_dim:", input_dim)

    print("\n" + "=" * 80)
    print("2. 测试 MLPRegressor")
    print("=" * 80)

    mlp = MLPRegressor(
        seq_len=seq_len,
        input_dim=input_dim,
        hidden_dim=128,
        dropout=0.2,
    )

    mlp_pred = mlp(x_batch)

    print(mlp)
    print("MLP 参数数量:", count_parameters(mlp))
    print("mlp_pred.shape:", mlp_pred.shape)

    print("\n" + "=" * 80)
    print("3. 测试 RNNRegressor")
    print("=" * 80)

    rnn = RNNRegressor(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=1,
        dropout=0.0,
    )

    rnn_pred = rnn(x_batch)

    print(rnn)
    print("RNN 参数数量:", count_parameters(rnn))
    print("rnn_pred.shape:", rnn_pred.shape)

    print("\n前 5 个 RNN 预测值:")
    print(rnn_pred[:5])

    print("\n前 5 个真实标签:")
    print(y_batch[:5])

    print("\n" + "=" * 80)
    print("4. 测试 GRURegressor")
    print("=" * 80)

    gru = GRURegressor(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=1,
        dropout=0.0,
    )

    gru_pred = gru(x_batch)

    print(gru)
    print("GRU 参数数量:", count_parameters(gru))
    print("gru_pred.shape:", gru_pred.shape)

    print("\n前 5 个 GRU 预测值:")
    print(gru_pred[:5])


    print("\n" + "=" * 80)
    print("5. 测试 LSTMRegressor")
    print("=" * 80)

    lstm = LSTMRegressor(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=1,
        dropout=0.0,
    )

    lstm_pred = lstm(x_batch)

    print(lstm)
    print("LSTM 参数数量:", count_parameters(lstm))
    print("lstm_pred.shape:", lstm_pred.shape)

    print("\n前 5 个 LSTM 预测值:")
    print(lstm_pred[:5])


if __name__ == "__main__":
    main()