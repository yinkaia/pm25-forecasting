from pathlib import Path
import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from dataset import get_dataloaders
from models import MLPRegressor, RNNRegressor, GRURegressor, LSTMRegressor, count_parameters


CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")


def set_seed(seed=42):
    """
    固定随机种子，尽量让每次训练结果可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(model_name, seq_len, input_dim, hidden_dim, num_layers, dropout):
    """
    根据 model_name 创建对应模型。
    """
    if model_name == "mlp":
        model = MLPRegressor(
            seq_len=seq_len,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    elif model_name == "rnn":
        model = RNNRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    
    elif model_name == "gru":
        model = GRURegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    elif model_name == "lstm":
        model = LSTMRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    return model


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
    max_grad_norm=None,
):
    """
    训练一个 epoch。
    """
    model.train()

    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(data_loader, desc="Train", leave=False)

    for x_batch, y_batch in progress_bar:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # 1. 前向传播
        y_pred = model(x_batch)

        # 2. 计算 loss
        loss = criterion(y_pred, y_batch)

        # 3. 清空上一轮梯度
        optimizer.zero_grad()

        # 4. 反向传播
        loss.backward()

        # 5. 梯度裁剪
        # RNN 训练时有时会出现梯度爆炸，裁剪可以让训练更稳定
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # 6. 更新参数
        optimizer.step()

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_samples
    return avg_loss


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    """
    在验证集或测试集上评估模型。
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    return avg_loss


def plot_loss(train_losses, val_losses, save_path, model_name):
    """
    画训练集和验证集 loss 曲线。
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"{model_name.upper()} Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "rnn", "gru", "lstm"],
        help="选择要训练的模型",
    )

    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================
    # 1. 设备
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("训练配置")
    print("=" * 80)
    print("model:", args.model)
    print("device:", device)
    print("seq_len:", args.seq_len)
    print("horizon:", args.horizon)
    print("batch_size:", args.batch_size)
    print("hidden_dim:", args.hidden_dim)
    print("num_layers:", args.num_layers)
    print("dropout:", args.dropout)
    print("lr:", args.lr)
    print("num_epochs:", args.num_epochs)
    print("max_grad_norm:", args.max_grad_norm)

    # =========================
    # 2. DataLoader
    # =========================
    train_loader, val_loader, test_loader = get_dataloaders(
        seq_len=args.seq_len,
        horizon=args.horizon,
        batch_size=args.batch_size,
    )

    x_batch, y_batch = next(iter(train_loader))
    input_dim = x_batch.shape[-1]

    print("\n" + "=" * 80)
    print("数据形状")
    print("=" * 80)
    print("x_batch.shape:", x_batch.shape)
    print("y_batch.shape:", y_batch.shape)
    print("input_dim:", input_dim)

    # =========================
    # 3. 模型
    # =========================
    model = build_model(
        model_name=args.model,
        seq_len=args.seq_len,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    print("\n" + "=" * 80)
    print("模型信息")
    print("=" * 80)
    print(model)
    print("可训练参数数量:", count_parameters(model))

    # =========================
    # 4. Loss 和优化器
    # =========================
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # =========================
    # 5. 训练循环
    # =========================
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    best_model_path = CHECKPOINT_DIR / f"{args.model}_best.pth"

    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)

    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_grad_norm=args.max_grad_norm,
        )

        val_loss = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch:03d}/{args.num_epochs}] "
            f"train_loss: {train_loss:.6f} "
            f"val_loss: {val_loss:.6f}"
        )

        # 保存验证集上表现最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "seq_len": args.seq_len,
                    "horizon": args.horizon,
                    "input_dim": input_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "best_val_loss": best_val_loss,
                },
                best_model_path,
            )

            print(
                f"  保存最佳模型: {best_model_path}, "
                f"best_val_loss: {best_val_loss:.6f}"
            )

    # =========================
    # 6. 用最佳模型在 test 上评估一次
    # =========================
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\n" + "=" * 80)
    print("训练完成")
    print("=" * 80)
    print("best_val_loss:", best_val_loss)
    print("test_loss:", test_loss)

    # =========================
    # 7. 保存 loss 曲线
    # =========================
    loss_curve_path = RESULTS_DIR / f"{args.model}_loss.png"
    plot_loss(
        train_losses=train_losses,
        val_losses=val_losses,
        save_path=loss_curve_path,
        model_name=args.model,
    )

    print("\n结果文件:")
    print("最佳模型:", best_model_path)
    print("loss 曲线:", loss_curve_path)


if __name__ == "__main__":
    main()