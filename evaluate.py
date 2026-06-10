from pathlib import Path
import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dataset import get_dataloaders
from models import MLPRegressor, RNNRegressor, GRURegressor, LSTMRegressor


PROCESSED_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")


def build_model_from_checkpoint(checkpoint):
    """
    根据 checkpoint 中保存的 model_name 创建对应模型。
    """
    model_name = checkpoint["model_name"]

    if model_name == "mlp":
        model = MLPRegressor(
            seq_len=checkpoint["seq_len"],
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            dropout=checkpoint["dropout"],
        )

    elif model_name == "rnn":
        model = RNNRegressor(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
        )
    elif model_name == "gru":
        model = GRURegressor(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
        )

    elif model_name == "lstm":
        model = LSTMRegressor(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
        )

    else:
        raise ValueError(f"不支持的模型: {model_name}")

    return model


@torch.no_grad()
def predict(model, data_loader, device):
    """
    使用模型在整个数据集上预测。

    返回:
        preds: 标准化后的预测值
        targets: 标准化后的真实值
    """
    model.eval()

    all_preds = []
    all_targets = []

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)

        y_pred = model(x_batch)

        all_preds.append(y_pred.cpu().numpy())
        all_targets.append(y_batch.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return preds, targets


def inverse_transform_target(values, scaler, target_index):
    """
    只对目标列 pm2.5 做反标准化。

    标准化:
        z = (x - mean) / scale

    反标准化:
        x = z * scale + mean
    """
    mean = scaler.mean_[target_index]
    scale = scaler.scale_[target_index]

    return values * scale + mean


def plot_predictions(y_true, y_pred, save_path, model_name, num_points=500):
    """
    画真实值 vs 预测值曲线。
    """
    plt.figure(figsize=(12, 5))

    plt.plot(y_true[:num_points], label="True PM2.5")
    plt.plot(y_pred[:num_points], label="Predicted PM2.5")

    plt.xlabel("Time step")
    plt.ylabel("PM2.5")
    plt.title(f"{model_name.upper()} True vs Predicted PM2.5 - First {num_points} Test Samples")
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
        help="选择要评估的模型",
    )

    parser.add_argument("--batch_size", type=int, default=64)

    return parser.parse_args()


def main():
    args = parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================
    # 1. 设备
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("评估配置")
    print("=" * 80)
    print("model:", args.model)
    print("device:", device)

    # =========================
    # 2. 加载 checkpoint
    # =========================
    checkpoint_path = CHECKPOINT_DIR / f"{args.model}_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print("\n" + "=" * 80)
    print("加载模型配置")
    print("=" * 80)
    print("checkpoint:", checkpoint_path)
    print("model_name:", checkpoint["model_name"])
    print("seq_len:", checkpoint["seq_len"])
    print("horizon:", checkpoint["horizon"])
    print("input_dim:", checkpoint["input_dim"])
    print("hidden_dim:", checkpoint["hidden_dim"])
    print("dropout:", checkpoint["dropout"])
    print("best_val_loss:", checkpoint["best_val_loss"])

    if checkpoint["model_name"] in ["rnn", "gru", "lstm"]:
        print("num_layers:", checkpoint["num_layers"]) 

    # =========================
    # 3. 构建模型并加载参数
    # =========================
    model = build_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # =========================
    # 4. 加载测试集
    # =========================
    _, _, test_loader = get_dataloaders(
        seq_len=checkpoint["seq_len"],
        horizon=checkpoint["horizon"],
        batch_size=args.batch_size,
    )

    print("\n" + "=" * 80)
    print("测试集信息")
    print("=" * 80)
    print("test 样本数:", len(test_loader.dataset))
    print("test batch 数:", len(test_loader))

    # =========================
    # 5. 模型预测
    # =========================
    preds_scaled, targets_scaled = predict(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    print("\n" + "=" * 80)
    print("标准化空间中的预测结果形状")
    print("=" * 80)
    print("preds_scaled.shape:", preds_scaled.shape)
    print("targets_scaled.shape:", targets_scaled.shape)

    # =========================
    # 6. 反标准化
    # =========================
    scaler = joblib.load(PROCESSED_DIR / "scaler.pkl")

    with open(PROCESSED_DIR / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    target_index = config["target_index"]

    preds = inverse_transform_target(
        values=preds_scaled,
        scaler=scaler,
        target_index=target_index,
    )

    targets = inverse_transform_target(
        values=targets_scaled,
        scaler=scaler,
        target_index=target_index,
    )

    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    print("\n" + "=" * 80)
    print("反标准化后的结果")
    print("=" * 80)
    print("preds.shape:", preds.shape)
    print("targets.shape:", targets.shape)

    print("\n前 10 个真实值:")
    print(targets[:10])

    print("\n前 10 个预测值:")
    print(preds[:10])

    # =========================
    # 7. 计算回归指标
    # =========================
    mae = mean_absolute_error(targets, preds)
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)

    abs_errors = np.abs(preds - targets)

    within_10 = np.mean(abs_errors <= 10) * 100
    within_20 = np.mean(abs_errors <= 20) * 100
    within_30 = np.mean(abs_errors <= 30) * 100

    print("\n" + "=" * 80)
    print(f"{args.model.upper()} 测试集评估指标，真实 PM2.5 单位")
    print("=" * 80)
    print(f"MAE:  {mae:.4f} μg/m³")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f} μg/m³")

    print("\n误差容忍范围内的命中率:")
    print(f"误差 <= 10 μg/m³: {within_10:.2f}%")
    print(f"误差 <= 20 μg/m³: {within_20:.2f}%")
    print(f"误差 <= 30 μg/m³: {within_30:.2f}%")

    # =========================
    # 8. 保存预测结果
    # =========================
    result_df = pd.DataFrame(
        {
            "target_pm25": targets,
            "pred_pm25": preds,
            "abs_error": abs_errors,
        }
    )

    result_csv_path = RESULTS_DIR / f"{args.model}_test_predictions.csv"
    result_df.to_csv(result_csv_path, index=False)

    # =========================
    # 9. 画预测曲线
    # =========================
    plot_path = RESULTS_DIR / f"{args.model}_true_vs_pred.png"
    plot_predictions(
        y_true=targets,
        y_pred=preds,
        save_path=plot_path,
        model_name=args.model,
        num_points=500,
    )

    print("\n" + "=" * 80)
    print("结果保存完成")
    print("=" * 80)
    print("预测结果 CSV:", result_csv_path)
    print("预测曲线:", plot_path)


if __name__ == "__main__":
    main()