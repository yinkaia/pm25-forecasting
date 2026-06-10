from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dataset import get_dataloaders


PROCESSED_DIR = Path("data/processed")


def inverse_transform_target(values, scaler, target_index):
    """
    把标准化后的 pm2.5 还原成真实 PM2.5。
    """
    mean = scaler.mean_[target_index]
    scale = scaler.scale_[target_index]

    return values * scale + mean


def main():
    seq_len = 24
    horizon = 1
    batch_size = 64

    # 1. 加载测试集
    _, _, test_loader = get_dataloaders(
        seq_len=seq_len,
        horizon=horizon,
        batch_size=batch_size,
    )

    # 2. 加载 scaler 和 config
    scaler = joblib.load(PROCESSED_DIR / "scaler.pkl")

    with open(PROCESSED_DIR / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    target_index = config["target_index"]

    all_preds_scaled = []
    all_targets_scaled = []

    for x_batch, y_batch in test_loader:
        # x_batch shape: [batch_size, seq_len, feature_dim]
        # target_index 对应 pm2.5 这一列
        #
        # naive baseline:
        # 直接用输入窗口最后一个小时的 pm2.5 作为预测值
        last_pm25 = x_batch[:, -1, target_index]

        # 变成 [batch_size, 1]，和 y_batch 保持一致
        y_pred = last_pm25.unsqueeze(1)

        all_preds_scaled.append(y_pred.numpy())
        all_targets_scaled.append(y_batch.numpy())

    preds_scaled = np.concatenate(all_preds_scaled, axis=0)
    targets_scaled = np.concatenate(all_targets_scaled, axis=0)

    # 3. 反标准化
    preds = inverse_transform_target(
        values=preds_scaled,
        scaler=scaler,
        target_index=target_index,
    ).reshape(-1)

    targets = inverse_transform_target(
        values=targets_scaled,
        scaler=scaler,
        target_index=target_index,
    ).reshape(-1)

    # 4. 计算指标
    mae = mean_absolute_error(targets, preds)
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)

    abs_errors = np.abs(preds - targets)

    within_10 = np.mean(abs_errors <= 10) * 100
    within_20 = np.mean(abs_errors <= 20) * 100
    within_30 = np.mean(abs_errors <= 30) * 100

    print("=" * 80)
    print("Naive Baseline 评估结果")
    print("=" * 80)
    print("预测方式: 直接用输入窗口最后一小时 PM2.5 预测下一小时 PM2.5")
    print(f"MAE:  {mae:.4f} μg/m³")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f} μg/m³")

    print("\n误差容忍范围内的命中率:")
    print(f"误差 <= 10 μg/m³: {within_10:.2f}%")
    print(f"误差 <= 20 μg/m³: {within_20:.2f}%")
    print(f"误差 <= 30 μg/m³: {within_30:.2f}%")

    print("\n前 10 个真实值:")
    print(targets[:10])

    print("\n前 10 个预测值:")
    print(preds[:10])


if __name__ == "__main__":
    main()