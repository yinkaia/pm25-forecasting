from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dataset import get_dataloaders


PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")


def inverse_transform_target(values, scaler, target_index):
    """
    把标准化后的 pm2.5 还原成真实 PM2.5。
    """
    mean = scaler.mean_[target_index]
    scale = scaler.scale_[target_index]

    return values * scale + mean


def compute_metrics(targets, preds):
    """
    计算回归任务常用指标。
    """
    mae = mean_absolute_error(targets, preds)
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)

    abs_errors = np.abs(preds - targets)

    within_10 = np.mean(abs_errors <= 10) * 100
    within_20 = np.mean(abs_errors <= 20) * 100
    within_30 = np.mean(abs_errors <= 30) * 100

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "within_10": within_10,
        "within_20": within_20,
        "within_30": within_30,
    }


def evaluate_naive_baseline(seq_len=24, horizon=1, batch_size=64):
    """
    Naive baseline:
    直接用输入窗口最后一小时的 PM2.5 预测下一小时 PM2.5。
    """
    _, _, test_loader = get_dataloaders(
        seq_len=seq_len,
        horizon=horizon,
        batch_size=batch_size,
    )

    scaler = joblib.load(PROCESSED_DIR / "scaler.pkl")

    with open(PROCESSED_DIR / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    target_index = config["target_index"]

    all_preds_scaled = []
    all_targets_scaled = []

    for x_batch, y_batch in test_loader:
        # x_batch: [batch_size, seq_len, feature_dim]
        # 取最后一个时间步的 pm2.5
        last_pm25 = x_batch[:, -1, target_index]

        # 变成 [batch_size, 1]
        y_pred = last_pm25.unsqueeze(1)

        all_preds_scaled.append(y_pred.numpy())
        all_targets_scaled.append(y_batch.numpy())

    preds_scaled = np.concatenate(all_preds_scaled, axis=0)
    targets_scaled = np.concatenate(all_targets_scaled, axis=0)

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

    return compute_metrics(targets, preds)


def evaluate_saved_model(model_name):
    """
    从 results/{model_name}_test_predictions.csv 读取预测结果，
    然后计算指标。
    """
    csv_path = RESULTS_DIR / f"{model_name}_test_predictions.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"找不到文件: {csv_path}\n"
            f"请先运行: python evaluate.py --model {model_name}"
        )

    df = pd.read_csv(csv_path)

    targets = df["target_pm25"].values
    preds = df["pred_pm25"].values

    return compute_metrics(targets, preds)


def plot_bar(df, metric, save_path):
    """
    画模型指标对比柱状图。
    """
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df[metric])
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.title(f"Model Comparison - {metric}")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_names = ["naive", "mlp", "rnn", "gru", "lstm"]

    records = []

    for model_name in model_names:
        print("=" * 80)
        print(f"计算 {model_name} 指标")
        print("=" * 80)

        if model_name == "naive":
            metrics = evaluate_naive_baseline(
                seq_len=24,
                horizon=1,
                batch_size=64,
            )
        else:
            metrics = evaluate_saved_model(model_name)

        record = {
            "model": model_name,
            **metrics,
        }

        records.append(record)

        print(f"MAE:       {metrics['MAE']:.4f}")
        print(f"RMSE:      {metrics['RMSE']:.4f}")
        print(f"within_10: {metrics['within_10']:.2f}%")
        print(f"within_20: {metrics['within_20']:.2f}%")
        print(f"within_30: {metrics['within_30']:.2f}%")

    comparison_df = pd.DataFrame(records)

    # 按 RMSE 从小到大排序，方便看谁整体更好
    comparison_df = comparison_df.sort_values("RMSE").reset_index(drop=True)

    print("\n" + "=" * 80)
    print("模型对比结果，按 RMSE 从小到大排序")
    print("=" * 80)
    print(comparison_df)

    # 保存 CSV
    csv_path = RESULTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)

    # 保存柱状图
    mae_plot_path = RESULTS_DIR / "model_mae_comparison.png"
    rmse_plot_path = RESULTS_DIR / "model_rmse_comparison.png"
    within10_plot_path = RESULTS_DIR / "model_within10_comparison.png"

    plot_bar(comparison_df, "MAE", mae_plot_path)
    plot_bar(comparison_df, "RMSE", rmse_plot_path)
    plot_bar(comparison_df, "within_10", within10_plot_path)

    print("\n" + "=" * 80)
    print("结果保存完成")
    print("=" * 80)
    print("模型对比 CSV:", csv_path)
    print("MAE 对比图:", mae_plot_path)
    print("RMSE 对比图:", rmse_plot_path)
    print("within_10 对比图:", within10_plot_path)


if __name__ == "__main__":
    main()