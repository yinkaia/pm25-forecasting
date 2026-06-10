import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


PROCESSED_DIR = Path("data/processed")


class PM25Dataset(Dataset):
    def __init__(self, csv_path, config_path, seq_len=24, horizon=1):
        """
        PM2.5 时间序列数据集

        参数:
            csv_path: 标准化后的 csv 文件路径
            config_path: config.json 文件路径
            seq_len: 使用过去多少个小时作为输入
            horizon: 预测未来第几个小时
                     horizon=1 表示预测下一小时
                     horizon=6 表示预测未来第 6 小时
        """
        self.seq_len = seq_len
        self.horizon = horizon

        # 1. 读取配置
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.feature_cols = config["feature_cols"]
        self.target_col = config["target_col"]
        self.target_index = config["target_index"]

        # 2. 读取数据
        df = pd.read_csv(csv_path)

        # 3. 保存时间列，后面画图可能会用
        self.datetimes = df["datetime"].values

        # 4. 只取模型需要的特征列
        self.data = df[self.feature_cols].values.astype("float32")

        # 5. 计算可以构造多少个样本
        self.num_samples = len(self.data) - self.seq_len - self.horizon + 1

        if self.num_samples <= 0:
            raise ValueError(
                f"数据太短，无法构造样本: len={len(self.data)}, "
                f"seq_len={self.seq_len}, horizon={self.horizon}"
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        返回一条样本:
            X: 过去 seq_len 小时的所有特征
            y: 未来 horizon 小时的 pm2.5
        """
        # 输入窗口:
        # idx 到 idx + seq_len - 1
        x_start = idx
        x_end = idx + self.seq_len

        # 标签位置:
        # 如果 horizon=1，就是窗口后面的第 1 个小时
        # 例如 X 是 0~23，y 就是 24
        y_index = idx + self.seq_len + self.horizon - 1

        x = self.data[x_start:x_end]
        y = self.data[y_index, self.target_index]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor([y], dtype=torch.float32)

        return x, y


def get_dataloaders(seq_len=24, horizon=1, batch_size=64):
    """
    创建 train / val / test 三个 DataLoader
    """
    config_path = PROCESSED_DIR / "config.json"

    train_dataset = PM25Dataset(
        csv_path=PROCESSED_DIR / "train_scaled.csv",
        config_path=config_path,
        seq_len=seq_len,
        horizon=horizon,
    )

    val_dataset = PM25Dataset(
        csv_path=PROCESSED_DIR / "val_scaled.csv",
        config_path=config_path,
        seq_len=seq_len,
        horizon=horizon,
    )

    test_dataset = PM25Dataset(
        csv_path=PROCESSED_DIR / "test_scaled.csv",
        config_path=config_path,
        seq_len=seq_len,
        horizon=horizon,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def main():
    seq_len = 24
    horizon = 1
    batch_size = 64

    train_loader, val_loader, test_loader = get_dataloaders(
        seq_len=seq_len,
        horizon=horizon,
        batch_size=batch_size,
    )

    print("=" * 80)
    print("DataLoader 构造完成")
    print("=" * 80)

    print("train 样本数:", len(train_loader.dataset))
    print("val 样本数:  ", len(val_loader.dataset))
    print("test 样本数: ", len(test_loader.dataset))

    print("train batch 数:", len(train_loader))
    print("val batch 数:  ", len(val_loader))
    print("test batch 数: ", len(test_loader))

    x_batch, y_batch = next(iter(train_loader))

    print("\n" + "=" * 80)
    print("取一个 batch 看看形状")
    print("=" * 80)
    print("x_batch.shape:", x_batch.shape)
    print("y_batch.shape:", y_batch.shape)

    print("\n第 1 条样本的 X shape:", x_batch[0].shape)
    print("第 1 条样本的 y:", y_batch[0])


if __name__ == "__main__":
    main()