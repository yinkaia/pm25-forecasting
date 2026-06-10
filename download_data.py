from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


def main():
    # 1. 从 UCI 下载 Beijing PM2.5 数据集
    # id=381 表示 UCI 里的 Beijing PM2.5 数据集
    beijing_pm25 = fetch_ucirepo(id=381)

    # 2. features 是输入特征
    X = beijing_pm25.data.features

    # 3. targets 是预测目标，这里就是 pm2.5
    y = beijing_pm25.data.targets

    # 4. 把输入特征和预测目标拼成一张完整表
    df = pd.concat([X, y], axis=1)

    # 5. 创建保存目录
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 6. 保存为 CSV 文件
    output_path = output_dir / "beijing_pm25.csv"
    df.to_csv(output_path, index=False)

    # 7. 打印信息，方便确认
    print("数据下载完成")
    print(f"保存路径: {output_path}")
    print(f"数据形状: {df.shape}")

    print("\n前 5 行数据:")
    print(df.head())

    print("\n列名:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()