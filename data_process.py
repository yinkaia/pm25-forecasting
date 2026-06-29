from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler    # 标准化数据的工具


RAW_DATA_PATH = Path("data/raw/beijing_pm25.csv")
OUTPUT_DIR = Path("data/processed")

TARGET_COL = "pm2.5"


def main():
    # 1. 读取原始数据
    df = pd.read_csv(RAW_DATA_PATH)

    print("=" * 80)
    print("1. 读取原始数据")
    print("=" * 80)
    print("原始数据形状:", df.shape)
    print("原始缺失值:")
    print(df.isna().sum())

    # 2. 构造 datetime 时间列
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])

    # 3. 按时间排序，确保时间顺序正确
    df = df.sort_values("datetime").reset_index(drop=True)

    print("\n" + "=" * 80)
    print("2. 时间范围")
    print("=" * 80)
    print("最早时间:", df["datetime"].min())
    print("最晚时间:", df["datetime"].max())

    # 4. 处理 pm2.5 缺失值
    # 对时间序列来说，我们优先保留完整时间轴，所以这里使用线性插值
    before_missing = df[TARGET_COL].isna().sum()

    df[TARGET_COL] = df[TARGET_COL].interpolate(method="linear")

    # 如果开头或结尾仍然有缺失值，用后一个/前一个有效值补齐
    df[TARGET_COL] = df[TARGET_COL].bfill().ffill()

    after_missing = df[TARGET_COL].isna().sum()

    print("\n" + "=" * 80)
    print("3. pm2.5 缺失值处理")
    print("=" * 80)
    print("处理前缺失数量:", before_missing)
    print("处理后缺失数量:", after_missing)

    # 5. 对 cbwd 风向做 one-hot 编码
    # 把 cbwd 风向，从 cbwd 列变成 cbwd_NE, cbwd_NW, cbwd_SE, cbwd_cv 四列
    df = pd.get_dummies(df, columns=["cbwd"], prefix="cbwd", dtype=int)

    print("\n" + "=" * 80)
    print("4. one-hot 编码后的列名")
    print("=" * 80)
    print(df.columns.tolist())

    # 6. 选择模型输入特征
    # 注意：这里包含 pm2.5 本身，因为我们要用过去的 pm2.5 预测未来的 pm2.5
    feature_cols = [
        "pm2.5",
        "DEWP",
        "TEMP",
        "PRES",
        "Iws",
        "Is",
        "Ir",
        "month",
        "day",
        "hour",
        "cbwd_NE",
        "cbwd_NW",
        "cbwd_SE",
        "cbwd_cv",
    ]

    print("\n" + "=" * 80)
    print("5. 模型输入特征")
    print("=" * 80)
    print(feature_cols)
    print("特征数量:", len(feature_cols))

    # 7. 检查是否还有缺失值
    print("\n" + "=" * 80)
    print("6. 处理后缺失值检查")
    print("=" * 80)
    print(df[feature_cols].isna().sum())

    # 8. 按时间顺序切分 train / val / test
    total_len = len(df)
    train_end = int(total_len * 0.7)
    val_end = int(total_len * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print("\n" + "=" * 80)
    print("7. 数据集切分")
    print("=" * 80)
    print("train:", train_df.shape, train_df["datetime"].min(), "->", train_df["datetime"].max())
    print("val:  ", val_df.shape, val_df["datetime"].min(), "->", val_df["datetime"].max())
    print("test: ", test_df.shape, test_df["datetime"].min(), "->", test_df["datetime"].max())

    # 9. 标准化
    # 只在训练集上 fit，验证集和测试集只 transform
    # 创建一个标准化器
    scaler = StandardScaler()   # 标准化值 = (原始值 - 训练集均值) / 训练集标准差

    # fit：计算训练集均值和标准差
    # transform：对训练集、验证集和测试集做标准化
    # fit_transform：先 fit 再 transform
    # 只用在训练集上 fit，验证集和测试集只 transform
    # 因为验证集和测试集的均值和标准差，应该用训练集的均值和标准差
    train_scaled_values = scaler.fit_transform(train_df[feature_cols])
    val_scaled_values = scaler.transform(val_df[feature_cols])
    test_scaled_values = scaler.transform(test_df[feature_cols])

    # 把标准化后的值，转换成 DataFrame
    # 把标准化后的数组重新包装成表格，并且把列名加回来
    train_scaled_df = pd.DataFrame(train_scaled_values, columns=feature_cols)
    val_scaled_df = pd.DataFrame(val_scaled_values, columns=feature_cols)
    test_scaled_df = pd.DataFrame(test_scaled_values, columns=feature_cols)

    # 保留 datetime，后面画图会用到
    train_scaled_df["datetime"] = train_df["datetime"].values
    val_scaled_df["datetime"] = val_df["datetime"].values
    test_scaled_df["datetime"] = test_df["datetime"].values

    # 10. 保存处理结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_scaled_df.to_csv(OUTPUT_DIR / "train_scaled.csv", index=False)
    val_scaled_df.to_csv(OUTPUT_DIR / "val_scaled.csv", index=False)
    test_scaled_df.to_csv(OUTPUT_DIR / "test_scaled.csv", index=False)

    # 保存一份未标准化但已经清洗和编码后的完整数据，方便后面查看
    df.to_csv(OUTPUT_DIR / "clean_data.csv", index=False)

    # 保存 scaler，后面反归一化预测结果时要用
    joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")

    # 保存特征列配置
    config = {
        "target_col": TARGET_COL,
        "feature_cols": feature_cols,
        "target_index": feature_cols.index(TARGET_COL),
    }

    with open(OUTPUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("8. 保存完成")
    print("=" * 80)
    print("保存目录:", OUTPUT_DIR)
    print("train_scaled.csv")
    print("val_scaled.csv")
    print("test_scaled.csv")
    print("clean_data.csv")
    print("scaler.pkl")
    print("config.json")

    print("\n标准化后训练集前 5 行:")
    print(train_scaled_df.head())


if __name__ == "__main__":
    main()