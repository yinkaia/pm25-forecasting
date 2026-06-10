import pandas as pd


def main():
    # 1. 读取原始数据
    path = "data/raw/beijing_pm25.csv"
    df = pd.read_csv(path)

    print("=" * 80)
    print("1. 数据形状")
    print("=" * 80)
    print(df.shape)

    print("\n" + "=" * 80)
    print("2. 列名")
    print("=" * 80)
    print(df.columns.tolist())

    print("\n" + "=" * 80)
    print("3. 前 5 行")
    print("=" * 80)
    print(df.head())

    print("\n" + "=" * 80)
    print("4. 后 5 行")
    print("=" * 80)
    print(df.tail())

    print("\n" + "=" * 80)
    print("5. 数据类型")
    print("=" * 80)
    print(df.dtypes)

    print("\n" + "=" * 80)
    print("6. 每一列缺失值数量")
    print("=" * 80)
    print(df.isna().sum())

    print("\n" + "=" * 80)
    print("7. pm2.5 基本统计")
    print("=" * 80)
    print(df["pm2.5"].describe())

    print("\n" + "=" * 80)
    print("8. 风向 cbwd 的类别统计")
    print("=" * 80)
    print(df["cbwd"].value_counts())

    print("\n" + "=" * 80)
    print("9. 时间范围")
    print("=" * 80)

    # 把 year/month/day/hour 合成一个真正的时间列
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])

    print("最早时间:", df["datetime"].min())
    print("最晚时间:", df["datetime"].max())

    print("\n" + "=" * 80)
    print("10. 检查时间间隔")
    print("=" * 80)

    # 计算相邻两行之间相差多久
    time_diff = df["datetime"].diff()

    print(time_diff.value_counts().head())


if __name__ == "__main__":
    main()