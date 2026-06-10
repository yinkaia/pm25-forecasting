# 基于 LSTM / GRU 的北京 PM2.5 时间序列预测系统

本项目使用 UCI Beijing PM2.5 数据集，基于过去 24 小时的 PM2.5 和气象特征，预测未来 1 小时的 PM2.5 浓度。

这是一个时间序列回归项目，重点练习：

```text
1. 时间序列数据处理
2. 缺失值处理
3. 类别特征 one-hot 编码
4. 标准化与反标准化
5. 滑动窗口构造序列样本
6. PyTorch Dataset / DataLoader
7. MLP / RNN / GRU / LSTM 回归建模
8. MAE / RMSE 评估
9. 真实值 vs 预测值曲线可视化
```

---

## 1. 项目目标

任务形式：

```text
输入：过去 24 小时的 PM2.5 + 气象特征
输出：未来 1 小时 PM2.5 浓度
任务类型：回归任务
模型：Naive baseline / MLP / RNN / GRU / LSTM
```

最终模型输入形状：

```text
x_batch.shape = [64, 24, 14]
y_batch.shape = [64, 1]
```

含义：

```text
64 条样本
每条样本使用过去 24 小时
每个小时有 14 个特征
预测未来 1 小时 PM2.5
```

---

## 2. 数据集

数据集：UCI Beijing PM2.5 Data

原始数据包含 2010-01-01 到 2014-12-31 的小时级 PM2.5 和气象数据。

原始字段包括：

```text
year, month, day, hour,
DEWP, TEMP, PRES, cbwd,
Iws, Is, Ir, pm2.5
```

字段含义：

```text
pm2.5：预测目标
DEWP：露点温度
TEMP：温度
PRES：气压
cbwd：风向
Iws：累计风速
Is：累计降雪
Ir：累计降雨
```

---

## 3. 项目结构

```text
pm25_forecasting/
├── data/
│   ├── raw/
│   └── processed/
├── checkpoints/
├── results/
├── download_data.py
├── explore_data.py
├── data_process.py
├── dataset.py
├── models.py
├── train.py
├── evaluate.py
├── naive_baseline.py
├── compare_models.py
├── requirements.txt
└── README.md
```

---

## 4. 环境准备

创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果需要单独安装 GPU 版 PyTorch，可以根据 CUDA 版本选择安装命令。例如 CUDA 11.8：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

检查 PyTorch 和 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## 5. 运行流程

### 5.1 下载数据

```bash
python download_data.py
```

### 5.2 探索数据

```bash
python explore_data.py
```

主要查看：

```text
1. 数据形状
2. 字段类型
3. 缺失值数量
4. pm2.5 分布
5. 风向类别
6. 时间范围和时间间隔
```

### 5.3 数据预处理

```bash
python data_process.py
```

预处理内容：

```text
1. 构造 datetime 时间列
2. 按时间排序
3. 对 pm2.5 缺失值做线性插值
4. 对 cbwd 风向做 one-hot 编码
5. 按时间顺序切分 train / val / test
6. 只使用训练集统计量做标准化
7. 保存 scaler.pkl 和 config.json
```

### 5.4 检查 Dataset

```bash
python dataset.py
```

滑动窗口构造方式：

```text
样本 1：
输入：第 1～24 小时
标签：第 25 小时 PM2.5

样本 2：
输入：第 2～25 小时
标签：第 26 小时 PM2.5
```

---

## 6. 模型训练

训练 MLP：

```bash
python train.py --model mlp
```

训练 RNN：

```bash
python train.py --model rnn
```

训练 GRU：

```bash
python train.py --model gru
```

训练 LSTM：

```bash
python train.py --model lstm
```

训练完成后，模型权重会保存到：

```text
checkpoints/
```

对应文件：

```text
mlp_best.pth
rnn_best.pth
gru_best.pth
lstm_best.pth
```

---

## 7. 模型评估

评估 MLP：

```bash
python evaluate.py --model mlp
```

评估 RNN：

```bash
python evaluate.py --model rnn
```

评估 GRU：

```bash
python evaluate.py --model gru
```

评估 LSTM：

```bash
python evaluate.py --model lstm
```

评估脚本会做：

```text
1. 加载最佳模型
2. 在测试集上预测
3. 将标准化后的预测值反标准化为真实 PM2.5
4. 计算 MAE / RMSE / MSE
5. 计算误差容忍范围内的命中率
6. 保存预测结果和预测曲线
```

---

## 8. Naive baseline

运行：

```bash
python naive_baseline.py
```

Naive baseline 的预测方式：

```text
直接用输入窗口最后一小时的 PM2.5，
作为下一小时 PM2.5 的预测值。
```

这个 baseline 很重要，因为 PM2.5 具有明显的短期连续性，上一小时 PM2.5 对下一小时预测非常强。

---

## 9. 模型对比

运行：

```bash
python compare_models.py
```

生成文件：

```text
results/model_comparison.csv
results/model_mae_comparison.png
results/model_rmse_comparison.png
results/model_within10_comparison.png
```

---

## 10. 实验结果

| Model |     MAE |    RMSE | within_10 | within_20 | within_30 |
| ----- | ------: | ------: | --------: | --------: | --------: |
| LSTM  | 10.8171 | 18.6296 |    64.49% |    86.96% |    94.05% |
| GRU   | 10.8594 | 18.7648 |    64.75% |    86.84% |    94.12% |
| RNN   | 10.8240 | 18.9772 |    64.89% |    87.02% |    93.98% |
| MLP   | 12.2280 | 19.8620 |    57.88% |    83.66% |    92.38% |
| Naive | 10.8414 | 19.9058 |    65.22% |    85.82% |    93.24% |

按 RMSE 排名：

```text
LSTM > GRU > RNN > MLP > Naive
```

---

## 11. 结果分析

实验结果说明：

```text
1. Naive baseline 在预测下一小时 PM2.5 时非常强，
   说明 PM2.5 具有明显的短期连续性。

2. MLP 没有显式建模时间顺序，
   因此整体效果不如 RNN / GRU / LSTM。

3. RNN 相比 MLP 有明显提升，
   说明按时间顺序建模是有效的。

4. GRU 和 LSTM 的 RMSE 更低，
   说明门控机制有助于减少较大预测误差。

5. 当前实验中，LSTM 的 RMSE 最低，
   综合表现最好。
```

---

## 12. 后续改进方向

后续可以继续尝试：

```text
1. 多步预测：过去 24 小时 → 未来 6 / 12 / 24 小时
2. 更长输入窗口：48 / 72 / 168 小时
3. 加入 Early Stopping
4. 调整 hidden_dim、num_layers、dropout
5. 尝试 Attention 或 Transformer
6. 分析高污染时段的预测误差
```

---

## 13. 项目总结

本项目完成了一个完整的时间序列回归流程：

```text
原始数据
↓
数据探索
↓
缺失值处理
↓
类别特征编码
↓
标准化
↓
滑动窗口构造样本
↓
Dataset / DataLoader
↓
Naive / MLP / RNN / GRU / LSTM 建模
↓
模型训练
↓
反标准化评估
↓
模型对比与结果分析
```

通过这个项目，可以更好地理解 RNN / GRU / LSTM 在时间序列预测任务中的使用方式，以及回归任务中 MAE / RMSE 等指标的意义。
