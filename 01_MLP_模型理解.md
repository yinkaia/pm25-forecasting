# 01_MLP_模型理解

## 复盘问题

1. 这个任务的输入是什么？
输入的内容：过去 24 小时的 14 个特征，也就是 X.shape = [24, 14]
如果一个 batch 有 64 条样本，则 x_batch.shape = [64, 24, 14]
其中：
- 64 = batch_size，一次训练取 64 条样本
- 24 = seq_len，每条样本看过去 24 小时
- 14 = feature_dim，每个小时有 14 个特征

14 个特征包括：
pm2.5      当前小时 PM2.5
DEWP       露点温度
TEMP       温度
PRES       气压
Iws        累计风速
Is         累计降雪
Ir         累计降雨
month      月份
day        日期
hour       小时
cbwd_NE    风向 NE
cbwd_NW    风向 NW
cbwd_SE    风向 SE
cbwd_cv    风向 cv
最重要的一点是：pm2.5 也被放进了输入特征。
因为任务是：用过去 24 小时的 PM2.5 和天气特征，预测未来 1 小时的 PM2.5

由于神经网络不能直接吃字符串，所以风向字符串在代码里做了 one-hot 编码：
df = pd.get_dummies(df, columns=["cbwd"], prefix="cbwd", dtype=int)
会把：
cbwd = NW
变成：
cbwd_NE = 0
cbwd_NW = 1
cbwd_SE = 0
cbwd_cv = 0

模型看到的不是原始数据，而是标准化后的数值。
比如原始 PM2.5 可能是：58 52 100
标准化后可能变成：-0.42 -0.51 0.15

dataset.py 不自己决定特征列，而是读取 data_process.py 保存的 configs.json：
config = {
    "target_col": TARGET_COL,
    "feature_cols": feature_cols,
    "target_index": feature_cols.index(TARGET_COL),
}

滑动窗口构造输入 X：

所以第一条样本是：idx = 0
X = 第 1 ~ 24 小时的 14 个特征
y = 第 25 小时的 PM2.5

第二条样本：idx = 1
X = 第 2 ~ 25 小时的 14 个特征
y = 第 26 小时的 PM2.5

第三条样本：idx = 2
X = 第 3 ~ 26 小时的 14 个特征
y = 第 27 小时的 PM2.5

输入在训练循环里，通过 DataLoader 取出一个 batch，然后把输入数据放到 GPU 或 CPU 上，
真正把输入送进模型的是这一句：y_pred = model(x_batch)


2. 这个任务的输出是什么？
输出是：未来 1 小时的 PM2.5 浓度。



3. 输入数据的 shape 是什么？
原始数据形状：(43824, 12)：
43824 行：43824 个小时的数据
12 列：原始字段数量

模型输入特征：定义了 14 个模型输入特征，处理后，每个小时会变成 14 个特征 [43824, 14]
43824 个小时
每个小时 14 个特征
原始字段的 year 没有用，因为更关心小时级别变化； 12 - 1 = 11
cbwd 风向从1 列变成了 4 列，因为神经网络不能直接吃字符串；12 - 1 - 1 + 4 = 14
cbwd ==>
cbwd_NE
cbwd_NW
cbwd_SE
cbwd_cv
如果 cbwd = NW ==>
cbwd_NE = 0
cbwd_NW = 1
cbwd_SE = 0
cbwd_cv = 0


train / val / test 切分后：
总长度：43824
train: 30676 行
val:    6574 行
test:   6574 行
因此，CSV 文件里：
train_scaled.csv.shape = [30676, 15]
val_scaled.csv.shape   = [6574, 15]
test_scaled.csv.shape  = [6574, 15]

dataset 里单条样本的输入 shape：[24, 14] [1]
单个样本会连续取 24 行，作为模型输入，每行 14 个特征。
预测的是一个 PM2.5 的值，所以是一个标量

DataLoader 之后 batch 的 shape：
DataLoader 会把 64 条样本合在一起，即：
x_batch.shape: torch.Size([64, 24, 14])
y_batch.shape: torch.Size([64, 1])

train 时模型实际拿到的输入 shape：[64, 24, 14]


4. 模型每一层的输入输出 shape 是什么？
MLP：
模型统一输入：[64, 24, 14]
nn.Flatten()：[64, 336]
nn.Linear(336, 128)：[64, 128]
nn.ReLU()：[64, 128]
nn.Dropout(0.2)：[64, 128]
nn.Linear(128, 64)：[64, 64]
nn.ReLU()、nn.Dropout(0.2)：[64, 64]
nn.Linear(64, 1)：[64, 1]
MLP 把过去 24 小时 x 14 个特征直接压平成 336 维向量，然后用全连接层预测下一小时 PM2.5

RNN：[batch_size, seq_len, input_dim]
模型统一输入：[64, 24, 14]，RNN 按 24 个时间步依次处理
RNN 处理 24 个时间步后输出：==>
rnn_output.shape = [64, 24, 128]，保存每个时间步的隐藏状态 
[batch_size, seq_len, hidden_dim]
h_n.shape        = [1, 64, 128]，保存最后一个时间步的最终隐藏状态 
[num_layers, batch_size, hidden_dim]
送入全连接层：nn.Linear(128, 1)：[64, 1]

GRU：
模型统一输入：[64, 24, 14]
GRU(input_size=14, hidden_size=128) ==>
gru_output.shape = [64, 24, 128]
h_n.shape        = [1, 64, 128]
h_n[-1] ==>
last_hidden = [64, 128]
Linear(128, 1) ==> [64, 1]


LSTM：
模型统一输入：[64, 24, 14] ==>
lstm_output.shape = [64, 24, 128]，每个时间步的隐藏状态
h_n.shape         = [1, 64, 128]，最后时间步的隐藏状态
c_n.shape         = [1, 64, 128]，最后时间步的细胞状态
h_n[-1] ==> 
last_hidden = [64, 128]
Linear(128, 1)：[64, 1]

四个模型的区别：
输入 x_batch.shape = [64, 24, 14]

MLP：不管时间序列，直接把所有特征摊平成一个大向量，再做预测。
- 优点：简单、快；
- 缺点：没有显示建模时间顺序

RNN：按时间一步一步读，每读一个小时，就更新一次隐藏状态。
- 优点：能处理时间顺序
- 缺点：记忆能力比较弱，序列长了容易忘掉前面的信息

GRU：RNN 的改进版，多了 “门”。
这些门大概决定：
- 哪些旧信息要保留
- 哪些旧信息要忘掉
- 哪些新信息要写进去
- 比 RNN 记忆能力更强；
- 比 LSTM 结构更简单，参数更少一些

LSTM：记忆能力更强的 RNN，比 GRU 更复杂，多了 h_n  隐藏状态 和 c_n 细胞状态。
- h_n：隐藏状态，当前对外输出的状态
- c_n：细胞状态，内部长期记忆

MLP：
把 24 小时全部摊平，当成普通表格特征来预测。

RNN：
按小时顺序读数据，用隐藏状态记住过去信息。

GRU：
在 RNN 基础上加入门控，学会保留和遗忘信息。

LSTM：
比 GRU 更复杂，有 hidden state 和 cell state，长期记忆能力更强。


5. 哪些参数会被训练？
MLP：全连接层的权重 weight 和 bias。
Linear(336, 128) 的 weight 和 bias
Linear(128, 64) 的 weight 和 bias
Linear（64， 1）的 weight 和 bias

RNN：RNN 层内部的循环参数 和 最后的 fc 全连接层参数。
- weight_ih_l0：输入 x 到隐藏状态 h 的权重
- weight_hh_l0：上一个隐藏状态 h 到当前隐藏状态 h 的权重
- bias_ih_l0：输入部分的偏置
- bias_hh_l0：隐藏状态部分的偏置
- fc.weight：隐藏状态到 PM2.5 预测值的权重
- fc.bias：输出层偏置
所以 RNN 训练的是：如何根据当前小时特征和历史隐藏状态更新记忆；以及如何根据最终记忆预测下一小时 PM2.5。

GRU：内部门控参数 和 最后的 fc 全连接层参数
- gru.weight_ih_l0
- gru.weight_hh_l0
- gru.bias_ih_l0
- gru.bias_hh_l0
- fc.weight
- fc.bias
所以 GRU 训练的是：门控参数 + 输出层参数。

LSTM：LSTM 内部门控参数 和 最后的 fc 全连接层参数。
- lstm.weight_ih_l0
- lstm.weight_hh_l0
- lstm.bias_ih_l0
- lstm.bias_hh_l0
- fc.weight
- fc.bias
所以 LSTM 训练的是：LSTM 内部的门控权重和偏置 + fc 输出层的权重和偏置。


6. loss 是怎么计算的？
训练时 loss 的计算流程是：
x_batch = [64, 24, 14]
y_batch = [64, 1]

y_pred = model(x_batch)
y_pred = [64, 1]

loss = MSELoss(y_pred, y_batch)，即 loss = mean((y_pred - y_batch)^2)

7. backward 后哪些参数会有梯度？
上述参与训练的参数，backward 后都会有梯度。

8. optimizer.step 后模型哪里发生了变化？
MLP：
- 3 个 Linear 层的参数
- 这些层学到的是：如何把过去 24 小时 x14 个特征组成的 336 维向量，映射成下一小时 PM2.5 预测值
- 不会变化的是：Flatten、ReLU、Dropout

RNN：
- 变化的同样是上述的参加训练的参数
- 会改变模型如何理解 “当前小时的 14 个特征”
- 会改变模型如何利用 “上一个时间步留下来的隐藏状态”
- 会改变模型如何把最后隐藏状态变成 PM 2.5 预测值
- 本质上变的是：它按照时间读取 24 小时数据的记忆机制，以及最后输出预测值的规则

GRU：
- 变化的同样是上述的参加训练的参数，也就是模型逐渐学会了：
- 哪些历史 PM2.5 信息要保留
- 哪些天气变化要重视
- 哪些旧信息可以忘掉
- 最终隐藏状态怎么变成 PM2.5 预测

LSTM：
- 变化的同样是上述的参加训练的参数，也就是模型逐渐学会了：
- 哪些长期信息要留在 cell state 里
- 哪些旧信息要忘掉
- 当前小时的新信息怎么写入
- 最后 hidden state 怎么输出
- 最终怎么预测下一小时 PM2.5


训练的本质就是重复做这件事：
1. 预测错乱
2. 根据错多少计算梯度
3. 调整参数
4. 下次尽量少错一点



# 项目的完整流程：
一、数据准备
1. download_data.py：下载原始数据到本地
2. explore_data.py：对数据做探索性检查，决定了后续应该怎么处理数据。
3. data_process.py：把原始 CSV 清洗成模型可以直接使用的 train、val、test scale 文件，同时保存 scaler 和 config 供后续构造滑动窗口、模型训练和预测反标准化使用。

二、核心组件
1. dataset.py：把处理好的表格数据，转换成 PyTorch 模型可以训练的事件序列样本,即滑动窗口样本 和 DataLoader
2. models.py：定义 PM2.5 时间序列预测项目里的 4 个模型（MLP、RNN、GRU、LSTM），并用一个 batch 数据测试它们能不能正常向前传播。

三、训练与评估
1. train.py：训练主脚本，根据命令行选择 MLP/RNN/GRU/LSTM，读取滑动窗口 DataLoader，用 MSELoss 和 Adam 训练模型，每轮用验证集挑选最佳 checkpoint，最后在测试集上算一次标准化空间的 test_loss，并保存 loss 曲线。
2. evaluate.py：最终评估脚本，加载保存好的最佳模型，在 test 测试集上做预测，把标准化啊啊啊后的预测值还原成真实 PM2.5，计算 MAE、RMSE 等指标，并保存预测结果和预测曲线。
3. naive_baseline.py：基线模型，直接用输入窗口最后一小时的 PM2.5 当做下一个 Pm2.5 的预测值，给模型提供一个最低对照标准。

四、对比与结果
1. compare_models.py：模型对比脚本，把 Naive baseline、MLP、RNN、GRU、LSTM 的测试集指标统一算出来，然后放到一张表里对比，并画柱状图。
2. record.txt：实验记录
3. results/ 目录：结果输出

五、其他文件
1. requrements.txt：依赖包列表
2. predict.py：预测脚本
3. checkpoints/：模型保存目录
4. data/：数据目录
