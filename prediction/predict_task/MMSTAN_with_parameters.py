import pickle as pkl
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from MMSTAN_model_changed import MMSTAN
from tqdm import tqdm
import sys

sys.path.append("/data2/tangyinzhou/MMSTAN")
from utils import *

spread_func = 'ICM'
threshold = 0
cut = 0

args = initialization()
Network = args.Network
model_dir = '../../data/{0}/save'.format(Network)
output_dir = args.output_save_dir
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

beta = args.beta
gamma = args.gamma
if cut == 0 and threshold == 0:
    graph_dir = '../../data/{0}/graphs/no_cut'.format(Network)
else:
    graph_dir = '../../data/{0}/graphs/th_{1}_cut_{2}'.format(Network, threshold, cut)

pre_fix = '{0}/macro_graph'.format(graph_dir)

unique_loc_list = np.load('../../data/{0}/region_file/unique_loc_list.npy'
                          .format(Network)).astype(int)
unique_loc_list = np.sort(unique_loc_list).tolist()

dynamic_feat = np.load('{0}/dynamic_feat.npy'.format(pre_fix))
dynamic_feat = dynamic_feat[unique_loc_list, :, :]

ill_cases = np.load('{0}/ill_cases.npy'.format(pre_fix))
ill_cases = ill_cases[unique_loc_list, :]

sus_cases = np.load('{0}/sus_cases.npy'.format(pre_fix))
sus_cases = sus_cases[unique_loc_list, :]

with open('{0}/g_with_all_loc.pkl'.format(pre_fix), 'rb') as fh:
    g = pkl.load(fh)
print('g')

static_feat = np.load('{0}/static_feat.npy'.format(pre_fix))
static_feat = static_feat[unique_loc_list]

valid_window = 20
test_window = 20

# history_window = 6
history_window = 10
pred_window = 7
slide_step = 1

normalize = True

dI_mean = np.load('{0}/dI_mean.npy'.format(pre_fix))
dI_mean = dI_mean[unique_loc_list]

dI_std = np.load('{0}/dI_std.npy'.format(pre_fix))
dI_std = dI_std[unique_loc_list]

dS_mean = np.load('{0}/dS_mean.npy'.format(pre_fix))
dS_mean = dS_mean[unique_loc_list]

dS_std = np.load('{0}/dS_std.npy'.format(pre_fix))
dS_std = dS_std[unique_loc_list]
# Split train-test

train_feat = dynamic_feat[:, :-valid_window - test_window, :]  # trainset为所有位置的所有特征，取相应的天数
val_feat = dynamic_feat[:, -valid_window - test_window:-test_window, :]
test_feat = dynamic_feat[:, -test_window:, :]


def prepare_data(data, sum_I, sum_S, history_window=5, pred_window=15,
                 slide_step=5):  # 输入参数有：数据data，I的总数，R的总数，历史窗、预测窗的大小和每一步滑动的大小
    # Data shape n_loc, timestep, n_feat
    # Reshape to n_loc, t, history_window*n_feat
    n_loc = data.shape[0]  # n_loc为location的个数
    timestep = data.shape[1]  # timestep为时间片的个数
    n_feat = data.shape[2]  # n_feat为特征的个数

    x = []
    y_I = []
    y_S = []
    last_I = []
    last_S = []
    concat_I = []
    concat_S = []
    for i in range(0, timestep, slide_step):  # 对于从0到timestep，每一次滑动slide_step来说，i为；
        if i + history_window + pred_window - 1 >= timestep or i + history_window >= timestep:  # 如果timestep给的太小了或者结束预测
            break  # 结束
        # 如果没有结束预测则有如下：
        x_data = data[:, i:i + history_window, :].reshape((n_loc, history_window * n_feat))
        # x.append(data[:, i:i + history_window, :].reshape(
        #     (n_loc, history_window * n_feat)))  # 将x后面追加上：将从i到i的历史窗大小的数据，令其reshape为第0维是位置个数，第一维是历史窗大小*特征个数
        for day in range(i, i + history_window):
            filename = '{0}/pooling_output/day_{1}.npy'.format(graph_dir, day + 7)
            pooling_output_max = np.load(filename)
            row = 0
            day_index = np.zeros(n_loc)
            for loc in range(n_loc):
                if str(loc) in unique_loc_list:
                    day_index[loc] = pooling_output_max[row]
                    row += 1
            day_index = day_index.reshape(n_loc, 1)
            x_data = np.concatenate((x_data, day_index), axis=1)
        x.append(x_data)
        concat_I.append(data[:, i + history_window - 1, 0])  # 在concat_I后追加上每一个位置的最后一天的dI
        concat_S.append(data[:, i + history_window - 1, 1])  # dR
        last_I.append(sum_I[:, i + history_window - 1])  # 在last_I后追加上最后一天的感染人数I的总数
        last_S.append(sum_S[:, i + history_window - 1])  # R

        y_I.append(data[:, i + history_window:i + history_window + pred_window, 0])  # y_I为对应数据的真实值
        y_S.append(data[:, i + history_window:i + history_window + pred_window, 1])

    x = np.array(x, dtype=np.float32).transpose((1, 0, 2))  # 转置
    last_I = np.array(last_I, dtype=np.float32).transpose((1, 0))
    last_S = np.array(last_S, dtype=np.float32).transpose((1, 0))
    concat_I = np.array(concat_I, dtype=np.float32).transpose((1, 0))
    concat_S = np.array(concat_S, dtype=np.float32).transpose((1, 0))
    y_I = np.array(y_I, dtype=np.float32).transpose((1, 0, 2))
    y_S = np.array(y_S, dtype=np.float32).transpose((1, 0, 2))  # 将其都进行转置，两个元素的是进行行列转置，三个元素的就是将loc和time进行转置
    return x, last_I, last_S, concat_I, concat_S, y_I, y_S  # 返回相应数据


train_x, train_I, train_S, train_cI, train_cS, train_yI, train_yS = prepare_data(train_feat,
                                                                                 ill_cases[:,
                                                                                 :-valid_window - test_window],
                                                                                 sus_cases[:,
                                                                                 :-valid_window - test_window],
                                                                                 history_window,
                                                                                 pred_window,
                                                                                 slide_step)
# 定义相关参数送入prepare函数，生成训练集的参数
val_x, val_I, val_S, val_cI, val_cS, val_yI, val_yS = prepare_data(val_feat,
                                                                   ill_cases[:,
                                                                   -valid_window - test_window:-test_window],
                                                                   sus_cases[:,
                                                                   -valid_window - test_window:-test_window],
                                                                   history_window,
                                                                   pred_window,
                                                                   slide_step)
# 生成验证集的参数
test_x, test_I, test_S, test_cI, test_cS, test_yI, test_yS = prepare_data(test_feat,
                                                                          ill_cases[:, -test_window:],
                                                                          sus_cases[:, -test_window:],
                                                                          history_window,
                                                                          pred_window,
                                                                          slide_step)

in_dim = 3 * history_window  # 输入维度为2倍的历史窗大小
hidden_dim1 = 100
hidden_dim2 = 100
gru_dim = 200
num_heads = 2
device = torch.device("cuda:3")  # 使用gpu训练

g = g.to(device)  # 将g拷贝到device上运行
model = MMSTAN(g, in_dim, hidden_dim1, hidden_dim2, gru_dim, num_heads, pred_window, device).to(device)  # 构建模型
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 确定优化器和学习率
criterion = nn.MSELoss()

print(model)

train_x = torch.tensor(train_x).to(device)
train_I = torch.tensor(train_I).to(device)
train_S = torch.tensor(train_S).to(device)
train_cI = torch.tensor(train_cI).to(device)
train_cS = torch.tensor(train_cS).to(device)
train_yI = torch.tensor(train_yI).to(device)
train_yS = torch.tensor(train_yS).to(device)

val_x = torch.tensor(val_x).to(device)
val_I = torch.tensor(val_I).to(device)
val_S = torch.tensor(val_S).to(device)
val_cI = torch.tensor(val_cI).to(device)
val_cS = torch.tensor(val_cS).to(device)
val_yI = torch.tensor(val_yI).to(device)
val_yS = torch.tensor(val_yS).to(device)

test_x = torch.tensor(test_x).to(device)
test_I = torch.tensor(test_I).to(device)
test_S = torch.tensor(test_S).to(device)
test_cI = torch.tensor(test_cI).to(device)
test_cS = torch.tensor(test_cS).to(device)
test_yI = torch.tensor(test_yI).to(device)
test_yS = torch.tensor(test_yS).to(device)  # 将数据全都拷贝到device上运行，并且转为为tensor

dI_mean = torch.tensor(dI_mean, dtype=torch.float32).to(device).reshape((dI_mean.shape[0], 1, 1))
dI_std = torch.tensor(dI_std, dtype=torch.float32).to(device).reshape((dI_mean.shape[0], 1, 1))
dS_mean = torch.tensor(dS_mean, dtype=torch.float32).to(device).reshape((dI_mean.shape[0], 1, 1))
dS_std = torch.tensor(dS_std, dtype=torch.float32).to(device).reshape(
    (dI_mean.shape[0], 1, 1))  # 将数据拷贝到device上运行，并且转化为tensor，并且将其reshape

# N = torch.tensor(static_feat[:, 0], dtype=torch.float32).to(device).unsqueeze(-1)
N = torch.tensor(static_feat, dtype=torch.float32).to(device).unsqueeze(-1)
# Train STAN_POOL

all_loss = []
all_loss_eff = []
file_name = '{0}/MMSTAN_his_{1}_pred_{2}_th_{3}_cut_{4}'.format(model_dir, history_window, pred_window, threshold, cut)
min_loss = 1e10  # 设置loss最小值
all_val_loss = []

epoch = 500 if normalize else 300  # epoch大小
scale = 0.1

# for epoch in tqdm(range(epoch)):  # 对于每一次的epoch
for epoch in tqdm(range(epoch)):  # 对于每一次的epoch
    model.train()  # 标志下面为训练阶段
    optimizer.zero_grad()  # 梯度清零

    active_pred, recovered_pred, phy_active, phy_recover, _, dbeta, dgamma = model(train_x,
                                                                                   train_cI,
                                                                                   train_cS,
                                                                                   N,
                                                                                   train_I,
                                                                                   train_S)
    # 调用model,进行前向计算，计算预测值，分别是短期和长期的预测值

    if normalize:  # 如果需要normalize
        phy_active = (phy_active - dI_mean) / dI_std  # 计算短期active值
        phy_active = torch.where(torch.isnan(phy_active), torch.full_like(phy_active, 0), phy_active)
        phy_recover = (phy_recover - dS_mean) / dS_std  # 计算短期recover值
        phy_recover = torch.where(torch.isnan(phy_recover), torch.full_like(phy_recover, 0), phy_recover)
    loss1 = criterion(active_pred.squeeze(), train_yI.squeeze()) \
            + criterion(recovered_pred.squeeze(), train_yS.squeeze())
    loss2 = scale * criterion(phy_active.squeeze(), train_yI.squeeze())
    loss3 = scale * criterion(phy_recover.squeeze(), train_yS.squeeze())  # 使用Loss函数，使用了两个正则化
    loss4 = 0.1 * (torch.sum(torch.abs(dbeta)) + torch.sum(torch.abs(dgamma)))
    loss = loss1 + loss2 + loss3 + loss4
    print(loss)
    loss.backward()  # 反向传播
    optimizer.step()
    all_loss.append(loss.item())  # 将所有的loss加到历史记录里面

    model.eval()  # 切换为评估模式，也就是使用val
    _, _, _, _, prev_h, _, _ = model(train_x,
                                     train_cI,
                                     train_cS,
                                     N,
                                     train_I,
                                     train_S)  # 得到预测的h

    val_active_pred, val_recovered_pred, val_phy_active, val_phy_recover, _, dbeta, dgamma = model(val_x,
                                                                                                   val_cI,
                                                                                                   val_cS,
                                                                                                   N,
                                                                                                   val_I,
                                                                                                   val_S,
                                                                                                   prev_h)  # 得到真实的值

    if normalize:
        val_phy_active = (val_phy_active - dI_mean) / dI_std
        val_phy_active = torch.where(torch.isnan(val_phy_active), torch.full_like(val_phy_active, 0), val_phy_active)
        val_phy_recover = (val_phy_recover - dS_mean) / dS_std
        val_phy_recover = torch.where(torch.isnan(val_phy_recover), torch.full_like(val_phy_recover, 0),
                                      val_phy_recover)
    val_loss = criterion(val_active_pred.squeeze(), val_yI.squeeze()) \
               + criterion(val_recovered_pred.squeeze(), val_yS.squeeze()) \
               + scale * criterion(val_phy_active.squeeze(), val_yI.squeeze()) \
               + scale * criterion(val_phy_recover.squeeze(), val_yS.squeeze()) \
               + 0.1 * (torch.sum(torch.abs(dbeta)) + torch.sum(torch.abs(dgamma)))
    print(val_loss)
    all_val_loss.append(val_loss.item())

    if val_loss < min_loss:  # 如果评估的loss符合要求
        state = {
            'state': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        all_loss_eff.append(loss.item())
        torch.save(state, file_name)
        min_loss = val_loss
        # all_val_loss.append(val_loss.item())

plt.plot(all_loss, c='r', label='Loss')
plt.plot(all_val_loss, c='g', label='val_Loss')
plt.legend()
plt.show()

file_name = '{0}/MMSTAN_his_{1}_pred_{2}_th_{3}_cut_{4}'.format(model_dir, history_window, pred_window, threshold, cut)
checkpoint = torch.load(file_name)
model.load_state_dict(checkpoint['state'])
optimizer.load_state_dict(checkpoint['optimizer'])  # 得到当loss最小时的模型参数
model.eval()

prev_x = torch.cat((train_x, val_x), dim=1)
prev_I = torch.cat((train_I, val_I), dim=1)
prev_R = torch.cat((train_S, val_S), dim=1)
prev_cI = torch.cat((train_cI, val_cI), dim=1)
prev_cR = torch.cat((train_cS, val_cS), dim=1)  # 生成输入数据

prev_active_pred, _, prev_phyactive_pred, _, h, _, _ = model(prev_x,
                                                             prev_cI,
                                                             prev_cR,
                                                             N,
                                                             prev_I,
                                                             prev_R)
# 得到长期和短期的输出数据,其实有用的只有h，也就是GRU的状态

test_pred_active, test_pred_recovered, test_pred_phy_active, test_pred_phy_recover, _, _, _ = model(test_x,
                                                                                                    test_cI,
                                                                                                    test_cS,
                                                                                                    N,
                                                                                                    test_I,
                                                                                                    test_S,
                                                                                                    h)  # 得到测试数据
beta_pred = model.alpha_scaled.detach().cpu().numpy()
gamma_pred = model.beta_scaled.detach().cpu().numpy()

beta_pred = np.mean(beta_pred)
gamma_pred = np.mean(gamma_pred)

if normalize:
    print('Estimated beta in SIR model is %.5f' % beta_pred)
    print('Estimated gamma in SIR model is %.5f' % gamma_pred)

# Cumulate predicted dI
pred_I = []
pred_dI = []
for i in range(test_pred_active.size(1)):  # 对于每一组数据
    if normalize:
        cur_pred = (test_pred_active[:, i, :].detach().cpu().numpy() * dI_std.reshape(-1, 1).detach().cpu().numpy()) + \
                   dI_mean.reshape(-1, 1).detach().cpu().numpy()
    else:
        cur_pred = test_pred_active[:, i, :].detach().cpu().numpy()  # 令cur_pred为当前这组数据的输出
    cur_pred = np.cumsum(cur_pred, axis=0)  # cur_pred按列累加
    test_I_numpy = test_I[:, i].detach().cpu().numpy().reshape(-1, 1)
    cur_pred = cur_pred + test_I_numpy  # cur_pred为原来的值加上测试集数据的I
    # cur_pred_d = cur_pred_d + test_cI[cur_loc, i].detach().cpu().item()
    pred_I.append(cur_pred)  # 将其累加到pred_I中
pred_I = np.array(pred_I)  # 转换为ndarray


def get_real_y(data, history_window=5, pred_window=15, slide_step=5):
    # Data shape n_loc, timestep, n_feat
    # Reshape to n_loc, t, history_window*n_feat
    n_loc = data.shape[0]
    timestep = data.shape[1]

    y = []
    for i in range(0, timestep, slide_step):  # 对于每一个有效的时间片
        if i + history_window + pred_window - 1 >= timestep or i + history_window >= timestep:  # 如果超出了时间片
            break
        y.append(data[:, i + history_window:i + history_window + pred_window])  # 如果没有超出范围，则将y追加上输入数据的输出信息
    y = np.array(y, dtype=np.float32).transpose((1, 0, 2))  # 将其转置得到真实数据
    return y


save_prefix = '{0}/prediction_output_his_{1}_pred_{2}'.format(graph_dir, history_window, pred_window)
mkdir(save_prefix)
I_true = get_real_y(ill_cases[:], history_window, pred_window, slide_step)
pred_true = I_true[:, -1, :]
pred_I = pred_I[-1, :, :]
STAN_filename = '{0}/MMSTAN.npy'.format(save_prefix)
np.save(STAN_filename, pred_I)
GT_filename = '{0}/Ground_Truth.npy'.format(save_prefix)
np.save(GT_filename, I_true)
cur_loc = 2
plt.plot(pred_true[cur_loc, :], c='r', label='Ground truth')
plt.plot(pred_I[cur_loc, :], c='b', label='Prediction')
plt.legend()
plt.show()


# 计算MSE
def cal_MSE(ground_value, pred_value):
    MSE = []
    for region in range(len(ground_value)):
        MSE_per = 0
        for i in range(len(ground_value[0])):
            MSE_per += (ground_value[region, i] - pred_value[region, i]) ** 2
        MSE_per = MSE_per / (i + 1)
        MSE.append(MSE_per)
    avg_MSE = np.mean(MSE)
    return MSE, avg_MSE


def cal_MAPE(ground, pred):
    ground, pred = np.array(ground).reshape(-1, 1), np.array(pred).reshape(-1, 1)
    return np.mean(np.abs((ground - pred) / ground)) * 100


MSE, avg_MSE = cal_MSE(pred_true, pred_I)
MAPE = cal_MAPE(pred_true, pred_I)


def cal_MAPE_region(ground, pred):
    MAPE = []
    for i in range(len(ground)):
        ground_per = ground[i]
        pred_per = pred[i]
        MAPE_per = np.mean(np.abs((ground_per - pred_per) / ground_per)) * 100
        MAPE.append(MAPE_per)
    MAPE_avg = np.mean(MAPE)
    return MAPE_avg


def cal_MAPE_day(ground, pred):
    MAPE = []
    for i in range(ground.shape[1]):
        ground_per = np.array(ground[:, i]).reshape(-1, 1)
        pred_per = np.array(pred[:, i]).reshape(-1, 1)
        MAPE_per = np.mean(np.abs((ground_per - pred_per) / ground_per)) * 100
        MAPE.append(MAPE_per)
    MAPE_avg = np.mean(MAPE)
    return MAPE_avg


def cal_SMAPE(actual, predicted):
    if not all([isinstance(actual, np.ndarray),
                isinstance(predicted, np.ndarray)]):
        actual, predicted = np.array(actual),
        np.array(predicted)

    return round(
        np.mean(
            np.abs(predicted - actual) /
            ((np.abs(predicted) + np.abs(actual)) / 2)
        ) * 100, 2
    )


SMAPE = cal_SMAPE(pred_true, pred_I)

MAPE_day = cal_MAPE_day(pred_true, pred_I)
MAPE_region = cal_MAPE_region(pred_true, pred_I)

print('MSE_list:', MSE)

# print('MAPE_day:', MAPE_day)
# print('MAPE_region:', MAPE_region)

print('MAPE:', MAPE)
print('SMAPE:', SMAPE)
print('Average MSE', avg_MSE)
