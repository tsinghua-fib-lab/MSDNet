import pickle as pkl
import numpy as np
from scipy.io import loadmat
import dgl
from tqdm import *
from utils import *

threshold = 0
cut = 0

args = initialization()
Network = args.Network
pop_num = args.pop_num  # 应该是所有人数之和，一会改
Region_num = args.region_num  # number of regions

if cut == 0 and threshold == 0:
    graph_dir = '../../data/{0}/graphs/no_cut'.format(Network)
    pop_info_dir = '../../data/{0}/trajectories/pop_info.pkl' \
        .format(Network)
else:
    graph_dir = '../../data/{0}/graphs/th_{1}_cut_{2}'.format(Network, threshold, cut)
    pop_info_dir = '../../data/{0}/trajectories/pop_info_th_{1}_cut_{2}.pkl' \
        .format(Network, threshold, cut)

ill_dir = '{0}/illness_spread'.format(graph_dir)
macro_graph_dir = '{0}/macro_graph'.format(graph_dir)
mkdir(macro_graph_dir)

simu_day = 100

with open(pop_info_dir, 'rb') as fh:
    pop_info = pkl.load(fh)
pop_info_and_trajectory = loadmat("../../data/{0}/region_file/population_xa.mat".format(Network))
population = pop_info_and_trajectory['population_xa']
Region_info = loadmat("../../data/{0}/region_file/centers_xa.mat".format(Network))  # load region information
Region_centers = np.array(Region_info['centers_xa'])  # let street in the region to be the edge

print('data loaded!')

loc_list = []
# 产生地点的列表
for i in range(len(population[0])):
    loc_list.append(str(i))

np.save('{0}/loc_list.npy'.format(macro_graph_dir), np.array(loc_list))

print('location list generated')

# 记录每个人的家的位置
home_location = []
for i in range(len(pop_info)):
    person_info = pop_info[i]
    home_location.append(str(person_info['home']))
home_location = np.array(home_location)

np.save('{0}/home_location.npy'.format(macro_graph_dir), home_location)

print('home location list generated')

# 产生两个地点之间的引力值大小
time_length = 107 * 24
loc_dist_map = np.zeros((time_length, len(loc_list), len(loc_list)))

for i in tqdm(range(pop_num), desc='generating weight'):  # 对于每个人
    trace = pop_info[i]['trace']  # trace为其轨迹
    home = home_location[i]  # home为其家所在区域
    for j in range(time_length):  # 对于每个时间片
        day = int(j / 24)
        destination = trace[j][0]  # destination为其目的地
        if not destination == int(home):  # 如果目的地和家不重合（发生移动）
            loc_dist_map[day, int(home), destination] += 1  # 累加轨迹权重

# loc_dist_map_sparse = sp.csr_matrix(loc_dist_map)
# sp.save_npz('{0}/loc_dist_map.npy'.format(save_dir), loc_dist_map_sparse)

sum_weight = []  # 记录每一个地点的population flow的大小

for time in tqdm(range(len(loc_dist_map)), desc='generalizing weight'):  # 对于每一天
    sum_weight_per_time = []
    for i in range(len(loc_list)):  # 对于每一个地点
        sum_weight_per = 0
        if loc_dist_map[time, i].any():  # 如果当前时间当前地点的权重不全为0
            sum_weight_per = loc_dist_map[time, i].sum()  # 计算当前时间当前地点的所有权重的和
        sum_weight_per_time.append(sum_weight_per)
    sum_weight.append(sum_weight_per_time)

sum_weight = np.array(sum_weight)

np.save('{0}/sum_weight.npy'.format(macro_graph_dir), sum_weight)

# 计算每个地点针对其他地点的归一化引力值大小
for time in tqdm(range(len(loc_dist_map)), desc='summing weight'):  # 对于每一天
    for i in range(len(loc_list)):  # 对于每一个地方
        if not sum_weight[time][i] == 0:
            loc_dist_map[time][i] = loc_dist_map[time][i] / sum_weight[time][i]

with open('{0}/loc_dist_map_weighted.pkl'.format(macro_graph_dir), 'wb') as f:
    pkl.dump(loc_dist_map, f)

# 计算符合要求的节点
adj_map = {}  # 有关系的节点图

loc_dist_map_added = np.zeros((len(loc_list), len(loc_list)))
for time in range(len(loc_dist_map)):  # 对于每一天
    loc_dist_map_added += loc_dist_map[time]

for i, each_loc in tqdm(enumerate(loc_list), desc='generating adj map'):  # 对于loc_list中的每一个地点
    adj_map[each_loc] = []
    a = np.array(loc_dist_map_added[i])  # a是当前地点的权重列表
    b = a.argsort()[-30:][::-1]  # b为最大的三个权重的位置坐标
    adj_list = []  # 当前位置符合权重要求的位置坐标
    for j in range(len(b)):  # 对于最大的三个权重位置坐标判断是否符合标准
        if not loc_dist_map_added[i, b[j]] == 0:
            adj_list.append(b[j])  # 将其加在列表中
            # print(each_loc, j, b[j], loc_dist_map_added[i][b[j]])
    adj_map[each_loc] = adj_list  # 记录当前位置的有效其他位置

np.save('{0}/adj_map.npy'.format(macro_graph_dir), np.array(adj_map))

# 生成图的关系
rows = []
cols = []
for each_loc in tqdm(adj_map, desc='generating graph'):  # 对于adj_map中的每一个元素，也就是每一个地方
    for each_loc2 in adj_map[each_loc]:  # 对于当前地方的每一个其他地方
        rows.append(loc_list.index(each_loc))  # rows追加上当前位置的索引
        cols.append(loc_list.index(str(each_loc2)))  # cols追加上其他位置的索引
g = dgl.graph((rows, cols), num_nodes=len(loc_list))
print(g)

output = open('{0}/g.pkl'.format(macro_graph_dir), 'wb')
pkl.dump(g, output)
output.close()

# 统计每一个时间片的S和I
step = 24
time_slot = simu_day
ill_cases = np.zeros((len(loc_list), time_slot))  # ill cases代表的是现有的确证病例数
sus_cases = np.zeros((len(loc_list), time_slot))  # confirmed cases代表的是所有曾经确诊过的病例数
new_cases = []  # new cases为新增病例数
static_feat = np.zeros(len(adj_map))  # static feat为其静态特性

for i, region_id in tqdm(enumerate(home_location), desc='generating static feature'):
    static_feat[int(region_id)] += 1

np.save('{0}/static_feat.npy'.format(macro_graph_dir), static_feat)

for i in tqdm(range(7, 107), desc='generating ill and sus cases'):  # 对于第i天
    illness_spread_day = np.load('{0}/day_{1}.npy'.format(ill_dir, i))

    for j in range(len(illness_spread_day)):  # 对于第j个人
        home = pop_info[j]['home']
        if illness_spread_day[j] == 0:
            sus_cases[home][i - 7] += 1
        else:
            ill_cases[home][i - 7] += 1
ill_cases = np.array(ill_cases)
sus_cases = np.array(sus_cases)

np.save('{0}/ill_cases.npy'.format(macro_graph_dir), ill_cases)

np.save('{0}/sus_cases.npy'.format(macro_graph_dir), sus_cases)

# 通过I和S计算dS和dI
dI = np.concatenate((np.zeros((ill_cases.shape[0], 1), dtype=np.float32), np.diff(ill_cases)), axis=-1)
dS = np.concatenate((np.zeros((sus_cases.shape[0], 1), dtype=np.float32), np.diff(sus_cases)), axis=-1)
print('I S dI dS')

np.save('{0}/dI.npy'.format(macro_graph_dir), dI)

np.save('{0}/dS.npy'.format(macro_graph_dir), dS)

# 定义normalizer
normalizer = {'S': {}, 'I': {}, 'dS': {}, 'dI': {}}
for i, each_loc in enumerate(loc_list):  # 对于每一个location的each_Loc,i为索引
    normalizer['S'][each_loc] = (np.mean(sus_cases[i]), np.std(sus_cases[i]))  # 每一个位置的S为对应的susceptible的均值和标准差
    normalizer['I'][each_loc] = (np.mean(ill_cases[i]), np.std(ill_cases[i]))  # I的均值和标准差
    normalizer['dI'][each_loc] = (np.mean(dI[i]), np.std(dI[i]))
    normalizer['dS'][each_loc] = (np.mean(dS[i]), np.std(dS[i]))  # 增量的均值和标准差
print('normalizer generated!')

# normalizer = np.array(normalizer)

np.save('{0}/normalizer.npy'.format(macro_graph_dir), np.array(normalizer))

normalize = True

dynamic_feat = np.concatenate((np.expand_dims(dI, axis=-1), np.expand_dims(dS, axis=-1)), axis=-1)

# Normalize
if normalize:
    for i, each_loc in enumerate(loc_list):
        if not (normalizer['dI'][each_loc][1] == 0 or normalizer['dS'][each_loc][1] == 0):
            dynamic_feat[i, :, 0] = (dynamic_feat[i, :, 0] - normalizer['dI'][each_loc][0]) / \
                                    normalizer['dI'][each_loc][1]
            dynamic_feat[i, :, 1] = (dynamic_feat[i, :, 1] - normalizer['dS'][each_loc][0]) / \
                                    normalizer['dS'][each_loc][1]

np.save('{0}/dynamic_feat.npy'.format(macro_graph_dir), dynamic_feat)
print('dynamic feature generated!')

dI_mean = []
dI_std = []
dS_mean = []
dS_std = []

for i, each_loc in enumerate(loc_list):
    dI_mean.append(normalizer['dI'][each_loc][0])
    dS_mean.append(normalizer['dS'][each_loc][0])
    dI_std.append(normalizer['dI'][each_loc][1])
    dS_std.append(normalizer['dS'][each_loc][1])

dI_mean = np.array(dI_mean)
dI_std = np.array(dI_std)
dS_mean = np.array(dS_mean)
dS_std = np.array(dS_std)

np.save('{0}/dI_mean.npy'.format(macro_graph_dir), dI_mean)
np.save('{0}/dI_std.npy'.format(macro_graph_dir), dI_std)
np.save('{0}/dS_mean.npy'.format(macro_graph_dir), dS_mean)
np.save('{0}/dS_std.npy'.format(macro_graph_dir), dS_std)

with open('{0}/loc_dist_map_weighted.pkl'.format(macro_graph_dir), 'rb') as fh:
    loc_dist_map = pkl.load(fh)

unique_loc_list = np.load('../../data/{0}/region_file/unique_loc_list.npy'
                          .format(Network)).astype(int)
unique_loc_list = np.sort(unique_loc_list).tolist()
unique_loc_dic = {}
dic_index = 0
for loc in unique_loc_list:
    unique_loc_dic[loc] = dic_index
    dic_index += 1

loc_dist_map = loc_dist_map[:, unique_loc_list, :]
loc_dist_map = loc_dist_map[:, :, unique_loc_list]

loc_dist_map_added = np.zeros((len(unique_loc_list), len(unique_loc_list)))
for time in range(len(loc_dist_map)):  # 对于每一天
    loc_dist_map_added += loc_dist_map[time]

adj_map = {}

for i, each_loc in enumerate(unique_loc_list):  # 对于loc_list中的每一个地点
    adj_map[i] = []
    a = np.array(loc_dist_map_added[i])  # a是当前地点的权重列表
    b = a.argsort()[-30:][::-1]  # b为最大的三个权重的位置坐标
    adj_list = []  # 当前位置符合权重要求的位置坐标
    for j in b:  # 对于最大的三个权重位置坐标判断是否符合标准
        if not loc_dist_map_added[i, j] == 0:
            adj_list.append(j)  # 将其加在列表中
            print(i, j, loc_dist_map_added[i][j])
    adj_map[i] = adj_list  # 记录当前位置的有效其他位置

rows = []
cols = []
for each_loc in adj_map:  # 对于adj_map中的每一个元素，也就是每一个地方
    for each_loc2 in adj_map[each_loc]:  # 对于当前地方的每一个其他地方
        rows.append(each_loc)  # rows追加上当前位置的索引
        cols.append(each_loc2)  # cols追加上其他位置的索引
g = dgl.graph((rows, cols), num_nodes=len(unique_loc_list))
print(g)

output = open('{0}/g_with_all_loc.pkl'.format(macro_graph_dir), 'wb')
pkl.dump(g, output)
output.close()
