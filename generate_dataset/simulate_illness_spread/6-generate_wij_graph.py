import pickle as pkl
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from tqdm import *
from utils import *

cut = 0
threshold = 0

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

save_dir = '{0}/wij_graph'.format(graph_dir)
mkdir(save_dir)
with open(pop_info_dir, 'rb') as fh:
    pop_info = pkl.load(fh)
pop_info_and_trajectory = loadmat("../../data/BJ-331/region_file/population_xa.mat".format(Network))
population = pop_info_and_trajectory['population_xa']

print('data loaded!')

loc_list = []
# 产生地点的列表
for i in range(len(population[0])):
    loc_list.append(str(i))

np.save('{0}/loc_list.npy'.format(save_dir), np.array(loc_list))

print('location list generated')

# 记录每个人的家的位置
home_location = []
for i in range(len(pop_info)):
    person_info = pop_info[i]
    home_location.append(str(person_info['home']))
home_location = np.array(home_location)

np.save('{0}/home_location.npy'.format(save_dir), home_location)

print('home location list generated')

# 产生两个地点之间的引力值大小
loc_dist_map = np.zeros((len(loc_list), len(loc_list)))
time_length = 100
for i in tqdm(range(pop_num), desc='generating weight'):  # 对于每个人
    trace = pop_info[i]['trace']  # trace为其轨迹
    home = home_location[i]  # home为其家所在区域
    for j in range(time_length):  # 对于每个时间片
        destination = trace[j][0]  # destination为其目的地
        loc_dist_map[int(home), destination] += 1  # 累加轨迹权重

loc_dist_map_sparse = sp.csr_matrix(loc_dist_map)
sp.save_npz('{0}/loc_dist_map.npy'.format(save_dir), loc_dist_map_sparse)
wij_graph = np.zeros((331, 331))
for region_idx, region_data in enumerate(loc_dist_map):
    sum_data = np.sum(region_data)
    if not sum_data == 0:
        normalized_region_data = region_data / sum_data
        wij_graph[region_idx] = normalized_region_data
wij_graph_sparse = sp.csc_matrix(wij_graph)

sp.save_npz('{0}/wij_graph.npz'.format(save_dir), wij_graph_sparse)
