from scipy.io import loadmat, savemat
from scipy.sparse import coo_matrix, save_npz, lil_matrix, load_npz, csc_matrix
import numpy as np
import random
import math
import pickle as pkl
import time
from tqdm import *
import sys

sys.path.append("/data2/tangyinzhou/MMSTAN")
from utils import *

args = initialization()
threshold = 0.5
cut = 0.25
beta = 0.08
gamma = 0.04
Simu_slot = args.simu_slot
time_slot = 60  # 10 minutes，时间片
trace_rho = args.rho  # parameter_ρ
trace_gamma = args.time_geo_gamma  # parameter_γ
trace_alpha = args.time_geo_alpha  # parameter_α
Network = args.Network
n_w = args.n_w  # 6.1
beta1 = args.beta1  # 3.67
beta2 = args.beta2  # 10
pop_num = args.pop_num  # 应该是所有人数之和，一会改
Region_num = args.region_num  # number of regions
graph_save_dir = '../../data/{0}/graphs/th_{1}_cut_{2}'.format(Network, threshold, cut)
no_cut_graph_save_dir = '../../data/{0}/graphs/no_cut'.format(Network)
trajectories_dir = '../../data/{0}/trajectories'.format(Network)
mkdir(trajectories_dir)
print('n_w={0},beta1={1},beta2={2}'.format(n_w, beta1, beta2))
Region_info = loadmat("../../data/{0}/region_file/streets_xa.mat".format(Network))  # load region information
Pop_info = loadmat('../../data/{0}/region_file/population_xa_test.mat'.format(Network))
Region_edge = np.array(Region_info['streets_xa'])  # let street in the region to be the edge
Region_pop = Pop_info['population_xa'].reshape((Region_num))
Pop_distribution = Region_pop / Region_pop.sum()  # 求region_pop的分布函数
Region_center_info = loadmat("../../data/{0}/region_file/centers_xa.mat".format(Network))
Region_center = Region_center_info['centers_xa']

with open('{0}/pop_info.pkl'.format(trajectories_dir), 'rb') as f:
    origin_pop_info = pkl.load(f)

pt_path = '{0}/p_t.npy'.format(trajectories_dir)
P_t = np.load(pt_path)

home_path = '{0}/home_location.npy'.format(trajectories_dir)
Home_location = np.load(home_path)


def stamp2array(time_stamp):
    return time.localtime(float(time_stamp))  # time_stamp格式化为本地时间


def get_p_t(now_time):
    now_time_tup = stamp2array(now_time)  # 将时间戳格式化为本地时间
    i = int(
        (now_time_tup.tm_wday * 24 * 60 + now_time_tup.tm_hour * 60 + now_time_tup.tm_min) / time_slot)  # 得到当前是在哪个时间戳
    return P_t[i]


def predict_next_place_time(n_w, p_t, beta1, beta2, current_location_type, over_threshold):
    if over_threshold:
        n_w = n_w * (1 - cut)
    p1 = 1 - n_w * p_t
    p2 = 1 - beta1 * n_w * p_t
    p3 = beta2 * n_w * p_t  # 分别计算提到的参数P1P2P3
    location_is_change = 0  # 标识标识位置是否改变
    new_location_type = 'undefined'  # 设置新位置的类型
    if current_location_type == 'home':  # 如果从家出发
        if random.uniform(0, 1) <= p1:  # 确定新位置的类型
            new_location_type = 'home'
            location_is_change = 0
        else:
            new_location_type = 'other'
            location_is_change = 1
    elif current_location_type == 'other':  # 如果从other出发，确定新位置类型
        p = random.uniform(0, 1)
        if p <= p2:
            new_location_type = 'other'
            location_is_change = 0
        elif random.uniform(0, 1) <= p3:
            new_location_type = 'other'
            location_is_change = 1
        else:
            new_location_type = 'home'
            location_is_change = 1
    if new_location_type == 'home':  # 如果新类型为home
        return 0, location_is_change
    else:
        return 2, location_is_change  # 返回信息


def negative_pow(k):
    p_k = {}
    for i, region in enumerate(k, 1):
        p_k[region[0]] = i ** (-trace_alpha)
    temp = sum(p_k.values())
    for key in p_k:
        p_k[key] = p_k[key] / temp
    return p_k


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def predict_next_place_location_simplify(P_new, region_history, current_region, home_region):
    rp = random.uniform(0, 1)  # 随机数
    prob_accum = 0  # 概率累积
    # print('rp', rp)
    if random.uniform(0, 1) < P_new:  # 一种情况，则进行explore
        # explore; the explore distance is depend on history->delta_r
        length = {}
        for i, cen in enumerate(Region_center):  # 对于region_center中的每一个元素，i为其索引，cen为其中心所在位置
            if i in region_history:  # 如果原来去过这个地方
                continue  # 找下一个区域
            length[i] = distance(cen, Region_center[current_region])  # 计算当前位置到新位置的中心的距离
        try:
            del length[home_region]
            del length[current_region]
        except KeyError:
            pass
        k = sorted(length.items(), key=lambda x: x[1], reverse=False)
        p_k = negative_pow(k)
        for i, key in enumerate(p_k):
            prob_accum += p_k[key]
            if prob_accum > rp:
                next_region = key
                region_history[key] = 1
                break
            else:
                continue
    else:  # 另一种情况，不进行explore
        # return
        region_history_sum = sum(region_history.values())
        for key in region_history:
            prob_accum += region_history[key] / region_history_sum
            if rp < prob_accum:
                next_region = key
                region_history[key] += 1
                break

    return next_region


def predict_next_place_location(region_history, current_location, home_region):
    s = len(region_history.values())  # s为原来去过的地方的总数
    if s == 0:  # 如果原来没去过地方
        p_new = 1  # 一定探索新地方
    else:  # 如果原来去过新地方
        p_new = trace_rho * s ** (-trace_gamma)  # 以概率探索新地方
    return predict_next_place_location_simplify(p_new, region_history, current_location, home_region)


def individual_trace_simulate(info, p_t, current_location_type, now_time, over_threshold):
    # region ID, time_slot
    simu_trace = info['trace']  # simu_trace为模拟的轨迹，为一个list
    now_type, location_change = predict_next_place_time(info['n_w'],
                                                        p_t,
                                                        info['beta1'],
                                                        info['beta2'],
                                                        current_location_type,
                                                        over_threshold)
    if location_change == 1:  # 如果位置改变了
        current_location = simu_trace[-1][0]
        if now_type == 0:  # 如果当前类型为home
            next_location = info['home']  # 下一个位置的信息就为home
            current_location_type = 'home'
        else:
            next_location = predict_next_place_location(info['region_history'],
                                                        current_location,
                                                        info['home'])  # 得到下一个位置的坐标数字
            current_location_type = 'other'  # 将当前位置设置为other
        info['feature']['move_num'] += 1  # 将移动数+1
        info['feature']['move_distance'] += distance(Region_center[next_location],
                                                     Region_center[current_location])
        # 将移动距离加上当前位置到下一个位置的距离
    else:
        next_location = simu_trace[-1][0]  # 得到下一个位置即可，不进行探索
    # simu_trace.append([next_location, now_time])
    new_trace = np.array([next_location, now_time]).reshape(1, 2)
    simu_trace = np.concatenate((simu_trace, new_trace), axis=0)
    return simu_trace, current_location_type


def simulate_trace(hour, start_time, pop_info, over_threshold):
    now_time = (hour) * 60 * time_slot + start_time  # 当前的时间
    p_t = get_p_t(now_time)  # 得到当前时间的Pt值
    # print('simulating trace of hour ', hour, '...')
    for i in range(pop_num):  # 对于每一个人，模拟这个小时的轨迹
        if hour == 0:
            current_location_type = 'home'  # 设定当前位置的类型
            pop_info[i]['trace'] = np.array([[pop_info[i]['home'], start_time]])
        else:
            current_location_type = pop_info[i]['current_location_type']
        simu_trace, current_location_type = individual_trace_simulate(pop_info[i],
                                                                      p_t,
                                                                      current_location_type,
                                                                      now_time,
                                                                      over_threshold)
        pop_info[i]['current_location_type'] = current_location_type
        pop_info[i]['trace'] = np.array(simu_trace)  # 计算这个时间片的位置
    print('finished trace simulate of hour ', hour)


def initial_pop_info(pop_info):
    for i in range(pop_num):  # 对于每一个人
        pop_info.append(
            {'n_w': n_w, 'beta1': beta1, 'beta2': beta2, 'home': Home_location[i], 'current_location_type': 'Null',
             'feature': {'move_num': 0, 'move_distance': 0},
             'region_history': {}})  # 创造每一个人的轨迹，其中包括每一个人去过地方的个数nw，两个beta和他家的位置，以及一系列特征，
        pop_info[i]['ill_history'] = []
        pop_info[i]['trace'] = []
    print("Initialized pop info")


def generate_contact_graph(pop_info, hour):
    file_dir = '{0}/user_contact_graphs'.format(graph_save_dir)
    mkdir(file_dir)
    filename = '{0}/hour_{1}.npz'.format(file_dir, hour)
    trace_array = []
    for info in pop_info:
        trace_array.append(info['trace'][-1, 0])
    trace_array = np.array(trace_array)
    row = []
    col = []
    not_home_flag_hour = np.array(trace_array - Home_location).astype(bool)
    not_home_list = np.array(np.where(not_home_flag_hour)).reshape(-1)
    not_home_user = np.zeros(pop_num)
    not_home_user[not_home_list] = 1
    not_home_user = not_home_user.astype(bool)  # 50000人中如果不在家为T
    for i in not_home_list:  # 对于每一个不在家的人i
        loc = trace_array[i]  # loc为他所在的区域
        temp_j = trace_array - loc
        loc_same = np.array(np.where(temp_j == 0)).reshape(-1)  # 得到在这个区域的人的索引
        loc_same_user = np.zeros(pop_num)
        loc_same_user[loc_same] = 1
        loc_same_user = loc_same_user.astype(bool)  # 在这个区域的人为T
        loc_same = np.logical_and(loc_same_user, not_home_user)  # 只有在这个区域且不在家的为T
        loc_same = np.array(np.where(loc_same)).reshape(-1)  # 得到在同一个区域且不在家的邻居的人数
        loc_same = np.delete(loc_same, np.where(loc_same == i))
        for k in loc_same:
            row.append(i)
            col.append(k)
    data = np.ones(len(row))
    contact_graph_coo = coo_matrix((data, (row, col)), shape=(pop_num, pop_num))
    contact_graph_csc = contact_graph_coo.tocsc()
    save_npz(filename, contact_graph_csc)


def simulate_illness_spread(day, contact_graph, feature_graph):
    # print('simulating illness spread of day ', day, '...')
    ill_num = np.zeros(pop_num)  # 定义染病的邻居数
    contact_graph = contact_graph.toarray()
    for index in tqdm(range(0, pop_num)):  # 对于每一个数
        person_row = contact_graph[index]
        index_row = np.array(np.where(person_row == 1)).reshape(-1)
        for person in index_row:
            if feature_graph[person] == 1:  # 如果另一个人染病
                ill_num[index] += 1  # 将其计入染病的邻居数中
    feature_graph_out = np.zeros((pop_num))
    for index in range(pop_num):  # 对于每一个人
        if feature_graph[index] == 0:  # 如果当前人不是患病的
            prob_ill = 1 - (1 - gamma) ** ill_num[index]
            if np.random.uniform(0, 1) < prob_ill:  # 有一定的概率患病
                feature_graph_out[index] = 1
        else:  # 如果是当前人是患病的
            if np.random.uniform(0, 1) > beta:  # 有一定的概率康复
                feature_graph_out[index] = 1
    return feature_graph_out


def save_ill_feature(pop_info, feature_graph):
    for i in range(len(feature_graph)):
        pop_info[i]['ill_history'].append(feature_graph[i])


def generate_day_contact_graph(day):
    print("day_contact_{0}".format(day))
    begin_hour = day * 24
    end_hour = (day + 1) * 24
    contact_graph_day_all = np.zeros((pop_num, pop_num))
    file_dir = '{0}/user_contact_graphs'.format(graph_save_dir)
    filename_1 = '{0}/hour_{1}.npz'.format(file_dir, begin_hour)
    contact_graph_1 = load_npz(filename_1)
    contact_graph_1 = contact_graph_1.tocsc()
    contact_graph_day = contact_graph_1.copy()
    for hour in tqdm(range(begin_hour + 1, end_hour), position=0, leave=False, colour='red'):
        filename_2 = '{0}/hour_{1}.npz'.format(file_dir, hour)
        contact_graph_2 = load_npz(filename_2)
        contact_graph_2 = contact_graph_2.tocsc()
        contact_graph_day += contact_graph_2
        contact_graph_day = contact_graph_day.tocoo()
        contact_point = np.array(np.where(contact_graph_day.data == 2))
        col = contact_graph_day.col[contact_point]
        row = contact_graph_day.row[contact_point]
        contact_graph_day_all[row, col] = 1
        # print('hour ', hour - 1, 'and hour ', hour, 'aggregated')
    contact_graph_day_all = csc_matrix(contact_graph_day_all)
    print('day ', day, 'finished!')
    day_contact_graph_dir = '{0}/user_contact_graphs_day'.format(graph_save_dir)
    mkdir(day_contact_graph_dir)
    save_filename = '{0}/day_{1}.npz'.format(day_contact_graph_dir, day)
    save_npz(save_filename, contact_graph_day_all)
    return contact_graph_day_all


def copy_pop_info(hour):
    for i in range(pop_num):  # 对于每一个人，模拟这个小时的轨迹
        info = origin_pop_info[i]
        loc, now = info['trace'][hour]
        if loc == info['home']:
            current_location_type = 'home'
        else:
            current_location_type = 'other'
        pop_info[i]['current_location_type'] = current_location_type
        append_info = np.array([loc, now]).reshape(1, 2)
        pop_info_trace = pop_info[i]['trace']
        if len(pop_info_trace) == 0:
            pop_info_trace = append_info
        else:
            pop_info_trace = np.concatenate((pop_info_trace, append_info), axis=0)
        # pop_info_trace = pop_info_trace + append_info
        pop_info[i]['trace'] = pop_info_trace  # 计算这个时间片的位置
    print('finished trace copy of hour ', hour)


def copy_hour_info(hour, start_time):
    for i in range(pop_num):  # 对于每一个人，模拟这个小时的轨迹
        info = origin_pop_info[i]
        loc, now = info['trace'][hour]
        if loc == info['home']:
            current_location_type = 'home'
        else:
            current_location_type = 'other'
        pop_info[i]['current_location_type'] = current_location_type
        append_info = np.array([loc, now]).reshape(1, 2)
        pop_info_trace = pop_info[i]['trace']
        if len(pop_info_trace) == 0:
            pop_info_trace = append_info
        else:
            pop_info_trace = np.concatenate((pop_info_trace, append_info), axis=0)
        # pop_info_trace = pop_info_trace + append_info
        pop_info[i]['trace'] = pop_info_trace  # 计算这个时间片的位置
    fromfile = '{0}/user_contact_graphs/hour_{1}.npz'.format(no_cut_graph_save_dir, hour)
    hour_contact_graph = load_npz(fromfile)
    mkdir('{0}/user_contact_graphs'.format(graph_save_dir))
    tofile = '{0}/user_contact_graphs/hour_{1}.npz'.format(graph_save_dir, hour)
    save_npz(tofile, hour_contact_graph)
    print('finished trace simulate of hour ', hour)


pop_info = []


def trace_and_illness_spread_simulate(pop_num, hour_length):
    start_time = 1621785600
    initial_pop_info(pop_info)
    over_threshold = False
    ill_num = args.init

    for hour in range(Simu_slot):  # 对于每一个小时
        day = hour // 24
        if day < 7:
            copy_pop_info(hour)
        else:
            if hour % 24 == 0:  # 如果当前是某一天的开始
                if ill_num > pop_num * threshold:  # 如果超过阈值
                    over_threshold = True
                else:  # 如果没超过阈值，未来一天的接触图和传播图与原来相同
                    over_threshold = False
            if not over_threshold:
                copy_hour_info(hour, start_time)
            else:
                simulate_trace(hour, start_time, pop_info, over_threshold)
                generate_contact_graph(pop_info, hour)
            if (hour + 1) % 24 == 0 and not hour == 24 * 7:
                day_contact_graph = generate_day_contact_graph(day)
                if not over_threshold:
                    fromfile = '{0}/illness_spread/day_{1}.npy'.format(no_cut_graph_save_dir, day)
                    feature_graph = np.load(fromfile)
                else:
                    feature_graph = simulate_illness_spread(day, day_contact_graph, feature_graph)
                ill_num = np.sum(feature_graph)
                mkdir('{0}/illness_spread'.format(graph_save_dir))
                tofile = '{0}/illness_spread/day_{1}.npy'.format(graph_save_dir, day)
                np.save(tofile, feature_graph)
    return pop_info


Pop_info = trace_and_illness_spread_simulate(pop_num, Simu_slot)
pop_info = np.array(Pop_info)

output = open('{0}/pop_info_th_{1}_cut_{2}.pkl'.format(trajectories_dir, threshold, cut), 'wb')
pkl.dump(pop_info, output)
output.close()
print('finished!')
