import numpy as np
import sys
from tqdm import *

sys.path.append("/data2/tangyinzhou/MMSTAN")
from utils import *

spread_func = 'ICM'
cut = 0
threshold = 0

args = initialization()
simu_slot = args.simu_slot
time_slot = 60  # 10 minutes，时间片
Network = 'BJ-331'
Region_num = 331  # number of regions

if cut == 0 and threshold == 0:
    graph_dir = '../../data/{0}/graphs/no_cut'.format(Network)
else:
    graph_dir = '../../data/{0}/graphs/th_{1}_cut_{2}'.format(Network, threshold, cut)
ill_dir = '{0}/illness_spread'.format(graph_dir)
home_loc = np.load('../../data/{0}/trajectories/home_location.npy'.format(Network))
save_dir = '{0}/region_illness_record'.format(graph_dir)
mkdir(save_dir)
region_ill = []
for day in trange(7, 107):
    region_day_ill = np.zeros((Region_num))
    illness_file = '{0}/day_{1}.npy'.format(ill_dir, day)
    illness_feature = np.load(illness_file)
    for user_idx, user_ill in enumerate(illness_feature):
        home = home_loc[user_idx]
        region_day_ill[home] += user_ill
    region_ill.append(list(region_day_ill))
region_ill = np.array(region_ill).reshape(-1, Region_num)
home_region = []
for region_idx in range(Region_num):
    region_ill_record = region_ill[:, region_idx].reshape(-1)
    if not np.sum(region_ill_record) == 0:
        home_region.append(region_idx)
        save_file = '{0}/region_{1}.npy'.format(save_dir, region_idx)
        np.save(save_file, region_ill_record)
home_region = np.array(home_region)
np.save('../../data/{0}/trajectories/home_region.npy'.format(Network), home_region)
