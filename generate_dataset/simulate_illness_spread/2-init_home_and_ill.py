from scipy.io import loadmat
import numpy as np
import random
from tqdm import *
from utils import *

args = initialization()
time_slot = 60  # 10 minutes
Network = args.Network
pop_num = args.pop_num
Region_num = args.region_num  # number of regions
save_dir = '../../data/{0}//trajectories'.format(Network)
mkdir(save_dir)

Pop_info = loadmat('../../data/{0}/region_file/population_xa.mat'.format(Network))
Region_pop = Pop_info['population_xa'].reshape((Region_num))
Pop_distribution = Region_pop / Region_pop.sum()
P_t_raw = loadmat("../../data/{0}/region_file/rhythm.mat".format(Network))['rhythm']


def p_t_process(p_t_raw):
    temp = p_t_raw.reshape((-1, int(time_slot / 10)))
    print(temp)
    p_t = np.sum(temp, axis=1)
    p_t = p_t.flatten()
    return p_t


P_t = p_t_process(P_t_raw)
pt_path = '{0}/p_t.npy'.format(save_dir)
np.save(pt_path, P_t)
P_t = np.load(pt_path)


def init_home_location(pop_distribution, pop_num):
    home_location = []
    for i in trange(pop_num):
        rp = random.uniform(0, 1)
        prob_accum = 0
        for j, r in enumerate(pop_distribution):
            prob_accum += r
            if prob_accum > rp:
                home_location.append(j)
                break
            else:
                continue
    print('Initialize home location successfully!')
    return home_location


home_path = '{0}/home_location.npy'.format(save_dir)
Home_location = init_home_location(Pop_distribution, pop_num)
np.save(home_path, Home_location)
Home_location = np.load(home_path)

original = np.zeros(pop_num)
init_ill = args.init
random_numbers = [random.randint(0, pop_num) for _ in range(init_ill)]
for ill in random_numbers:
    original[ill] = 1
original_path = '{0}/original.npy'.format(save_dir)
np.save(original_path, original)
