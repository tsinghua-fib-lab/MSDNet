import numpy as np
import scipy.sparse as sp
from tqdm import *
import networkx as nx
import util
import sys

sys.path.append("/data2/tangyinzhou/MMSTAN")
from utils import *

args = initialization()
Network = args.Network
graph_dir = args.graph_save_dir
cut = 0
threshold = 0

if cut == 0 and threshold == 0:
    graph_dir = '../../data/{0}/graphs/no_cut'.format(Network)
else:
    graph_dir = '../../data/{0}/graphs/th_{1}_cut_{2}'.format(Network, threshold, cut)

contact_graph_dir = '{0}/user_contact_graphs_day'.format(graph_dir)
region_graph_dir = '{0}/region_contact_graphs_day'.format(graph_dir)
mkdir(region_graph_dir)

unique_loc_list = np.load('../../data/{0}/region_file/unique_loc_list.npy'.format(Network))
home_dict = np.load('../../data/{0}/region_file/loc_dict.npy'.format(Network), allow_pickle=True)
home_dict = np.array(home_dict).reshape(-1)[0]
for day in trange(7, 107):
    filename = '{0}/day_{1}.npz'.format(contact_graph_dir, day)
    time_contact_graph = sp.load_npz(filename)
    time_contact_graph_coo = time_contact_graph.tocoo()
    degree = len(time_contact_graph_coo.data) / 50000
    dense = time_contact_graph.toarray()
    dir_name = '{0}/day_{1}'.format(region_graph_dir, day)
    mkdir(dir_name)
    for dict in home_dict:
        graph = dense[home_dict[dict]]
        graph = graph[:, home_dict[dict]]
        graph = np.int8(graph > 0)
        filename_region = '{0}/region_{1}.npy'.format(dir_name, dict)
        np.save(filename_region, graph)
