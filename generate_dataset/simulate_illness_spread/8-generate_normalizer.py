import numpy as np
from tqdm import *
import networkx as nx
import pickle as pkl
from utils import *

args = initialization()
Network = args.Network
graph_dir = args.graph_save_dir

graph_dir = '../../data/{0}/graphs/no_cut'.format(Network)
contact_graph_dir = '{0}/user_contact_graphs_day'.format(graph_dir)
region_graph_dir = '{0}/region_contact_graphs_day'.format(graph_dir)
region_nx_dir = '{0}/region_contact_graphs_day_nx'.format(graph_dir)
mkdir(region_graph_dir)
mkdir(region_nx_dir)

degree_record = np.zeros((331, 100))

unique_loc_list = np.load('../../data/{0}/region_file/unique_loc_list.npy'.format(Network))

for day in trange(7, 107):
    matrix_filename_day = '{0}/day_{1}'.format(region_graph_dir, day)
    for index, loc in enumerate(unique_loc_list):
        matrix_filename = '{0}/region_{1}.npy'.format(matrix_filename_day, loc)
        contact_graph = np.load(matrix_filename)
        G = nx.from_numpy_matrix(contact_graph)
        num_nodes = len(G.nodes)
        avg_degree = (len(G.edges) * 2) / num_nodes
        degree_record[loc, day - 7] = avg_degree

normalizer = {}
for i, each_loc in enumerate(unique_loc_list):  # 对于每一个location的each_Loc,i为索引
    normalizer[each_loc] = (np.mean(degree_record[i]), np.std(degree_record[i]))  # 每一个位置的S为对应的susceptible的均值和标准差
with open('../../data/BJ-331/region_file/normalizer.pkl', 'wb') as f:
    pkl.dump(normalizer, f)
