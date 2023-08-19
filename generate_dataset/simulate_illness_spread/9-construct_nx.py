import numpy as np
import scipy.sparse as sp
from tqdm import *
import networkx as nx
import util
import sys
import pickle as pkl

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
region_nx_dir = '{0}/region_contact_graphs_day_nx'.format(graph_dir)
mkdir(region_graph_dir)
mkdir(region_nx_dir)

unique_loc_list = np.load('../../data/{0}/region_file/unique_loc_list.npy'.format(Network))
with open('../../data/BJ-331/region_file/normalizer.pkl', 'rb') as f:
    normalizer = pkl.load(f)

for day in trange(7, 107):
    matrix_filename_day = '{0}/day_{1}'.format(region_graph_dir, day)
    nx_filename_day = '{0}/day_{1}'.format(region_nx_dir, day)
    mkdir(nx_filename_day)
    for index, loc in enumerate(unique_loc_list):
        (mean, std) = normalizer[loc]
        matrix_filename = '{0}/region_{1}.npy'.format(matrix_filename_day, loc)
        nx_filename = '{0}/region_{1}_nx.gpickle'.format(nx_filename_day, loc)
        contact_graph = np.load(matrix_filename)
        G = nx.from_numpy_matrix(contact_graph)
        num_nodes = len(G.nodes)
        avg_degree = (len(G.edges) * 2) / num_nodes
        if mean == 0 and std == 0:
            degree = 0
        else:
            degree = (avg_degree - mean) / std
        G.graph['avg_degree'] = degree
        for u in util.node_iter(G):
            util.node_dict(G)[u]['degree'] = G.degree(u)
        nx.write_gpickle(G, nx_filename)
