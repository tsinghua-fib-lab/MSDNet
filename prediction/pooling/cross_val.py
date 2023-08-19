'''
DIFFPOOL回归任务版的交叉验证数据生成程序
'''
import networkx as nx
import numpy as np
import torch

import pickle
import random

from graph_sampler import GraphSampler


def prepare_val_data(graphs, args, val_idx, max_nodes=0):
    random.shuffle(graphs)  # 将graph的信息打乱
    val_size = len(graphs) // 5  # 得到val的大小，将数据集的十分之一作为val集
    train_graphs = graphs[:val_idx * val_size]  # 得到train为除了val之外的
    if val_idx < 9:
        train_graphs = train_graphs + graphs[(val_idx + 1) * val_size:]  # 构建train set
    val_graphs = graphs[val_idx * val_size: (val_idx + 1) * val_size]  # 构建val set
    print('Num training graphs: ', len(train_graphs),
          '; Num validation graphs: ', len(val_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ',
          max([G.number_of_nodes() for G in graphs]), ', '
                                                      "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])),
          ', '
          "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, \
           dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim
