import networkx as nx
import numpy as np
import torch
import torch.utils.data
import pickle as pkl
import util


class GraphSampler(torch.utils.data.Dataset):
    ''' Sample graphs and nodes in graph
    '''

    def __init__(self, G_list, features='default', normalize=True, assign_feat='default', max_num_nodes=0):
        self.adj_all = []  # 所有图的邻接矩阵
        self.len_all = []  # 所有图的节点个数
        self.feature_all = []  # 所有图的节点特征
        self.label_all = []

        self.assign_feat_all = []  # 所有图依据什么聚合，这个矩阵就是什么

        self.max_num_nodes = max_num_nodes  # 最大节点数

        # self.feat_dim = util.node_dict(G_list[0])[0]['feat'].shape[0]  # feat_dim为一个节点特征的维度数
        self.max_num_nodes = max([G.number_of_nodes() for G in G_list])  # 最大节点数
        self.feat_dim = 1

        for G in G_list:  # 对于G_list中的每一个G来说
            # adj = np.array(nx.to_numpy_matrix(G))  # adj为图G的邻接矩阵
            adj = np.array(nx.to_numpy_matrix(G))
            self.adj_all.append(adj)  # 记录下本张图的邻接矩阵
            self.len_all.append(G.number_of_nodes())  # 记录本张图的节点数
            self.label_all.append(G.graph['label'])  # 记录图的label
            # feat matrix: max_num_nodes x feat_dim
            f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)  # f为最大节点数乘以节点特征的矩阵
            for i, u in enumerate(G.nodes()):  # 对于图G中的第i个节点u
                f[i, :] = util.node_dict(G)[u]['feat']  # f的第i行为节点名称为u的节点的feat
            self.feature_all.append(f)  # 将本张图所有节点的特征加入记录
            self.assign_feat_all.append(self.feature_all[-1])  # 将所有特征的最后一个维度加到记录中去

        self.feat_dim = self.feature_all[0].shape[1]  # feat_dim为节点特征的维度
        self.assign_feat_dim = self.assign_feat_all[0].shape[1]  # assign_feat_dim为a节点聚合依据矩阵的特征维度

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        # use all nodes for aggregation (baseline)

        return {'adj': adj_padded,
                'feats': self.feature_all[idx].copy(),
                'label': self.label_all[idx],
                'num_nodes': num_nodes,
                'assign_feats': self.assign_feat_all[idx].copy()}
