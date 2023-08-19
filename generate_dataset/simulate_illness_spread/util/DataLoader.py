import torch, tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader


class BaseDataset(Dataset):
    def __init__(self, config, device=torch.device('cpu'), static_feat=None):
        super(BaseDataset, self).__init__()

        self.config = config
        self.data = config['data'].to(device)
        self.min_infected = config['min_infected']

        self.static_feat = static_feat
        self.index = np.arange(self.data.shape[0])
        if self.min_infected > 0:
            self.index = list(set(self.index) & set(np.where((self.data[..., 0] >= 0).sum(-1) >= self.min_infected)[0]))
        if 'init_degree_window' in config:
            self.init_degree_window = config['init_degree_window']
            self.index = list(set(self.index) & set(np.where(self.static_feat.degree >= self.init_degree_window[0])[0]) \
                              & set(np.where(self.static_feat.degree <= self.init_degree_window[1])[0]))
        self.samples, self.num_nodes, _ = self.data.shape
        self.I = self.data[..., 0].long()
        self.R = self.data[..., 1].long()
        self.t1, self.t2 = 0, 0

    def get_state(self, index, t):
        index = self.index[index]
        assert (t >= 0)
        I = ((0 <= self.I[index]) & (self.I[index] <= t)) & ((t < self.R[index]) | (self.R[index] < 0))
        R = (0 <= self.R[index]) & (self.R[index] <= t)
        S = (t < self.I[index]) | (self.I[index] < 0)
        return_value = torch.stack([S, I, R], dim=-1).float()
        return return_value

    def get_all(self, t):
        return_value = torch.stack([self.get_state(index, t) for index in range(len(self))], dim=0)
        return return_value

    def __getitem__(self, index):
        index = self.index[index]
        return_dict = {
            'init': self.get_state(index, self.t1).float(),
            'final': self.get_state(index, self.t2).float(),
            't1': torch.LongTensor([self.t1]),
            't2': torch.LongTensor([self.t2])
        }
        return return_dict

    def __len__(self):
        return len(self.index)

    def set_time(self, init_time, final_time):
        self.t1, self.t2 = init_time, final_time


class StaticFeat:
    def __init__(self, config):
        self.config = config

        self.edges = config['edges']  # [(s, d, contact)]
        self.edge_weight = {}
        self.num_nodes = config['num_nodes']

        self.degree = torch.zeros([self.num_nodes, 1])

        for s, t, c in self.edges:
            pair1, pair2 = (min(s, t), max(s, t)), (max(s, t), min(s, t))
            self.edge_weight[pair1], self.edge_weight[pair2] = c, c
            self.degree[s] += c

        self.normed_degree = self.degree / self.degree.max()
        self.edge_index = torch.LongTensor(list(self.edge_weight.keys())).T
        self.edge_weight = torch.FloatTensor([list(self.edge_weight.values())]).T

        # self.communities = config['communities']
        # self.same_community = \
        #    torch.Tensor([self.communities[s] == self.communities[t] for s, t, in self.edge_index.T]).T.long()


class StochasticDataset(Dataset):
    def __init__(self, config):
        super(StochasticDataset, self).__init__()
        self.config = config
        self.data = config['data']
        self.t1, self.t2 = 0, 0

    def get_state(self, index, t):
        return self.data[index, t]

    def __getitem__(self, index):
        return_dict = {
            'init': self.get_state(index, self.t1).float(),
            'final': self.get_state(index, self.t2).float(),
            't1': torch.LongTensor([self.t1]),
            't2': torch.LongTensor([self.t2])
        }
        return return_dict

    def __len__(self):
        return self.data.shape[0]

    def set_time(self, init_time, final_time):
        self.t1, self.t2 = init_time, final_time


'''
class BaseDataset2(Dataset):
    def __init__(self, config):
        super(BaseDataset2, self).__init__()
        
        self.config = config
        self.data = config['data']
        self.init = self.data[:, 0].long()
        self.I = self.data[:, 1].long()
        self.R = self.data[:, 2].long()
        self.t1, self.t2 = 0, 0
        
    def get_state(self, index, t):
        assert(t >= 0)
        t = t + 1
        I = ((self.I[index] < t) & (self.I[index] >= 0)) & (self.R[index] >= t)
        I[self.init[index].bool() & (self.R[index] >= t)] = 1
        R = (self.R[index] < t) & (self.R[index] >= 0)
        S = torch.logical_not(I) & torch.logical_not(R)
        return torch.stack([S, I, R], dim=-1).long()
    
    def get_all(self, t):
        return torch.stack([self.get_state(index, t) for index in range(self.__len__())], dim=0)

    def __getitem__(self, index):
        init = self.get_state(index, self.t1)
        final = self.get_state(index, self.t2)
        
        return_dict = {
            'init':self.get_state(index, self.t1).float(),
            'final':self.get_state(index, self.t2).float(),
            't1':torch.LongTensor([self.t1]),
            't2':torch.LongTensor([self.t2])
        }
        return return_dict

    def __len__(self):
        return self.init.shape[0]
    
    def set_time(self, init_time, final_time):
        self.t1, self.t2 = init_time, final_time
        if type(self.t2) == list:
            self.t2 = np.random.randint(*final_time)
'''
