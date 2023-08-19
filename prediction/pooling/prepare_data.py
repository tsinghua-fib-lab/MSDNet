import networkx as nx
import numpy as np
import torch

import pickle
import random

from graph_sampler_pred import GraphSampler


def prepare_test_data(graphs, args, val_idx, max_nodes=0):
    dataset_sampler = GraphSampler(graphs, normalize=False, max_num_nodes=1887, features=args.feature_type)
    test_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    return test_dataset_loader, dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim
