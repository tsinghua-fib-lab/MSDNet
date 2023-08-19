'''
Encoders of diffpool regression
'''
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np


# from set2set import Set2Set


# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
                 dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)  # 将adj和x相乘
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)  # 将其乘上权重矩阵
        if self.bias is not None:
            y = y + self.bias  # 加上偏置
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)  # 进行归一化
            # print(y[0][0])
        return y


class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            input_dim, hidden_dim, embedding_dim, num_layers,
            add_self, normalize=True, dropout=dropout)
        self.act = nn.LeakyReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                                 label_dim, num_aggs=self.num_aggs)

        for m in self.modules():  # 对于self里面的每一个module
            print(m)
            if isinstance(m, GraphConv):  # 如果m为GraphConv类型的
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))  # 初始化权重值
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)  # 初始化权重值

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
                          normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                               normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                       normalize_embedding=normalize, dropout=dropout, bias=self.bias)
             for i in range(num_layers - 2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                              normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs  # 为pred_input_dim乘上需要聚合的个数
        if len(pred_hidden_dims) == 0:  # 如果预测不需要隐藏层
            pred_model = nn.Linear(pred_input_dim, label_dim)  # 预测模型为一个输入到输出的线性层
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        对于batch_num_nodes中的每个num_nodes，对应列的第一个num_nodes项是1，其余为0(需要屏蔽)。
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]  # 对于batch_num_nodes中的每一个num，有为其长度的1
        batch_size = len(batch_num_nodes)  # 得到batch的size大小
        out_tensor = torch.zeros(batch_size, max_nodes)  # 初始化一个batch_size乘以max_nodes的maks矩阵
        for i, mask in enumerate(packed_masks):  # 对于packed_masks中的第i个mask来说
            out_tensor[i, :batch_num_nodes[i]] = mask  # 得到其有效数据的mask
        return out_tensor.unsqueeze(2).cuda()  # 返回mask矩阵

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''

        x = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        # out_all = []
        # out, _ = torch.max(x, dim=1)
        # out_all.append(out)
        for i in range(len(conv_block)):
            x = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x, adj)

        # 以下为新加
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        # 新加结束

        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]  # 图中有多少个节点
        if batch_num_nodes is not None:  # batch_num_nodes为每一个batch图中有多少有效节点
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x = self.conv_first(x, adj)  # 经过第一层卷积层
        x = self.act(x)  # 经过激活层
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)  # out为x按照dim=1的维度得到的最大值
        out_all.append(out)
        for i in range(self.num_layers - 2):
            x = self.conv_block[i](x, adj)  # 进行中间层的前向传播
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out, _ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x, adj)  # 经过最后一层卷积层

        # 以下为新加
        x = self.act(x)  # 经过激活层
        if self.bn:
            x = self.apply_bn(x)
        # 新加结束

        # x = self.act(x)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)  # 将每一层得到的输出拼接起来
        else:
            output = out
        ypred = self.pred_model(output)  # 将结果经过预测层
        # print(output.size())
        return ypred  # 返回预测值

    def loss(self, pred, label, type='softmax'):
        # softmax + CE
        if type == 'softmax':
            return F.mse_loss(pred.squeeze(), label.squeeze())
        elif type == 'margin':
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1, 1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)

        # return F.binary_cross_entropy(F.sigmoid(pred[:,0]), label.float())


class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
                 pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True,
                 assign_input_dim=-1, args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling_to_generate_data
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                                    num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat,
                                                    args=args)
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True

        # GC
        self.conv_first_after_pool = nn.ModuleList()
        self.conv_block_after_pool = nn.ModuleList()
        self.conv_last_after_pool = nn.ModuleList()
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
                self.pred_input_dim, hidden_dim, embedding_dim, num_layers,
                add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(conv_first2)
            self.conv_block_after_pool.append(conv_block2)
            self.conv_last_after_pool.append(conv_last2)

        # assignment
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = nn.ModuleList()
        self.assign_conv_block_modules = nn.ModuleList()
        self.assign_conv_last_modules = nn.ModuleList()
        self.assign_pred_modules = nn.ModuleList()
        assign_dim = int(max_num_nodes * assign_ratio)
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                normalize=True)
            assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
            assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)

            # next pooling_to_generate_data layer
            assign_input_dim = self.pred_input_dim
            assign_dim = int(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(assign_conv_first)
            self.assign_conv_block_modules.append(assign_conv_block)
            self.assign_conv_last_modules.append(assign_conv_last)
            self.assign_pred_modules.append(assign_pred)

        self.pred_model = self.build_pred_layers(self.pred_input_dim * (num_pooling + 1), pred_hidden_dims,
                                                 5, num_aggs=self.num_aggs)
        self.out_model = nn.Linear(5, label_dim)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('leaky_relu'))
                if m.bias is not None:
                    m.bias.data = nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x, adj, batch_num_nodes, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x

        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []

        embedding_tensor = self.gcn_forward(x, adj,
                                            self.conv_first, self.conv_block, self.conv_last, embedding_mask)

        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None

            self.assign_tensor = self.gcn_forward(x_a, adj,
                                                  self.assign_conv_first_modules[i], self.assign_conv_block_modules[i],
                                                  self.assign_conv_last_modules[i],
                                                  embedding_mask)
            # [batch_size x num_nodes x next_lvl_num_nodes]
            self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred_modules[i](self.assign_tensor))
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask

            # update pooled features and adj matrix
            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x

            embedding_tensor = self.gcn_forward(x, adj,
                                                self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                                                self.conv_last_after_pool[i])

            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                # out = torch.mean(embedding_tensor, dim=1)
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)

        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        yout = self.pred_model(output)
        ypred = self.out_model(yout)

        return ypred, yout

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop - 1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).cuda())
            self.link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[(1 - adj_mask).bool()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            # print('linkloss: ', self.link_loss)
            return loss + self.link_loss
        return loss
