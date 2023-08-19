import torch
import os
import argparse
import prepare_data
import encoders
import util
import networkx as nx
import numpy as np
from torch.autograd import Variable
from tqdm import *
import sys

sys.path.append("/data2/tangyinzhou/MMSTAN")
from utils import *

threshold = 0.75
cut = 0.5

args1 = initialization()
Network = args1.Network

if cut == 0 and threshold == 0:
    graph_dir = '../../data/{0}/graphs/no_cut'.format(Network)
else:
    graph_dir = '../../data/{0}/graphs/th_{1}_cut_{2}'.format(Network, threshold, cut)

model_dir = '../../data/{0}/save'.format(Network)


def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')  # 定义命令行解析器对象
    io_parser = parser.add_mutually_exclusive_group(required=False)  # 创建一个互斥组为io_parser
    io_parser.add_argument('--dataset', dest='dataset',
                           help='Input dataset.')  # 增加命令行参数
    benchmark_parser = io_parser.add_argument_group()  # 创建一个组
    benchmark_parser.add_argument('--bmname', dest='bmname',
                                  help='Name of the benchmark dataset')  # 在组里加上一个命令行参数
    io_parser.add_argument('--pkl', dest='pkl_fname',
                           help='Name of the pkl data file')

    softpool_parser = parser.add_argument_group()  # softpool组
    softpool_parser.add_argument('--assign-ratio', dest='assign_ratio', type=float,
                                 help='ratio of number of nodes in consecutive layers')
    softpool_parser.add_argument('--num-pool', dest='num_pool', type=int,
                                 help='number of pooling_to_generate_data layers')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
                        const=True, default=False,
                        help='Whether link prediction side objective is used')

    parser.add_argument('--datadir', dest='datadir',
                        help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
                        help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda',
                        help='CUDA.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
                        help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
                        help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
                        help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
                        help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
                        help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
                        help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
                        help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
                        help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
                        help='Number of graph convolution layers before each pooling_to_generate_data')
    parser.add_argument('--nobn', dest='bn', action='store_const',
                        const=False, default=True,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
                        help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--no-log-graph', dest='log_graph', action='store_const',
                        const=False, default=True,
                        help='Whether disable log graph')

    parser.add_argument('--method', dest='method',
                        help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--name-suffix', dest='name_suffix',
                        help='suffix added to the output filename')

    parser.set_defaults(datadir='data',
                        logdir='log',
                        dataset='syn1v2',
                        max_nodes=100,
                        cuda='2',
                        feature_type='default',
                        lr=0.001,
                        clip=2.0,
                        batch_size=16,
                        num_epochs=1000,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=0,
                        input_dim=10,
                        hidden_dim=60,
                        output_dim=60,
                        num_classes=1,
                        num_gc_layers=3,
                        dropout=0.0,
                        method='soft-assign',
                        name_suffix='',
                        assign_ratio=0.1,
                        num_pool=1,
                        bmname='ENZYMES',
                        )
    return parser.parse_args()


def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname
    else:
        name = args.dataset
    name += '_' + args.method
    if args.method == 'soft-assign':
        name += '_l' + str(args.num_gc_layers) + 'x' + str(args.num_pool)
        name += '_ar' + str(int(args.assign_ratio * 100))
        if args.linkpred:
            name += '_lp'
    else:
        name += '_l' + str(args.num_gc_layers)
    name += '_h' + str(args.hidden_dim) + '_o' + str(args.output_dim)
    if not args.bias:
        name += '_nobias'
    if len(args.name_suffix) > 0:
        name += '_' + args.name_suffix
    return name


def gen_train_plt_name(args):
    return './results/' + gen_prefix(args) + '.png'


def evaluate(dataset, model, args, name='Test', max_num_examples=None):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.03)
    file_name = '{0}/prediction_pooling'.format(model_dir)
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['state'])
    optimizer.load_state_dict(checkpoint['optimizer'])  # 得到当loss最小时的模型参数
    model.eval()
    # labels = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataset):  # 对于dataset中的第batch_idx条数据data来说
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            h0 = Variable(data['feats'].float()).cuda()
            # labels.append(data['label'].float().numpy())
            batch_num_nodes = data['num_nodes'].int().numpy()
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()

            ypred, yout = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            del adj, h0, batch_num_nodes, assign_input
            torch.cuda.empty_cache()
            if max_num_examples is not None:
                if (batch_idx + 1) * args.batch_size > max_num_examples:
                    break

    # labels = np.hstack(labels)  # 把labels矩阵按行连接

    return ypred, yout


def main():
    args = arg_parse()  # 得到输入的参数
    save_filename_pre = '{0}/pooling_output'.format(graph_dir)
    mkdir(save_filename_pre)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda  # 设置使用的cudaGPU
    print('CUDA', args.cuda)
    i = 0
    unique_loc_list = np.load('../../data/{0}/region_file/unique_loc_list.npy'.format(Network))
    for day in trange(7, 107):
        graphs = []
        for region in unique_loc_list:
            filename = '{0}/region_contact_graphs_day_nx/' \
                       'day_{1}/region_{2}_nx.gpickle' \
                .format(graph_dir, day, region)
            G = nx.read_gpickle(filename)
            graphs.append(G)

        for G in graphs:  # 对于每一张图G
            # G.graph['label'] = G.graph['avg_degree_normalized']
            G.graph['label'] = G.graph['avg_degree']
            for u in G.nodes():  # 对于每一个节点u
                util.node_dict(G)[u]['feat'] = np.array(
                    util.node_dict(G)[u]['degree'])  # 令节点的feat属性为节点的label属性
        # print('loaded ' + str(day))
        test_dataset, max_num_nodes, input_dim, assign_input_dim = \
            prepare_data.prepare_test_data(graphs, args, i, max_nodes=args.max_nodes)

        model = encoders.SoftPoolingGcnEncoder(
            max_num_nodes, input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
            args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
            bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
            assign_input_dim=assign_input_dim).cuda()

        test_pred, test_output = evaluate(test_dataset, model, args)
        test_output = test_output.cpu().data.numpy()
        print(test_output.shape)
        save_filename = '{0}/day_{1}.npy'.format(save_filename_pre, day)
        np.save(save_filename, test_output)


if __name__ == "__main__":
    main()
