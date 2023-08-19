import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorboardX
from tensorboardX import SummaryWriter
import random
import time
import cross_val
import encoders
import util
from tqdm import *
from utils import *

print('start')
args1 = initialization()
Network = args1.Network
graph_dir = '../../data/{0}/graphs/no_cut'.format(Network)
save_dir = '../../data/{0}/save'.format(Network)
mkdir(save_dir)
print(args1)


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
                        lr=1e-2,
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


def log_assignment(assign_tensor, writer, epoch, batch_idx):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8, 6), dpi=300)

    # has to be smaller than args.batch_size
    for i in range(len(batch_idx)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(assign_tensor.cpu().data.numpy()[batch_idx[i]], cmap=plt.get_cmap('BuPu'))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")
    plt.tight_layout()
    fig.canvas.draw()

    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('assignment', data, epoch)


def log_graph(adj, batch_num_nodes, writer, epoch, batch_idx, assign_tensor=None):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8, 6), dpi=300)

    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i + 1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()
        G = nx.from_numpy_matrix(adj_matrix)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color='#336699',
                edge_color='grey', width=0.5, node_size=300,
                alpha=0.7)
        ax.xaxis.set_visible(False)

    plt.tight_layout()
    fig.canvas.draw()

    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('graphs', data, epoch)

    # colored according to assignment
    assignment = assign_tensor.cpu().data.numpy()
    fig = plt.figure(figsize=(8, 6), dpi=300)

    num_clusters = assignment.shape[2]
    all_colors = np.array(range(num_clusters))

    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i + 1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()

        label = np.argmax(assignment[batch_idx[i]], axis=1).astype(int)
        label = label[: batch_num_nodes[batch_idx[i]]]
        node_colors = all_colors[label]

        G = nx.from_numpy_matrix(adj_matrix)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color=node_colors,
                edge_color='grey', width=0.4, node_size=50, cmap=plt.get_cmap('Set1'),
                vmin=0, vmax=num_clusters - 1,
                alpha=0.8)

    plt.tight_layout()
    fig.canvas.draw()

    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('graphs_colored', data, epoch)


def evaluate(dataset, model, args, name='Validation', max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):  # 对于dataset中的第batch_idx条数据data来说
        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        h0 = Variable(data['feats'].float()).cuda()
        labels.append(data['label'].float().numpy())
        label = Variable(data['label'].float()).cuda()
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()

        ypred, _ = model(h0, adj, batch_num_nodes, assign_x=assign_input)

        preds.append(ypred.squeeze().cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break
    # print(ypred)
    # print(label)
    labels = np.hstack(labels)  # 把labels矩阵按行连接
    preds = np.hstack(preds)

    result = {'mse': metrics.mean_absolute_error(labels, preds)}
    print(name, " mse:", result['mse'])

    return result


def gen_train_plt_name(args):
    return './results/' + gen_prefix(args) + '.png'


def train(dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
          mask_nodes=True):
    writer_batch_idx = [0, 3, 6, 9]
    file_name = '{0}/prediction_pooling'.format(save_dir)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-3)
    iter = 0
    best_val_result = {
        'epoch': 0,
        'loss': 0,
        'mse': 1e10}
    test_result = {
        'epoch': 0,
        'loss': 0,
        'mse': 1e10}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []
    for epoch in range(args.num_epochs):  # 对于每一个epoch
        total_time = 0
        avg_loss = 0.0
        model.train()
        print('Epoch: ', epoch)
        for batch_idx, data in enumerate(dataset):  # 对于dataset中的每一条数据data，其索引为batch_idx
            begin_time = time.time()  # 计时开始
            # begin_time = 0
            model.zero_grad()  # 梯度清零
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()  # 将adj传到cuda上
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()  # 将feature传到cuda上，作为第0层的输入
            label = Variable(data['label'].float()).cuda()  # 将label传到cuda上，作为每一张图的标记
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None  # 得到每一张图的节点个数
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()  # 将assign_feats传到cuda上

            ypred, yout = model(h0, adj, batch_num_nodes, assign_x=assign_input)  # 得到输出结果ypred
            Before = list(model.parameters())[0].clone()
            # last_Before = list(model.parameters())[-1].clone()
            loss = model.loss(ypred, label)  # 计算相应的loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            After = list(model.parameters())[0].clone()
            # last_After = list(model.parameters())[-1].clone()
            iter += 1
            avg_loss += loss

            elapsed = time.time() - begin_time
            total_time += elapsed

        avg_loss /= batch_idx + 1
        if writer is not None:
            writer.add_scalar('loss/avg_loss', avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar('loss/linkpred_loss', model.link_loss, epoch)
        print('Avg loss: ', avg_loss, '; epoch time: ', total_time)
        print('模型的第0层更新幅度：', torch.sum(After - Before))
        # print('模型的第-1层更新幅度：', torch.sum(last_After - last_Before))
        result = evaluate(dataset, model, args, name='Train', max_num_examples=100)
        train_accs.append(result['mse'])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model, args, name='Validation')  # 在验证集上进行评估
            val_accs.append(val_result['mse'])
        if val_result['mse'] < best_val_result['mse'] + 1e-10:  # 如果在验证集上的结果比最好的还要好
            best_val_result['mse'] = val_result['mse']
            best_val_result['epoch'] = epoch
            best_val_result['loss'] = avg_loss
            state = {
                'state': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, file_name)
        if test_dataset is not None:
            test_result = evaluate(test_dataset, model, args, name='Test')
            test_result['epoch'] = epoch
        if writer is not None:
            writer.add_scalar('mse/train_mse', result['mse'], epoch)
            writer.add_scalar('mse/val_mse', val_result['mse'], epoch)
            writer.add_scalar('loss/best_val_loss', best_val_result['loss'], epoch)
            if test_dataset is not None:
                writer.add_scalar('mse/test_mse', test_result['mse'], epoch)

        print('Best val result: ', best_val_result)
        best_val_epochs.append(best_val_result['epoch'])
        best_val_accs.append(best_val_result['mse'])
        if test_dataset is not None:
            print('Test result: ', test_result)
            test_epochs.append(test_result['epoch'])
            test_accs.append(test_result['mse'])

    matplotlib.style.use('seaborn')
    plt.switch_backend('agg')
    plt.figure()
    plt.plot(train_epochs, util.exp_moving_avg(train_accs, 0.85), '-', lw=1)
    if test_dataset is not None:
        plt.plot(best_val_epochs, best_val_accs, 'bo', test_epochs, test_accs, 'go')
        plt.legend(['train', 'val', 'test'])
    else:
        plt.plot(best_val_epochs, best_val_accs, 'bo')
        plt.legend(['train', 'val'])
    plt.savefig(gen_train_plt_name(args), dpi=600)
    plt.close()
    matplotlib.style.use('default')

    return model, val_accs


def benchmark_task_val(args, writer=None, feat='node-label'):
    unique_loc_list = np.load('../../data/{0}/region_file/unique_loc_list.npy'
                              .format(Network))
    all_vals = []
    graphs = []
    for day in trange(20):
        random.shuffle(unique_loc_list)
        loc_list = unique_loc_list[0:100]
        # loc_list = unique_loc_list
        for loc in loc_list:
            filename = '{0}/region_contact_graphs_day_nx/' \
                       'day_{1}/region_{2}_nx.gpickle' \
                .format(graph_dir, day * 5 + 7, loc)
            G = nx.read_gpickle(filename)
            graphs.append(G)
        # print('load ' + str(day) + ' graph')

    print('Using node labels')
    for G in graphs:  # 对于每一张图G
        # G.graph['label'] = G.graph['avg_degree_normalized']
        G.graph['label'] = G.graph['avg_degree']
        for u in G.nodes():  # 对于每一个节点u
            # util.node_dict(G)[u]['feat'] = np.array(util.node_dict(G)[u]['degree_normalized'])  # 令节点的feat属性为节点的label属性
            util.node_dict(G)[u]['feat'] = np.array(util.node_dict(G)[u]['degree'])  # 令节点的feat属性为节点的label属性

    for i in range(1):
        train_dataset, val_dataset, max_num_nodes, input_dim, assign_input_dim = \
            cross_val.prepare_val_data(graphs, args, i, max_nodes=args.max_nodes)
        # if args.method == 'soft-assign':
        print('Method: soft-assign')
        model = encoders.SoftPoolingGcnEncoder(
            max_num_nodes,
            input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
            args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
            bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
            assign_input_dim=assign_input_dim).cuda()

        print(model)
        _, val_accs = train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=None,
                            writer=writer)
        all_vals.append(np.array(val_accs))
    all_vals = np.vstack(all_vals)
    all_vals = np.mean(all_vals, axis=0)
    print(all_vals)
    print(np.max(all_vals))
    print(np.argmax(all_vals))


def main():
    prog_args = arg_parse()  # 得到输入的参数

    # export scalar data to JSON for external processing
    path = os.path.join(prog_args.logdir, gen_prefix(prog_args))
    if os.path.isdir(path):  # 如果path为一个目录，也就是有当前的目录
        print('Remove existing log dir: ', path)  # 输出目录
        # shutil.rmtree(path)  # 删除目录里面的内容
        path = path + '_new1'
    writer = SummaryWriter(path)  # 定义文件位置

    os.environ['CUDA_VISIBLE_DEVICES'] = prog_args.cuda  # 设置使用的cudaGPU
    print('CUDA', prog_args.cuda)

    benchmark_task_val(prog_args, writer=writer)

    writer.close()


if __name__ == "__main__":
    main()
