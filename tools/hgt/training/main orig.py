# This file is a modified version of:
# - https://github.com/acbull/pyHGT/blob/master/OAG/train_paper_venue.py
# - https://github.com/acbull/pyHGT/blob/master/OAG/train_paper_field.py

# For single-classification task (i.e. main-node is associated with one sub-node) keep the defualt of multi_lable_task: "False".
# For mult-classification task (i.e. main-node is associated with multiple sub-nodes) set the value of multi_lable_task to "True".

import argparse
import os
import json
import time
import multiprocessing as mp
import numpy as np
import torch.nn as nn
import networkx as nx
from tools.hgt.pyHGT.datax import *
from tools.hgt.pyHGT.model import GNN, Classifier
from tools.hgt.pyHGT.attention import *
from tools.hgt.utils.utils import randint, ndcg_at_k, mean_reciprocal_rank, logger
#import torch
from torch.utils.tensorboard import SummaryWriter



from warnings import filterwarnings
filterwarnings("ignore")

parser = argparse.ArgumentParser(
    description='Training GNN on main_node - sub_node classification task')

"""Dataset arguments"""
parser.add_argument('--graph_dir', type=str, default='storage/OAG_grap_reddit_10000.pk',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='storage',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--graph_params_dir', type=str, default='config/HGT_graph_params_Reddit.json',
                    help='The address of the graph params file')
parser.add_argument('--main_node', type=str, default='post',
                    help='The name of the main node in the graph')
parser.add_argument('--predicted_node_name', type=str, default='subreddit',
                    help='The name of the node that its values to be predicted')
parser.add_argument('--edge_name', type=str, default='post_subreddit',
                    help='The name of edge')
parser.add_argument('--exract_attention', type=bool, default=False,
                    help='extract the attention lists')
parser.add_argument('--show_tensor_board', type=bool, default=False,
                    help='show tensor board')

"""Model arguments """
parser.add_argument('--multi_lable_task', type=bool, default=True,
                    help='Multi label classification task')
parser.add_argument('--conv_name', type=str, default='hgt',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=400,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=4,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout ratio')
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=128,
                    help='How many nodes to be sampled per layer per type')

"""Optimization arguments"""
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--data_percentage', type=float, default=1.0,
                    help='Percentage of training and validation data to use')
parser.add_argument('--n_epoch', type=int, default=15,
                    help='Number of epoch to run')
parser.add_argument('--n_pool', type=int, default=4,
                    help='Number of process to sample subgraph')
parser.add_argument('--n_batch', type=int, default=32,
                    help='Number of batch (sampled graphs) for each epoch')
parser.add_argument('--repeat', type=int, default=2,
                    help='How many time to train over a singe batch (reuse data)')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of output nodes for training')
parser.add_argument('--clip', type=float, default=0.25,
                    help='Gradient Norm Clipping')
parser.add_argument('--include_fake_edges', type=bool, default=False,
                    help='Include fake edges')
parser.add_argument('--remove_edges', type=bool, default=False,
                    help='remove weak edges')
parser.add_argument('--restructure_at_epoch', type=int, default=10,
                    help='restructure graph at after x epoch')

writer = SummaryWriter()

start_time = time.time()

        
args = parser.parse_args()
logger(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# replace with read gRAPH
graph = nx.read_gpickle(args.graph_dir)
graph.graph['edge_list'] = get_edge_list(graph)
logger(graph.graph['meta'])
with open(args.graph_params_dir) as json_file:
    graph_params = json.load(json_file)

weight_thresholds = graph_params['weight_split_range']['valid_range']

train_range = {w: True for w in graph.graph['weights'] if w !=
               None and w < weight_thresholds['start']}
valid_range = {w: True for w in graph.graph['weights'] if w !=
               None and w >= weight_thresholds['start'] and w <= weight_thresholds['end']}
test_range = {w: True for w in graph.graph['weights'] if w !=
              None and w > weight_thresholds['end']}

rev_edge_name = f'rev_{args.edge_name}'

"""cand_list stores all the sub-nodes, which is the classification domain."""
# selected_edges = [(u,v) for u,v,e in graph.edges(data=True) if e['edge_type'] == args.edge_name]
cand_list = [u for u,v,e in graph.edges(data=True) if e['edge_type'] == rev_edge_name]
cand_list = list(set(cand_list))
# cand_list = list(graph.edge_list[args.predicted_node_name]
#                  [args.main_node][args.edge_name].keys())

if not args.multi_lable_task:
    """
        Using Negative Log Likelihood Loss (torch.nn.NLLLoss()), since each main-node can be associated with one sub-node.
        consider using CrossEntropy (log-softmax + NLL) 
    """
    criterion = nn.NLLLoss()
else:
    """
        Use KL Divergence here, since each main node can be associated with multiple predicted sub nodes.
        Thus this task is a multi-label classification.
    """
    criterion = nn.KLDivLoss(reduction='batchmean')


def node_classification_sample(seed, pairs, weight_range):
    """
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (main-nodes) and their weight.
    """
    np.random.seed(seed)
    target_ids = np.random.choice(
        list(pairs.keys()), args.batch_size, replace=False)
    target_info = []
    for target_id in target_ids:
        _, _weight = pairs[target_id]
        target_info += [[target_id, _weight]]

    """(2) Based on the seed nodes, sample a subgraph with 'sampled_depth' and 'sampled_number'"""
    feature, weights, edge_list, indxs, _ = sample_subgraph(graph, weight_range, graph_params,
                                                            inp={args.main_node: np.array(
                                                                target_info)},
                                                            sampled_depth=args.sample_depth, sampled_number=args.sample_width)

    """(3) Mask out the edge between the output target nodes (main-node) with output source nodes (sub-node)"""
    masked_edge_list = []
    for i in edge_list[args.main_node][args.predicted_node_name][rev_edge_name]:
        if i[0] >= args.batch_size:
            masked_edge_list += [i]
    edge_list[args.main_node][args.predicted_node_name][rev_edge_name] = masked_edge_list

    masked_edge_list = []
    for i in edge_list[args.predicted_node_name][args.main_node][args.edge_name]:
        if i[1] >= args.batch_size:
            masked_edge_list += [i]
    edge_list[args.predicted_node_name][args.main_node][args.edge_name] = masked_edge_list

    """(4) Transform the subgraph into torch Tensor (edge_index is in format of pytorch_geometric)"""
    node_feature, node_type, edge_weight, edge_index, edge_type, node_dict, edge_dict = \
        to_torch(feature, weights, edge_list, graph, args.include_fake_edges)
    """
        (5) Prepare the labels for each output target node (main-node), and their index in sampled graph.
            (node_dict[type][0] stores the start index of a specific type of nodes)
    """
    if not args.multi_lable_task:
        ylabel = torch.zeros(args.batch_size, dtype=torch.long)
        for x_id, target_id in enumerate(target_ids):
            ylabel[x_id] = cand_list.index(pairs[target_id][0])
    else:
        ylabel = np.zeros([args.batch_size, len(cand_list)])
        for x_id, target_id in enumerate(target_ids):
            for source_id in pairs[target_id][0]:
                ylabel[x_id][cand_list.index(source_id)] = 1
        ylabel /= ylabel.sum(axis=1).reshape(-1, 1)

    x_ids = np.arange(args.batch_size) + node_dict[args.main_node][0]
    return node_feature, node_type, edge_weight, edge_index, edge_type, x_ids, ylabel, node_dict, edge_dict, indxs


def prepare_data(pool):
    """Sampled and prepare training and validation data using multi-process parallization."""
    jobs = []
    for batch_id in np.arange(args.n_batch):
        p = pool.apply_async(node_classification_sample, args=(randint(),
                                                               sel_train_pairs, train_range))
        jobs.append(p)
    p = pool.apply_async(node_classification_sample, args=(randint(),
                                                           sel_valid_pairs, valid_range))
    jobs.append(p)
    return jobs


train_pairs = {}
valid_pairs = {}
test_pairs = {}

# source node: venue - target node: paper
# pairs = {paper_id: [venue_id, year], ....}
"""Prepare all the source nodes (sub-nodes) associated with each target node (main-node) as dict"""
selected_edges = [(u, v, e['weight']) for u,v,e in graph.edges(data=True) if e['edge_type'] == args.edge_name]

if not args.multi_lable_task:
    for target_id, source_id, _weight in selected_edges:
        if _weight in train_range:
            if target_id not in train_pairs:
                train_pairs[target_id] = [source_id, _weight]
        elif _weight in valid_range:
            if target_id not in valid_pairs:
                valid_pairs[target_id] = [source_id, _weight]
        else:
            if target_id not in test_pairs:
                test_pairs[target_id] = [source_id, _weight]
else:
    for target_id, source_id, _weight in selected_edges:
        if _weight in train_range:
            if target_id not in train_pairs:
                train_pairs[target_id] = [[], _weight]
            train_pairs[target_id][0] += [source_id]
        elif _weight in valid_range:
            if target_id not in valid_pairs:
                valid_pairs[target_id] = [[], _weight]
            valid_pairs[target_id][0] += [source_id]
        else:
            if target_id not in test_pairs:
                test_pairs[target_id] = [[], _weight]
            test_pairs[target_id][0] += [source_id]

np.random.seed(43)
"""Only train and valid with a certain percentage of data, if necessary."""
sel_train_pairs = {p: train_pairs[p] for p in np.random.choice(list(
    train_pairs.keys()), int(len(train_pairs) * args.data_percentage), replace=False)}
sel_valid_pairs = {p: valid_pairs[p] for p in np.random.choice(list(
    valid_pairs.keys()), int(len(valid_pairs) * args.data_percentage), replace=False)}
# there is no sel_test_paris here as it is not costy whatever size it has

"""Initialize GNN (model is specified by conv_name) and Classifier"""
# TODO: make in_dim dynamic
# in_dim = graph.graph['main_node_embedding_length'] + 401
in_dim = 768 + 768 +  1
# TODO: make use_RTE dynamic
num_relations = len(graph.graph['meta']) + 3 if args.include_fake_edges else len(graph.graph['meta']) + 1
gnn = GNN(in_dim=in_dim,
          n_hid=args.n_hid,
          num_types=len(graph.graph['node_types']),
          num_relations=num_relations,
          n_heads=args.n_heads,
          n_layers=args.n_layers,
          dropout=args.dropout,
          conv_name=args.conv_name,
          use_RTE=False).to(device)
logger(f"GNN configuration: \n in_dim = {in_dim}, n_hid = {args.n_hid}, \
             num_types = {len(graph.graph['node_types'])}, num_relations = {num_relations}, \
            n_heads = {args.n_heads}, n_layers = {args.n_layers}, dropout = {args.dropout}")

classifier = Classifier(args.n_hid, len(cand_list)).to(device)

model = nn.Sequential(gnn, classifier)

if args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters())
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters())
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(model.parameters())

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 1000, eta_min=1e-6)

stats = []
res = []
best_val = 0
train_step = 1500
# pool = _ThreadPool(4)
pool = _ProcessPool(args.n_pool)
st = time.time()
jobs = prepare_data(pool)
edges_attentions = defaultdict( #target_type
    lambda: defaultdict(  #source_type
        lambda: defaultdict(  #edge_type
            lambda: defaultdict(  #target_id
            lambda: defaultdict(  #source_id
                lambda: 0.0
            )
))))
for epoch in np.arange(args.n_epoch) + 1:
    if (args.include_fake_edges or args.remove_edges) and epoch == args.restructure_at_epoch + 1:
        add_fake_edges(graph, edges_attentions) if args.include_fake_edges else remove_edges(graph, edges_attentions)
        

    """Prepare Training and Validation Data"""
    train_data = [job.get() for job in jobs[:-1]]
    valid_data = jobs[-1].get()
    pool.close()
    pool.join()
    """After the data is collected, close the pool and then reopen it."""
    pool = mp.Pool(args.n_pool)
    jobs = prepare_data(pool)
    et = time.time()
    logger('Data Preparation: %.1fs' % (et - st))

    """Train (weight < x1)"""
    model.train()
    train_losses = []
    torch.cuda.empty_cache()
    for repeat in range(args.repeat):
        for node_feature, node_type, edge_weight, edge_index, edge_type, x_ids, ylabel, node_dict, edge_dict, indxs in train_data:
            attention, node_rep = gnn.forward(node_feature.to(device), node_type.to(device),
                                   edge_weight.to(device), edge_index.to(device), edge_type.to(device))
            if  (args.include_fake_edges or args.remove_edges) and epoch == args.restructure_at_epoch and repeat == args.repeat - 1:
                handle_attention(graph, attention, edges_attentions, edge_index, node_type, node_dict, indxs)
            res = classifier.forward(node_rep[x_ids])
            if not args.multi_lable_task:
                loss = criterion(res, ylabel.to(device))
            else:
                loss = criterion(res, torch.FloatTensor(ylabel).to(device))

            optimizer.zero_grad()
            torch.cuda.empty_cache()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]
            train_step += 1
            scheduler.step(train_step)
            del res, loss
    writer.add_scalar('training loss', np.average(train_losses), epoch)
    writer.add_histogram('classifier.linear.bias', classifier.linear.bias, epoch)

    """Valid (x1 <= weight <= x2)"""
    model.eval()
    with torch.no_grad():
        node_feature, node_type, edge_weight, edge_index, edge_type, x_ids, ylabel, node_dict, edge_dict, indxs = valid_data
        _, node_rep = gnn.forward(node_feature.to(device), node_type.to(device),
                               edge_weight.to(device), edge_index.to(device), edge_type.to(device))
        res = classifier.forward(node_rep[x_ids])
        if not args.multi_lable_task:
            loss = criterion(res, ylabel.to(device))
        else:
            loss = criterion(res, torch.FloatTensor(ylabel).to(device))

        """Calculate Valid NDCG. Update the best model based on highest NDCG score."""
        valid_res = []
        for ai, bi in zip(ylabel, res.argsort(descending=True)):
            if not args.multi_lable_task:
                valid_res += [(bi == ai).int().tolist()]
            else:
                valid_res += [ai[bi.cpu().numpy()]]
        valid_ndcg = np.average([ndcg_at_k(resi, len(resi))
                                 for resi in valid_res])
        writer.add_scalar('NDCG', valid_ndcg, epoch)
        valid_mrr = np.average(mean_reciprocal_rank(valid_res))
        writer.add_scalar('MRR', valid_mrr, epoch)
        if valid_ndcg > best_val:
            best_val = valid_ndcg
            torch.save(model, os.path.join(
                args.model_dir, f'{args.main_node}_{args.predicted_node_name}_{args.conv_name}'))
            logger('UPDATE!!!')

        st = time.time()
        logger(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid NDCG: %.4f") %
              (epoch, (st-et), optimizer.param_groups[0]['lr'], np.average(train_losses),
               loss.cpu().detach().tolist(), valid_ndcg))
        stats += [[np.average(train_losses), loss.cpu().detach().tolist()]]
        del res, loss
    del train_data, valid_data
"""Evaluate the trained model via test set (weight > x2)"""
logger(dict(get_meta_graph(graph)))
with torch.no_grad():
    test_res = []
    for _ in range(10):
        node_feature, node_type, edge_weight, edge_index, edge_type, x_ids, ylabel, node_dict, edge_dict, indxs = \
            node_classification_sample(randint(), test_pairs, test_range)
        _, main_node_rep = gnn.forward(node_feature.to(device), node_type.to(device),
                                    edge_weight.to(device), edge_index.to(device), edge_type.to(device))
        main_node_rep = main_node_rep[x_ids]
        res = classifier.forward(main_node_rep)
        for ai, bi in zip(ylabel, res.argsort(descending=True)):
            if not args.multi_lable_task:
                test_res += [(bi == ai).int().tolist()]
            else:
                test_res += [ai[bi.cpu().numpy()]]
    test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
    logger('Last Test NDCG: %.4f' % np.average(test_ndcg))
    test_mrr = mean_reciprocal_rank(test_res)
    logger('Last Test MRR:  %.4f' % np.average(test_mrr))

best_model = torch.load(os.path.join(
    args.model_dir, f'{args.main_node}_{args.predicted_node_name}_{args.conv_name}'))
best_model.eval()
gnn, classifier = best_model
with torch.no_grad():
    test_res = []
    for _ in range(10):
        node_feature, node_type, edge_weight, edge_index, edge_type, x_ids, ylabel, node_dict, edge_dict, indxs = \
            node_classification_sample(randint(), test_pairs, test_range)
        _, main_node_rep = gnn.forward(node_feature.to(device), node_type.to(device),
                                    edge_weight.to(device), edge_index.to(device), edge_type.to(device))
        main_node_rep = main_node_rep[x_ids]
        res = classifier.forward(main_node_rep)
        for ai, bi in zip(ylabel, res.argsort(descending=True)):
            if not args.multi_lable_task:
                test_res += [(bi == ai).int().tolist()]
            else:
                test_res += [ai[bi.cpu().numpy()]]
    test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
    logger('Best Test NDCG: %.4f' % np.average(test_ndcg))
    test_mrr = mean_reciprocal_rank(test_res)
    logger('Best Test MRR:  %.4f' % np.average(test_mrr))


end_time = time.time()
logger('Done after %.1fs' % (end_time-start_time))