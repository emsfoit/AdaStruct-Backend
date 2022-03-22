# This file is a modified version of:
# - https://github.com/acbull/pyHGT/blob/master/OAG/train_paper_venue.py
# - https://github.com/acbull/pyHGT/blob/master/OAG/train_paper_field.py

# For single-classification task (i.e. main-node is associated with one sub-node) keep the defualt of multi_lable_task: "False".
# For mult-classification task (i.e. main-node is associated with multiple sub-nodes) set the value of multi_lable_task to "True".

from ast import arg
import os
import time
# import multiprocessing as mp
# from pathos.multiprocessing import ProcessingPool as Pool
from pathos.pools import _ProcessPool
import pathos.multiprocessing as mp
import numpy as np
import torch.nn as nn
import networkx as nx
from tools.hgt.pyHGT.datax import *
from tools.hgt.pyHGT.model import GNN, Classifier
from tools.hgt.pyHGT.attention import *
from tools.hgt.utils.utils import randint, ndcg_at_k, mean_reciprocal_rank, logger, dotdict
import traceback

#import torch
from torch.utils.tensorboard import SummaryWriter



from warnings import filterwarnings
filterwarnings("ignore")



def run(args, graph_params):
    class HGTTraining:

        def __init__(self, args, graph_params):
            print(x)
            self.args = args
            self.graph_params = graph_params
            self.start_time = time.time()
            logger(self.args)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # replace with read self.graph
            self.graph = nx.read_gpickle(self.args.graph_dir)
            self.graph.graph['edge_list'] = {}
            # get_edge_list(self.graph)
            logger(self.graph.graph['meta'])

            self.weight_thresholds = self.graph_params['weight_split_range']['valid_range']

            self.train_range = {w: True for w in self.graph.graph['weights'] if w !=
                        None and w < self.weight_thresholds['start']}
            self.valid_range = {w: True for w in self.graph.graph['weights'] if w !=
                        None and w >= self.weight_thresholds['start'] and w <= self.weight_thresholds['end']}
            self.test_range = {w: True for w in self.graph.graph['weights'] if w !=
                        None and w > self.weight_thresholds['end']}

            self.rev_edge_name = f'rev_{self.args.edge_name}'

            """self.cand_list stores all the sub-nodes, which is the classification domain."""
            # self.selected_edges = [(u,v) for u,v,e in self.graph.edges(data=True) if e['edge_type'] == self.args.edge_name]
            self.cand_list = [u for u,v,e in self.graph.edges(data=True) if e['edge_type'] == self.rev_edge_name]
            self.cand_list = list(set(self.cand_list))
            # self.cand_list = list(self.graph.edge_list[self.args.predicted_node_name]
            #                  [self.args.main_node][self.args.edge_name].keys())

            if not self.args.multi_lable_task:
                """
                    Using Negative Log Likelihood Loss (torch.nn.NLLLoss()), since each main-node can be associated with one sub-node.
                    consider using CrossEntropy (log-softmax + NLL) 
                """
                self.criterion = nn.NLLLoss()
            else:
                """
                    Use KL Divergence here, since each main node can be associated with multiple predicted sub nodes.
                    Thus this task is a multi-label classification.
                """
                self.criterion = nn.KLDivLoss(reduction='batchmean')


            self.train_pairs = {}
            self.valid_pairs = {}
            self.test_pairs = {}

            # source node: venue - target node: paper
            # pairs = {paper_id: [venue_id, year], ....}
            """Prepare all the source nodes (sub-nodes) associated with each target node (main-node) as dict"""
            self.selected_edges = [(u, v, e['weight']) for u,v,e in self.graph.edges(data=True) if e['edge_type'] == self.args.edge_name]

            if not self.args.multi_lable_task:
                for target_id, source_id, _weight in self.selected_edges:
                    if _weight in self.train_range:
                        if target_id not in self.train_pairs:
                            self.train_pairs[target_id] = [source_id, _weight]
                    elif _weight in self.valid_range:
                        if target_id not in self.valid_pairs:
                            self.valid_pairs[target_id] = [source_id, _weight]
                    else:
                        if target_id not in self.test_pairs:
                            self.test_pairs[target_id] = [source_id, _weight]
            else:
                for target_id, source_id, _weight in self.selected_edges:
                    if _weight in self.train_range:
                        if target_id not in self.train_pairs:
                            self.train_pairs[target_id] = [[], _weight]
                        self.train_pairs[target_id][0] += [source_id]
                    elif _weight in self.valid_range:
                        if target_id not in self.valid_pairs:
                            self.valid_pairs[target_id] = [[], _weight]
                        self.valid_pairs[target_id][0] += [source_id]
                    else:
                        if target_id not in self.test_pairs:
                            self.test_pairs[target_id] = [[], _weight]
                        self.test_pairs[target_id][0] += [source_id]

            np.random.seed(43)
            """Only train and valid with a certain percentage of data, if necessary."""
            self.train_pairs = {p: self.train_pairs[p] for p in np.random.choice(list(
                self.train_pairs.keys()), int(len(self.train_pairs) * self.args.data_percentage), replace=False)}
            self.valid_pairs = {p: self.valid_pairs[p] for p in np.random.choice(list(
                self.valid_pairs.keys()), int(len(self.valid_pairs) * self.args.data_percentage), replace=False)}
            # there is no sel_test_paris here as it is not costy whatever size it has

            """Initialize GNN (self.model is specified by conv_name) and Classifier"""
            # TODO: make in_dim dynamic
            # in_dim = self.graph.graph['main_node_embedding_length'] + 401
            in_dim = 768 + 768 +  1
            # TODO: make use_RTE dynamic
            num_relations = len(self.graph.graph['meta']) + 3 if self.args.include_fake_edges else len(self.graph.graph['meta']) + 1
            gnn = GNN(in_dim=in_dim,
                    n_hid=self.args.n_hid,
                    num_types=len(self.graph.graph['node_types']),
                    num_relations=num_relations,
                    n_heads=self.args.n_heads,
                    n_layers=self.args.n_layers,
                    dropout=self.args.dropout,
                    conv_name=self.args.conv_name,
                    use_RTE=False).to(self.device)
            logger(f"GNN configuration: \n in_dim = {in_dim}, n_hid = {self.args.n_hid}, \
                        num_types = {len(self.graph.graph['node_types'])}, num_relations = {num_relations}, \
                        n_heads = {self.args.n_heads}, n_layers = {self.args.n_layers}, dropout = {self.args.dropout}")

            classifier = Classifier(self.args.n_hid, len(self.cand_list)).to(self.device)

            self.model = nn.Sequential(gnn, classifier)

            if self.args.optimizer == 'adamw':
                self.optimizer = torch.optim.AdamW(self.model.parameters())
            elif self.args.optimizer == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters())
            elif self.args.optimizer == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            elif self.args.optimizer == 'adagrad':
                self.optimizer = torch.optim.Adagrad(self.model.parameters())

            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 1000, eta_min=1e-6)

            stats = []
            res = []
            best_val = 0
            train_step = 1500
            # self.pool = _ThreadPool(4)
            self.pool = _ProcessPool(4)
            st = time.time()
            jobs = self.prepare_data(self.pool)
            self.edges_attentions = defaultdict( #target_type
                lambda: defaultdict(  #source_type
                    lambda: defaultdict(  #edge_type
                        lambda: defaultdict(  #target_id
                        lambda: defaultdict(  #source_id
                            lambda: 0.0
                        )
            ))))



        def node_classification_sample(self, seed, pairs, weight_range):
            """
                sub-self.graph sampling and label preparation for node classification:
                (1) Sample batch_size number of output nodes (main-nodes) and their weight.
            """
            try:
                np.random.seed(seed)
                target_ids = np.random.choice(
                    list(pairs.keys()), self.args.batch_size, replace=False)
                target_info = []
                for target_id in target_ids:
                    _, _weight = pairs[target_id]
                    target_info += [[target_id, _weight]]

                """(2) Based on the seed nodes, sample a subgraph with 'sampled_depth' and 'sampled_number'"""
                feature, weights, edge_list, indxs, _ = sample_subgraph(self.graph, weight_range, self.graph_params,
                                                                        inp={self.args.main_node: np.array(
                                                                            target_info)},
                                                                        sampled_depth=self.args.sample_depth, sampled_number=self.args.sample_width)

                """(3) Mask out the edge between the output target nodes (main-node) with output source nodes (sub-node)"""
                masked_edge_list = []
                for i in edge_list[self.args.main_node][self.args.predicted_node_name][self.rev_edge_name]:
                    if i[0] >= self.args.batch_size:
                        masked_edge_list += [i]
                edge_list[self.args.main_node][self.args.predicted_node_name][self.rev_edge_name] = masked_edge_list

                masked_edge_list = []
                for i in edge_list[self.args.predicted_node_name][self.args.main_node][self.args.edge_name]:
                    if i[1] >= self.args.batch_size:
                        masked_edge_list += [i]
                edge_list[self.args.predicted_node_name][self.args.main_node][self.args.edge_name] = masked_edge_list

                """(4) Transform the subself.graph into torch Tensor (edge_index is in format of pytorch_geometric)"""
                node_feature, node_type, edge_weight, edge_index, edge_type, node_dict, edge_dict = \
                    to_torch(feature, weights, edge_list, self.graph, self.args.include_fake_edges)
                """
                    (5) Prepare the labels for each output target node (main-node), and their index in sampled self.graph.
                        (node_dict[type][0] stores the start index of a specific type of nodes)
                """
                if not self.args.multi_lable_task:
                    ylabel = torch.zeros(self.args.batch_size, dtype=torch.long)
                    for x_id, target_id in enumerate(target_ids):
                        ylabel[x_id] = self.cand_list.index(pairs[target_id][0])
                else:
                    ylabel = np.zeros([self.args.batch_size, len(self.cand_list)])
                    for x_id, target_id in enumerate(target_ids):
                        for source_id in pairs[target_id][0]:
                            ylabel[x_id][self.cand_list.index(source_id)] = 1
                    ylabel /= ylabel.sum(axis=1).reshape(-1, 1)

                x_ids = np.arange(self.args.batch_size) + node_dict[self.args.main_node][0]
                return node_feature, node_type, edge_weight, edge_index, edge_type, x_ids, ylabel, node_dict, edge_dict, indxs

            except Exception as e:
                logger(e)
                logger(traceback.format_exc())
                return False


    
        def prepare_data(self, pool):
            g = 4
            def add(x, y, u):
                x = g
                return x, 3, 5
            """Sampled and prepare training and validation data using multi-process parallization."""
            jobs = []
            results = pool.starmap(add, [(1,2,3), (1,2,3)])
            y = list(results)
            ms = pool.starmap(node_classification_sample, [(
                    "self.graph", self.args, self.graph_params, self.rev_edge_name, self.cand_list, randint(), self.train_pairs, self.train_range
                )])
            x = list(ms)
            for batch_id in np.arange(self.args.n_batch):
                results = self.pool.imap(node_classification_sample, [(
                    self.graph, self.args, self.graph_params, self.rev_edge_name, self.cand_list, randint(), self.train_pairs, self.train_range
                )])
                x = list(results)
                p = self.pool.apply_async(node_classification_sample, args=(self.graph, self.args, self.graph_params, self.rev_edge_name, self.cand_list, randint(),
                                                                    self.train_pairs, self.train_range))
                node_feature, node_type, edge_weight, edge_index, edge_type, x_ids, ylabel, node_dict, edge_dict, indxs = p.get()
                jobs.append(p)
            p = self.pool.apply_async(node_classification_sample, args=(self.graph, self.args, self.graph_params, self.rev_edge_name, self.cand_list, randint(),
                                                                self.valid_pairs, self.valid_range))
            jobs.append(p)
            return jobs


        def train(self):
            for epoch in np.arange(self.args.n_epoch) + 1:
                if (self.args.include_fake_edges or self.args.remove_edges) and epoch == self.args.restructure_at_epoch + 1:
                    add_fake_edges(self.graph, self.edges_attentions) if self.args.include_fake_edges else remove_edges(self.graph, self.edges_attentions)
                
                train_data = []
                for job in jobs[:-1]:
                    try:
                        train_data.append(job.get())
                    except Exception as e:
                        print(e)
                """Prepare Training and Validation Data"""
                # train_data = [job.get() for job in jobs[:-1]]
                valid_data = jobs[-1].get()
                self.pool.close()
                self.pool.join()
                """After the data is collected, close the self.pool and then reopen it."""
                self.pool = mp.Pool(self.args.n_pool)
                jobs = self.prepare_data(self.pool)
                et = time.time()
                logger('Data Preparation: %.1fs' % (et - st))

                """Train (weight < x1)"""
                self.model.train()
                train_losses = []
                torch.cuda.empty_cache()
                for repeat in range(self.args.repeat):
                    for node_feature, node_type, edge_weight, edge_index, edge_type, x_ids, ylabel, node_dict, edge_dict, indxs in train_data:
                        attention, node_rep = gnn.forward(node_feature.to(self.device), node_type.to(self.device),
                                            edge_weight.to(self.device), edge_index.to(self.device), edge_type.to(self.device))
                        if  (self.args.include_fake_edges or self.args.remove_edges) and epoch == self.args.restructure_at_epoch and repeat == self.args.repeat - 1:
                            handle_attention(self.graph, attention, self.edges_attentions, edge_index, node_type, node_dict, indxs)
                        res = classifier.forward(node_rep[x_ids])
                        if not self.args.multi_lable_task:
                            loss = self.criterion(res, ylabel.to(self.device))
                        else:
                            loss = self.criterion(res, torch.FloatTensor(ylabel).to(self.device))

                        self.optimizer.zero_grad()
                        torch.cuda.empty_cache()
                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                        self.optimizer.step()

                        train_losses += [loss.cpu().detach().tolist()]
                        train_step += 1
                        self.scheduler.step(train_step)
                        del res, loss
                # writer.add_scalar('training loss', np.average(train_losses), epoch)
                # writer.add_histogram('classifier.linear.bias', classifier.linear.bias, epoch)

                """Valid (x1 <= weight <= x2)"""
                self.model.eval()
                with torch.no_grad():
                    node_feature, node_type, edge_weight, edge_index, edge_type, x_ids, ylabel, node_dict, edge_dict, indxs = valid_data
                    _, node_rep = gnn.forward(node_feature.to(self.device), node_type.to(self.device),
                                        edge_weight.to(self.device), edge_index.to(self.device), edge_type.to(self.device))
                    res = classifier.forward(node_rep[x_ids])
                    if not self.args.multi_lable_task:
                        loss = self.criterion(res, ylabel.to(self.device))
                    else:
                        loss = self.criterion(res, torch.FloatTensor(ylabel).to(self.device))

                    """Calculate Valid NDCG. Update the best self.model based on highest NDCG score."""
                    valid_res = []
                    for ai, bi in zip(ylabel, res.self.argsort(descending=True)):
                        if not self.args.multi_lable_task:
                            valid_res += [(bi == ai).int().tolist()]
                        else:
                            valid_res += [ai[bi.cpu().numpy()]]
                    valid_ndcg = np.average([ndcg_at_k(resi, len(resi))
                                            for resi in valid_res])
                    # writer.add_scalar('NDCG', valid_ndcg, epoch)
                    valid_mrr = np.average(mean_reciprocal_rank(valid_res))
                    # writer.add_scalar('MRR', valid_mrr, epoch)
                    if valid_ndcg > best_val:
                        best_val = valid_ndcg
                        torch.save(self.model, os.path.join(
                            self.args.model_dir, f'{self.args.main_node}_{self.args.predicted_node_name}_{self.args.conv_name}'))
                        logger('UPDATE!!!')

                    st = time.time()
                    logger(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid NDCG: %.4f") %
                        (epoch, (st-et), self.optimizer.param_groups[0]['lr'], np.average(train_losses),
                        loss.cpu().detach().tolist(), valid_ndcg))
                    stats += [[np.average(train_losses), loss.cpu().detach().tolist()]]
                    del res, loss
                del train_data, valid_data
            """Evaluate the trained self.model via test set (weight > x2)"""
            # logger(dict(get_meta_self.graph(self.graph)))
            with torch.no_grad():
                test_res = []
                for _ in range(10):
                    node_feature, node_type, edge_weight, edge_index, edge_type, x_ids, ylabel, node_dict, edge_dict, indxs = \
                        node_classification_sample(self.graph, self.args, self.graph_params, self.rev_edge_name, self.cand_list, randint(), self.test_pairs, self.test_range)
                    _, main_node_rep = gnn.forward(node_feature.to(self.device), node_type.to(self.device),
                                                edge_weight.to(self.device), edge_index.to(self.device), edge_type.to(self.device))
                    main_node_rep = main_node_rep[x_ids]
                    res = classifier.forward(main_node_rep)
                    for ai, bi in zip(ylabel, res.self.argsort(descending=True)):
                        if not self.args.multi_lable_task:
                            test_res += [(bi == ai).int().tolist()]
                        else:
                            test_res += [ai[bi.cpu().numpy()]]
                test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
                logger('Last Test NDCG: %.4f' % np.average(test_ndcg))
                test_mrr = mean_reciprocal_rank(test_res)
                logger('Last Test MRR:  %.4f' % np.average(test_mrr))

            model = torch.load(os.path.join(
                self.args.model_dir, f'{self.args.main_node}_{self.args.predicted_node_name}_{self.args.conv_name}'))
            model.eval()
            gnn, classifier = model
            with torch.no_grad():
                test_res = []
                for _ in range(10):
                    node_feature, node_type, edge_weight, edge_index, edge_type, x_ids, ylabel, node_dict, edge_dict, indxs = \
                        node_classification_sample(self.graph, self.args, self.graph_params, self.rev_edge_name, self.cand_list, randint(), self.test_pairs, self.test_range)
                    _, main_node_rep = gnn.forward(node_feature.to(self.device), node_type.to(self.device),
                                                edge_weight.to(self.device), edge_index.to(self.device), edge_type.to(self.device))
                    main_node_rep = main_node_rep[x_ids]
                    res = classifier.forward(main_node_rep)
                    for ai, bi in zip(ylabel, res.self.argsort(descending=True)):
                        if not self.args.multi_lable_task:
                            test_res += [(bi == ai).int().tolist()]
                        else:
                            test_res += [ai[bi.cpu().numpy()]]
                test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
                logger('Best Test NDCG: %.4f' % np.average(test_ndcg))
                test_mrr = mean_reciprocal_rank(test_res)
                logger('Best Test MRR:  %.4f' % np.average(test_mrr))


            end_time = time.time()
            logger('Done after %.1fs' % (end_time-self.start_time))

    m = HGTTraining(args, graph_params)
    m.train()
if __name__ == "__main__":
    default_args = {
        "graph_dir":"storage/OAG_grap_reddit_10000.pk",
        "model_dir":"storage",
        "main_node":"post",
        "predicted_node_name":"subreddit",
        "edge_name":"post_subreddit",
        "exract_attention":False,
        "show_tensor_board":False,
        "multi_lable_task":True,
        "conv_name":"hgt",
        "n_hid":400,
        "n_heads":8,
        "n_layers":4,
        "dropout":0.2,
        "sample_depth":6,
        "sample_width":128,
        "optimizer":"adamw",
        "data_percentage":1.0,
        "n_epoch":30,
        "n_pool":4,
        "n_batch":32,
        "repeat":2,
        "batch_size":256,
        "clip":0.25,
        "include_fake_edges":False,
        "remove_edges":False,
        "restructure_at_epoch":10
    }
    graph_Settings = {
        "nodes":[
            {
                "name":"post",
                "df":"post_author_subreddits",
                "features":[
                    {
                    "feature_name":"id",
                    "column_name":"post_id"
                    },
                    {
                    "feature_name":"title",
                    "column_name":"post_title"
                    },
                    {
                    "feature_name":"time",
                    "column_name":"post_created_utc"
                    }
                ]
            },
            {
                "name":"author",
                "df":"authors",
                "features":[
                    {
                    "feature_name":"id",
                    "column_name":"author"
                    }
                ]
            },
            {
                "name":"comment",
                "df":"comments",
                "features":[
                    {
                    "feature_name":"id",
                    "column_name":"comment_id"
                    },
                    {
                    "feature_name":"body",
                    "column_name":"comment_body"
                    },
                    {
                    "feature_name":"time",
                    "column_name":"comment_created_utc"
                    },
                    {
                    "feature_name":"parent_id",
                    "column_name":"comment_parent_id"
                    }
                ],
                "parent":{
                    "df":"comments",
                    "features":[
                    {
                        "feature_name":"id",
                        "column_name":"comment_id"
                    },
                    {
                        "feature_name":"parent_id",
                        "column_name":"comment_parent_id"
                    }
                    ]
                }
            },
            {
                "name":"subreddit",
                "df":"post_author_subreddits",
                "features":[
                    {
                    "feature_name":"id",
                    "column_name":"subreddit_id"
                    },
                    {
                    "feature_name":"title",
                    "column_name":"subreddit"
                    }
                ]
            }
        ],
        "edges":[
            {
                "name":"post_author",
                "df":"post_author_subreddits",
                "source":"post",
                "target":"author"
            },
            {
                "name":"post_subreddit",
                "df":"post_author_subreddits",
                "source":"post",
                "target":"subreddit"
            },
            {
                "name":"post_comment",
                "df":"comments",
                "source":"post",
                "target":"comment"
            },
            {
                "name":"comment_comment",
                "df":"comments",
                "source":"comment",
                "target":"comment",
                "self_edge":True
            },
            {
                "name":"author_comment",
                "df":"comments",
                "source":"author",
                "target":"comment"
            }
        ],
        "main_node":"post",
        "node_to_calculate_repitition":"comment",
        "emb":[
            {
                "node":"post",
                "feature":"title",
                "model":"XLNetTokenizer",
                "min_number_of_words":1
            },
            {
                "node":"comment",
                "feature":"body",
                "model":"XLNetTokenizer",
                "min_number_of_words":1
            },
            {
                "node":"subreddit",
                "feature":"title",
                "model":"XLNetTokenizer",
                "min_number_of_words":1
            }
        ],
        "weight":{
            "df":"post_author_subreddits",
            "feature":"post_created_utc"
        },
        "weight_split_range":{
            "train_range":{
                "start":1293924080,
                "end":1293930237
            },
            "valid_range":{
                "start":1293930237,
                "end":1293939316
            },
            "test_range":{
                "start":1293939316,
                "end":1293944394
            }
        }
    }
    
    m = run(dotdict(default_args), graph_Settings) 
    # HGTTraining(dotdict(default_args), graph_Settings)
    # x = m.train()
    # runapp(dotdict(default_args), graph_Settings)