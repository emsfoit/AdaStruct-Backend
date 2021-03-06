import numpy as np
import scipy.sparse as sp
import torch
import json
import os
import glob
import dill

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import pandas as pd
import networkx as nx
from numpy import linalg
from time import gmtime, strftime

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return [1. / (r[0] + 1) if r.size else 0. for r in rs]


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def randint():
    return np.random.randint(2**32 - 1)



def feature_OAG(layer_data, graph):
    feature = {}
    times   = {}
    indxs   = {}
    texts   = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs  = np.array(list(layer_data[_type].keys()))
        tims  = np.array(list(layer_data[_type].values()))[:,1]
        
        if 'node_emb' in graph.node_feature[_type]:
            feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'node_emb']), dtype=np.float)
        else:
            feature[_type] = np.zeros([len(idxs), 400])
        feature[_type] = np.concatenate((feature[_type], list(graph.node_feature[_type].loc[idxs, 'emb']),\
            np.log10(np.array(list(graph.node_feature[_type].loc[idxs, 'citation'])).reshape(-1, 1) + 0.01)), axis=1)
        
        times[_type]   = tims
        indxs[_type]   = idxs
        
        if _type == 'paper':
            texts = np.array(list(graph.node_feature[_type].loc[idxs, 'title']), dtype=np.str)
    return feature, times, indxs, texts



def get_files_into_dict(dir, sep=',', dtype=str):
    """ returns a dict of dataframes with file names 

    Parameters
    ----------
    dir : str
        directory of the files 
    sep : str
        features sep in the input files (default = ",")

    Returns
    -------
    dict_dfs
        a dict of dataframes: {file_name1: df1, file_name2: df2, ...}
    """

    files = glob.glob(dir + "/*")
    dict_dfs = {}
    for f in files:
        df = pd.read_csv(f, sep=sep, dtype=dtype)
        dict_dfs[os.path.basename(f).split(".")[0]] = df

    return dict_dfs


def convert_series_to_array(x, sep=' ', dtype='<U9'):
    array = np.array(x.split(sep)).astype(float)
    array = np.around(array, decimals=6).astype(dtype)
    return array


# log_file_name = "logs/{time}.txt".format(time=strftime("%m_%d__%H_%M_%S", gmtime()))
# f = open(log_file_name, "w")
# f.close()
def logger(data):
    print(data)
    # with open(log_file_name, "a") as file_object:
    #     data = "\n {}".format(data)
    #     file_object.write(data)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
