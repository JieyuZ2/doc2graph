import os
import csv
import json
import random
import itertools
import collections
import time
from datetime import datetime
from tqdm import tqdm
import math
import numpy as np
import scipy.sparse as sp
from decimal import *
import torch
from src.logger import myLogger
# matplotlib.use('Agg')


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(itertools.islice(it, size)), ())


def init_embed(n, dim):
    return np.random.uniform(-0.5 / dim, 0.5 / dim, size=(n, dim))


def l2_loss(tensor):
    return 0.5*((tensor ** 2).sum())


def softmax(x):
    e_x = np.exp(x - np.max(x))  # for numberation stablity
    return e_x / e_x.sum()


def print_config(config, logger=None):
    config = vars(config)
    info = "Running with the following configs:\n"
    for k, v in config.items():
        to_add = "\t{} : {}\n".format(k, str(v))
        if len(to_add) < 1000:
            info += to_add
    info.rstrip()
    if not logger:
        print("\n" + info)
    else:
        logger.info("\n" + info)


def init_logger(args):
    if args.prefix:
        base = os.path.join('log', args.prefix)
        log_dir = os.path.join(base, args.suffix)
    else:
        tag = f'{args.dataset}_'
        comment = args.suffix if args.suffix else datetime.now().strftime('%b_%d_%H-%M-%S')
        log_dir = os.path.join('log', tag + comment)
    args.log_dir = log_dir
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    logger = myLogger(name='exp', log_path=os.path.join(log_dir, 'log.txt'))
    logger.setLevel(args.log_level)
    return logger


def exec_time(func):
    def new_func(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Cost {} seconds.".format(end - start))
        return result

    return new_func


def save_checkpoint(state, modelpath, modelname, logger=None, del_others=True):
    if del_others:
        for dirpath, dirnames, filenames in os.walk(modelpath):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                if path.endswith('pth.tar'):
                    if logger is None:
                        print(f'rm {path}')
                    else:
                        logger.warning(f'rm {path}')
                    os.system("rm -rf '{}'".format(path))
            break
    path = os.path.join(modelpath, modelname)
    if logger is None:
        print('saving model to {}...'.format(path))
    else:
        logger.warning('saving model to {}...'.format(path))
    torch.save(state, path)


def read_graph(link_file):
    """read data from files

    Args:
        link_file: link file path

    Returns:
        node_num: int, number of nodes in the graph
        graph: dict, node_id -> list of neighbors in the graph
    """

    graph = collections.defaultdict(set)
    nodes = set()
    num_link = 0
    with open(link_file) as fin:
        for l in fin:
            num_link += 1
            n1, n2 = map(int, l.strip().split('\t'))
            nodes.add(n1)
            nodes.add(n2)
            graph[n1].add(n2)
            graph[n2].add(n1)
    graph = {k:list(v) for k, v in graph.items()}
    return list(nodes), graph, num_link


def read_nodes(node_file):
    """read pretrained node embeddings
    """

    id2name = []
    name2id = {}
    embedding_matrix = None
    with open(node_file, "r") as fin:
        vecs = []
        for l in fin:
            l = l.strip().split('\t')
            if len(l)==2:
                name, embed = l
                vecs.append([float(i) for i in embed.split(',')])
            else:
                name = l[0]
            id2name.append(name)
            name2id[name] = id
        if len(vecs)==len(id2name):
            embedding_matrix = np.array(vecs)
    return embedding_matrix, id2name, name2id


def sublist(lst1, lst2):
    # whether lst1 is sublist of lst2
    return set(lst1) <= set(lst2)


def read_emd(filename, n_node, n_embed):
    """use the pretrain node embeddings
    """
    with open(filename, "r") as f:
        lines = f.readlines()[1:]  # skip the first line
    node_embed = np.random.rand(n_node, n_embed)
    for line in lines:
        emd = line.split()
        node_embed[int(float(emd[0])), :] = str_list_to_float(emd[1:])
    return node_embed


def build_adj(graph, num_node):
    row_id_list = []
    col_id_list = []
    data_list = []
    for node, neighbors in graph.items():
        for n in neighbors:
            row_id_list.append(node)
            col_id_list.append(n)
            data_list.append(1)
    dim = num_node
    return sp.csr_matrix((data_list, (row_id_list, col_id_list)), shape=(dim, dim))


def join_int(l):
    return ','.join([str(i) for i in l])


def str_list_to_float(str_list):
    return [float(item) for item in str_list]


def makeDist(graphpath, power=0.75):
    edgedistdict = collections.defaultdict(int)
    nodedistdict = collections.defaultdict(int)

    weightsum = 0
    negprobsum = 0
    with open(graphpath, "r") as graphfile:
        # can work with weighted edge, but currently we do not have
        weight = 1
        for l in graphfile:
            line = l.rstrip().split('\t')
            node1, node2 = int(line[0]), int(line[1])
            edgedistdict[tuple([node1, node2])] = weight
            nodedistdict[node1] += weight
            weightsum += weight
            negprobsum += np.power(weight, power)

    for node, outdegree in nodedistdict.items():
        nodedistdict[node] = np.power(outdegree, power) / negprobsum

    for edge, weight in edgedistdict.items():
        edgedistdict[edge] = weight / weightsum

    return edgedistdict, nodedistdict


class VoseAlias(object):
    """
    Adding a few modifs to https://github.com/asmith26/Vose-Alias-Method
    """

    def __init__(self, dist):
        """
        (VoseAlias, dict) -> NoneType
        """
        self.dist = dist
        self.alias_initialisation()

    def alias_initialisation(self):
        """
        Construct probability and alias tables for the distribution.
        """
        # Initialise variables
        n = len(self.dist)
        self.table_prob = {}   # probability table
        self.table_alias = {}  # alias table
        scaled_prob = {}       # scaled probabilities
        small = []             # stack for probabilities smaller that 1
        large = []             # stack for probabilities greater than or equal to 1

        # Construct and sort the scaled probabilities into their appropriate stacks
        # print("1/2. Building and sorting scaled probabilities for alias table...")
        for o, p in self.dist.items():
            scaled_prob[o] = Decimal(p) * n

            if scaled_prob[o] < 1:
                small.append(o)
            else:
                large.append(o)

        # print("2/2. Building alias table...")
        # Construct the probability and alias tables
        while small and large:
            s = small.pop()
            l = large.pop()

            self.table_prob[s] = scaled_prob[s]
            self.table_alias[s] = l

            scaled_prob[l] = (scaled_prob[l] + scaled_prob[s]) - Decimal(1)

            if scaled_prob[l] < 1:
                small.append(l)
            else:
                large.append(l)

        # The remaining outcomes (of one stack) must have probability 1
        while large:
            self.table_prob[large.pop()] = Decimal(1)

        while small:
            self.table_prob[small.pop()] = Decimal(1)
        self.listprobs = list(self.table_prob)

    def alias_generation(self):
        """
        Yields a random outcome from the distribution.
        """
        # Determine which column of table_prob to inspect
        col = random.choice(self.listprobs)
        # Determine which outcome to pick in that column
        if self.table_prob[col] >= random.uniform(0, 1):
            return col
        else:
            return self.table_alias[col]

    def sample_n(self, size):
        """
        Yields a sample of size n from the distribution, and print the results to stdout.
        """
        for i in range(size):
            yield self.alias_generation()

    def sample_from(self, candidates, size):
        probs = np.array([self.dist[i] for i in candidates]) + 1e-10
        probs /= np.linalg.norm(probs, ord=1)
        return np.random.choice(candidates, size, p=probs).tolist()


