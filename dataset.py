from timeit import default_timer

import numpy as np
import networkx as nx
import torch
import shutil

import argparse
import time
import os
import sys
import os.path as osp
from shutil import copy
import copy as cp

import torch_geometric.utils
from torch_geometric import seed_everything
from networkx import Graph
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.profile import profileit, timeit
from tqdm import tqdm
import pdb

from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as ssp
from torch.nn import BCEWithLogitsLoss

from torch_sparse import coalesce, SparseTensor

from torch_geometric.datasets import Planetoid, AttributedGraphDataset
from torch_geometric.data import Dataset, InMemoryDataset, Data
from torch_geometric.utils import to_undirected
from torch_geometric import transforms as T

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

import warnings
from scipy.sparse import SparseEfficiencyWarning

from baselines.gnn_link_pred import train_gnn
from baselines.mf import train_mf
from baselines.n2v import run_n2v
from custom_losses import auc_loss, hinge_auc_loss
from data_utils import load_splitted_data, read_label, read_edges
from models import SAGE, DGCNN, GCN, GIN
from ogbl_baselines.gnn_link_pred import train_gae_ogbl
from ogbl_baselines.mf import train_mf_ogbl
from ogbl_baselines.mlp_on_n2v import train_n2v_emb
from ogbl_baselines.n2v import run_and_save_n2v
from profiler_utils import profile_helper
from utils import get_pos_neg_edges, extract_enclosing_subgraphs, construct_pyg_graph, k_hop_subgraph, do_edge_split, \
    Logger, AA, CN, PPR, calc_ratio_helper, do_seal_edge_split

warnings.simplefilter('ignore', SparseEfficiencyWarning)
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)



class SEALDataset(InMemoryDataset):
    def __init__(self, root, data, split_edge, num_hops, percent=100, split='train',
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0,
                 max_nodes_per_hop=None, directed=False, rw_kwargs=None, device='cpu', pairwise=False,
                 pos_pairwise=False, neg_ratio=1, reprocess=True):
        print('SEALDataset')
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent
        self.split = split
        self.processed_files_suffix = '' # '_alwayz_reprocess_kostil' if reprocess else ''
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        self.device = device
        self.N = self.data.num_nodes
        self.E = self.data.edge_index.size()[-1]
        self.sparse_adj = SparseTensor(
            row=self.data.edge_index[0].to(self.device), col=self.data.edge_index[1].to(self.device),
            value=torch.arange(self.E, device=self.device),
            sparse_sizes=(self.N, self.N))
        self.rw_kwargs = rw_kwargs
        self.pairwise = pairwise
        self.pos_pairwise = pos_pairwise
        self.neg_ratio = neg_ratio
        super(SEALDataset, self).__init__(root)
        if not self.rw_kwargs.get('calc_ratio', False):
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.percent == 100:
            name = 'SEAL_{}_data{}'.format(self.split, self.processed_files_suffix)
        else:
            name = 'SEAL_{}_data_{}{}'.format(self.split, self.percent, self.processed_files_suffix)
        name += '.pt'
        return [name]

    def process(self):
        pos_edge, neg_edge = get_pos_neg_edges(self.split, self.split_edge,
                                               self.data.edge_index,
                                               self.data.num_nodes,
                                               self.percent, neg_ratio=self.neg_ratio)

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight,
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )

        if self.directed:
            A_csc = A.tocsc()
        else:
            A_csc = None

        # Extract enclosing subgraphs for pos and neg edges

        rw_kwargs = {
            "rw_m": self.rw_kwargs.get('m'),
            "rw_M": self.rw_kwargs.get('M'),
            "sparse_adj": self.sparse_adj,
            "edge_index": self.data.edge_index,
            "device": self.device,
            "data": self.data,
        }

        if self.rw_kwargs.get('calc_ratio', False):
            print(f"Calculating preprocessing stats for {self.split}")
            calc_ratio_helper(pos_edge, neg_edge, A, self.data.x, -1, self.num_hops, self.node_label,
                              self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc, rw_kwargs, self.split,
                              args.dataset, args.seed)
            exit()

        if not self.pairwise:
            print("Setting up Positive Subgraphs")
            pos_list = extract_enclosing_subgraphs(
                pos_edge, A, self.data.x, 1, self.num_hops, self.node_label,
                self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc, rw_kwargs, is_positive=True)
            print("Setting up Negative Subgraphs")
            neg_list = extract_enclosing_subgraphs(
                neg_edge, A, self.data.x, 0, self.num_hops, self.node_label,
                self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc, rw_kwargs, is_positive=False)
            torch.save(self.collate(pos_list + neg_list), self.processed_paths[0])
            del pos_list, neg_list
        else:
            if self.pos_pairwise:
                pos_list = extract_enclosing_subgraphs(
                    pos_edge, A, self.data.x, 1, self.num_hops, self.node_label,
                    self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc, rw_kwargs)
                torch.save(self.collate(pos_list), self.processed_paths[0])
                del pos_list
            else:
                neg_list = extract_enclosing_subgraphs(
                    neg_edge, A, self.data.x, 0, self.num_hops, self.node_label,
                    self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc, rw_kwargs)
                torch.save(self.collate(neg_list), self.processed_paths[0])
                del neg_list


class SEALDynamicDataset(Dataset):
    def __init__(self, root, data, split_edge, num_hops, percent=100, split='train',
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0,
                 max_nodes_per_hop=None, directed=False, rw_kwargs=None, device='cpu', pairwise=False,
                 pos_pairwise=False, neg_ratio=1, **kwargs):
        print('SEALDynamicDataset')
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = percent
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        self.rw_kwargs = rw_kwargs
        self.device = device
        self.N = self.data.num_nodes
        self.E = self.data.edge_index.size()[-1]
        self.sparse_adj = SparseTensor(
            row=self.data.edge_index[0].to(self.device), col=self.data.edge_index[1].to(self.device),
            value=torch.arange(self.E, device=self.device),
            sparse_sizes=(self.N, self.N))
        self.pairwise = pairwise
        self.pos_pairwise = pos_pairwise
        self.neg_ratio = neg_ratio
        super(SEALDynamicDataset, self).__init__(root)

        pos_edge, neg_edge = get_pos_neg_edges(split, self.split_edge,
                                               self.data.edge_index,
                                               self.data.num_nodes,
                                               self.percent, neg_ratio=self.neg_ratio)
        if self.pairwise:
            if self.pos_pairwise:
                self.links = pos_edge.t().tolist()
                self.labels = [1] * pos_edge.size(1)
            else:
                self.links = neg_edge.t().tolist()
                self.labels = [0] * neg_edge.size(1)
        else:
            self.links = torch.cat([pos_edge, neg_edge], 1).t().tolist()
            self.labels = [1] * pos_edge.size(1) + [0] * neg_edge.size(1)

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight,
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        self.A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )
        if self.directed:
            self.A_csc = self.A.tocsc()
        else:
            self.A_csc = None

        self.unique_nodes = {}
        if self.rw_kwargs.get('M'):
            print("Start caching random walk unique nodes")
            # if in dynamic SWEAL mode, need to cache the unique nodes of random walks before get() due to below error
            # RuntimeError: Cannot re-initialize CUDA in forked subprocess.
            # To use CUDA with multiprocessing, you must use the 'spawn' start method
            for link in self.links:
                rw_M = self.rw_kwargs.get('M')
                starting_nodes = []
                [starting_nodes.extend(link) for _ in range(rw_M)]
                start = torch.tensor(starting_nodes, dtype=torch.long, device=device)
                rw = self.sparse_adj.random_walk(start.flatten(), self.rw_kwargs.get('m'))
                self.unique_nodes[tuple(link)] = torch.unique(rw.flatten()).tolist()
            print("Finish caching random walk unique nodes")

    def __len__(self):
        return len(self.links)

    def len(self):
        return self.__len__()

    def get(self, idx):
        src, dst = self.links[idx]
        y = self.labels[idx]

        rw_kwargs = {
            "rw_m": self.rw_kwargs.get('m'),
            "rw_M": self.rw_kwargs.get('M'),
            "sparse_adj": self.sparse_adj,
            "edge_index": self.data.edge_index,
            "device": self.device,
            "data": self.data,
            "unique_nodes": self.unique_nodes
        }

        if not rw_kwargs['rw_m']:
            tmp = k_hop_subgraph(src, dst, self.num_hops, self.A, self.ratio_per_hop,
                                 self.max_nodes_per_hop, node_features=self.data.x,
                                 y=y, directed=self.directed, A_csc=self.A_csc)
            data = construct_pyg_graph(*tmp, self.node_label)
        else:
            data = k_hop_subgraph(src, dst, self.num_hops, self.A, self.ratio_per_hop,
                                  self.max_nodes_per_hop, node_features=self.data.x,
                                  y=y, directed=self.directed, A_csc=self.A_csc, rw_kwargs=rw_kwargs)

        return data