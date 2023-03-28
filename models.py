import math
import numpy as np
import torch
from torch.nn import (ModuleList, Linear, Conv1d, MaxPool1d, Embedding, ReLU,
                      Sequential, BatchNorm1d as BN)
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, global_sort_pool, global_add_pool, global_mean_pool, MLP, \
    global_max_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import dropout_adj
from torch_scatter import scatter_add, scatter_max
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
from torch_geometric.utils import softmax


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset,
                 use_feature=False, node_embedding=None, dropout=0.5, dropedge=0.0):
        super(GCN, self).__init__()

        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        self.convs.append(GCNConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.dropedge = dropedge
        self.mlp = MLP([hidden_channels, hidden_channels, 1], dropout=dropout, batch_norm=False)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, num_nodes, z, edge_index, batch, x=None, edge_weight=None, node_id=None, edge_mask=None):
        edge_index, _ = dropout_adj(edge_index, p=self.dropedge,
                                    force_undirected=True,
                                    num_nodes=num_nodes,
                                    training=self.training)

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)

        # center pooling
        _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
        x_src = x[center_indices]
        x_dst = x[center_indices + 1]
        x = (x_src * x_dst)

        # sum pool
        # x = global_add_pool(x, batch)

        # max pool
        # x = global_max_pool(x, batch)

        x = self.mlp(x)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset=None,
                 use_feature=False, node_embedding=None, dropout=0.5, dropedge=0.0):
        super(SAGE, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        self.convs.append(SAGEConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.dropedge = dropedge
        self.mlp = MLP([hidden_channels, hidden_channels, 1], dropout=dropout, batch_norm=False)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, num_nodes, z, edge_index, batch, x=None, edge_weight=None, node_id=None, edge_mask=None):
        edge_index, _ = dropout_adj(edge_index, p=self.dropedge,
                                    force_undirected=True,
                                    num_nodes=num_nodes,
                                    training=self.training)
        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        if True:  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            x_src = x[center_indices]
            x_dst = x[center_indices + 1]
            x = (x_src * x_dst)
            x = self.mlp(x)
        else:  # max pooling
            x = global_max_pool(x, batch)
            x = self.mlp(x)

        return x


# An end-to-end deep learning architecture for graph classification, AAAI-18.
class DGCNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, k=0.6, train_dataset=None,
                 dynamic_train=False, GNN=GCNConv, use_feature=False,
                 node_embedding=None, dropedge=0.0):
        super(DGCNN, self).__init__()

        self.use_feature = use_feature
        self.node_embedding = node_embedding

        if k <= 1:  # Transform percentile to number.
            if train_dataset is None:
                k = 30
            else:
                if dynamic_train:
                    sampled_train = train_dataset[:1000]
                else:
                    sampled_train = train_dataset
                num_nodes = sorted([g.num_nodes for g in sampled_train])
                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)
        self.k = int(k)

        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim

        self.convs.append(GNN(initial_channels, hidden_channels))
        for i in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)

        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.dropedge = dropedge
        self.mlp = MLP([dense_dim, 128, 1], dropout=0.5, batch_norm=False)

    def forward(self, num_nodes, z, edge_index, batch, x=None, edge_weight=None, node_id=None, edge_mask=None):
        edge_index, _ = dropout_adj(edge_index, p=self.dropedge,
                                    force_undirected=True,
                                    num_nodes=num_nodes,
                                    training=self.training)

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        xs = [x]

        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index, edge_weight))]
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = self.mlp(x)
        return x


class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset,
                 use_feature=False, node_embedding=None, dropout=0.5,
                 jk=True, train_eps=False, dropedge=0.0):
        super(GIN, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.jk = jk

        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        self.conv1 = GINConv(
            Sequential(
                Linear(initial_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                BN(hidden_channels),
            ),
            train_eps=train_eps)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        BN(hidden_channels),
                    ),
                    train_eps=train_eps))

        self.dropout = dropout
        if self.jk:
            self.mlp = MLP([num_layers * hidden_channels, hidden_channels, 1], dropout=0.5, batch_norm=False)
        else:
            self.mlp = MLP([hidden_channels, hidden_channels, 1], dropout=0.5, batch_norm=False)

        self.dropedge = dropedge

    def forward(self, num_nodes, z, edge_index, batch, x=None, edge_weight=None, node_id=None, edge_mask=None):
        edge_index, _ = dropout_adj(edge_index, p=self.dropedge,
                                    force_undirected=True,
                                    num_nodes=num_nodes,
                                    training=self.training)

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        if self.jk:
            x = global_mean_pool(torch.cat(xs, dim=1), batch)
        else:
            x = global_mean_pool(xs[-1], batch)
        x = self.mlp(x)

        return x


class GCN_WalkPooling(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset,
                 use_feature=False, node_embedding=None, dropout=0.5, dropedge=0.0):
        super().__init__()

        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        self.convs.append(GCNConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.pooling = WalkPooling(hidden_channels, hidden_channels, heads=2, walk_len=7)

        self.dropout = dropout
        self.dropedge = dropedge
        self.mlp = MLP([72, hidden_channels, 1], dropout=dropout, batch_norm=False)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, num_nodes, z, edge_index, batch, x=None, edge_weight=None, node_id=None, edge_mask=None):
        assert edge_mask is not None # kostil
        edge_index, _ = dropout_adj(edge_index, p=self.dropedge,
                                    force_undirected=True,
                                    num_nodes=num_nodes,
                                    training=self.training)

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)

        # center pooling
        # [1135, 32]
        x = self.pooling(x, edge_index, edge_mask, batch)
        # _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
        # x_src = x[center_indices]
        # x_dst = x[center_indices + 1]
        # x = (x_src * x_dst)
        # 32, 32

        # sum pool
        # x = global_add_pool(x, batch)

        # max pool
        # x = global_max_pool(x, batch)

        x = self.mlp(x)
        return x


class WalkPooling(MessagePassing):
    def __init__(self, in_channels: int, hidden_channels: int, heads: int = 1, walk_len: int = 6):
        super(WalkPooling, self).__init__()
        print('wp params', in_channels, hidden_channels, heads, walk_len)

        self.hidden_channels = hidden_channels
        self.heads = heads
        self.walk_len = walk_len

        # the linear layers in the attention encoder
        self.lin_key1 = Linear(in_channels, hidden_channels)
        self.lin_query1 = Linear(in_channels, hidden_channels)
        self.lin_key2 = Linear(hidden_channels, heads * hidden_channels)
        self.lin_query2 = Linear(hidden_channels, heads * hidden_channels)

    def attention_mlp(self, x, edge_index):
        query = self.lin_key1(x).reshape(-1,self.hidden_channels)
        key = self.lin_query1(x).reshape(-1,self.hidden_channels)

        query = F.leaky_relu(query,0.2)
        key = F.leaky_relu(key,0.2)

        query = F.dropout(query, p=0.5, training=self.training)
        key = F.dropout(key, p=0.5, training=self.training)

        query = self.lin_key2(query).view(-1, self.heads, self.hidden_channels)
        key = self.lin_query2(key).view(-1, self.heads, self.hidden_channels)

        row, col = edge_index
        weights = (query[row] * key[col]).sum(dim=-1) / np.sqrt(self.hidden_channels)

        return weights

    def weight_encoder(self, x, edge_index, edge_mask):
        # print('weight_encoder', edge_mask.shape, torch.logical_not(edge_mask).sum())
        # print(x.shape, edge_index.shape)
        weights = self.attention_mlp(x, edge_index)

        omega = torch.sigmoid(weights[torch.logical_not(edge_mask)])

        row, col = edge_index
        num_nodes = torch.max(edge_index)+1

        # edge weights of the plus graph
        weights_p = softmax(weights,edge_index[1])

        # edge weights of the minus graph
        weights_m = weights - scatter_max(weights, col, dim=0, dim_size=num_nodes)[0][col]
        weights_m = torch.exp(weights_m)
        weights_m = weights_m * edge_mask.view(-1,1)
        norm = scatter_add(weights_m, col, dim=0, dim_size=num_nodes)[col] + 1e-16
        weights_m = weights_m / norm
        # print('weight_encoder end')

        return weights_p, weights_m, omega

    def forward(self, x, edge_index, edge_mask, batch):

        #encode the node representation into edge weights via attention mechanism
        weights_p, weights_m, omega = self.weight_encoder(x, edge_index, edge_mask)

        # pytorch geometric set the batch adjacency matrix to
        # be the diagonal matrix with each graph's adjacency matrix
        # stacked in the diagonal. Therefore, calculating the powers
        # of the stochastic matrix directly will cost lots of memory.
        # We compute the powers of stochastic matrix as follows
        # Let A = diag ([A_1,\cdots,A_n]) be the batch adjacency matrix,
        # we set x = [x_1,\cdots,x_n]^T be the batch feature matrix
        # for the i-th graph in the batch with n_i nodes, its feature
        # is a n_i\times n_max matrix, where n_max is the largest number
        # of nodes for all graphs in the batch. The elements of x_i are
        # (x_i)_{x,y} = \delta_{x,y}.
        device = x.device

        # number of graphs in the batch
        batch_size = torch.max(batch)+1

        # for node i in the batched graph, index[i] is i's id in the graph before batch
        index = torch.zeros(batch.size(0),1,dtype=torch.long)

        # numer of nodes in each graph
        _, counts = torch.unique(batch, sorted=True, return_counts=True)

        # maximum number of nodes for all graphs in the batch
        max_nodes = torch.max(counts)

        # set the values in index
        id_start = 0
        for i in range(batch_size):
            index[id_start:id_start+counts[i]] = torch.arange(0,counts[i],dtype=torch.long).view(-1,1)
            id_start = id_start+counts[i]

        index = index.to(device)

        #the output graph features of walk pooling
        nodelevel_p = torch.zeros(batch_size,(self.walk_len*self.heads)).to(device)
        nodelevel_m = torch.zeros(batch_size,(self.walk_len*self.heads)).to(device)
        linklevel_p = torch.zeros(batch_size,(self.walk_len*self.heads)).to(device)
        linklevel_m = torch.zeros(batch_size,(self.walk_len*self.heads)).to(device)
        graphlevel = torch.zeros(batch_size,(self.walk_len*self.heads)).to(device)
        # a link (i,j) has two directions i->j and j->i, and
        # when extract the features of the link, we usually average over
        # the two directions. indices_odd and indices_even records the
        # indices for a link in two directions
        indices_odd = torch.arange(0,omega.size(0),2).to(device)
        indices_even = torch.arange(1,omega.size(0),2).to(device)

        omega = torch.index_select(omega, 0 ,indices_even)\
        + torch.index_select(omega,0,indices_odd)

        #node id of the candidate (or perturbation) link
        link_ij, link_ji = edge_index[:,torch.logical_not(edge_mask)]
        node_i = link_ij[indices_odd]
        node_j = link_ij[indices_even]

        # compute the powers of stochastic matrix
        for head in range(self.heads):

            # x on the plus graph and minus graph
            x_p = torch.zeros(batch.size(0),max_nodes,dtype=x.dtype).to(device)
            x_p = x_p.scatter_(1,index,1)
            x_m = torch.zeros(batch.size(0),max_nodes,dtype=x.dtype).to(device)
            x_m = x_m.scatter_(1,index,1)

            # propagage once
            x_p = self.propagate(edge_index, x= x_p, norm = weights_p[:,head])
            x_m = self.propagate(edge_index, x= x_m, norm = weights_m[:,head])
            # print(x_p.shape, nodelevel_p.shape)

            # start from tau = 2
            for i in range(self.walk_len):
                x_p = self.propagate(edge_index, x= x_p, norm = weights_p[:,head])
                x_m = self.propagate(edge_index, x= x_m, norm = weights_m[:,head])

                # print('x_p', x_p.shape)
                # print(node_i,index[node_i].view(-1), node_j,index[node_j])

                # returning probabilities around i + j
                nodelevel_p_w = x_p[node_i,index[node_i].view(-1)] + x_p[node_j,index[node_j].view(-1)]
                nodelevel_m_w = x_m[node_i,index[node_i].view(-1)] + x_m[node_j,index[node_j].view(-1)]
                # print('nodelevel_p', nodelevel_p.shape)
                # print('nodelevel_p_w', nodelevel_p_w.shape)
                nodelevel_p[:,head*self.walk_len+i] = nodelevel_p_w.view(-1)
                nodelevel_m[:,head*self.walk_len+i] = nodelevel_m_w.view(-1)

                # transition probabilities between i and j
                linklevel_p_w = x_p[node_i,index[node_j].view(-1)] + x_p[node_j,index[node_i].view(-1)]
                linklevel_m_w = x_m[node_i,index[node_j].view(-1)] + x_m[node_j,index[node_i].view(-1)]
                linklevel_p[:,head*self.walk_len+i] = linklevel_p_w.view(-1)
                linklevel_m[:,head*self.walk_len+i] = linklevel_m_w.view(-1)

                # graph average of returning probabilities
                diag_ele_p = torch.gather(x_p,1,index)
                diag_ele_m = torch.gather(x_m,1,index)

                graphlevel_p = scatter_add(diag_ele_p, batch, dim = 0)
                graphlevel_m = scatter_add(diag_ele_m, batch, dim = 0)

                graphlevel[:,head*self.walk_len+i] = (graphlevel_p-graphlevel_m).view(-1)

        feature_list = graphlevel
        feature_list = torch.cat((feature_list,omega),dim=1)
        feature_list = torch.cat((feature_list,nodelevel_p),dim=1)
        feature_list = torch.cat((feature_list,nodelevel_m),dim=1)
        feature_list = torch.cat((feature_list,linklevel_p),dim=1)
        feature_list = torch.cat((feature_list,linklevel_m),dim=1)


        return feature_list

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
