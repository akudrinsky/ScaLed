from timeit import default_timer

import torch
import shutil

import argparse
import time
import os
import sys
import os.path as osp
from shutil import copy
import copy as cp

from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.profile import timeit
import pdb

import scipy.sparse as ssp

from torch_geometric.datasets import Planetoid, AttributedGraphDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

import warnings
from scipy.sparse import SparseEfficiencyWarning

from baselines.gnn_link_pred import train_gnn
from baselines.mf import train_mf
from baselines.n2v import run_n2v
from data_utils import read_label, read_edges
from models import SAGE, DGCNN, GCN, GIN, GCN_WalkPooling
from ogbl_baselines.gnn_link_pred import train_gae_ogbl
from ogbl_baselines.mf import train_mf_ogbl
from ogbl_baselines.mlp_on_n2v import train_n2v_emb
from ogbl_baselines.n2v import run_and_save_n2v
from profiler_utils import profile_helper
from utils import get_pos_neg_edges, do_edge_split, Logger
from test_utils import *
from train_utils import *
from dataset import *
import shutil

warnings.simplefilter('ignore', SparseEfficiencyWarning)
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)


def run_sweal(args, device):
    if args.override_data:
        shutil.rmtree('dataset/')

    if args.save_appendix == '':
        args.save_appendix = '_' + time.strftime("%Y_%m_%d_%H_%M_%S") + f'_seed{args.seed}'
        if args.m and args.M:
            args.save_appendix += f'_m{args.m}_M{args.M}_dropedge{args.dropedge}_seed{args.seed}'

    if args.data_appendix == '':
        if args.m and args.M:
            args.data_appendix = f'_m{args.m}_M{args.M}_dropedge{args.dropedge}_seed{args.seed}'
        else:
            args.data_appendix = '_h{}_{}_rph{}_seed{}'.format(
                args.num_hops, args.node_label, ''.join(str(args.ratio_per_hop).split('.')), args.seed)
            if args.max_nodes_per_hop is not None:
                args.data_appendix += '_mnph{}'.format(args.max_nodes_per_hop)
        if args.use_valedges_as_input:
            args.data_appendix += '_uvai'

    args.res_dir = os.path.join('results/{}{}'.format(args.dataset, args.exp_name))
    print('Results will be saved in ' + args.res_dir)
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)
    if not args.keep_old:
        # Backup python files.
        copy('seal_link_pred.py', args.res_dir)
        copy('utils.py', args.res_dir)
    log_file = os.path.join(args.res_dir, 'log.txt')
    # Save command line input.
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
        f.write(cmd_input)
    print('Command line input: ' + cmd_input + ' is saved.')
    with open(log_file, 'a') as f:
        f.write('\n' + cmd_input)
    with open(os.path.join(args.res_dir, 'comment.txt'), 'w') as f:
        f.write(args.comment)

    # S[W]EAL Dataset prep + Training Flow
    if args.dataset.startswith('ogbl'):
        dataset = PygLinkPropPredDataset(name=args.dataset)
        split_edge = dataset.get_edge_split()
        data = dataset[0]
    elif args.dataset.startswith('attributed'):
        dataset_name = args.dataset.split('-')[-1]
        path = osp.join('dataset', dataset_name)
        dataset = AttributedGraphDataset(path, dataset_name)
        split_edge = do_edge_split(dataset, args.fast_split, val_ratio=args.split_val_ratio,
                                   test_ratio=args.split_test_ratio, neg_ratio=args.neg_ratio)
        data = dataset[0]
        data.edge_index = split_edge['train']['edge'].t()

    elif args.dataset in ['Cora', 'Pubmed', 'CiteSeer']:
        path = osp.join('dataset', args.dataset)
        dataset = Planetoid(path, args.dataset)
        split_edge = do_edge_split(dataset, args.fast_split, val_ratio=args.split_val_ratio,
                                   test_ratio=args.split_test_ratio, neg_ratio=args.neg_ratio)
        data = dataset[0]
        data.edge_index = split_edge['train']['edge'].t()
        import networkx as nx
        G = nx.Graph()
        G.add_edges_from(data.edge_index.T.detach().numpy())
    elif args.dataset in ['USAir', 'NS', 'Power', 'Celegans', 'Router', 'PB', 'Ecoli', 'Yeast']:
        # We consume the dataset split index as well
        file_name = os.path.join('data', 'link_prediction', args.dataset.lower())
        node_id_mapping = read_label(file_name)
        edges = read_edges(file_name, node_id_mapping)

        import networkx as nx
        G = nx.Graph(edges)
        edges_coo = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data = Data(edge_index=edges_coo.view(2, -1))
        data.edge_index = to_undirected(data.edge_index)
        data.num_nodes = torch.max(data.edge_index) + 1

        split_edge = do_edge_split(data, args.fast_split, val_ratio=args.split_val_ratio,
                                   test_ratio=args.split_test_ratio, neg_ratio=args.neg_ratio, data_passed=True)
        data.edge_index = split_edge['train']['edge'].t()

        # backward compatibility
        class DummyDataset:
            def __init__(self, root):
                self.root = root
                self.num_features = 0

            def __repr__(self):
                return args.dataset

            def __len__(self):
                return 1

        dataset = DummyDataset(root=f'dataset/{args.dataset}/SEALDataset_{args.dataset}')
        print("Finish reading from file")
    else:
        raise NotImplementedError(f'dataset {args.dataset} is not yet supported.')

    if args.dataset_stats:
        if args.dataset in ['USAir', 'NS', 'Power', 'Celegans', 'Router', 'PB', 'Ecoli', 'Yeast']:
            print(f'Dataset: {dataset}:')
            print('======================')
            print(f'Number of graphs: {len(dataset)}')
            print(f'Number of node features: {dataset.num_features}')
            print(f'Number of nodes: {G.number_of_nodes()}')
            print(f'Number of edges: {G.number_of_edges()}')
            degrees = [x[1] for x in G.degree]
            print(f'Average node degree: {sum(degrees) / len(G.nodes):.2f}')
            print(f'Average clustering coeffiecient: {nx.average_clustering(G)}')
            print(f'Is undirected: {data.is_undirected()}')
            print(f'Training {args.model}')
        else:
            print(f'Dataset: {dataset}:')
            print('======================')
            print(f'Number of graphs: {len(dataset)}')
            print(f'Number of features: {dataset.num_features}')
            print(f'Number of nodes: {data.num_nodes}')
            print(f'Number of edges: {G.number_of_edges()}')
            print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
            print(f'Average clustering coeffiecient: {nx.average_clustering(G)}')
            print(f'Is undirected: {data.is_undirected()}')

    if args.dataset.startswith('ogbl-citation'):
        args.eval_metric = 'mrr'
        directed = True
    elif args.dataset.startswith('ogbl'):
        args.eval_metric = 'hits'
        directed = False
    else:  # assume other datasets are undirected
        args.eval_metric = 'auc'
        directed = False

    if args.use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        if not directed:
            val_edge_index = to_undirected(val_edge_index)
        data.edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=int)
        data.edge_weight = torch.cat([data.edge_weight, val_edge_weight], 0)

    evaluator = None
    if args.dataset.startswith('ogbl'):
        evaluator = Evaluator(name=args.dataset)
    if args.eval_metric == 'hits':
        loggers = {
            'Hits@20': Logger(args.runs, args),
            'Hits@50': Logger(args.runs, args),
            'Hits@100': Logger(args.runs, args),
        }
    elif args.eval_metric == 'mrr':
        loggers = {
            'MRR': Logger(args.runs, args),
        }
    elif args.eval_metric == 'auc':
        loggers = {
            'AUC': Logger(args.runs, args),
            'AP': Logger(args.runs, args)
        }

    if args.use_heuristic:
        # Test link prediction heuristics.
        num_nodes = data.num_nodes
        if 'edge_weight' in data:
            edge_weight = data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(data.edge_index.size(1), dtype=int)

        A = ssp.csr_matrix((edge_weight, (data.edge_index[0], data.edge_index[1])),
                           shape=(num_nodes, num_nodes))

        pos_val_edge, neg_val_edge = get_pos_neg_edges('valid', split_edge,
                                                       data.edge_index,
                                                       data.num_nodes, neg_ratio=args.neg_ratio)
        pos_test_edge, neg_test_edge = get_pos_neg_edges('test', split_edge,
                                                         data.edge_index,
                                                         data.num_nodes, neg_ratio=args.neg_ratio)
        pos_val_pred, pos_val_edge = eval(args.use_heuristic)(A, pos_val_edge)
        neg_val_pred, neg_val_edge = eval(args.use_heuristic)(A, neg_val_edge)
        pos_test_pred, pos_test_edge = eval(args.use_heuristic)(A, pos_test_edge)
        neg_test_pred, neg_test_edge = eval(args.use_heuristic)(A, neg_test_edge)

        if args.eval_metric == 'hits':
            results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, evaluator)
        elif args.eval_metric == 'mrr':
            results = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, evaluator)
        elif args.eval_metric == 'auc':
            val_pred = torch.cat([pos_val_pred, neg_val_pred])
            val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int),
                                  torch.zeros(neg_val_pred.size(0), dtype=int)])
            test_pred = torch.cat([pos_test_pred, neg_test_pred])
            test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int),
                                   torch.zeros(neg_test_pred.size(0), dtype=int)])
            results = evaluate_auc(val_pred, val_true, test_pred, test_true)

        for key, result in results.items():
            loggers[key].add_result(0, result)
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics()
            with open(log_file, 'a') as f:
                print(key, file=f)
                loggers[key].print_statistics(f=f)
        exit()

    # SEAL.
    path = dataset.root + '_seal{}'.format(args.data_appendix)
    use_coalesce = True if args.dataset == 'ogbl-collab' else False
    if not args.dynamic_train and not args.dynamic_val and not args.dynamic_test:
        args.num_workers = 0

    rw_kwargs = {}
    if args.m and args.M:
        rw_kwargs = {
            "m": args.m,
            "M": args.M
        }
    if args.calc_ratio:
        rw_kwargs.update({'calc_ratio': True})

    if not any([args.train_gae, args.train_mf, args.train_n2v]):
        print("Setting up Train data")
        dataset_class = 'SEALDynamicDataset' if args.dynamic_train else 'SEALDataset'
        if not args.pairwise:
            # тут SEALDataset
            train_dataset = eval(dataset_class)(
                path,
                data,
                split_edge,
                num_hops=args.num_hops,
                percent=args.train_percent,
                split='train',
                use_coalesce=use_coalesce,
                node_label=args.node_label,
                ratio_per_hop=args.ratio_per_hop,
                max_nodes_per_hop=args.max_nodes_per_hop,
                directed=directed,
                rw_kwargs=rw_kwargs,
                device=device,
                neg_ratio=args.neg_ratio,
            )
        else:
            pos_path = f'{path}_pos_edges'
            train_positive_dataset = eval(dataset_class)(
                pos_path,
                data,
                split_edge,
                num_hops=args.num_hops,
                percent=args.train_percent,
                split='train',
                use_coalesce=use_coalesce,
                node_label=args.node_label,
                ratio_per_hop=args.ratio_per_hop,
                max_nodes_per_hop=args.max_nodes_per_hop,
                directed=directed,
                rw_kwargs=rw_kwargs,
                device=device,
                pairwise=args.pairwise,
                pos_pairwise=True,
                neg_ratio=args.neg_ratio,
            )
            neg_path = f'{path}_neg_edges'
            train_negative_dataset = eval(dataset_class)(
                neg_path,
                data,
                split_edge,
                num_hops=args.num_hops,
                percent=args.train_percent,
                split='train',
                use_coalesce=use_coalesce,
                node_label=args.node_label,
                ratio_per_hop=args.ratio_per_hop,
                max_nodes_per_hop=args.max_nodes_per_hop,
                directed=directed,
                rw_kwargs=rw_kwargs,
                device=device,
                pairwise=args.pairwise,
                pos_pairwise=False,
                neg_ratio=args.neg_ratio,
            )
    max_viz = 3
    if max_viz:  # visualize some graphs
        import networkx as nx
        from torch_geometric.utils import to_networkx
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        for i, g in enumerate(loader):
            if i >= max_viz:
                break
            f = plt.figure(figsize=(20, 20))
            g = g.to(device)
            node_size = 100
            with_labels = True
            G = to_networkx(g, node_attrs=['z'])
            labels = {i: G.nodes[i]['z'] for i in range(len(G))}
            colors = ['green' if labels[i] != 1 else 'red' for i in range(len(G))]
            nx.draw(G, node_size=node_size, arrows=True, with_labels=with_labels,
                    labels=labels, node_color=colors)
            os.makedirs(f'{args.res_dir}/graphs/', exist_ok=True)
            f.savefig(f'{args.res_dir}/graphs/train_graph_{i}.png')

    if not any([args.train_gae, args.train_mf, args.train_n2v]):
        print("Setting up Val data")
        dataset_class = 'SEALDynamicDataset' if args.dynamic_val else 'SEALDataset'
        val_dataset = eval(dataset_class)(
            path,
            data,
            split_edge,
            num_hops=args.num_hops,
            percent=args.val_percent,
            split='valid',
            use_coalesce=use_coalesce,
            node_label=args.node_label,
            ratio_per_hop=args.ratio_per_hop,
            max_nodes_per_hop=args.max_nodes_per_hop,
            directed=directed,
            rw_kwargs=rw_kwargs,
            device=device
        )
        print("Setting up Test data")
        dataset_class = 'SEALDynamicDataset' if args.dynamic_test else 'SEALDataset'
        test_dataset = eval(dataset_class)(
            path,
            data,
            split_edge,
            num_hops=args.num_hops,
            percent=args.test_percent,
            split='test',
            use_coalesce=use_coalesce,
            node_label=args.node_label,
            ratio_per_hop=args.ratio_per_hop,
            max_nodes_per_hop=args.max_nodes_per_hop,
            directed=directed,
            rw_kwargs=rw_kwargs,
            device=device
        )

    if args.calc_ratio:
        print("Finished calculating ratio of datasets.")
        exit()

    max_z = 1000  # set a large max_z so that every z has embeddings to look up

    if not any([args.train_gae, args.train_mf, args.train_n2v]):
        if args.pairwise:
            train_pos_loader = DataLoader(train_positive_dataset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers)
            train_neg_loader = DataLoader(train_negative_dataset, batch_size=args.batch_size * args.neg_ratio,
                                          shuffle=True, num_workers=args.num_workers)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers)

        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 num_workers=args.num_workers)

    if args.train_node_embedding:
        emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
    elif args.pretrained_node_embedding:
        weight = torch.load(args.pretrained_node_embedding)
        emb = torch.nn.Embedding.from_pretrained(weight)
        emb.weight.requires_grad = False
    else:
        emb = None

    seed_everything(args.seed)  # reset rng for model weights
    for run in range(args.runs):
        if args.pairwise:
            train_dataset = train_positive_dataset
        if args.train_gae:
            if not args.dataset.startswith('ogbl'):
                train_gnn(device, data, split_edge, args)
            else:
                train_gae_ogbl(args, device, data, split_edge)
            exit()
        if args.train_n2v:
            if not args.dataset.startswith('ogbl'):
                run_n2v(device, data, split_edge, args.epochs, args.lr, args.hidden_channels, args.neg_ratio,
                        args.batch_size, args.num_workers, args)
            else:
                run_and_save_n2v(args, device, data)  # saves n2v embeddings
                train_n2v_emb(args, device, data, split_edge)  # trains MLP on above saved n2v embeddings
            exit()
        if args.train_mf:
            if not args.dataset.startswith('ogbl'):
                train_mf(data, split_edge, device, args.log_steps, args.num_layers, args.hidden_channels, args.dropout,
                         args.batch_size, args.lr, args.epochs, args.eval_steps, args.runs, args.seed, args)
            else:
                train_mf_ogbl(args, split_edge, data)
            exit()
        if args.model == 'DGCNN':
            model = DGCNN(args.hidden_channels, args.num_layers, max_z, args.sortpool_k,
                          train_dataset, args.dynamic_train, use_feature=args.use_feature,
                          node_embedding=emb, dropedge=args.dropedge).to(device)
        elif args.model == 'SAGE':
            model = SAGE(args.hidden_channels, args.num_layers, max_z, train_dataset,
                         args.use_feature, node_embedding=emb, dropedge=args.dropedge).to(device)
        elif args.model == 'GCN':
            model = GCN(args.hidden_channels, args.num_layers, max_z, train_dataset,
                        args.use_feature, node_embedding=emb, dropedge=args.dropedge).to(device)
        elif args.model == 'GIN':
            model = GIN(args.hidden_channels, args.num_layers, max_z, train_dataset,
                        args.use_feature, node_embedding=emb).to(device)
        elif args.model == 'GCN_WalkPooling':
            model = GCN_WalkPooling(args.hidden_channels, args.num_layers, max_z, train_dataset,
                        args.use_feature, node_embedding=emb, dropedge=args.dropedge).to(device)
        parameters = list(model.parameters())
        if args.train_node_embedding:
            torch.nn.init.xavier_uniform_(emb.weight)
            parameters += list(emb.parameters())
        optimizer = torch.optim.Adam(params=parameters, lr=args.lr)
        total_params = sum(p.numel() for param in parameters for p in param)
        print(f'Total number of parameters is {total_params}')
        if args.model == 'DGCNN':
            print(f'SortPooling k is set to {model.k}')
        with open(log_file, 'a') as f:
            print(f'Total number of parameters is {total_params}', file=f)
            if args.model == 'DGCNN':
                print(f'SortPooling k is set to {model.k}', file=f)

        start_epoch = 1
        if args.continue_from is not None:
            model.load_state_dict(
                torch.load(os.path.join(args.res_dir,
                                        'run{}_model_checkpoint{}.pth'.format(run + 1, args.continue_from)))
            )
            optimizer.load_state_dict(
                torch.load(os.path.join(args.res_dir,
                                        'run{}_optimizer_checkpoint{}.pth'.format(run + 1, args.continue_from)))
            )
            start_epoch = args.continue_from + 1
            args.epochs -= args.continue_from

        if args.only_test:
            results = test(evaluator, model, val_loader, device, emb, test_loader, args)
            for key, result in results.items():
                loggers[key].add_result(run, result)
            for key, result in results.items():
                valid_res, test_res = result
                print(key)
                print(f'Run: {run + 1:02d}, '
                      f'Valid: {100 * valid_res:.2f}%, '
                      f'Test: {100 * test_res:.2f}%')
            pdb.set_trace()
            exit()

        if args.test_multiple_models:
            model_paths = [
            ]  # enter all your pretrained .pth model paths here
            models = []
            for path in model_paths:
                m = cp.deepcopy(model)
                m.load_state_dict(torch.load(path))
                models.append(m)
            Results = test_multiple_models(models, val_loader, device, emb, test_loader, evaluator, args)
            for i, path in enumerate(model_paths):
                print(path)
                with open(log_file, 'a') as f:
                    print(path, file=f)
                results = Results[i]
                for key, result in results.items():
                    loggers[key].add_result(run, result)
                for key, result in results.items():
                    valid_res, test_res = result
                    to_print = (f'Run: {run + 1:02d}, ' +
                                f'Valid: {100 * valid_res:.2f}%, ' +
                                f'Test: {100 * test_res:.2f}%')
                    print(key)
                    print(to_print)
                    with open(log_file, 'a') as f:
                        print(key, file=f)
                        print(to_print, file=f)
            pdb.set_trace()
            exit()

        # Training starts
        all_stats = []
        for epoch in range(start_epoch, start_epoch + args.epochs):
            if args.profile:
                # this gives the stats for exactly one training epoch
                loss, stats = profile_train(model, train_loader, optimizer, device, emb, train_dataset, args)
                all_stats.append(stats)
            else:
                if not args.pairwise:
                    loss = train_bce(model, train_loader, optimizer, device, emb, train_dataset, args)
                else:
                    loss = train_pairwise(model, train_pos_loader, train_neg_loader, optimizer, device, emb,
                                          train_dataset,
                                          args)

            if epoch % args.eval_steps == 0:
                results = test(evaluator, model, val_loader, device, emb, test_loader, args)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    if args.checkpoint_training:
                        model_name = os.path.join(
                            args.res_dir, 'run{}_model_checkpoint{}.pth'.format(run + 1, epoch))
                        optimizer_name = os.path.join(
                            args.res_dir, 'run{}_optimizer_checkpoint{}.pth'.format(run + 1, epoch))
                        torch.save(model.state_dict(), model_name)
                        torch.save(optimizer.state_dict(), optimizer_name)

                    for key, result in results.items():
                        valid_res, test_res = result
                        to_print = (f'Run: {run + 1:02d}, Epoch: {epoch:02d}, ' +
                                    f'Loss: {loss:.4f}, Valid: {100 * valid_res:.2f}%, ' +
                                    f'Test: {100 * test_res:.2f}%')
                        print(key)
                        print(to_print)
                        with open(log_file, 'a') as f:
                            print(key, file=f)
                            print(to_print, file=f)

        if args.profile:
            stats_suffix = f'{args.model}_{args.dataset}{args.data_appendix}_seed_{args.seed}'
            profile_helper(all_stats, model, train_dataset, stats_suffix)

        for key in loggers.keys():
            print(key)
            loggers[key].add_info(args.epochs, args.runs)
            loggers[key].print_statistics(run)
            with open(log_file, 'a') as f:
                print(key, file=f)
                loggers[key].print_statistics(run, f=f)

    for key in loggers.keys():
        print(key)
        loggers[key].add_info(args.epochs, args.runs)
        loggers[key].print_statistics()
        with open(log_file, 'a') as f:
            print(key, file=f)
            loggers[key].print_statistics(f=f)
    print(f'Total number of parameters is {total_params}')
    print(f'Results are saved in {args.res_dir}')

    if args.delete_dataset:
        if os.path.exists(path):
            shutil.rmtree(path)

    print("fin.")


@timeit()
def run_sweal_with_run_profiling(args, device):
    start = default_timer()
    run_sweal(args, device)
    end = default_timer()
    print(f"Time taken for run: {end - start:.2f} seconds")


def main():
    # Data settings
    parser = argparse.ArgumentParser(description='OGBL (SEAL)')
    parser.add_argument('--dataset', type=str, default='ogbl-collab')
    parser.add_argument('--fast_split', action='store_true',
                        help="for large custom datasets (not OGB), do a fast data split")
    # GNN settings
    parser.add_argument('--model', type=str, default='DGCNN')
    parser.add_argument('--sortpool_k', type=float, default=0.6)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    # Subgraph extraction settings
    parser.add_argument('--num_hops', type=int, default=1)
    parser.add_argument('--ratio_per_hop', type=float, default=1.0)
    parser.add_argument('--max_nodes_per_hop', type=int, default=None)
    parser.add_argument('--node_label', type=str, default='drnl',
                        help="which specific labeling trick to use")
    parser.add_argument('--use_feature', action='store_true',
                        help="whether to use raw node features as GNN input")
    parser.add_argument('--use_edge_weight', action='store_true',
                        help="whether to consider edge weight in GNN")
    # Training settings
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--train_percent', type=float, default=100)
    parser.add_argument('--val_percent', type=float, default=100)
    parser.add_argument('--test_percent', type=float, default=100)
    parser.add_argument('--dynamic_train', action='store_true',
                        help="dynamically extract enclosing subgraphs on the fly")
    parser.add_argument('--dynamic_val', action='store_true')
    parser.add_argument('--dynamic_test', action='store_true')
    parser.add_argument('--num_workers', type=int, default=16,
                        help="number of workers for dynamic mode; 0 if not dynamic")
    parser.add_argument('--train_node_embedding', action='store_true',
                        help="also train free-parameter node embeddings together with GNN")
    parser.add_argument('--pretrained_node_embedding', type=str, default=None,
                        help="load pretrained node embeddings as additional node features")
    # Testing settings
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--checkpoint_training', action='store_true')
    parser.add_argument('--data_appendix', type=str, default='',
                        help="an appendix to the data directory")
    parser.add_argument('--save_appendix', type=str, default='',
                        help="an appendix to the save directory")
    parser.add_argument('--keep_old', action='store_true',
                        help="do not overwrite old files in the save directory")
    parser.add_argument('--delete_dataset', action='store_true',
                        help="delete existing datasets folder before running new command")
    parser.add_argument('--continue_from', type=int, default=None,
                        help="from which epoch's checkpoint to continue training")
    parser.add_argument('--only_test', action='store_true',
                        help="only test without training")
    parser.add_argument('--test_multiple_models', action='store_true',
                        help="test multiple models together")
    parser.add_argument('--use_heuristic', type=str, default=None,
                        help="test a link prediction heuristic (CN or AA)")
    parser.add_argument('--dataset_stats', action='store_true',
                        help="Print dataset statistics", default=True)
    parser.add_argument('--m', type=int, default=0, help="Set rw length")
    parser.add_argument('--M', type=int, default=0, help="Set number of rw")
    parser.add_argument('--dropedge', type=float, default=.0, help="Drop Edge Value for initial edge_index")
    parser.add_argument('--cuda_device', type=int, default=0, help="Only set available the passed GPU")

    parser.add_argument('--calc_ratio', action='store_true', help="Calculate overall sparsity ratio")
    parser.add_argument('--pairwise', action='store_true',
                        help="Choose to override the BCE loss to pairwise loss functions")
    parser.add_argument('--loss_fn', type=str, help="Choose the loss function")
    parser.add_argument('--neg_ratio', type=int, default=1,
                        help="Compile neg_ratio times the positive samples for compiling neg_samples"
                             "(only for Training data)")
    parser.add_argument('--profile', action='store_true', help="Run the PyG profiler for each epoch")
    parser.add_argument('--split_val_ratio', type=float, default=0.05)
    parser.add_argument('--split_test_ratio', type=float, default=0.1)
    parser.add_argument('--train_mlp', action='store_true',
                        help="Train using structure unaware mlp")
    parser.add_argument('--train_gae', action='store_true', help="Train GAE on the dataset")
    parser.add_argument('--base_gae', type=str, default='', help='Choose base GAE model', choices=['GCN', 'SAGE'])

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=1)  # we can set this to value in dataset_split_num as well
    parser.add_argument('--dataset_split_num', type=int, default=1)  # This is maintained for WalkPool Datasets only

    parser.add_argument('--train_n2v', action='store_true', help="Train node2vec on the dataset")
    parser.add_argument('--train_mf', action='store_true', help="Train MF on the dataset")

    parser.add_argument('--comment', type=str, help="Comment something about training model", default='')
    parser.add_argument('--exp_name', type=str, help="Exp name", default='')
    parser.add_argument('--override_data', action='store_true', help="If override dataset", default='')

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')

    seed_everything(args.seed)

    if args.profile and not torch.cuda.is_available():
        raise Exception("CUDA needs to be enabled to run PyG profiler")

    if args.profile:
        run_sweal_with_run_profiling(args, device)
    else:
        start = default_timer()
        run_sweal(args, device)
        end = default_timer()
        print(f"Time taken for run: {end - start:.2f} seconds")

if __name__ == '__main__':
    # import pprofile
    # profiler = pprofile.Profile()
    # with profiler:
    #     main()
    # profiler.print_stats()

    # profiler.dump_stats("profstats.txt")
    main()
