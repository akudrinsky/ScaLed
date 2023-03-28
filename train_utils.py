import torch
from torch_geometric.profile import profileit
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss

import warnings
from scipy.sparse import SparseEfficiencyWarning

from custom_losses import auc_loss, hinge_auc_loss

warnings.simplefilter('ignore', SparseEfficiencyWarning)
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)


@profileit()
def profile_train(model, train_loader, optimizer, device, emb, train_dataset, args):
    # normal training with BCE logit loss with profiling enabled
    model.train()

    total_loss = 0
    pbar = tqdm(train_loader, ncols=70)
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        num_nodes = data.num_nodes
        logits = model(num_nodes, data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_dataset)


def train_bce(model, train_loader, optimizer, device, emb, train_dataset, args):
    # normal training with BCE logit loss
    model.train()

    total_loss = 0
    pbar = tqdm(train_loader, ncols=70)
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        num_nodes = data.num_nodes
        edge_mask = data.edge_mask
        logits = model(num_nodes, data.z, data.edge_index, data.batch, x, edge_weight, node_id, edge_mask)
        loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_dataset)


def train_pairwise(model, train_positive_loader, train_negative_loader, optimizer, device, emb, train_dataset, args):
    # pairwise training with AUC loss + many others from PLNLP paper
    model.train()

    total_loss = 0
    pbar = tqdm(train_positive_loader, ncols=70)
    train_negative_loader = iter(train_negative_loader)

    for indx, data in enumerate(pbar):
        pos_data = data.to(device)
        optimizer.zero_grad()

        pos_x = pos_data.x if args.use_feature else None
        pos_edge_weight = pos_data.edge_weight if args.use_edge_weight else None
        pos_node_id = pos_data.node_id if emb else None
        pos_num_nodes = pos_data.num_nodes
        pos_logits = model(pos_num_nodes, pos_data.z, pos_data.edge_index, data.batch, pos_x, pos_edge_weight,
                           pos_node_id)

        neg_data = next(train_negative_loader).to(device)
        neg_x = neg_data.x if args.use_feature else None
        neg_edge_weight = neg_data.edge_weight if args.use_edge_weight else None
        neg_node_id = neg_data.node_id if emb else None
        neg_num_nodes = neg_data.num_nodes
        neg_logits = model(neg_num_nodes, neg_data.z, neg_data.edge_index, neg_data.batch, neg_x, neg_edge_weight,
                           neg_node_id)
        loss_fn = get_loss(args.loss_fn)
        loss = loss_fn(pos_logits, neg_logits, args.neg_ratio)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_dataset)


def get_loss(loss_function):
    if loss_function == 'auc_loss':
        return auc_loss
    elif loss_function == 'hinge_auc_loss':
        return hinge_auc_loss
    else:
        raise NotImplementedError(f'Loss function {loss_function} not implemented')