import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from GCNII_layer import GCNIIdenseConv
import math
from models.common_blocks import batch_norm
import os
from torch_geometric.utils import remove_self_loops, add_self_loops
import numpy as np
import random

def load_data(dataset="Cora"):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
    if dataset in ["Cora", "Citeseer", "Pubmed"]:
        data = Planetoid(path, dataset, T.NormalizeFeatures())[0]
        num_nodes = data.x.size(0)
        edge_index, _ = remove_self_loops(data.edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        if isinstance(edge_index, tuple):
            data.edge_index = edge_index[0]
        else:
            data.edge_index = edge_index
        return data
    else:
        raise Exception(f'the dataset of {dataset} has not been implemented')

def remove_feature(data, miss_rate):
    num_nodes = data.x.size(0)
    erasing_pool = torch.arange(num_nodes)[~data.train_mask]
    size = int(len(erasing_pool) * miss_rate)
    idx_erased = np.random.choice(erasing_pool, size=size, replace=False)
    x = data.x
    x[idx_erased] = 0.
    return x

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

seeds = [100, 200, 300, 400, 500]

parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
parser.add_argument('--type_norm', type=str, default="group", help='{None, batch, group, pair}')
parser.add_argument('--miss_rate', type=float, default=0.)
args = parser.parse_args()

dataset = 'Cora'
data = load_data(dataset)
if args.miss_rate > 0.:
    data.x = remove_feature(data, args.miss_rate)

print(data.train_mask.sum())
print(data.val_mask.sum())
print(data.test_mask.sum())

###################hyperparameters
nlayer = args.layer
dropout = 0.6
alpha = 0.1
lamda = 0.5
hidden_dim = 64
weight_decay1 = 0.01
weight_decay2 = 5e-4
lr = 0.01
patience = 100
## set parameters used in group norm
num_groups = 10 # 10
if args.layer == 2:
    skip_weight = 0.005
elif args.layer == 64:
    skip_weight = 0.0005
else:
    skip_weight = 0.001
type_norm = args.type_norm
num_features = 1433
num_classes = 7
#####################

GConv = GCNIIdenseConv


class GCNII_model(torch.nn.Module):
    def __init__(self):
        super(GCNII_model, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.layers_bn = torch.nn.ModuleList([])
        self.convs.append(torch.nn.Linear(num_features, hidden_dim))
        self.type_norm = type_norm
        if self.type_norm in ['None', 'batch', 'pair']:
            skip_connect = False
        else:
            skip_connect = True
        for i in range(nlayer):
            self.convs.append(GConv(hidden_dim, hidden_dim))
            self.layers_bn.append(batch_norm(hidden_dim, self.type_norm, skip_connect, num_groups, skip_weight))

        self.convs.append(torch.nn.Linear(hidden_dim,num_classes))

        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())
        self.non_reg_params += list(self.layers_bn[0:].parameters())


    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        _hidden = []
        x = F.dropout(x, dropout ,training=self.training)
        x = self.convs[0](x)
        x = F.relu(x)
        _hidden.append(x)
        for i,con in enumerate(self.convs[1:-1]):
            x = F.dropout(x, dropout ,training=self.training)
            beta = math.log(lamda/(i+1)+1)
            x = con(x, edge_index,alpha, _hidden[0],beta,edge_weight)
            x = self.layers_bn[i](x)
            x = F.relu(x)
        x = F.dropout(x, dropout ,training=self.training)
        x = self.convs[-1](x)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
acc_test_list = []
for seed in seeds:
    set_seed(seed)
    model, data = GCNII_model().to(device), data.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.reg_params, weight_decay=weight_decay1),
        dict(params=model.non_reg_params, weight_decay=weight_decay2)
    ], lr=lr)

    def train():
        model.train()
        optimizer.zero_grad()
        loss_train = F.nll_loss(model()[data.train_mask], data.y[data.train_mask])
        loss_train.backward()
        optimizer.step()
        return loss_train.item()


    @torch.no_grad()
    def test():
        model.eval()
        logits = model()
        loss_val = F.nll_loss(logits[data.val_mask], data.y[data.val_mask]).item()
        for _, mask in data('val_mask'):
            pred = logits[mask].max(1)[1]
            val_accs = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        for _, mask in data('test_mask'):
            pred = logits[mask].max(1)[1]
            test_accs = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        return loss_val, val_accs, test_accs


    best_val_loss = 9999999
    best_val_acc = 0.
    test_acc = 0
    bad_counter = 0
    best_epoch = 0
    for epoch in range(1, 1500):
        loss_tra = train()
        loss_val,acc_val_tmp, acc_test_tmp = test()
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            test_acc = acc_test_tmp
            bad_counter = 0
            best_epoch = epoch
        else:
            bad_counter+=1
        if epoch%20 == 0:
            log = 'Epoch: {:03d}, Train loss: {:.4f}, Val loss: {:.4f}, Test acc: {:.4f}'
            print(log.format(epoch, loss_tra, loss_val, test_acc))
        if bad_counter == patience:
            break
    log = 'best Epoch: {:03d}, Val loss: {:.4f}, Test acc: {:.4f}'
    acc_test_list.append(test_acc)
    print(log.format(best_epoch, best_val_loss, test_acc))

print('test acc of 5 seeds: ', acc_test_list)
print('avg test acc and std: ', np.mean(acc_test_list), np.std(acc_test_list))