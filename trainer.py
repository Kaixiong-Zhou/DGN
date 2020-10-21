import torch
import os
from models.GCN import GCN
from models.simpleGCN import simpleGCN
from models.GAT import GAT
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor

import torch_geometric.transforms as T
import torch.nn.functional as F
import glob
from torch_geometric.utils import remove_self_loops, add_self_loops
import numpy as np
from MI.kde import mi_kde


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
    elif dataset in ['CoauthorCS']:
        data = Coauthor(path, 'cs', T.NormalizeFeatures())[0]
        num_nodes = data.x.size(0)
        edge_index, _ = remove_self_loops(data.edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        if isinstance(edge_index, tuple):
            data.edge_index = edge_index[0]
        else:
            data.edge_index = edge_index

        # devide training validation and testing set
        train_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        val_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        test_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        train_num = 40
        val_num = 150
        for i in range(15): # number of labels
            index = (data.y == i).nonzero()[:,0]
            perm = torch.randperm(index.size(0))
            train_mask[index[perm[:train_num]]] = 1
            val_mask[index[perm[train_num:(train_num+val_num)]]] = 1
            test_mask[index[perm[(train_num+val_num):]]] = 1
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
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


def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()

class trainer(object):
    def __init__(self, args):
        self.dataset = args.dataset
        self.device = torch.device(f'cuda:{args.cuda_num}' if args.cuda else 'cpu')
        if self.dataset in ["Cora", "Citeseer", "Pubmed", 'CoauthorCS']:
            self.data = load_data(self.dataset)
            self.loss_fn = torch.nn.functional.nll_loss
        else:
            raise Exception(f'the dataset of {self.dataset} has not been implemented')

        self.miss_rate = args.miss_rate
        if self.miss_rate > 0.:
            self.data.x = remove_feature(self.data, self.miss_rate)

        self.type_model = args.type_model
        self.epochs = args.epochs
        self.grad_clip = args.grad_clip
        self.weight_decay = args.weight_decay
        if self.type_model == 'GCN':
            self.model = GCN(args)
        elif self.type_model == 'simpleGCN':
            self.model = simpleGCN(args)
        elif self.type_model == 'GAT':
            self.model = GAT(args)
        else:
            raise Exception(f'the model of {self.type_model} has not been implemented')

        self.data.to(self.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.seed = args.random_seed
        self.type_norm = args.type_norm
        self.skip_weight = args.skip_weight


    def train_net(self):
        try:
            loss_train = self.run_trainSet()
            acc_train, acc_valid, acc_test = self.run_testSet()
            return loss_train, acc_train, acc_valid, acc_test
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
            else:
                raise e

    def compute_MI(self):
        self.model.eval()
        data_x = self.data.x.data.cpu().numpy()
        with torch.no_grad():
            layers_self = self.model(self.data.x, self.data.edge_index)
        layer_self = layers_self.data.cpu().numpy()
        MI_XiX = mi_kde(layer_self, data_x, var=0.1)
        return MI_XiX



    def train_compute_MI(self):
        best_acc = 0
        for epoch in range(self.epochs):
            loss_train, acc_train, acc_valid, acc_test = self.train_net()
            print('Epoch: {:02d}, Loss: {:.4f}, val_acc: {:.4f}, test_acc:{:.4f}'.format(epoch, loss_train,
                                                                                         acc_valid, acc_test))
            if best_acc < acc_valid:
                best_acc = acc_valid
                self.model.cpu()
                self.save_model(self.type_model, self.dataset)
                self.model.to(self.device)

        # reload the best model
        state_dict = self.load_model(self.type_model, self.dataset)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

        # evaluate the saved model
        acc_train, acc_valid, acc_test = self.run_testSet()
        print('val_acc: {:.4f}, test_acc:{:.4f}'.format(acc_valid, acc_test))

        # compute the instance information gain
        MI_XiX = self.compute_MI()

        # compute the intra-group and inter-group distances first
        # then obtain the group distance ratio
        dis_intra, dis_inter = self.dis_cluster()
        if self.dataset == 'CoauthorCS' and self.type_model == 'simpleGCN':
            # if the intra-group and inter-group distances are close, we assign them the same values
            # and have the distance ratio of 1.
            distance_gap = dis_inter - dis_intra
            dis_ratio = 1. if distance_gap < 0.35 else dis_inter / dis_intra
        else:
            dis_ratio = dis_inter / dis_intra
        # if both dis_inter and dis_intra are close to zero, the value of dis_ratio is nan
        # in this case, we assign the distance ratio to 1.
        dis_ratio = 1. if np.isnan(dis_ratio) else dis_ratio

        return acc_test, MI_XiX, dis_ratio




    def dis_cluster(self):
        self.model.eval()
        with torch.no_grad():
            X = self.model(self.data.x, self.data.edge_index)
        X_labels = []
        for i in range(self.model.num_classes):
            X_label = X[self.data.y == i].data.cpu().numpy()
            h_norm = np.sum(np.square(X_label), axis=1, keepdims=True)
            h_norm[h_norm == 0.] = 1e-3
            X_label = X_label / np.sqrt(h_norm)
            X_labels.append(X_label)

        dis_intra = 0.
        for i in range(self.model.num_classes):
            x2 = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
            dists = x2 + x2.T - 2 * np.matmul(X_labels[i], X_labels[i].T)
            dis_intra += np.mean(dists)
        dis_intra /= self.model.num_classes

        dis_inter = 0.
        for i in range(self.model.num_classes-1):
            for j in range(i+1, self.model.num_classes):
                x2_i = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
                x2_j = np.sum(np.square(X_labels[j]), axis=1, keepdims=True)
                dists = x2_i + x2_j.T - 2 * np.matmul(X_labels[i], X_labels[j].T)
                dis_inter += np.mean(dists)
        num_inter = float(self.model.num_classes * (self.model.num_classes-1) / 2)
        dis_inter /= num_inter

        print('dis_intra: ', dis_intra)
        print('dis_inter: ', dis_inter)
        return dis_intra, dis_inter

    def run_trainSet(self):
        self.model.train()
        logits = self.model(self.data.x, self.data.edge_index)
        logits = F.log_softmax(logits[self.data.train_mask], 1)
        loss = self.loss_fn(logits, self.data.y[self.data.train_mask])
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return loss.item()

    def run_testSet(self):
        self.model.eval()
        # torch.cuda.empty_cache()
        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index)
        logits = F.log_softmax(logits, 1)
        acc_train = evaluate(logits, self.data.y, self.data.train_mask)
        acc_valid = evaluate(logits, self.data.y, self.data.val_mask)
        acc_test = evaluate(logits, self.data.y, self.data.test_mask)
        return acc_train, acc_valid, acc_test


    def filename(self, filetype='logs', type_model='GCN', dataset='PPI'):
        filedir = f'./{filetype}/{dataset}'
        if not os.path.exists(filedir):
            os.makedirs(filedir)

        num_layers = int(self.model.num_layers)
        type_norm = self.type_norm
        miss_rate = int(self.miss_rate * 10)
        seed = int(self.seed)

        if type_norm == 'group':
            group = self.model.num_groups
            skip_weight = int(self.model.skip_weight * 1e3)

            filename = f'{filetype}_{type_model}_{type_norm}' \
                f'L{num_layers}M{miss_rate}S{seed}G{group}S{skip_weight}.pth.tar'
        else:

            filename = f'{filetype}_{type_model}_{type_norm}' \
                       f'L{num_layers}M{miss_rate}S{seed}.pth.tar'

        filename = os.path.join(filedir, filename)
        return filename

    def get_saved_info(self, path=None):
        paths = glob.glob(path)
        paths.sort()

        def get_numbers(items, delimiter, idx, replace_word, must_contain=''):
            return list(set([int(
                name.split(delimiter)[idx].replace(replace_word, ''))
                for name in items if must_contain in name]))

        basenames = [os.path.basename(path.rsplit('.', 2)[0]) for path in paths]
        epochs = get_numbers(basenames, '_', 2, 'epoch')
        epochs.sort()
        return epochs

    def load_model(self, type_model='GCN', dataset='PPI'):
        filename = self.filename(filetype='params', type_model=type_model, dataset=dataset)
        if os.path.exists(filename):
            print('load model: ', type_model, filename)
            return torch.load(filename)
        else:
            return None

    def save_model(self, type_model='GCN', dataset='PPI'):
        filename = self.filename(filetype='params', type_model=type_model, dataset=dataset)
        state = self.model.state_dict()
        torch.save(state, filename)
        print('save model to', filename)

