from options.base_options import BaseOptions, reset_weight
from trainer import trainer
import torch
import os
import numpy as np
import random

def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_num)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

seeds = [100, 200, 300, 400, 500] #  + [123, 50, 150, 250, 350, 450]

# layers_GCN = list(range(1, 10, 1)) + list(range(10, 31, 5))
layers_GCN = [2, 15, 30]
# layers_SGCN = [1, 5] + list(range(10, 121, 10))
layers_SGCN = [5, 60, 120]


def main(args):

    if args.type_model in ['GCN', 'GAT', 'GCNII']:
        layers = layers_GCN
    else:
        layers = layers_SGCN

    acc_test_layers = []
    MI_XiX_layers = []
    dis_ratio_layers = []
    for layer in layers:
        args.num_layers = layer
        if args.type_norm == 'group':
            args = reset_weight(args)
        acc_test_seeds = []
        MI_XiX_seeds = []
        dis_ratio_seeds =  []
        for seed in seeds:
            args.random_seed = seed
            set_seed(args)
            trnr = trainer(args)
            acc_test, MI_XiX, dis_ratio = trnr.train_compute_MI()
            acc_test_seeds.append(acc_test)
            MI_XiX_seeds.append(MI_XiX)
            dis_ratio_seeds.append(dis_ratio)
        avg_acc_test = np.mean(acc_test_seeds)
        avg_MI_XiX = np.mean(MI_XiX_seeds)
        avg_dis_ratio = np.mean(dis_ratio_seeds)

        acc_test_layers.append(avg_acc_test)
        MI_XiX_layers.append(avg_MI_XiX)
        dis_ratio_layers.append(avg_dis_ratio)

    print(f'experiment results of {args.type_norm} applied in {args.type_model} on dataset {args.dataset}')
    print('number of layers: ', layers)
    print('test accuracies: ', acc_test_layers)
    print('instance information gain: ', MI_XiX_layers)
    print('group distance ratio: ', dis_ratio_layers)

if __name__ == "__main__":
    args = BaseOptions().initialize()
    main(args)