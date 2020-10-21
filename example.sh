## type_norm: the type of normalization layer, we include ['None', 'batch', 'pair', 'group']
## type_model: the type of GNN model, we include ['GCN', 'GAT', 'simpleGCN']
## dataset: we include ['Cora', 'Citeseer', 'Pubmed', 'CoauthorCS']
## miss_rate: the missing rate of input features:
## 0. corresponds to the original dataset; 1. means removing the features in validation and testing sets

python main.py --cuda_num=0 --type_norm='group' --type_model='GAT' --dataset='Pubmed' --miss_rate=1.