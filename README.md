# Towards Deeper Graph Neural Networks with Differentiable Group Normalization

This is an authors' implementation of "Towards Deeper Graph Neural Networks with Differentiable Group Normalization" in Pytorch.

Authors: Kaixiong Zhou, Xiao Huang, Yuening Li, Daochen Zha, Rui Chen and Xia Hu

Paper: https://arxiv.org/abs/2006.06972

Accepted by NeurIPS 2020.

## Introduction

This work presents two metrics to quantify the over-smoothing in GNNs: (1) Group distance ratio and (2) Instance information gain.

Based on these two metrics, we provide the differentiable group normalization (DGN), a general module applied between the 
graph convolutional layers, to relieve the over-smoothing issue. DGN softly clusters nodes and normalizes each group independently, 
which prevents the distinct groups from having the close node representations.

Detailed information about the metrics and DGN is provided in (https://arxiv.org/abs/2006.06972).

## Requirements

python == 3.6

torch == 1.3.1

torch-geometric==1.3.2

## Train over GCN, GAT or SGCN backbone networks

To train GNN model, measure the group distance ratio as well as instance information gain, run:
```
python main.py --cuda_num=0 --type_norm='group' --type_model='GCN' --dataset='Cora' --miss_rate=0.
```
Hyperparameter explanations:

--type_norm: the type of normalization layer. We include ['None', 'batch', 'pair', 'group'] for none normalization, 
batch normalization, pair normalization and differentiable group normalization, respectively. 

--type_model: the type of GNN model. We include ['GCN', 'GAT', 'simpleGCN']

--dataset: we include ['Cora', 'Citeseer', 'Pubmed', 'CoauthorCS']

--miss_rate: the missing rate of input features.
The value of 0. corresponds to the original dataset. The value of 1. means removing the features in validation and testing sets


## Train over GCNII backbone networks

We directly use the provided implementation of GCNII. After the model configuration, run:
```
python GCNII_Cora.py --type_norm='group'
```


## Citation

If using this code, please cite our paper.
```
@inproceedings{zhou2020towards,
  title={Towards Deeper Graph Neural Networks with Differentiable Group Normalization},
  author={Zhou, Kaixiong and Huang, Xiao and Li, Yuening and Zha, Daochen and Chen, Rui and Hu, Xia},
  booktitle={Advances in neural information processing systems},
  year={2020}
}
```