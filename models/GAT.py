from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from models.common_blocks import batch_norm

class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()

        self.dataset = args.dataset
        self.num_layers = args.num_layers
        self.num_feats = args.num_feats
        self.num_classes = args.num_classes
        self.dim_hidden = args.dim_hidden
        self.dropout = args.dropout
        self.layers_GCN = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])
        self.type_norm = args.type_norm
        self.skip_weight = args.skip_weight
        self.num_groups = args.num_groups

        # build up the convolutional layers
        if self.num_layers == 1:
            self.layers_GCN.append(GATConv(self.num_feats, self.num_classes, heads=1, concat=True, dropout=self.dropout,
                                           bias=False))
        elif self.num_layers == 2:
            self.layers_GCN.append(GATConv(self.num_feats, self.dim_hidden, heads=1, concat=True, dropout=self.dropout,
                                           bias=False))
            self.layers_GCN.append(GATConv(self.dim_hidden, self.num_classes, heads=1, concat=True, dropout=self.dropout,
                                           bias=False))
        else:
            self.layers_GCN.append(GATConv(self.num_feats, self.dim_hidden,heads=1, concat=True, dropout=self.dropout,
                                           bias=False))
            for _ in range(self.num_layers - 2):
                self.layers_GCN.append(GATConv(self.dim_hidden, self.dim_hidden, heads=1, concat=True, dropout=self.dropout,
                                           bias=False))
            self.layers_GCN.append(GATConv(self.dim_hidden, self.num_classes, heads=1, concat=True, dropout=self.dropout,
                                           bias=False))

        for i in range(self.num_layers):
            dim_out = self.layers_GCN[i].out_channels
            if self.type_norm in ['None', 'batch', 'pair']:
                skip_connect = False
            else:
                skip_connect = True
            self.layers_bn.append(batch_norm(dim_out, self.type_norm, skip_connect, self.num_groups, self.skip_weight))


    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            if i == 0 or i == self.num_layers-1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers_GCN[i](x, edge_index)
            x = self.layers_bn[i](x)
            x = F.relu(x)

        return x