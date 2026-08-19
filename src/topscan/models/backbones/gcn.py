import torch.nn as nn
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(
            self,
            in_feats,
            n_hidden,
            n_classes,
            n_layers,
            activation,
            dropout,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            self.convs.append(
                dglnn.GraphConv(in_hidden, out_hidden, "both")
            )

            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_hidden))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, graph, feat):
        h = feat

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)
            h = conv

            if i < self.n_layers - 1:
                h = self.norms[i](h)
                h = self.activation(h)
                h = self.dropout(h)

        return h
