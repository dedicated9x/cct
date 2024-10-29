# -*- coding: utf-8 -*-
"""
Custom implementation of a Graph Attention Network for reaction classification
"""

import torch
from torch import nn

from src_.featurization.utils import ORDERED_ATOM_OH_KEYS, ORDERED_BOND_OH_KEYS, ATOM_PROPS, BOND_PROPS
from src_.models.modules import MultiHeadGraphConvLayer
from src_.models.utils import construct_oh_tensors


class GAT(nn.Module):
    """
    Graph Attention Network - a classification model based on https://arxiv.org/abs/1710.10903.
    It uses a reaction graph as an input and outputs softmax classification for each graph.
    """

    def __init__(self, output_dim: int = 1, hidden_dim: int = 512, n_mols: int = 2,
                 n_conv_layers: int = 4, n_fc_layers: int = 3, residual: bool = True,
                 fc_hidden_dim: int = 256, fc_batch_norm: bool = True, embedding_bias: bool = True):
        """
        :param output_dim: output dimensionality (for binary classification: 1 or 2)
        :param hidden_dim: hidden dimensionality of GCN layers
        :param n_mols: number of molecules in an input reaction (should be 2 for substrates/products)
        :param n_conv_layers: number of GCN layers
        :param n_fc_layers: number of final fully connected layers
        :param residual: whether to use residual connections between graph layers
        :param fc_hidden_dim: hidden dimensionality of final fully connected layers
        :param fc_batch_norm: add Batch Normalization after each fully connected layer
        :param embedding_bias: use bias in atom embedding layer
        """
        super(GAT, self).__init__()
        self.output_dim = output_dim
        assert n_fc_layers >= 1

        self.n_mols = n_mols
        self.total_atom_oh_len = sum(len(ATOM_PROPS[feat_key]) + 1 for feat_key in ORDERED_ATOM_OH_KEYS) * n_mols
        self.total_bond_oh_len = sum(len(BOND_PROPS[feat_key]) + 1 for feat_key in ORDERED_BOND_OH_KEYS) * n_mols

        self.atom_embedding = nn.Linear(self.total_atom_oh_len, hidden_dim, bias=embedding_bias)

        # define GCN layers
        self.conv_layers = []
        for i in range(n_conv_layers):
            conv = MultiHeadGraphConvLayer(input_bond_dim=self.total_bond_oh_len,
                                           input_atom_dim=hidden_dim, output_dim=hidden_dim,
                                           residual=residual and i % 2 != 0)  # every second layer is residual
            self.conv_layers.append(conv)
            setattr(self, f'MultiHeadGraphConv_{i + 1}', conv)

        # define final fully-connected layers
        self.fc_layers = []
        for i in range(n_fc_layers):
            in_dim = hidden_dim if i == 0 else fc_hidden_dim
            out_dim = fc_hidden_dim if i < n_fc_layers - 1 else output_dim
            if fc_batch_norm and i < n_fc_layers - 1:
                fc = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim)
                )
            else:
                fc = nn.Linear(in_dim, out_dim)
            self.fc_layers.append(fc)
            setattr(self, f'FullyConnected_{i + 1}', fc)

    def forward(self, x: dict) -> torch.Tensor:
        """
        Forward method to be used by default by user.
        :param x: dictionary with input features
        :return: model prediction (number of dimensions depending on 'output_dim' in config)
        """
        out = self.forward_logits(x)
        return self.prediction_from_logits(out)

    def prediction_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        :param logits: predictions of model as logits (before final activation)
        :return: final model prediction
        """
        if self.output_dim == 1:
            return torch.sigmoid(logits).squeeze(-1)
        else:
            return nn.functional.softmax(logits, dim=-1)

    def forward_logits(self, x) -> torch.Tensor:
        x_atom, x_bond = x['atom'], x['bond']

        atom_feats, adj_feats = construct_oh_tensors(x_atom, x_bond, n_mols=self.n_mols)
        atom_feats = self.atom_embedding(atom_feats)

        adj_mask = torch.max(adj_feats, dim=-1)[0]
        adj_soft_mask = (-adj_mask + 1.0) * -1e9

        atom_mask = torch.max(adj_mask, dim=-1)[0].unsqueeze(-1)

        for gat_layer in self.conv_layers:
            atom_mask_exp = atom_mask.expand(atom_feats.shape)
            atom_feats = gat_layer(atom_feats, adj_feats, adj_mask, adj_soft_mask, atom_mask_exp)

        # pool nodes
        atom_mask_exp = atom_mask.expand(atom_feats.shape).bool()
        # making sure that for each reaction in the batch we only use feature nodes corresponding to this reaction
        dummy_minus_inf = torch.tensor(-float('inf'), dtype=atom_feats.dtype).to(atom_feats.device)
        atom_feats = torch.where(atom_mask_exp, atom_feats, dummy_minus_inf)
        atom_feats = atom_feats.max(dim=1)[0]

        # final fully connected layers
        for fc in self.fc_layers[:-1]:
            atom_feats = fc(atom_feats)
            atom_feats = torch.relu(atom_feats)

        atom_feats = self.fc_layers[-1](atom_feats)
        return atom_feats


if __name__ == "__main__":
    from src_.data.conditions_prediction_dataset import ConditionsPredictionToyTask
    from src_.featurization.gat_featurizer import GatGraphFeaturizer

    # TODO wrocic z tym do normalnosci
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    dataset = ConditionsPredictionToyTask()
    featurizer = GatGraphFeaturizer(n_jobs=1)
    data_x = featurizer.load(dataset.feat_dir)
    batch_dict = {col: data_x[col][0: 0 + 100] for col in data_x.keys()}
    X = featurizer.unpack(batch_dict)
    model = GAT()
    model = model.to(device)
    print("Model prediction:")
    print(model.forward(X))
