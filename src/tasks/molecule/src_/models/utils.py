# -*- coding: utf-8 -*-
"""
Utility functions to use in pytorch models
"""
import torch
from torch.autograd import Variable

from src_.featurization.utils import ORDERED_ATOM_OH_KEYS, ATOM_PROP2OH, ORDERED_BOND_OH_KEYS, BOND_PROP2OH


def to_one_hot(y: torch.Tensor, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    if type(y_tensor) != torch.LongTensor:
        y_tensor = y_tensor.type(torch.LongTensor).to(y.device)
    y_tensor = y_tensor.view(-1, 1)

    n_dims = n_dims + 1 if n_dims is not None else int(torch.max(y_tensor)) + 2
    if y.dim() < 2 and y.sum() == 0:
        y_one_hot = torch.zeros((1, n_dims - 1), device=y.device)
    else:
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims, device=y.device).scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot[:, 1:]
        y_one_hot = y_one_hot.view(*y.shape, -1)

    return y_one_hot


def construct_oh_tensors(x_atom: torch.Tensor, x_bond: torch.Tensor, n_mols: int):
    """
    Constructs merged one-hot tensors that represents reaction atoms and bonds.
    """
    oh_atom_feats = []
    i = 0
    # we concatenate atom features from each molecule in the reaction
    for _ in range(n_mols):
        for key in ORDERED_ATOM_OH_KEYS:
            oh_feat = to_one_hot(x_atom[:, :, i], n_dims=len(ATOM_PROP2OH[key]) + 1)
            oh_atom_feats.append(oh_feat)
            i += 1
    atom_feats = torch.cat(oh_atom_feats, dim=-1)

    oh_bond_feats = []
    i = 0
    # we concatenate bond features from each molecule in the reaction
    for _ in range(n_mols):
        for key in ORDERED_BOND_OH_KEYS:
            oh_feat = to_one_hot(x_bond[:, :, :, i], n_dims=len(BOND_PROP2OH[key]) + 1)
            oh_bond_feats.append(oh_feat)
            i += 1
    adj_feats = torch.cat(oh_bond_feats, dim=-1)

    return atom_feats, adj_feats
