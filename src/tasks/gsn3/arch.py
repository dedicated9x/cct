import matplotlib.pyplot as plt
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(Attention, self).__init__()

        QKV_DIM = 16 # TODO przenies to gdzie indziej

        self.list_Wqs = nn.ModuleList([nn.Linear(hidden_dim, QKV_DIM) for i in range(num_heads)])
        self.list_Wks = nn.ModuleList([nn.Linear(hidden_dim, QKV_DIM) for i in range(num_heads)])
        self.list_Wvs = nn.ModuleList([nn.Linear(hidden_dim, QKV_DIM) for i in range(num_heads)])


    def forward(self, x):
        # TODO: implement Attention; return both result of attention mechanism and
        # attention weights (for visualization).
        # x shape: (seqlen, batch, hiddendim)

        for Wq, Wk, Wv in zip(self.list_Wqs, self.list_Wks, self.list_Wvs):
            Q = Wq(x)
            K = Wk(x)
            V = Wv(x)

            torch.save(Q.cpu(), "/home/admin2/Documents/repos/cct/.EXCLUDED/outputs/qkv/Q.pt")
            torch.save(K.cpu(), "/home/admin2/Documents/repos/cct/.EXCLUDED/outputs/qkv/K.pt")
            torch.save(V.cpu(), "/home/admin2/Documents/repos/cct/.EXCLUDED/outputs/qkv/V.pt")

        layer = nn.Linear(128, 16).cuda()



        result, att_weights = x, None  # placeholder
        pass
        return result, att_weights


class FeedForward(nn.Module):
    def __init__(self, hidden_dim, d_ff):
        super(FeedForward, self).__init__()
        # TODO: implement FeedForward layer
        pass

    def forward(self, x):
        # TODO: implement
        # x shape: (seqlen, batch, hiddendim)
        result = x  # placeholder
        pass
        return result


class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, d_ff, num_heads, use_attention=True,
                 use_feedforward=True):
        super(EncoderLayer, self).__init__()
        self.attention = Attention(hidden_dim, num_heads)
        # TODO: implement a single encoder layer, using Attention and FeedForward.
        pass

    def forward(self, x):
        # x shape: (seqlen, batch, hiddendim)
        # TODO: implement a single encoder layer, using Attention and FeedForward.
        x = self.attention(x)

        result, att_weights = x, None  # placeholder
        pass
        return result, att_weights

# Code from https://www.tensorflow.org/tutorials/text/transformer
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def get_positional_encoding(n_positions, n_dimensions):
    angle_rads = get_angles(np.arange(n_positions)[:, np.newaxis],
                            np.arange(n_dimensions)[np.newaxis, :],
                            n_dimensions)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    positional_encoding = angle_rads

    # output shape: (seqlen, hiddendim)
    return torch.tensor(positional_encoding, dtype=torch.float)


class EncoderModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, d_ff, output_dim, n_layers,
                 num_heads, use_attention=True, use_feedforward=True,
                 use_positional=True):
        super(EncoderModel, self).__init__()

        self._use_positional = use_positional
        self.embedding_layer = nn.Embedding(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, d_ff, num_heads, use_attention,
                         use_feedforward) for i in range(n_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, return_att_weights=False):
        # x shape: (seqlen, batch)
        hidden = self.embedding_layer(x)
        # hidden shape: (seqlen, batch, hiddendim)

        if self._use_positional:
            positional_encoding = get_positional_encoding(
                n_positions=hidden.shape[0],
                n_dimensions=hidden.shape[-1]
            )

            # Move to proper accelerator
            positional_encoding = positional_encoding.to(self.device)

            # reshaping to (seqlen, 1, hiddendim)
            positional_encoding = torch.reshape(
                positional_encoding,
                (hidden.shape[0], 1, hidden.shape[-1])
            )
            hidden = hidden + positional_encoding

        list_att_weights = []
        for layer in self.layers:
            hidden, att_weights = layer(hidden)
            list_att_weights.append(att_weights)

        result = self.output_layer(hidden)

        if return_att_weights:
            return result, list_att_weights
        else:
            return result