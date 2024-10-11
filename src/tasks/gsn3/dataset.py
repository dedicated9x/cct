import matplotlib.pyplot as plt
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device type', DEVICE)

N_TOKENS = 16
SEQ_LEN = 64

MAX_COUNT = 9
OUTPUT_DIM = MAX_COUNT + 1

def get_single_example(n_tokens=None, seqlen=None):
  if n_tokens is None:
    n_tokens = N_TOKENS
  if seqlen is None:
    seqlen = SEQ_LEN
  seq = np.random.randint(low=0, high=n_tokens, size=(seqlen,))
  label = [min(list(seq[:i]).count(seq[i]), MAX_COUNT) for i, x in enumerate(seq)]
  label = np.array(label)
  return seq, label

seq, label = get_single_example()
print('Sequence:', seq)
print('Labels:', label)
print('Sequence and labels interleaved:\n', np.stack((seq, label)).transpose())