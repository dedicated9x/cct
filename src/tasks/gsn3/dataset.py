import numpy as np

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device type', DEVICE)

N_TOKENS = 16
SEQ_LEN = 64

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def get_single_example(n_tokens=None, seqlen=None, max_count=None):
  if n_tokens is None:
    n_tokens = N_TOKENS
  if seqlen is None:
    seqlen = SEQ_LEN
  seq = np.random.randint(low=0, high=n_tokens, size=(seqlen,))
  label = [min(list(seq[:i]).count(seq[i]), max_count) for i, x in enumerate(seq)]
  label = np.array(label)
  return seq, label

# TODO for the sake of same random result
seq, label = get_single_example(max_count=9)
