import torch
import numpy as np


def get_single_example(
        n_tokens,
        seqlen,
        max_count
):
  seq = np.random.randint(low=0, high=n_tokens, size=(seqlen,))
  label = [min(list(seq[:i]).count(seq[i]), max_count) for i, x in enumerate(seq)]
  label = np.array(label)
  return seq, label

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# TODO for the sake of same random result
seq, label = get_single_example(n_tokens=16, seqlen=64, max_count=9)
