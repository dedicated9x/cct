import torch

import torch.nn.functional as F

def chunkwise_softmax_2d_and_reshape(x, chunk_size: int):
    assert abs(x.shape[1] % chunk_size) < 1e-6
    assert x.dim() == 2
    batch_size = x.shape[0]
    logits = logits_flattened.reshape(batch_size, int(x.shape[1] / chunk_size), chunk_size)
    preds = F.softmax(logits, dim=2)
    return preds

def loss_counting_explicit(counts, preds):
    loss_explicit = 0
    for r, y in zip(counts, preds):
        loss = 0
        for i in range(6):
            for j in range(10):
                loss += y[i, j] * (j - r[i]) ** 2
        loss_explicit += loss
    return loss_explicit

def loss_counting(counts, preds):
    repeated_counts = counts.unsqueeze(2).repeat(1, 1, 10)
    j_indices = torch.arange(10).unsqueeze(0).repeat(6, 1)
    loss = (preds * ((j_indices - repeated_counts) ** 2)).sum()
    return loss

torch.manual_seed(0)

counts = torch.tensor([[0, 0, 0, 0, 4, 6],
                       [3, 0, 7, 0, 0, 0],
                       [0, 0, 8, 0, 2, 0]], dtype=torch.int32)

logits_flattened = torch.rand(3, 60)

preds = chunkwise_softmax_2d_and_reshape(logits_flattened, chunk_size=10)

loss = loss_counting(counts, preds)
loss_explicit = loss_counting_explicit(counts, preds)


