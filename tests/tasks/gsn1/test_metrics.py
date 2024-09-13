import torch

from src.tasks.gsn1.metrics import bcewithlogits_multilabel, chunkwise_softmax_2d_and_reshape, loss_counting

def generate_tensor(height, width, ones_per_row):
    # Initialize an empty tensor of zeros with the specified dimensions
    tensor = torch.zeros(height, width, dtype=torch.int32)

    # Loop through each row and randomly assign exactly 'ones_per_row' ones
    for i in range(height):
        # Get random indices to place ones in the row
        ones_indices = torch.randperm(width)[:ones_per_row]
        # Set the random positions to one
        tensor[i, ones_indices] = 1

    return tensor

def bcewithlogits_multilabel_unstable(batch_logits, batch_targets):
    batch_preds = torch.sigmoid(batch_logits)
    # print(batch_preds)

    loss_batch = 0
    for y_hat, y in zip(batch_preds, batch_targets):
        loss_sample = 0
        for i in range(6):
            loss_sample += -(y[i] * torch.log(y_hat[i]) + (1 - y[i]) * torch.log(1 - y_hat[i]))
        loss_batch += loss_sample

    return loss_batch


def test_bcewithlogits_multilabel():
    torch.manual_seed(0)

    # batch_size = 3
    batch_targets = generate_tensor(height=3, width=6, ones_per_row=2)
    batch_logits = torch.randn(3, 6)
    # print(batch_targets)

    loss_batch_unstable = bcewithlogits_multilabel_unstable(batch_logits, batch_targets)
    loss_batch = bcewithlogits_multilabel(batch_logits, batch_targets)

    assert abs(loss_batch - loss_batch_unstable) < 1e-6

def loss_counting_explicit(counts, preds):
    loss_explicit = 0
    for r, y in zip(counts, preds):
        loss = 0
        for i in range(6):
            for j in range(10):
                loss += y[i, j] * (j - r[i]) ** 2
        loss_explicit += loss
    return loss_explicit

def test_loss_counting():
    torch.manual_seed(0)

    counts = torch.tensor([[0, 0, 0, 0, 4, 6],
                           [3, 0, 7, 0, 0, 0],
                           [0, 0, 8, 0, 2, 0]], dtype=torch.int32)

    logits_flattened = torch.rand(3, 60)

    preds = chunkwise_softmax_2d_and_reshape(x=logits_flattened, chunk_size=10)

    loss = loss_counting(counts, preds)
    loss_explicit = loss_counting_explicit(counts, preds)

    assert abs(loss - loss_explicit) < 1e-4

if __name__ == "__main__":
    test_loss_counting()