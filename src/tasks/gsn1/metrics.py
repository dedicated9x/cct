import torch
import torch.nn as nn


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

torch.manual_seed(0)

# Parameters
height = 3
width = 6
ones_per_row = 2

# Generate the tensor
batch_targets = generate_tensor(height, width, ones_per_row)
print(batch_targets)

batch_logits = torch.randn(3, 6)
print(batch_logits)

batch_preds = torch.sigmoid(batch_logits)
print(batch_preds)

""" way 1"""
loss_batch = 0
for y_hat, y in zip(batch_preds, batch_targets):
    loss_sample = 0
    for i in range(6):
        loss_sample += -(y[i] * torch.log(y_hat[i]) + (1-y[i]) * torch.log(1 - y_hat[i]))
    loss_batch += loss_sample

# TODO narazie jszcze nie widzimy, co wlasciwie bedzie wchodzic funkcji
# sc - single class
loss_batch_v2 = 0
for batch_logits_sc, batch_targets_sc in zip(batch_logits.T, batch_targets.T):
    batch_logits_sc = batch_logits_sc.unsqueeze(0)
    batch_targets_sc = batch_targets_sc.unsqueeze(0).float()
    bce_sc = nn.BCEWithLogitsLoss(reduction="none")(batch_logits_sc, batch_targets_sc)
    loss_batch_v2 += bce_sc.sum()

a = 2




