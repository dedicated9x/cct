import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


logits = torch.Tensor([[-0.3, 0.2]])
targets_dense = torch.Tensor([[1, 0]])


logits_1 = logits
preds_1 = torch.sigmoid(logits)
preds_0 = 1 - preds_1
preds = torch.stack((preds_0, preds_1), dim=1)
nlogpreds = -torch.log(preds)
targets_sparse = torch.stack((1 - targets_dense, targets_dense), dim=1)
print(f"{logits=}")
print(f"{logits_1=}")
print(f"{preds_1=}")
print(f"{preds_0=}")
print(f"{preds=}")
print(f"{nlogpreds=}")
print(f"{targets_sparse=}")

bce_v1 = nn.BCEWithLogitsLoss(reduction="none")(logits, targets_dense)
bce_v2 = nn.BCELoss(reduction="none")(torch.sigmoid(logits), targets_dense)

print(f"{bce_v1=}")
print(f"{bce_v2=}")

