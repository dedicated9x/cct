import torch
import torch.nn as nn


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

def bcewithlogits_multilabel(batch_logits, batch_targets):
    """
    Args:
        batch_logits (torch.Tensor): The raw model outputs (logits) of shape (batch_size, num_labels).
        batch_targets (torch.Tensor): The target tensor of shape (batch_size, num_labels) containing binary labels.
    """
    # sl - single label
    loss_batch = 0
    for batch_logits_sl, batch_targets_sl in zip(batch_logits.T, batch_targets.T):
        batch_logits_sl = batch_logits_sl.unsqueeze(0)
        batch_targets_sl = batch_targets_sl.unsqueeze(0).float()
        bce_sl = nn.BCEWithLogitsLoss(reduction="none")(batch_logits_sl, batch_targets_sl)
        loss_batch += bce_sl.sum()

    return loss_batch

def convert_topk_to_binary(tensor, k):
    """
    Convert the top k values in each row of a tensor to 1, and the rest to 0.

    Parameters:
    tensor (torch.Tensor): Input tensor.
    k (int): Number of top values to convert to 1 in each row.

    Returns:
    torch.Tensor: Binary tensor with top k values as 1 and the rest as 0.
    """
    # Create a zero tensor with the same shape as the input tensor
    binary_tensor = torch.zeros_like(tensor)

    # Get the indices of the top k largest values in each row
    topk_vals, topk_indices = torch.topk(tensor, k, dim=1)

    # Set the top k values to 1 in the binary tensor
    binary_tensor.scatter_(1, topk_indices, 1)

    return binary_tensor

if __name__ == '__main__':
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

    # batch_size = 3
    batch_targets = generate_tensor(height=3, width=6, ones_per_row=2)
    batch_logits = torch.randn(3, 6)
    print(batch_targets)

    loss_batch = bcewithlogits_multilabel_unstable(batch_logits, batch_targets)
    loss_batch_v2 = bcewithlogits_multilabel(batch_logits, batch_targets)

    print(loss_batch)
    print(loss_batch_v2)




