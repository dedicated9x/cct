import torch
import torch.nn as nn
import torch.nn.functional as F


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

def chunkwise_softmax_2d_and_reshape(x, chunk_size: int):
    assert abs(x.shape[1] % chunk_size) < 1e-6
    assert x.dim() == 2
    batch_size = x.shape[0]
    logits = x.reshape(batch_size, int(x.shape[1] / chunk_size), chunk_size)
    preds = F.softmax(logits, dim=2)
    return preds

# TODO zapytac preview skad pomysl na taki glupi loss
def loss_counting(counts, preds, device: str = "cpu"):
    repeated_counts = counts.unsqueeze(2).repeat(1, 1, 10)
    j_indices = torch.arange(10).unsqueeze(0).repeat(6, 1).to(device)
    loss = (preds * ((j_indices - repeated_counts) ** 2)).sum()
    return loss

if __name__ == '__main__':
    pass




