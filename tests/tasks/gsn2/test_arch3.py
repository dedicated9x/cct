import torch
import numpy as np
import pytest

from src.tasks.gsn2.dataset import ImagesDataset
from src.tasks.gsn2.arch import Backbone

@pytest.mark.parametrize("n_layers, expected_shape", [
    (1, torch.Size([1, 64, 32, 32])),
    (2, torch.Size([1, 128, 32, 32])),
    (3, torch.Size([1, 256, 32, 32])),
    (4, torch.Size([1, 512, 32, 32])),
])
def test_backbone(n_layers, expected_shape):
    torch.manual_seed(42)
    np.random.seed(42)

    ds = ImagesDataset(split="train", size=1000)
    _x = ds[0].get_torch_tensor()

    model = Backbone(n_layers=n_layers)
    model.eval()
    output = model(_x)

    assert output.shape == expected_shape, f"Expected {expected_shape}, but got {output.shape}"
