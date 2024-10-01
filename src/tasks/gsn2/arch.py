from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import ResNet, BasicBlock, load_state_dict_from_url
from src.tasks.gsn2.dataset import ImagesDataset

# Modified ShapeClassificationNet
class Head(nn.Module):
    def __init__(
            self,
            n_layers: int,
            filters: int,
            out_channels: int,
            n_anchor_sizes: int
    ):
        super(Head, self).__init__()

        if n_layers >= 2:
            self.conv_block = nn.ModuleList()
            for i in range(n_layers - 1):
                self.conv_block.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1),
                        nn.ReLU(),
                    )
                )
        else:
            pass

        self.last_conv = nn.Conv2d(
            in_channels=filters,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

        self.n_layers = n_layers
        self.n_anchor_sizes = n_anchor_sizes
        self.last_size_last_layer = int(out_channels / n_anchor_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_layers >= 2:
            for layer in self.conv_block:
                x = layer(x)
        else:
            pass
        x = self.last_conv(x)

        x = x.permute(0, 2, 3, 1)
        x = x.unflatten(-1, (self.n_anchor_sizes, self.last_size_last_layer))
        x = x.flatten(start_dim=1, end_dim=3)
        return x



class MyNet32(ResNet):
    def __init__(
            self,
            n_layers_backbone: int,
            n_layers_clf_head: int,
            n_layers_reg_head: int,
            anchor_sizes: List[Tuple[int, int]]
    ):
        super(MyNet32, self).__init__(block=BasicBlock, layers=[2, 2, 2, 2])
        assert n_layers_backbone in [1, 2, 3, 4]

        state_dict = load_state_dict_from_url(
            'https://download.pytorch.org/models/resnet18-f37072fd.pth',
            progress=True
        )
        self.load_state_dict(state_dict)

        self.n_layers = n_layers_backbone
        self.scale_factor = 2 ** (n_layers_backbone - 1)
        # TODO opcja z mode='nearest'
        self.scale_mode = "bilinear"
        self.filters = 64 * self.scale_factor

        n_anchor_sizes = len(anchor_sizes)
        n_classes = 10
        self.classification_head = Head(
            n_layers=n_layers_clf_head,
            filters=self.filters,
            out_channels=n_anchor_sizes * n_classes,
            n_anchor_sizes=n_anchor_sizes
        )
        self.box_regression_head = Head(
            n_layers=n_layers_reg_head,
            filters=self.filters,
            out_channels=n_anchor_sizes * 4,
            n_anchor_sizes=n_anchor_sizes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(1, 3, 1, 1)
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.scale_mode, align_corners=False)
        else:
            pass

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) # 1
        if self.n_layers >= 2:
            x = self.layer2(x) # 2
        if self.n_layers >= 3:
            x = self.layer3(x) # 2
        if self.n_layers >= 4:
            x = self.layer4(x) # 2

        classification_output = self.classification_head(x)
        box_regression_output = self.box_regression_head(x)
        return {
            "classification_output": classification_output,
            "box_regression_output": box_regression_output
        }


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    anchor_sizes = [
        (19, 19),
        (19, 15),
        (19, 13),
        (19, 11),
        (19, 5),
    ]

    ds = ImagesDataset(split="train")
    _x = ds[0].get_torch_tensor()

    # for n_layers in [1, 2, 3, 4]:
    for n_layers_ in [2]:
        model = MyNet32(
            n_layers_backbone=n_layers_,
            n_layers_clf_head=2,
            n_layers_reg_head=2,
            anchor_sizes=anchor_sizes
        )
        model.eval()
        output1 = model(_x)
        print(output1["classification_output"].shape)
        print(output1["box_regression_output"].shape)

# TODO test na backbone (ponizej)

"""
1 torch.Size([1, 64, 32, 32]) 64
2 torch.Size([1, 128, 32, 32]) 128
3 torch.Size([1, 256, 32, 32]) 256
4 torch.Size([1, 512, 32, 32]) 512
"""