from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from  pathlib import Path
from torchvision.models.resnet import ResNet, BasicBlock, load_state_dict_from_url
from src.tasks.gsn2.dataset import ImagesDataset
from src.tasks.gsn2.structures import MnistBox
from src.tasks.gsn2.anchor_set import AnchorSet

class DigitDetectionModelOutput:

    def __init__(
        self,
        anchors: List[MnistBox],
        classification_output: torch.Tensor,
        box_regression_output: torch.Tensor,
    ):
        self.anchors = anchors
        self.classification_output = classification_output
        self.box_regression_output = box_regression_output

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


class Backbone(ResNet):
    def __init__(
            self,
            n_layers: int,
    ):
        super(Backbone, self).__init__(block=BasicBlock, layers=[2, 2, 2, 2])
        assert n_layers in [1, 2, 3, 4]

        state_dict = load_state_dict_from_url(
            'https://download.pytorch.org/models/resnet18-f37072fd.pth',
            progress=True
        )
        self.load_state_dict(state_dict)

        self.n_layers = n_layers
        self.scale_factor = 2 ** (n_layers - 1)
        # TODO opcja z mode='nearest'
        self.scale_mode = "bilinear"


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
        return x


class MyNet32(nn.Module):
    def __init__(
            self,
            n_layers_backbone: int,
            n_layers_clf_head: int,
            n_layers_reg_head: int,
            anchor_sizes: List[Tuple[int, int]],
            anchors: List[MnistBox]
    ):
        super(MyNet32, self).__init__()

        self.backbone = Backbone(n_layers_backbone)

        filters = 64 * self.backbone.scale_factor

        n_anchor_sizes = len(anchor_sizes)
        n_classes = 10
        self.classification_head = Head(
            n_layers=n_layers_clf_head,
            filters=filters,
            out_channels=n_anchor_sizes * n_classes,
            n_anchor_sizes=n_anchor_sizes
        )
        self.box_regression_head = Head(
            n_layers=n_layers_reg_head,
            filters=filters,
            out_channels=n_anchor_sizes * 4,
            n_anchor_sizes=n_anchor_sizes
        )

        self.anchors = anchors

    def forward(self, x: torch.Tensor) -> DigitDetectionModelOutput:
        x = self.backbone(x)
        classification_output = self.classification_head(x)
        box_regression_output = self.box_regression_head(x)

        output = DigitDetectionModelOutput(
            anchors=self.anchors,
            classification_output=classification_output,
            box_regression_output=box_regression_output
        )

        return output

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

    ds = ImagesDataset(split="train", size=1000)
    anchor_set = AnchorSet(anchor_sizes, k_grid=2)

    _x = ds[0].get_torch_tensor()

    for n_layers_ in [2]:
        model = MyNet32(
            n_layers_backbone=n_layers_,
            n_layers_clf_head=2,
            n_layers_reg_head=2,
            anchor_sizes=anchor_sizes,
            anchors=anchor_set.list_mnistboxes
        )
        model.eval()
        output1 = model(_x)

        # Check if same as before
        path_file = Path(__file__).parents[3] / ".EXCLUDED/outputs/tensor.pt"
        # torch.save(output1["classification_output"], path_file)
        assert (torch.load(path_file) == output1.classification_output).all().item()

