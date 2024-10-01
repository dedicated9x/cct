import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
from torchvision.models.resnet import ResNet, BasicBlock, load_state_dict_from_url
from src.tasks.gsn2.dataset import ImagesDataset

class MyNet32(ResNet):
    def __init__(self, n_layers: int):
        super(MyNet32, self).__init__(block=BasicBlock, layers=[2, 2, 2, 2])
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
        self.n_filters = 64 * self.scale_factor

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


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    ds = ImagesDataset(split="train")
    _x = ds[0].get_torch_tensor()

    for n_layers in [1, 2, 3, 4]:
        model = MyNet32(n_layers=n_layers)
        model.eval()
        output1 = model(_x)
        print(output1.shape, model.n_filters)
        # print(output1.mean())

