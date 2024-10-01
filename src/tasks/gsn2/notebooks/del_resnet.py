import torch
from torchvision.models.resnet import ResNet, BasicBlock, load_state_dict_from_url


class MyResNet(ResNet):
    def __init__(self, *args, **kwargs):
        super(MyResNet, self).__init__(*args, **kwargs)

        state_dict = load_state_dict_from_url(
            'https://download.pytorch.org/models/resnet18-f37072fd.pth',
            progress=True
        )
        self.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        #
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

torch.manual_seed(42)
model = MyResNet(
    block=BasicBlock,
    layers=[2, 2, 2, 2],
)


model.eval()

input_tensor = torch.rand(1, 3, 128, 128)
output1 = model(input_tensor)
print(output1.shape)
print(output1.mean())
"""
torch.Size([1, 64, 32, 32])
tensor(0.4769, grad_fn=<MeanBackward0>)
"""

