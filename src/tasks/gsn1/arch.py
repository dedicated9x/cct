from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# Modified ShapeClassificationNet
class ShapeClassificationNet(nn.Module):
    def __init__(
            self,
            out_features: int,
            n_conv_layers: int,
            maxpool_placing: str = None,
        ):
        assert maxpool_placing in ["first_conv", "all_conv", None]

        super(ShapeClassificationNet, self).__init__()

        # TODO conv
        # list_conv_layers = []
        # for i, n_channels in zip(range(n_conv_layers), [32, 64, 128]):


        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) if maxpool_placing in ["first_conv", "all_conv"] else nn.Identity()
        )

        self.fc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) if maxpool_placing == "all_conv" else nn.Identity()
        )

        self.fc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) if maxpool_placing == "all_conv" else nn.Identity()
        )

        # Global Average Pooling will replace the fully connected layer
        # Output Layer
        self.head = nn.Linear(128, out_features)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply convolutional layers with ReLU activation
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        # Apply Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Reduce spatial dimensions to 1x1
        x = x.view(x.size(0), -1)             # Flatten to (batch_size, 128)

        # Apply dropout
        x = self.dropout(x)

        # Output layer
        x = self.head(x)

        return x

# Funkcja main
def main():
    # Tworzymy model
    model = ShapeClassificationNet(out_features=6)

    # Przykładowy losowy batch (batch_size=1, kanał=1, wysokość=28, szerokość=28)
    random_input = torch.randn(32, 1, 28, 28)  # szum
    print("Random input shape:", random_input.shape)

    # Przepuszczenie batcha przez sieć
    output = model(random_input)

    # Wyświetlenie wyników
    print("Output shape:", output.shape)
    print("Output:", output[13])


if __name__ == "__main__":
    main()
