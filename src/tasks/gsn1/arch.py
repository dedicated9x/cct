import torch
import torch.nn as nn
import torch.nn.functional as F




# Modified ShapeClassificationNet
class ShapeClassificationNet(nn.Module):
    def __init__(
            self,
            out_features: int,
            maxpool_placing: str = None
        ):
        assert maxpool_placing in ["first_conv", "all_conv", None]
        self.maxpool_placing = maxpool_placing

        super(ShapeClassificationNet, self).__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        if maxpool_placing in ["first_conv", "all_conv"]:
            self.pool = nn.MaxPool2d(2, 2)
        else:
            self.pool = nn.Identity()

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        if maxpool_placing == "all_conv":
            self.pool2 = nn.MaxPool2d(2, 2)
        else:
            self.pool2 = nn.Identity()

        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        if maxpool_placing == "all_conv":
            self.pool3 = nn.MaxPool2d(2, 2)
        else:
            self.pool3 = nn.Identity()

        # Global Average Pooling will replace the fully connected layer
        # Output Layer
        self.head = nn.Linear(128, out_features)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply convolutional layers with ReLU activation
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))             # No pooling after this layer
        x = F.relu(self.bn3(self.conv3(x)))             # No pooling after this layer

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
