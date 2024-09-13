import torch
import torch.nn as nn
import torch.nn.functional as F


# Definicja sieci klasyfikującej kształty
class ShapeClassificationNet(nn.Module):
    def __init__(self, out_features: int):
        super(ShapeClassificationNet, self).__init__()
        # Warstwa konwolucyjna 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)  # warstwa pooling

        # Warstwa konwolucyjna 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Warstwa konwolucyjna 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Warstwa w pełni połączona
        self.fc1 = nn.Linear(128 * 3 * 3, 128)

        self.head = nn.Linear(128, out_features)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(-1, 128 * 3 * 3)  # Flatten

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.head(x)

        # No sigmoid for better stability
        # x = torch.sigmoid(x)

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

# TODO