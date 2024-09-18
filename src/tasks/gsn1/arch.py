import torch
import torch.nn as nn
import torch.nn.functional as F


def _create_scheme_from_params(
        n_conv_layers: int,
        n_channels_first_layer: int,
        n_channels_last_layer: int,
        maxpool_placing: str,
):
    out_channels = [min(n_channels_first_layer * 2 ** i, n_channels_last_layer) for i in range(n_conv_layers)]
    in_channels = [1] + out_channels[:-1]
    if maxpool_placing == "all_conv":
        add_maxpools = [True] * n_conv_layers
    elif maxpool_placing == "first_conv":
        add_maxpools = [True] + [False] * (n_conv_layers-1)
    else:
        add_maxpools = [False] * n_conv_layers

    layers_scheme = list(zip(in_channels, out_channels, add_maxpools))
    return layers_scheme


# Modified ShapeClassificationNet
class ShapeClassificationNet(nn.Module):
    def __init__(
            self,
            out_features: int,
            n_conv_layers: int,
            n_channels_first_layer: int,
            n_channels_last_layer: int,
            maxpool_placing: str,
    ):
        assert n_conv_layers >= 2
        assert maxpool_placing in ["first_conv", "all_conv", None]

        super(ShapeClassificationNet, self).__init__()

        layers_scheme = _create_scheme_from_params(
            n_conv_layers, n_channels_first_layer,
            n_channels_last_layer, maxpool_placing
        )


        self.conv_block = nn.ModuleList()

        for i, (in_channels, out_channels, add_maxpool) in enumerate(layers_scheme):
            self.conv_block.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2) if add_maxpool else nn.Identity()
                )
            )

        # Global Average Pooling will replace the fully connected layer
        # Output Layer
        self.head = nn.Linear(128, out_features)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply convolutional layers
        for conv_layer in self.conv_block:
            x = conv_layer(x)

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
    model = ShapeClassificationNet(
        out_features=6,
        n_conv_layers=3,
        n_channels_first_layer = 32,
        n_channels_last_layer = 128,
        maxpool_placing = "first_conv"
    )

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
