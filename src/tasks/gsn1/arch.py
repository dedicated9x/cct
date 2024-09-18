from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _create_conv_block_scheme_from_params(
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

def _create_fc_block_scheme_from_params(
    n_fc_layers: int,
    in_features_first_fc_layer: int,
    fc_hidden_dim: int,
    out_features: int
):
    in_features = [in_features_first_fc_layer] + [fc_hidden_dim] * (n_fc_layers - 1)
    out_features = [fc_hidden_dim] * (n_fc_layers - 1) + [out_features]
    if n_fc_layers > 1:
        add_dropouts = [False] + [True] * (n_fc_layers - 1)
    else:
        add_dropouts = [True]
    add_relus = [True] * (n_fc_layers - 1) + [False]

    layers_scheme = list(zip(in_features, out_features, add_dropouts, add_relus))
    return layers_scheme


# Modified ShapeClassificationNet
class ShapeClassificationNet(nn.Module):
    def __init__(
            self,
            out_features: int,
            input_shape: List[int],
            n_conv_layers: int,
            n_channels_first_conv_layer: int,
            n_channels_last_conv_layer: int,
            maxpool_placing: str,
            pooling_method: str,
            n_fc_layers: int,
            fc_hidden_dim: int
    ):
        assert len(input_shape) == 3
        assert n_conv_layers >= 2
        assert maxpool_placing in ["first_conv", "all_conv", None]
        assert pooling_method in ["adaptive_avg", "fc"]

        super(ShapeClassificationNet, self).__init__()

        # Create conv block
        conv_block_scheme = _create_conv_block_scheme_from_params(
            n_conv_layers, n_channels_first_conv_layer,
            n_channels_last_conv_layer, maxpool_placing
        )
        self.conv_block = nn.ModuleList()
        for i, (in_channels, out_channels, add_maxpool) in enumerate(conv_block_scheme):
            self.conv_block.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2) if add_maxpool else nn.Identity()
                )
            )

        # Create "pooling" fc layer (neck).
        if pooling_method == "fc":
            _x = torch.randn([1] + input_shape)
            for conv_layer in self.conv_block:
                _x = conv_layer(_x)
            in_features_first_fc_layer = _x.shape[1:]
        else:
            in_features_first_fc_layer = self.conv_block[-1][0].out_channels

        fc_block_scheme = _create_fc_block_scheme_from_params(
            n_fc_layers, in_features_first_fc_layer,
            fc_hidden_dim, out_features
        )

        self.fc_block = nn.ModuleList()
        for i, (in_features, out_features, add_dropout, add_relu) in enumerate(fc_block_scheme):
            self.fc_block.append(
                nn.Sequential(
                    nn.Dropout(0.5) if add_dropout else nn.Identity(),
                    nn.Linear(in_features, out_features),
                    nn.ReLU() if add_relu else nn.Identity(),
                )
            )

        # Store params required during `forward` step.
        self.pooling_method = pooling_method

    def forward(self, x):
        # Apply convolutional layers
        for conv_layer in self.conv_block:
            x = conv_layer(x)

        # Apply Global Average Pooling or first fully connected layer
        if self.pooling_method == "adaptive_avg":
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
        else:
            raise NotImplementedError

        # Apply fully connected layers
        for fc_layer in self.fc_block:
            x = fc_layer(x)

        return x

# Funkcja main
def main():
    # Tworzymy model
    model = ShapeClassificationNet(
        out_features=6,
        input_shape=[1, 28, 28],
        n_conv_layers=3,
        n_channels_first_conv_layer = 32,
        n_channels_last_conv_layer = 128,
        maxpool_placing = "first_conv",
        pooling_method="adaptive_avg",
        n_fc_layers=1,
        fc_hidden_dim=None
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
