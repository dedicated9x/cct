import pytest
import torch
from src.tasks.gsn1.arch import ShapeClassificationNet

# Common keyword arguments for the model
common_kwargs = dict(
    out_features=6,
    input_shape=[1, 28, 28],
    n_conv_layers=3,
    n_channels_first_conv_layer=32,
    n_channels_last_conv_layer=128,
    maxpool_placing="first_conv",
    fc_hidden_dim=512
)

# Parameterized test to run over different pooling methods and number of fc layers
@pytest.mark.parametrize("pooling_method, n_fc_layers", [
    ("adaptive_avg", 1),
    ("fc", 1),
    ("adaptive_avg", 2),
    ("fc", 2)
])
def test_shape_classification_net(pooling_method, n_fc_layers):
    # Initialize the model with the provided parameters
    model = ShapeClassificationNet(
        **common_kwargs,
        pooling_method=pooling_method,
        n_fc_layers=n_fc_layers,
    )

    # Generate a random input tensor
    input_size = (32, 1, 28, 28)
    input_tensor = torch.randn(input_size)

    # Forward pass through the model
    output = model(input_tensor)

    # Assert the output shape is as expected
    assert output.shape == torch.Size([32, 6]), f"Unexpected output shape: {output.shape}"

if __name__ == "__main__":
    test_shape_classification_net("fc", 1)