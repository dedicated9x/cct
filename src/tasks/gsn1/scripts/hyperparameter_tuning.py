import torch
from src.tasks.gsn1.arch import ShapeClassificationNet


def _sample_from_distribution(distribution):
    # Extract keys and values
    elements = list(distribution.keys())
    probs = torch.tensor(list(distribution.values()))

    # Sample one element based on the probabilities
    sampled_idx = torch.multinomial(probs, 1).item()

    # Return the corresponding element
    return elements[sampled_idx]

def _sample_params_variation():
    distributions = {
        "n_conv_layers": [3, 5, 7, 9],
        "n_channels_first_conv_layer": [16, 32, 64],
        "n_channels_last_conv_layer": [64, 128, 256, 512],
        "maxpool_placing": {
            "none": 0.2,
            "first_conv": 0.4,
            "even_convs": 0.4
        },
        "pooling_method": {
            "adaptive_avg": 0.66,
            "fc": 0.34
        },
        "n_fc_layers": [1, 2, 3],
        "fc_hidden_dim": [128, 512]
    }
    list_to_uniform = lambda x: {elem: 1 / len(x) for elem in x}

    # convert all values to distribution format
    distributions = {
        k: v if isinstance(v, dict) else list_to_uniform(v)
        for k, v in distributions.items()
    }
    variation = {
        k: _sample_from_distribution(v)
        for k, v in distributions.items()
    }

    return variation

def _is_valid_variation(variation):
    _v = variation
    if _v["n_conv_layers"] >= 9 and _v['maxpool_placing'] == 'even_convs':
        return False
    else:
        return True

def sample_params_variation():
    is_valid = False
    while not is_valid:
        variation = _sample_params_variation()
        is_valid = _is_valid_variation(variation)
    return variation

if __name__ == '__main__':
    for i in range(1000):
        variation = sample_params_variation()
        print(i)

        model = ShapeClassificationNet(
            **variation,
            out_features=6,
            input_shape=[1, 28, 28],
            )

        input_size = (32, 1, 28, 28)
        input_tensor = torch.randn(input_size)

        # Forward pass through the model
        output = model(input_tensor)

        # Assert the output shape is as expected
        assert output.shape == torch.Size([32, 6]), f"Unexpected output shape: {output.shape}"
