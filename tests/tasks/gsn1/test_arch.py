import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tasks.gsn1.arch import ShapeClassificationNet

def get_forward_layers(model, input_size):
    layers = []

    def hook(module, input, output):
        # Record the module type and output shape
        layers.append((type(module), output.shape))

    hooks = []
    # Register hooks on all leaf modules
    for module in model.modules():
        if len(list(module.children())) == 0 and not isinstance(module, nn.Identity):
            hooks.append(module.register_forward_hook(hook))

    # Create a dummy input with the specified size
    dummy_input = torch.randn(input_size)

    # Run the model forward pass
    model(dummy_input)

    # Remove all hooks
    for h in hooks:
        h.remove()

    return layers

class ShapeClassificationNetOriginal(nn.Module):
    def __init__(self, out_features: int):
        super(ShapeClassificationNetOriginal, self).__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)  # Apply pooling only after the first conv layer

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()


        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()


        # Global Average Pooling will replace the fully connected layer
        # Output Layer
        self.head = nn.Linear(128, out_features)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply convolutional layers with ReLU activation
        x = self.pool(self.relu1(self.bn1(self.conv1(x))))  # Max pooling after first conv layer
        x = self.relu2(self.bn2(self.conv2(x)))             # No pooling after this layer
        x = self.relu3(self.bn3(self.conv3(x)))             # No pooling after this layer

        # Apply Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Reduce spatial dimensions to 1x1
        x = x.view(x.size(0), -1)             # Flatten to (batch_size, 128)

        # Apply dropout
        x = self.dropout(x)

        # Output layer
        x = self.head(x)

        return x

def compare_layers(layers1, layers2):
    if len(layers1) != len(layers2):
        return False

    for (type1, shape1), (type2, shape2) in zip(layers1, layers2):
        if type1 != type2 or shape1 != shape2:
            return False
    return True



def test_ShapeClassificationNet():
    model_original = ShapeClassificationNetOriginal(out_features=6)
    model_refactored = ShapeClassificationNet(
        out_features=6,
        maxpool_placing="first_conv",
        n_conv_layers=3
    )


    # Define the input size (batch_size, channels, height, width)
    input_size = (32, 1, 28, 28)  # Example for MNIST-sized images

    layers_original = get_forward_layers(model_original, input_size)
    layers_refactored = get_forward_layers(model_refactored, input_size)

    are_models_equivalent = compare_layers(layers_original, layers_refactored)

    assert are_models_equivalent

if __name__ == "__main__":
    test_ShapeClassificationNet()