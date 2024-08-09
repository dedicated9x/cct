import numpy as np
import torch
import pandas as pd

def pprint_sample(_dict):
    # Set Pandas display options for better readability
    pd.set_option('display.expand_frame_repr', False)  # Don't wrap to the next line

    data = {
        "Key": [],
        "Type": [],
        "Dtype": [],
        "Shape": [],
        "Value": [],
        "Min": [],
        "Max": [],
        "Mean": []
    }

    for k, v in _dict.items():
        dtype = type(v)
        shape = ""
        value = ""
        min_val, max_val, mean_val = None, None, None

        try:
            dtype = v.dtype
        except AttributeError:
            pass

        try:
            shape = v.shape
        except AttributeError:
            pass

        try:
            if v.numel() < 10:
                value = v
        except AttributeError:
            pass

        try:
            if v.size < 10:
                value = v
        except (AttributeError, TypeError):
            pass

        try:
            if v.shape == () or v.shape == (1,):
                value = v
        except AttributeError:
            pass

        try:
            if hasattr(v, 'min') and hasattr(v, 'max') and hasattr(v, 'mean'):
                min_val = v.min().item()
                max_val = v.max().item()
                mean_val = v.mean().item()
        except AttributeError:
            pass

        data["Key"].append(k)
        data["Type"].append(str(type(v)))
        data["Dtype"].append(str(dtype))
        data["Shape"].append(str(shape))
        data["Value"].append(str(value))
        data["Min"].append(min_val)
        data["Max"].append(max_val)
        data["Mean"].append(mean_val)

    df = pd.DataFrame(data)
    print(df)

    # Set Pandas display options back to default
    pd.set_option('display.expand_frame_repr', True)

if __name__ == '__main__':
    # Example usage with numpy and torch
    sample_dict = {
        'array_small': np.array([1, 2, 3]),
        'array_large': np.random.randn(100),
        'tensor_small': torch.tensor([1.0, 2.0, 3.0]),
        'tensor_large': torch.randn(100),
        'scalar': torch.tensor(5.0),
        'list': [1, 2, 3, 4],
        'string': 'example'
    }

    pprint_sample(sample_dict)
