import torch
import omegaconf
import pandas as pd

from src.tasks.gsn3.arch import EncoderModel
from src.tasks.gsn3.train import get_test_dataset



def print_predictions(test_X, test_Y, model):
    predicted_logits = model.forward(test_X)
    predicted_logits_ = predicted_logits.argmax(dim=-1)

    idx_sample = 13

    # Convert tensors to lists
    test_X_list = test_X[:, idx_sample].cpu().detach().numpy().tolist()
    test_Y_list = test_Y[:, idx_sample].cpu().detach().numpy().tolist()
    predicted_logits_list = predicted_logits_[:, idx_sample].cpu().detach().numpy().tolist()

    # Create a DataFrame
    df = pd.DataFrame({
        'test_X': test_X_list,
        'test_Y': test_Y_list,
        'predicted_logits': predicted_logits_list
    })

    # To display all rows in the DataFrame
    pd.set_option('display.max_rows', None)

    print(df)

def train(config: omegaconf.DictConfig):
    n_tokens = 16
    max_count = 9
    seq_len = 64


    model = EncoderModel(
        n_tokens, config.hidden_dim, config.ff_dim,
        config.n_layers, config.n_heads, output_dim=(max_count + 1)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_X, test_Y = get_test_dataset(n_tokens, max_count, seq_len, device)
    model.to(device)

    # Load the saved state_dict
    if config.ckpt_path is not None:
        model.load_state_dict(torch.load(config.ckpt_path))

    print_predictions(test_X, test_Y, model)


if __name__ == '__main__':
    config = {
        "hidden_dim": 512,
        "ff_dim": 512,
        "n_heads": 8,
        "n_layers": 1,
        "batch_size": 64,
        "lr": 0.0001,
        "num_steps": 2000,
        "ckpt_path": "/tmp/20241015_083926.ckpt"
        # "ckpt_path": None
    }

    # Convert to DictConfig
    config = omegaconf.OmegaConf.create(config)

    train(config)
