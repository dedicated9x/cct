import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
import torch
import omegaconf
import seaborn as sns

from src.tasks.gsn3.arch import EncoderModel
from src.tasks.gsn3.train import get_test_dataset



def _plot_attention(att_weights, test_X, token_idx):
    att_single_token_all_heads = [att_weights[head_idx_][token_idx, :] for head_idx_ in range(len(att_weights))]
    att_single_token_all_heads = torch.stack(att_single_token_all_heads)

    assert abs(att_single_token_all_heads[0, :].sum().item() - 1) < 1e-5

    # Zakładam, że att_single_token_all_heads i test_X są już zdefiniowane

    # Konwertuj tensor atencji na numpy i transponuj go, aby uzyskać 64 wiersze i 8 kolumn
    attention_data = att_single_token_all_heads.cpu().detach().numpy().T  # Kształt [64, 8]

    # Konwertuj test_X na numpy
    test_X_labels = test_X.cpu().numpy()  # Kształt [64]

    # Tworzenie heatmapy
    plt.figure(figsize=(10, 12))
    ax = sns.heatmap(attention_data, annot=False, cmap='viridis', yticklabels=test_X_labels)

    # Ustawienia osi
    plt.xlabel('Głowy atencji')
    plt.ylabel('Tokeny (test_X)')
    plt.title('Heatmapa wizualizująca atencję')

    # Upewnij się, że etykiety osi Y są w poprawnej kolejności
    plt.yticks(rotation=0)  # Opcjonalnie, aby etykiety były poziome

    # Wyświetl wykres
    plt.show()


def plot_attention(test_X, test_Y, model):
    _, att_weights = model.forward(test_X, return_att_weights=True)

    sample_idx = 13

    test_X = test_X[:, sample_idx]
    test_Y = test_Y[:, sample_idx]

    att_weights = att_weights[0]
    att_weights = [att_weights[head_idx][sample_idx] for head_idx in range(len(att_weights))]

    _plot_attention(att_weights, test_X, token_idx=55)


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

    plot_attention(test_X, test_Y, model)


if __name__ == '__main__':
    config = {
        "hidden_dim": 512,
        "ff_dim": 512,
        "n_heads": 8,
        "n_layers": 1,
        "batch_size": 64,
        "lr": 0.0001,
        "num_steps": 2000,
        # "num_steps": 200,
        "ckpt_path": "/tmp/20241015_083926.ckpt"
        # "ckpt_path": None
    }

    # Convert to DictConfig
    config = omegaconf.OmegaConf.create(config)

    train(config)