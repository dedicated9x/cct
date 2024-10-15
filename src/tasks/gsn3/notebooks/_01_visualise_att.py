from time import time


import torch
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')

from matplotlib.collections import LineCollection
import torch
import omegaconf
import torch.nn as nn
import torch.optim as optim
import wandb
import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from src.tasks.gsn3.dataset import get_single_example
from src.tasks.gsn3.arch import EncoderModel
from src.tasks.gsn3.train import _train_model, get_test_dataset


def plot_attention(att_weights, test_X, token_idx, head_idx):
    # Zakładam, że masz już zdefiniowane att_weights i test_X
    # Jeśli nie, proszę je zdefiniować zgodnie z Twoimi danymi

    # Wybierz indeks tokena, który chcesz zwizualizować (0 <= idx < 64)
    idx = token_idx  # Możesz zmienić na dowolny indeks
    head = head_idx  # Wybierz głowę attention (0 <= head < 8)

    # Pobierz wagi attention dla wybranego tokena i głowy
    weights = att_weights[head][idx].detach().cpu().numpy()  # Rozmiar [64]

    # Normalizacja wag dla celów wizualizacji (opcjonalne)
    weights = weights / weights.max()

    # Przygotuj dane do wizualizacji
    tokens = test_X.cpu().numpy()  # Konwertuj tensory na numpy
    token_labels = [str(t) for t in tokens]

    # Ustawienia pozycji dla lewej i prawej kolumny
    y_pos = np.arange(len(tokens))
    x_left = np.full(len(tokens), 0)
    x_right = np.full(len(tokens), 1)

    # Tworzenie figur i osi
    fig, ax = plt.subplots(figsize=(10, 20))

    # TODO usun te linie
    # Create an array of values from 0 to 2*pi
    # x = np.linspace(0, 2 * np.pi, 100)
    # y = np.sin(x)
    # ax.plot(x, y)
    #
    # plt.show()
    # return

    # Rysowanie tokenów po lewej i prawej stronie
    for i, token in enumerate(token_labels):
        ax.text(x_left[i] - 0.05, y_pos[i], token, ha='right', va='center')
        ax.text(x_right[i] + 0.05, y_pos[i], token, ha='left', va='center')

    # Przygotowanie linii reprezentujących wagi attention
    lines = []
    colors = []
    for j in range(len(tokens)):
        # Linia od tokena idx po lewej do tokena j po prawej
        lines.append([(x_left[idx], y_pos[idx]), (x_right[j], y_pos[j])])
        # Kolor zależny od wagi
        colors.append(plt.cm.viridis(weights[j]))

    # Tworzenie kolekcji linii
    lc = LineCollection(lines, colors=colors, linewidths=2)

    # Dodanie linii do wykresu
    ax.add_collection(lc)

    # Ukrycie osi
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-1, len(tokens))
    ax.axis('off')

    # Dodanie kolorbarwy do interpretacji wag
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=weights.min(), vmax=weights.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.set_label('Waga attention')

    plt.title(f'Attention dla tokena na pozycji {idx} w głowie {head}')
    plt.tight_layout()
    plt.show()


def _analyze2(test_X, test_Y, model):
    _, att_weights = model.forward(test_X, return_att_weights=True)

    sample_idx = 13

    test_X = test_X[:, sample_idx]
    test_Y = test_Y[:, sample_idx]

    att_weights = att_weights[0]
    att_weights = [att_weights[head_idx][sample_idx] for head_idx in range(len(att_weights))]


    plot_attention(att_weights, test_X, token_idx=1, head_idx=0)
    plot_attention(att_weights, test_X, token_idx=1, head_idx=1)
    plot_attention(att_weights, test_X, token_idx=1, head_idx=2)
    plot_attention(att_weights, test_X, token_idx=1, head_idx=3)
    plot_attention(att_weights, test_X, token_idx=1, head_idx=4)
    plot_attention(att_weights, test_X, token_idx=1, head_idx=5)
    plot_attention(att_weights, test_X, token_idx=1, head_idx=6)
    plot_attention(att_weights, test_X, token_idx=1, head_idx=7)
    return

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

    _analyze2(test_X, test_Y, model)


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