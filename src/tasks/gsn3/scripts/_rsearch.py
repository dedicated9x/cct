import omegaconf
import wandb

from rsearch import sample_configs
from src.tasks.gsn3.train import train

config = {
    "hidden_dim": 128,
    "ff_dim": 256,
    "n_heads": 8,
    "n_layers": 2,
    "batch_size": 7,
    "lr": 0.001,
    "num_steps": 1000
}

distributions = {
    "hidden_dim": [64, 128, 256, 512],
    "ff_dim": [128, 256, 512],
    "n_heads": [2, 4, 8],
    "n_layers": [1, 2, 4, 6],
    "batch_size": [8, 16, 32, 64, 128],
    "lr": [0.0003, 0.001, 0.003],
    "num_steps": [1000, 2000, 4000],
}

# Convert to DictConfig
config = omegaconf.OmegaConf.create(config)

list_configs = sample_configs(config, distributions, n_iters=100)

for config_ in list_configs:
    try:
        wandb.finish()
        # Your code that might raise an exception
        train(config_)
    except Exception as e:
        # Handle any exception
        print(f"An error occurred: {e}")