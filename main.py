from pathlib import Path

import omegaconf
import pytorch_lightning as pl

from src.trainer import get_trainer
from src.module import ShapesModule


def build_shapes_config() -> omegaconf.DictConfig:
    root = Path(__file__).resolve().parent

    config_dict = {
        "num_workers": 4,
        "main": {
            # Keep disabled by default because current torch/checkpoint behavior
            # may fail on loading "best" checkpoint in test stage.
            "is_tested": False,
        },
        "trainer": {
            "device": [0],
            "max_epochs": 20,
            "monitored_metric": {
                "name": "Val/Acc",
                "mode": "max",
            },
            "wandb": False,
            "limit_train_batches": 1.0,
            "limit_val_batches": 1.0,
            "limit_test_batches": 1.0,
            "batch_size": 16,
            "ckpt_path": None,
            "tag": None,
        },
        "optimizer": {
            "lr": 0.00001,
        },
        "paths": {
            "root": str(root),
            "data": str(root / "data"),
        },
        "dataset": {
            "augmentations": {
                "prob_rotation": 1.0,
                "prob_mirroring": 1.0,
            },
            "split": {
                "seed": 42,
                "train_size": 0.8,
                "val_size": 0.1,
                "test_size": 0.1,
            },
            "visualization_mode": False,
        },
        "model": {
            "n_conv_layers": 3,
            "n_channels_first_conv_layer": 32,
            "n_channels_last_conv_layer": 128,
            "maxpool_placing": "first_conv",
            "pooling_method": "adaptive_avg",
            "n_fc_layers": 1,
            "fc_hidden_dim": 128,
        },
    }

    return omegaconf.OmegaConf.create(config_dict)


def run_shapes_experiment() -> None:
    config = build_shapes_config()
    print(omegaconf.OmegaConf.to_yaml(config))

    pl.seed_everything(1234)

    module = ShapesModule(config)
    trainer = get_trainer(config=config)
    trainer.fit(
        model=module,
        ckpt_path=config.trainer.ckpt_path,
    )
    if config.main.is_tested:
        trainer.test(
            model=module,
            ckpt_path=config.trainer.ckpt_path or "best",
        )


if __name__ == "__main__":
    run_shapes_experiment()
