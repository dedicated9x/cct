import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers
from pathlib import Path
import os

def get_logging_dir():
    repo_name = Path(__file__).parents[2].name
    logdir = f'/tmp/wandb_pl_logs/{repo_name}'
    os.makedirs(logdir, exist_ok=True)
    return logdir

def get_trainer(config):

    # Checkpoints
    if config.trainer.monitored_metric is not None:
        metric_name = config.trainer.monitored_metric.name
        callbacks = [pl.callbacks.ModelCheckpoint(
            monitor=metric_name,
            filename='{epoch}-' + f'{{{metric_name}:.2f}}',
            mode=config.trainer.monitored_metric.mode,
            auto_insert_metric_name=("/" not in metric_name)
        )]
    else:
        callbacks = None

    # Configure loggers
    if config.trainer.wandb:
        module_name = config.module._target_.split(".")[-1]
        logger = pytorch_lightning.loggers.WandbLogger(
            project=module_name,
            name=module_name.lower(),
            save_dir=get_logging_dir(),
        )

        if config.trainer.tag is not None:
            logger.experiment.tags = logger.experiment.tags + (config.trainer.tag,)
    else:
        logger = pytorch_lightning.loggers.TensorBoardLogger(
            save_dir=get_logging_dir()
        )

    # GPUs
    if torch.cuda.is_available():
        gpus = config.trainer.device
    else:
        gpus = None

    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=config.trainer.max_epochs,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
        limit_train_batches=config.trainer.limit_train_batches,
    )

    return trainer