import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger, WandbLogger

def get_logging_dir():
    repo_name = Path(__file__).parents[2].name
    logdir = f"/tmp/pl_logs/{repo_name}"
    os.makedirs(logdir, exist_ok=True)
    return logdir


def get_outputs_dir(config):
    outputs_dir = Path(config.paths.data) / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir


def _setup_mlflow_env() -> None:
    # /tmp/.env is expected in the target runtime (same pattern as clothes-classificator).
    load_dotenv("/tmp/.env", override=False)

    # TODO przeniesc do zmiennych lokalnych
    os.environ.setdefault("SSO_CLIENT_ID", "mlflow-caise-platform")
    os.environ.setdefault("SSO_URL","https://sso.task.gda.pl/auth/realms/citask/protocol/openid-connect/token")
    os.environ.setdefault("MLFLOW_TRACKING_URI", "https://mlflow.caise.apl.task.gda.pl/pl0158-01/")
    os.environ.setdefault("MLFLOW_TRACKING_AUTH", "mlflow_oauth2_client.MlFlowAuthProvider")

    required_vars = (
        "SSO_CLIENT_ID",
        "SSO_URL",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_AUTH",
        "MLFLOW_SSO_USER",
        "MLFLOW_SSO_PASSWORD",
    )
    missing = [name for name in required_vars if not os.getenv(name)]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            f"Missing MLflow SSO environment variables: {missing_list}. "
            "Set them in the shell and/or /tmp/.env."
        )

def get_trainer(config):
    outputs_dir = get_outputs_dir(config)
    checkpoint_base_dir = outputs_dir

    # Configure loggers
    logger_name = str(getattr(config.trainer, "logger", "tensorboard")).lower()
    if logger_name == "mlflow":
        _setup_mlflow_env()
        logger = MLFlowLogger(
            experiment_name=str(config.trainer.experiment_name),
            run_name=str(config.trainer.run_name),
            tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
            tags={"tag": str(config.trainer.tag)},
        )
        checkpoint_base_dir = outputs_dir / str(logger.run_id)
    elif logger_name == "wandb":
        logger = WandbLogger(
            project=str(config.trainer.experiment_name),
            name=str(config.trainer.run_name),
            save_dir=get_logging_dir(),
        )
    else:
        logger = TensorBoardLogger(save_dir=get_logging_dir())

    # Checkpoints
    if config.trainer.monitored_metric is not None:
        metric_name = config.trainer.monitored_metric.name
        callbacks = [pl.callbacks.ModelCheckpoint(
            monitor=metric_name,
            dirpath=str(checkpoint_base_dir / "checkpoints"),
            filename='{epoch}-' + f'{{{metric_name}:.2f}}',
            mode=config.trainer.monitored_metric.mode,
            auto_insert_metric_name=("/" not in metric_name)
        )]
    else:
        callbacks = None

    # Devices (PyTorch Lightning 2.x API)
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = config.trainer.device
    else:
        accelerator = "cpu"
        devices = 1

    trainer = pl.Trainer(
        default_root_dir=str(outputs_dir),
        accelerator=accelerator,
        devices=devices,
        max_epochs=config.trainer.max_epochs,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        limit_test_batches=config.trainer.limit_test_batches,
    )

    return trainer