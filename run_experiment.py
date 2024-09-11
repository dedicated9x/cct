import os
import hydra
import omegaconf
import pytorch_lightning as pl
from pathlib import Path

from src.common.dispatch import modulename2cls
from src.common.get_trainer import get_trainer

def prepare_wandb_logger():
    repo_name = Path(__file__).parent.name
    wandb_logdir = f'/tmp/wandb_logs/{repo_name}'
    os.makedirs(wandb_logdir, exist_ok=True)
    os.environ['WANDB_DIR'] = wandb_logdir

# @hydra.main(version_base="1.2", config_path="src/tasks/flowers/conf", config_name="base")
@hydra.main(version_base="1.2", config_path="src/tasks/gsn1/conf", config_name="base")
def main(config: omegaconf.DictConfig) -> None:
    print(omegaconf.OmegaConf.to_yaml(config))

    prepare_wandb_logger()
    config.trainer.wandb = True

    pl.seed_everything(1234)
    # TODO mozna to zrobic jakims hydra.instantiate
    module_cls = modulename2cls(name=config.main.module_name)
    model = module_cls(config=config)
    trainer = get_trainer(config=config)
    trainer.fit(model)
    if config.main.is_tested:
        trainer.test(
            model=model,
            ckpt_path="best"
        )

if __name__ == '__main__':
    main()

