import os
import hydra
import omegaconf
import pytorch_lightning as pl
from pathlib import Path

from src.common.dispatch import modulename2cls
from src.common.get_trainer import get_trainer


# @hydra.main(version_base="1.2", config_path="src/tasks/flowers/conf", config_name="base")
# @hydra.main(version_base="1.2", config_path="src/tasks/gsn1/conf", config_name="00_shapes_base")
@hydra.main(version_base="1.2", config_path="src/tasks/gsn1/conf", config_name="01_counts_base")
def main(config: omegaconf.DictConfig) -> None:
    print(omegaconf.OmegaConf.to_yaml(config))

    # config.trainer.wandb = True
    config.trainer.max_epochs = 3

    pl.seed_everything(1234)
    # TODO mozna to zrobic jakims hydra.instantiate
    module_cls = modulename2cls(name=config.module.name)
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

