import hydra
from  hydra.utils import instantiate
import omegaconf
import pytorch_lightning as pl

from src.common.get_trainer import get_trainer

"""
/tmp/wandb_pl_logs/cct/lightning_logs/version_2/checkpoints/18-0.90.ckpt
flowers/conf/base -> 0.26, 0.50, 0.80
gsn1/conf/01_counts_base -> 0.26, 0.41, 0.52
"""

# @hydra.main(version_base="1.2", config_path="src/tasks/flowers/conf", config_name="base")
# @hydra.main(version_base="1.2", config_path="src/tasks/gsn1/conf", config_name="00_shapes_base")
# @hydra.main(version_base="1.2", config_path="src/tasks/gsn1/conf", config_name="01_counts_base")
# @hydra.main(version_base="1.2", config_path="src/tasks/gsn1/conf", config_name="02_counts_encoded_base")
# @hydra.main(version_base="1.2", config_path="src/tasks/gsn2/conf", config_name="00_base")
@hydra.main(version_base="1.2", config_path="src/tasks/molecule/conf", config_name="00_base")
def main(config: omegaconf.DictConfig) -> None:
    run_experiment(config)

def run_experiment(config: omegaconf.DictConfig) -> None:
    print(omegaconf.OmegaConf.to_yaml(config))

    # config.trainer.wandb = True

    # config.trainer.ckpt_path = "/tmp/wandb_pl_logs/cct/default/version_3/checkpoints/14-0.97.ckpt"
    # config.trainer.max_epochs = 40
    config.trainer.max_epochs = 10
    # config.trainer.batch_size = 6


    # config.trainer.limit_train_batches = 2
    # config.trainer.limit_val_batches = 2
    # config.trainer.limit_test_batches = 2

    pl.seed_everything(1234)

    module = instantiate(config.module, config)
    trainer = get_trainer(config=config)
    trainer.fit(
        model=module,
        ckpt_path=config.trainer.ckpt_path
    )
    if config.main.is_tested:
        trainer.test(
            model=module,
            ckpt_path=config.trainer.ckpt_path or "best"
        )

if __name__ == '__main__':
    main()

