from typing import Dict, List, Union

import hydra
import wandb
import omegaconf

from scripts.generate_rsearch_script import sample_params_variation
from run_experiment import run_experiment

def sample_configs(
    base_config: omegaconf.DictConfig,
    distributions: Dict[str, Union[Dict, List]],
    n_iters: int
):
    list_configs = []
    # TODO add filter
    dummy_filter = lambda x: True
    for _ in range(n_iters):
        dict_variation = sample_params_variation(distributions, dummy_filter)
        next_config = base_config.copy()
        for k, v in dict_variation.items():
            omegaconf.OmegaConf.update(next_config, k, v)
        list_configs.append(next_config)
    return list_configs

# @hydra.main(version_base="1.2", config_path="src/tasks/gsn1/conf", config_name="00_shapes_base")
@hydra.main(version_base="1.2", config_path="src/tasks/gsn1/conf", config_name="03_shapes_rsearch_v2")
def main(config: omegaconf.DictConfig) -> None:
    # TODO remove
    config.trainer.max_epochs = 5

    config_random_search = config.random_search
    del config.random_search
    distributions = omegaconf.OmegaConf.to_container(config_random_search.distributions, resolve=True)

    list_configs = sample_configs(
        base_config=config,
        distributions=distributions,
        n_iters=config_random_search.n_iters
    )

    for config in list_configs:
        try:
            wandb.finish()
            # Your code that might raise an exception
            run_experiment(config)
        except Exception as e:
            # Handle any exception
            print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()