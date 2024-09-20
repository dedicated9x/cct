import hydra
import omegaconf
import importlib

from run_experiment import run_experiment

def sample_config(
        base_config: omegaconf.DictConfig,
        distributions: omegaconf.DictConfig
):
    a = 2
    pass

@hydra.main(version_base="1.2", config_path="src/tasks/gsn1/conf", config_name="03_shapes_rsearch")
def run_random_search(config: omegaconf.DictConfig) -> None:
    # TODO wczytac go i wygenerowac configi
    # TODO usunac artefakt w postaci configu dla samego tuningu


    for i in range(config.random_search.n_iters):
        distributions = config.random_search.distributions
        with omegaconf.open_dict(config):
            del config['random_search']

        config = sample_config(config, distributions)

        run_experiment(config)

if __name__ == '__main__':
    run_random_search()


# TODO moze to przeniesc do foleru "scripts" w glownym catalogu
"""
func_module = "src.tasks.gsn1.scripts.hyperparameter_tuning"
module = importlib.import_module(func_module)
func = getattr(module, cfg.func_name)
result = func(variation)
"""