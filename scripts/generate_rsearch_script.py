from typing import Callable

import torch
import hydra
import omegaconf
from pathlib import Path
from  hydra.utils import instantiate


def _sample_from_distribution(distribution):
    # Extract keys and values
    elements = list(distribution.keys())
    probs = torch.tensor(list(distribution.values()))

    # Sample one element based on the probabilities
    sampled_idx = torch.multinomial(probs, 1).item()

    # Return the corresponding element
    return elements[sampled_idx]

def _sample_params_variation(distributions):
    list_to_uniform = lambda x: {elem: 1 / len(x) for elem in x}

    # Convert all values to distribution format
    distributions = {
        k: v if isinstance(v, dict) else list_to_uniform(v)
        for k, v in distributions.items()
    }

    variation = {
        k: _sample_from_distribution(v)
        for k, v in distributions.items()
    }

    return variation

def sample_params_variation(distributions, filter_):
    # TODO ten filtr musi leciec z zewnatrz
    is_valid = False
    while not is_valid:
        variation = _sample_params_variation(distributions)
        is_valid = filter_(variation)
    return variation

def sample_script_line(distributions, filter_):
    dict_variation = sample_params_variation(distributions, filter_)
    str_variation = " ".join([f"{k}={v}" for k, v in dict_variation.items()])
    script_line = f"python run_experiment.py {str_variation}"
    return script_line


# @hydra.main(version_base="1.2", config_path="../src/tasks/gsn1/conf", config_name="03_shapes_rsearch")
@hydra.main(version_base="1.2", config_path="../src/tasks/gsn1/conf", config_name="04_counts_encoded_rsearch")
def generate_random_search_script(config: omegaconf.DictConfig) -> None:
    distributions = config.random_search.distributions
    distributions = omegaconf.OmegaConf.to_container(distributions, resolve=True)

    list_script_lines = []
    for i in range(config.random_search.n_iters):
        script_line = sample_script_line(
            distributions,
            filter_=instantiate(config.random_search.filter)
        )
        list_script_lines.append(script_line)
    script_text = ";\n".join(list_script_lines)
    script_text = f"#!/bin/bash\n\n{script_text}"
    with open(str(Path(__file__).parent / "run_rsearch.sh"), "w") as outfile:
        outfile.write(script_text)

    print(script_text)


if __name__ == '__main__':
    generate_random_search_script()

"""
with omegaconf.open_dict(config):
    del config['random_search']
"""