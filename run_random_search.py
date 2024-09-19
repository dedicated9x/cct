import os
import hydra
import omegaconf
import pytorch_lightning as pl
from pathlib import Path

from src.common.dispatch import modulename2cls
from src.common.get_trainer import get_trainer

@hydra.main(version_base="1.2", config_path="src/tasks/gsn1/conf", config_name="00_shapes_base")
def run_random_search(config: omegaconf.DictConfig) -> None:
    # TODO najpierw to same paramsy musialyby sie znalezc w configu
    # TODO zrobic osobnego yamla dla tuningu
    # TODO wczytac go i wygenerowac configi
    # TODO usunac artefakt w postaci configu dla samgeo tuningu
    a = 2

if __name__ == '__main__':
    run_random_search()


# TODO moze to przeniesc do foleru "scripts" w glownym catalogu
