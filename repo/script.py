import hydra
from omegaconf import DictConfig, OmegaConf
import os

@hydra.main(version_base="1.2", config_path="some_dir/conf", config_name="load_this_conf")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()
