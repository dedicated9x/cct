import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf/app", config_name="base", version_base="1.2")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()
