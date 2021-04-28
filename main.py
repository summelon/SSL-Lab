import hydra
from omegaconf import DictConfig, OmegaConf

import my_run


@hydra.main(config_path="hydra_configs/", config_name="base")
def my_app(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    my_run.run(cfg)

    return


if __name__ == "__main__":
    my_app()
