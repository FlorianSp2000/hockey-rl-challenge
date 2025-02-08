# src/main.py
from src.utils.logger import Logger
from src import train, test, selfplay, hp_tuning
from src.utils.config_loader import load_config

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Merge Hydra config with argparse arguments
    log_dir_datetimed = HydraConfig.get().run.dir
    logger = Logger(log_dir_datetimed, cfg)

    mode = cfg.mode.name
    config = OmegaConf.to_container(cfg, resolve=True)
    print(f"Config: {config}")

    if mode == "train":
        train.run(config, logger)
    elif mode == "test":
        test.run(config, logger)
    elif mode == "selfplay":
        selfplay.run(config, logger)
    elif mode == "hp_tuning":
        hp_tuning.run(config, logger)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()

    # python main.py mode=train n_eval_envs=1 parallelize=False
