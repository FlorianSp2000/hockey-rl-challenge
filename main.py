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

    mode = cfg.mode.name
    config = OmegaConf.to_container(cfg, resolve=True)
    print(f"Config: {config}")

    if mode == "train":
        if len(cfg.seed) > 1:
            print(f"Starting Ablation study over {len(cfg.seed)} seeds")
            for seed in cfg.seed:
                config["seed"] = seed
                log_dir_datetimed = HydraConfig.get().run.dir + "_s" + str(seed)
                logger = Logger(log_dir_datetimed, cfg)
                train.run(config, logger)
        else:
            logger = Logger(log_dir_datetimed, cfg)
            config["seed"] = cfg.seed[0]
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
