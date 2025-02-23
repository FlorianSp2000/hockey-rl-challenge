# src/main.py
from src.utils.logger import Logger
from src import train, test, selfplay, hp_tuning

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import src.utils.hydra_utils  # This imports and registers the custom resolver

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Merge Hydra config with argparse arguments
    log_dir_datetimed = HydraConfig.get().run.dir

    mode = cfg.mode.name
    config = OmegaConf.to_container(cfg, resolve=True)
    print(f"Config Complete: \n{config}")

    if mode == "train":
        if len(cfg.seed) > 1:
            print(f"Starting Ablation study over {len(cfg.seed)} seeds")
            for seed in cfg.seed:
                config["seed"] = seed
                log_dir_datetimed = HydraConfig.get().run.dir + "_s" + str(seed)
                # if checkpoint is present overwrite logging directory with checkpoint directory
                logger = create_logger(cfg, log_dir_datetimed)
                train.run(config, logger)
        else:
            logger = create_logger(cfg, log_dir_datetimed)
            config["seed"] = cfg.seed[0]
            train.run(config, logger)
            
    elif mode == "test":
        logger = Logger(log_dir_datetimed, cfg) # We don't want to write the selfplay results into the potential checkpoint directory
        test.run(config, logger)

    elif mode == "selfplay":
        print("start selfplay")
        logger = Logger(log_dir_datetimed, cfg) # We don't want to write the selfplay results into the potential checkpoint directory
        config["seed"] = cfg.seed[0]
        selfplay.run(config, logger)

    elif mode == "hp_tuning":
        logger = create_logger(cfg, log_dir_datetimed)
        config["seed"] = cfg.seed[0]
        hp_tuning.run(config, logger)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def create_logger(cfg, log_dir):
    if cfg.checkpoint.load_from is not None:
        return Logger(cfg.checkpoint.load_from, cfg)
    return Logger(log_dir, cfg)


if __name__ == "__main__":
    main()

    # python main.py mode=train n_eval_envs=1 parallelize=False
