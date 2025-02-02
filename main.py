# src/main.py
import argparse
import yaml
from src.utils.logger import Logger
from src import train, test, selfplay, hp_tuning
from src.utils.config_loader import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    # parser.add_argument("--load_model", type=str, help="Path to trained model (optional)")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = Logger(config)

    mode = config["mode"]
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

    # python main.py --config configs/SAC/train.yaml
