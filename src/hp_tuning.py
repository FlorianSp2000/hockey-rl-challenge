# src/hp_tuning.py
import optuna
from src.train import run
from src.utils.logger import Logger
from src.utils.config_loader import load_config

def objective(trial, config, logger):
    # Update config with trial parameters
    config["batch_size"] = trial.suggest_categorical("batch_size", config["hp_tuning"]["batch_size"])
    config["learning_rate"] = trial.suggest_loguniform("learning_rate", config["hp_tuning"]["learning_rate_min"], config["hp_tuning"]["learning_rate_max"])

    final_reward = run(config, logger)
    return final_reward

def run(config, logger):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, config, logger), n_trials=config["hp_tuning"]["n_trials"])
