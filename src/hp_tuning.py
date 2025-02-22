# src/hp_tuning.py
import os
import optuna

import hydra
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from omegaconf import DictConfig, OmegaConf
from hydra.utils import call

from src.train import run
from src.utils.env_wrapper import HockeySB3Wrapper, create_parallel_envs
from src.utils.logger import print_system_info
from src.utils.algo_wrapper import AlgoWrapper
from src.utils.callbacks import (
    CustomEvalCallback,
)

import numpy as np

def objective(trial, config, logger):
    """Objective function for Optuna hyperparameter tuning."""
    
    train_kwargs = config["algorithm"]["params"]
    # Sample hyperparameters
    train_kwargs["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True) #trial.suggest_loguniform("learning_rate", 1e-5, 3e-3)
    train_kwargs["batch_size"] = trial.suggest_categorical("batch_size", [256, 512])
    train_kwargs["gamma"] = trial.suggest_float("gamma", 0.97, 0.995)
    train_kwargs["num_critics"] = trial.suggest_categorical("num_critics", [2, 3, 4, 5])
    train_kwargs["learning_starts"] = trial.suggest_int("learning_starts", 100, 5100, step=1000)
    # net_arch: {pi: [256, 256], qf: [256, 256]}
    net_arch = trial.suggest_categorical("net_arch", [
                (256, 256),
                (400, 300),
                (512, 512)
            ]
    )
    train_kwargs["net_arch"]["pi"] = net_arch
    train_kwargs["net_arch"]["qf"] = net_arch

    config["algorithm"]["params"] = train_kwargs
    # Use a unique logging directory per trial
    config["tensorboard_log"] = f"{logger.log_dir}/trial_{trial.number}"

    # Create environment
    num_cpus, num_gpus = print_system_info()
    opponent_type = "strong" # EVALUATING AGAINST STRONG OPP
    env = create_parallel_envs(config, n_envs=num_cpus-config['n_eval_envs'], opponent_type=opponent_type)
    eval_env = create_parallel_envs(config, n_envs=config['n_eval_envs'], opponent_type=opponent_type)

    # Create the model
    algo_wrapper = AlgoWrapper(config)  # Assuming this is defined in your project
    model = algo_wrapper.create_or_load_model(env=env)
    
    # Setup evaluation callback
    eval_freq = max(config['mode']["eval_freq"] // num_cpus-config['n_eval_envs'], 1) if config['parallelize'] else config['mode']["eval_freq"]
    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        best_model_save_path=f"{logger.log_dir}/best_model",
        n_eval_episodes=config['mode']["n_eval_episodes"], # default in SB3 is 5
        log_path=f"{logger.log_dir}/results",
        deterministic=True,
        render=False,
    )

    # Train the agent
    model.learn(
        total_timesteps=config['mode']['total_timesteps'],
        callback=[eval_callback],
        tb_log_name="",
        reset_num_timesteps=config["checkpoint"]["reset_num_timesteps"],
    )
    mean_reward = eval_callback.best_mean_reward
    return mean_reward

def read_optuna_results_from_db(study_name='sac_hockey_optimization', storage_path='hockey-rl-challenge\logs\SAC\sb3\hp_tuning\optuna_studies.db'):
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{storage_path}")
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    print(df)

    return df

def run(config, logger):
    import pprint
    pprint.pprint(config)
    hp_kwargs = config["mode"]
    print(f"hp_kwargs: {hp_kwargs}")

    sampler = TPESampler(n_startup_trials=hp_kwargs["n_startup_trials"])
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(
        n_startup_trials=hp_kwargs["n_startup_trials"],
        n_warmup_steps=hp_kwargs["n_evaluations"] // 3,
    )

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction=hp_kwargs["direction"],
        study_name=hp_kwargs["study_name"],
        storage=hp_kwargs["storage"],
        load_if_exists=True,
    )
    try:
        study.optimize(
            lambda trial: objective(trial, config, logger),
            n_trials=hp_kwargs["n_trials"],
            n_jobs=hp_kwargs["n_jobs"],
        )
    except KeyboardInterrupt:
        pass

    print(f"Number of finished trials: {len(study.trials)}")
    
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))
