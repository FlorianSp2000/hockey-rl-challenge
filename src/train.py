# src/train.py
import time


import gymnasium as gym
from stable_baselines3.common.logger import configure

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import src # trigger custom env registration
from src.utils.algo_wrapper import AlgoWrapper
from src.utils.callbacks import (
    CustomEvalCallback,
    CustomTensorboardCallback,
    WinRateCheckpointCallback
)
from src.utils.env_wrapper import HockeySB3Wrapper, create_parallel_envs
from src.utils.logger import print_system_info
from src.utils.model_io import get_enumerated_path
import hockey.hockey_env as h_env

def run(config, logger):
    num_cpus, num_gpus = print_system_info()
    # synchronize custom logger with SB3 logger
    config["tensorboard_log"] = logger.log_dir
    
    # Load opponent mode from config
    opponent_type = config.get("opponent", "weak")  # Default: weak opponent

    algo_wrapper = AlgoWrapper(config)
    print(f"Opponent type: {opponent_type}")

    if config["parallelize"]:
        print("Parallelizing environment")
        env = create_parallel_envs(config, n_envs=num_cpus-config['n_eval_envs'], opponent_type=opponent_type)
        eval_env = create_parallel_envs(config, n_envs=config['n_eval_envs'], opponent_type=opponent_type)
        print("Parallelized environments created")
    else:
        env = HockeySB3Wrapper(h_env.HockeyEnv(), opponent_type)
        eval_env = Monitor(HockeySB3Wrapper(h_env.HockeyEnv(), opponent_type))

    agent = algo_wrapper.create_or_load_model(env=env)

    # synchronize custom logger with SB3 logger
    sb3_logger = configure(logger.log_dir, ["tensorboard", "stdout"])
    agent.set_logger(sb3_logger)

    training_callback = CustomTensorboardCallback(n_envs=num_cpus-config['n_eval_envs'])
    # account for vectorized environments
    eval_freq = max(config['mode']["eval_freq"] // num_cpus-config['n_eval_envs'], 1) if config['parallelize'] else config['mode']["eval_freq"]
    print(f"eval_freq is: {eval_freq}")
    
    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=config['mode']["n_eval_episodes"], # default in SB3 is 5
        log_path=logger.log_dir,
        deterministic=True,
        render=False,
    )
    winrate_callback = WinRateCheckpointCallback(
        save_path=f"{logger.log_dir}/checkpoints",
        win_rate_threshold=config['win_rate_threshold_w_to_s'],
        verbose=1
    )

    start_time = time.time()
    print("### Starting Training ###")
    agent.learn(
        total_timesteps=config["mode"]["total_timesteps"],
        callback=[training_callback, eval_callback, winrate_callback],
        tb_log_name="",
        reset_num_timesteps=config["checkpoint"]["reset_num_timesteps"],
    )

    end_time = time.time()
    training_duration = (end_time - start_time) / 60  # Convert seconds to minutes

    # Log training time in minutes
    logger.log_scalar(
        "time/Training Duration (minutes)",
        training_duration,
    )

    print(f"🕒 Training completed in {training_duration:.2f} minutes")
    logger.writer.flush()

    # Close the environment
    env.close()
    eval_env.close()
    del env, eval_env
    
    model_path = get_enumerated_path(f"{logger.log_dir}/final_model.zip")
    agent.save(model_path)
    print(f"Model saved at: {model_path}")

    logger.close()
