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
    # CustomEvalCallback,
    # ComprehensiveTrainingCallback,
    CustomTensorboardCallback,
)
from src.utils.env_wrapper import HockeySB3Wrapper, create_parallel_envs
from src.utils.logger import print_system_info
import hockey.hockey_env as h_env

def run(config, logger):
    num_cpus, num_gpus = print_system_info()
    # synchronize custom logger with SB3 logger TODO: still necessary?
    config["tensorboard_log"] = logger.log_dir

    # Load opponent mode from config
    opponent_type = config.get("opponent", "weak")  # Default: weak opponent

    algo_wrapper = AlgoWrapper(config)
    print(f"Opponent type: {opponent_type}")

    if config["parallelize"]:
        print("Parallelizing environment")
        env = create_parallel_envs(config, n_envs=num_cpus-config['n_eval_envs'], opponent_type="weak")
        eval_env = create_parallel_envs(config, n_envs=config['n_eval_envs'], opponent_type="weak")
        print("Parallelized environments created")
    else:
        env = HockeySB3Wrapper(h_env.HockeyEnv(), opponent_type)
        eval_env = Monitor(HockeySB3Wrapper(h_env.HockeyEnv(), opponent_type))

    agent = algo_wrapper.get_model(env=env)

    # synchronize custom logger with SB3 logger
    sb3_logger = configure(logger.log_dir, ["tensorboard", "stdout"])
    agent.set_logger(sb3_logger)

    training_callback = CustomTensorboardCallback(n_envs=num_cpus-config['n_eval_envs'])
    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_freq=config['mode']["eval_interval"],
        n_eval_episodes=5, # default: 5
        log_path=logger.log_dir,
        deterministic=True,
        render=False,
    )

    start_time = time.time()
    print("### Starting Training ###")
    agent.learn(
        total_timesteps=config['mode']["total_timesteps"],
        callback=[training_callback, eval_callback],
        tb_log_name="",
    )

    end_time = time.time()
    training_duration = (end_time - start_time) / 60  # Convert seconds to minutes

    # Log training time in minutes
    logger.log_scalar(
        "time/Training Duration (minutes)",
        training_duration,
    )

    print(f"🕒 Training completed in {training_duration:.2f} minutes")

    model_path = f"{logger.log_dir}/final_model.zip"
    agent.save(model_path)
    print(f"Model saved at: {model_path}")

    logger.close()
