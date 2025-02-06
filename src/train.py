# src/train.py
import time
import os
import torch
import multiprocessing

import gymnasium as gym
# from src.utils.logger import Logger
from src.utils.algo_wrapper import AlgoWrapper
from src.utils.callbacks import CustomEvalCallback, ComprehensiveTrainingCallback, CustomTensorboardCallback
from src.utils.env_wrapper import HockeySB3Wrapper
from stable_baselines3.common.logger import configure
# import sb3 evalcallback
from stable_baselines3.common.callbacks import EvalCallback

def print_system_info():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    
    print(f"Available CPU cores: {num_cpus}")
    print(f"Available GPUs: {num_gpus}")
    
    if num_gpus > 0:
        for i in range(num_gpus):
            print(f"🔹 GPU {i}: {torch.cuda.get_device_name(i)}")


def run(config, logger):
    print_system_info()
    # synchronize custom logger with SB3 logger
    config['tensorboard_log'] = logger.log_dir

    # Load opponent mode from config
    opponent_type = config.get("opponent", "weak")  # Default: weak opponent
    
    algo_wrapper = AlgoWrapper(config)
    print(f"Opponent type: {opponent_type}")    

    env = HockeySB3Wrapper(opponent_type)
    eval_env = HockeySB3Wrapper(opponent_type)

    agent = algo_wrapper.get_model(env=env)

    # synchronize custom logger with SB3 logger
    sb3_logger = configure(logger.log_dir, ["tensorboard", "stdout"])
    agent.set_logger(sb3_logger)

    # train_callback = ComprehensiveTrainingCallback(custom_logger=logger)
    # eval_callback = CustomEvalCallback(
    #     eval_env=eval_env,
    #     custom_logger=logger,
    #     eval_freq=config["eval_interval"],
    #     best_model_save_path=logger.log_dir,
    #     log_path=logger.log_dir,
    #     deterministic=True,
    #     render=False,
    # )
    training_callback = CustomTensorboardCallback()
    eval_callback = EvalCallback(eval_env=eval_env, eval_freq=config["eval_interval"], log_path=logger.log_dir, deterministic=True, render=False)

    start_time = time.time()
    print("### Starting Training ###")
    agent.learn(total_timesteps=config["total_timesteps"], callback=[training_callback, eval_callback], tb_log_name="")

    end_time = time.time()
    training_duration = (end_time - start_time) / 60  # Convert seconds to minutes

    # Log training time in minutes
    # TODO: log not showing up in tensorboard
    logger.log_scalar("Training/Duration (minutes)", training_duration, step=0) # config["total_timesteps"]
    # logger.writer.flush()
    print(f"🕒 Training completed in {training_duration:.2f} minutes")

    model_path = f"{logger.log_dir}/final_model.zip"
    agent.save(model_path)
    print(f"Model saved at: {model_path}")

    logger.close()




