# src/train.py
import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback
from src.utils.logger import Logger
from src.utils.algo_wrapper import AlgoWrapper
from utils.callbacks import CustomEvalCallback

def run(config):
    base_log_dir = config["log_dir"]
    logger = Logger(base_log_dir, config)  # Now logs to a unique run directory

    env = gym.make(config["env"])
    algo_wrapper = AlgoWrapper(config)
    model = algo_wrapper.get_model(env)

    eval_env = gym.make(config["env"])
    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        logger=logger,
        eval_freq=config["eval_interval"],
        best_model_save_path=logger.log_dir,  # Ensure best models are saved in the run directory
        log_path=logger.log_dir,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=config["total_timesteps"], callback=eval_callback)

    # Save the trained model uniquely per run
    model_path = f"{logger.log_dir}/final_model.zip"
    model.save(model_path)
    print(f"Model saved at: {model_path}")

    logger.close()

