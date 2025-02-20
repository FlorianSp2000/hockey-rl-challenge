# src/selfplay.py
import time
from stable_baselines3.common.logger import configure
from src.utils.selfplay_utils import ModelPool
from src.utils.callbacks import SelfPlayCallback, CustomTensorboardCallback
from src.utils.env_wrapper import create_parallel_envs
from src.utils.logger import print_system_info
from src.utils.model_io import get_enumerated_path
from src.utils.algo_wrapper import AlgoWrapper

def run(config, logger):
    num_cpus, num_gpus = print_system_info()
    config["tensorboard_log"] = logger.log_dir
    
    # Initialize model pool
    model_pool = ModelPool(
        pool_size=config['mode']['pool_size'],  # Adjust pool size as needed
        save_dir=logger.log_dir,
        start_relative_strength=config['mode']['start_relative_strength'],
        curriculum_thresholds = config['mode']['curriculum'],
        game_decision_margins = config['mode']['game_decision_margin'],
        k_factors = config['mode']['k_factors'],

    )
    
    num_of_train_envs = num_cpus-config['n_eval_envs']
    # Create environments with model pool
    env = create_parallel_envs(
        config,
        n_envs=num_of_train_envs,
        model_pool=model_pool
    )
    
    # Create or load agent
    algo_wrapper = AlgoWrapper(config)
    agent = algo_wrapper.create_or_load_model(env=env)
    
    # Setup logging
    sb3_logger = configure(logger.log_dir, ["tensorboard", "stdout"])
    agent.set_logger(sb3_logger)
    
    # Setup callbacks
    callbacks = [
        SelfPlayCallback(
            model_pool=model_pool,
            save_path=f"{logger.log_dir}/checkpoints",
            selfplay_env=env,
            opponent_share_constraint=config['mode']["opponent_share_constraint"],
            opponent_switch_freq=config['mode']["opponent_switch_freq"],
            n_envs=num_of_train_envs,
            mixing_ratio=config['mode']["mixing_ratio"],
            verbose=1
        )
    ]
    
    # Train
    start_time = time.time()
    print("### Starting Self-play Training ###")
    agent.learn(
        total_timesteps=config["mode"]["total_timesteps"],
        callback=callbacks,
        tb_log_name="",
        reset_num_timesteps=config["checkpoint"]["reset_num_timesteps"],
    )
    
    # Save final model and cleanup
    training_duration = (time.time() - start_time) / 60
    logger.log_scalar("time/Training Duration (minutes)", training_duration)
    print(f"🕒 Training completed in {training_duration:.2f} minutes")
    
    model_path = get_enumerated_path(f"{logger.log_dir}/final_model.zip")
    agent.save(model_path)
    print(f"Model saved at: {model_path}")
    agent.save_replay_buffer(f"{model_path.split('.')[0]}_replay_buffer")

    env.close()
    logger.close()