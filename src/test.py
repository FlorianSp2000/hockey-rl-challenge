from pathlib import Path
from stable_baselines3.common.monitor import Monitor
from src.custom_sb3.SAC_ERE import SACERE
from stable_baselines3 import SAC
from pathlib import Path
import numpy as np
import hockey.hockey_env as h_env
import gymnasium as gym

from src.utils.env_wrapper import HockeySB3Wrapper
from src.utils.algo_wrapper import AlgoWrapper
import os
from dotenv import load_dotenv
load_dotenv('../.env')

# checkpoint_path = Path('').resolve().parent / Path('models\\run_2025-02-22_18-01-57\\model_1150023')
    

def run(config, logger):
    config["tensorboard_log"] = logger.log_dir
    
    n_games = config['mode']['n_test_episodes']
    opponent_spec = config['mode']['opponent_checkpoint'].get('load_from', None)
    opponent_type = config['opponent'] # weak or strong

    if not opponent_spec:
        opponent_spec = opponent_type # overwrite with basic opponent
    else:
        opponent_spec = os.path.join(config['mode']['opponent_checkpoint']['load_from'], config['mode']['opponent_checkpoint']['model_name'])
    print(f"opponent_spec: {opponent_spec}")
    change_sides = config['mode'].get('change_sides', False)
    render = config['mode'].get('render', False)
    algorithm_name = config['algorithm']['name']


    algo_wrapper = AlgoWrapper(config)
    model_class = algo_wrapper.get_sb3_class(algorithm_name)
    opponent = algo_wrapper.load_opponent(opponent_spec, model_class) # opponent with .act() method
    
    # by passing test_opponent we set the opponent in the buffer internally
    env = HockeySB3Wrapper(h_env.HockeyEnv(), 
                           rank=0, 
                           change_sides=change_sides, 
                           opponent_type=opponent_type,
                           test_opponent=opponent,
                           )
    
    model = algo_wrapper.load_model_from_checkpoint(env, 
                checkpoint_path=config['checkpoint']['load_from'], 
                sb3_model_class=model_class,
                model_name=config['checkpoint'].get('model_name', 'final_model')  # Allow specifying a custom model name
            )

    wins, losses, draws = 0, 0, 0
    for game in range(n_games):
        obs, _ = env.reset()
        done = False
        # print(f"env.playing_as_agent2 = {env.playing_as_agent2}")
        while not done:
            if render:
                env.unwrapped.render(mode="human")
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(np.array(action))
        
        # print(f"reward: {reward}")
        # print(f"info: {info}")
        if info['winner'] == 1:
            wins += 1
        elif info['winner'] == -1:
            losses += 1
        else:
            draws += 1
        
        print(f"Game {game + 1}/{n_games}: Winner = {info['winner']}")

    print(f"Results over {n_games} games: Wins={wins}, Losses={losses}, Draws={draws}")


# How to Use
# python main.py algorithm=sac mode=test 
# checkpoint.load_from=models/run_2025-02-22_18-01-57 
# checkpoint.model_name=model_4250085 
# algorithm.params.replay_buffer_class=ERE 
# mode.opponent_checkpoint.load_from=models/run_2025-02-22_18-01-57 mode.opponent_checkpoint.model_name=model_1150023 
# mode.render=True mode.change_sides=false