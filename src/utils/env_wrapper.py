import gymnasium as gym
import numpy as np
import hockey.hockey_env as h_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC
from src.custom_sb3.SAC_ERE import SACERE
from src.utils.selfplay_utils import ModelRecord
from typing import Dict
from gymnasium import spaces

ENV_TYPES = {'subproc': SubprocVecEnv, 'dummy': DummyVecEnv}

class HockeySB3Wrapper(gym.Wrapper):
    def __init__(self, env, rank, opponent_type="weak", model_class=None, model_pool=None, change_sides=True):
        super().__init__(env) # Initialize the parent class with the environment
        # Define observation and action space
        self.observation_space = self.env.observation_space
        self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)
        self.rank = rank
        # self.mode = mode
        # make player change sides
        self.change_sides = change_sides
        self.playing_as_agent2 = False
        # selfplay  
        # self.opponent = h_env.BasicOpponent(weak=(opponent_type == "weak"))
        self.model_class = model_class
        self.model_pool = model_pool
        self.current_opponent_id = None
        self.set_opponent(opponent_type)
        print(f"self.opponent.weak: {self.opponent.weak}")

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset() # left player

        if self.change_sides:
            self.playing_as_agent2 = bool(np.random.choice([0, 1]))
        
        if self.playing_as_agent2:
            self.obs_opponent = obs  # Store original observation for opponent
            obs = self.env.obs_agent_two()
        else:
            self.obs_opponent = self.env.obs_agent_two()

        return obs, info

    def step(self, action):
        # make sure action is of shape (4,)
        # assert action.shape == (4,), f"Invalid action shape: {action.shape}"
        opponent_action = self.opponent.act(self.obs_opponent)

        if self.playing_as_agent2:
            combined_action = np.hstack([opponent_action, action])
        else:
            combined_action = np.hstack([action, opponent_action])
        obs, reward, done, truncated, info = self.env.step(combined_action)
        
        if self.playing_as_agent2:
            self.obs_opponent = obs  # Store original observation for opponent
            obs = self.env.obs_agent_two()
            info = self.env.get_info_agent_two()
            reward = self.env.get_reward_agent_two(info)
        else:
            self.obs_opponent = self.env.obs_agent_two()

        return obs, reward, done, truncated, info

    def render(self, mode="human"):
        return self.env.render()

    def close(self):
        self.env.close()
    
    def set_opponent(self, opponent_type_or_id):
        """Set the current opponent"""
        if opponent_type_or_id in ["weak", "strong", "basic_weak", "basic_strong"]:
            opponent_type_or_id = opponent_type_or_id.split('_')[-1]
            self.opponent = h_env.BasicOpponent(weak=(opponent_type_or_id == "weak"))
            self.current_opponent_id = f"basic_{opponent_type_or_id}"
        elif self.model_pool and opponent_type_or_id in self.model_pool.models:
            model_path = self.model_pool.get_model_path(opponent_type_or_id)
            self.opponent = SelfPlayOpponent(self.model_class, model_path)
            self.current_opponent_id = opponent_type_or_id
        else:
            raise ValueError(f"Invalid opponent type or ID: {opponent_type_or_id}")
    
    def synchronize_model_pools(self, new_model_pool_models: Dict[str, ModelRecord]):
        """Method to copy model pool from callback (source of truth) to buffer callsback"""
        self.model_pool.models = new_model_pool_models
        # if self.rank == 0:
        #     print(f"self.model_pool.models after sync: {self.model_pool.models}")
        
        


def make_env(seed, rank, model_pool=None, opponent_type="weak", model_class=None):
    """
    Utility function for creating multiple environments for parallel execution.
    This function is required for SubprocVecEnv, which spawns environments.
    """
    def _init():
        env = h_env.HockeyEnv()
        env = HockeySB3Wrapper(env, rank, opponent_type=opponent_type, model_pool=model_pool, model_class=model_class)
        env = Monitor(env)
        return env
    set_random_seed(seed)
    return _init

def create_parallel_envs(config, n_envs, model_pool=None, opponent_type="weak"):
    model_class = None
    if config['algorithm']['name'].lower() == 'sac':
        model_class = SAC
        if config['algorithm']['params']['replay_buffer_class'] == 'ERE':
            model_class = SACERE   
    print(f"model_class: {model_class}")
    env_fns = [make_env(config["seed"], i, model_pool, opponent_type, model_class) for i in range(n_envs - 1)]
    env = ENV_TYPES[config["vector_type"]](env_fns)
    return env



class SelfPlayOpponent:
    def __init__(self, model_class, model_path: str):
        self.model = model_class.load(model_path)
        self.weak = False  # For compatibility with BasicOpponent interface
    
    def act(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        assert action.shape == (4,), f"Invalid action shape: {action.shape}"
        return action
