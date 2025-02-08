import gymnasium as gym
import numpy as np
import hockey.hockey_env as h_env
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

class HockeySB3Wrapper(gym.Wrapper):
    def __init__(self, env, opponent_type="weak"):
        super().__init__(env) # Initialize the parent class with the environment
        self.opponent = h_env.BasicOpponent(weak=(opponent_type == "weak"))

        # Define observation and action space
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        self.obs_opponent = self.env.obs_agent_two()
        return obs, info

    def step(self, action):
        opponent_action = self.opponent.act(self.obs_opponent)
        combined_action = np.hstack([action, opponent_action])
        obs, reward, done, truncated, info = self.env.step(combined_action)
        
        self.obs_opponent = self.env.obs_agent_two()  # Update opponent's obs
        return obs, reward, done, truncated, info

    def render(self, mode="human"):
        return self.env.render()

    def close(self):
        self.env.close()

def make_env(seed, rank, opponent_type="weak"):
    """
    Utility function for creating multiple environments for parallel execution.
    This function is required for SubprocVecEnv, which spawns environments.
    """
    def _init():
        env = h_env.HockeyEnv()
        env = HockeySB3Wrapper(env, opponent_type=opponent_type)
        env = Monitor(env)
        return env
    set_random_seed(seed + rank)
    return _init

def create_parallel_envs(config, n_envs, opponent_type="weak"):
    env_fns = [make_env(config["seed"], i, opponent_type) for i in range(n_envs - 1)]
    env = SubprocVecEnv(env_fns)
    return env


# # Reregister the environment as old registry is invalid (laserhockey instead of hockey module)
# register(
#     id='HockeyWrapped-v0',
#     entry_point='hockey.hockey_env:HockeyEnv',
#     kwargs={'mode': 0}
# )
