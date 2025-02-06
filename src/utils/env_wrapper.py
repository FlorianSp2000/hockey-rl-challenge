import gymnasium as gym
import numpy as np
import hockey.hockey_env as h_env

class HockeySB3Wrapper(gym.Env):
    def __init__(self, opponent_type="weak"):
        super(HockeySB3Wrapper, self).__init__()
        self.env = h_env.HockeyEnv()
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
