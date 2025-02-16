import os
from typing import Optional, Union, Dict, Any
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

class CustomTensorboardCallback(BaseCallback):
    def __init__(self, n_envs=1, verbose=0):
        super().__init__(verbose)
        self.n_envs = n_envs
        # print(f"n_envs: {n_envs}")
        # self.episode_rewards = [[] for _ in range(n_envs)]  # Track rewards per env
        # self.episode_lengths = [[] for _ in range(n_envs)]  # Track lengths per env
        self.all_rewards = []  # Stores completed episode rewards
        self.all_lengths = []  # Stores completed episode lengths
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.reset_episode_rewards()

    def reset_episode_rewards(self):
        self.current_episode_reward_closeness = [0] * self.n_envs
        self.current_episode_reward_touch = [0] * self.n_envs
        self.current_episode_reward_direction = [0] * self.n_envs

    def _on_step(self) -> bool:
        infos = self.locals['infos']
        dones = self.locals['dones']
        # print(f"len(infos): {len(infos)}")
        episode_finished = False  # Flag to track if at least one environment finished

        for i in range(min(self.n_envs, len(infos))):
            info = infos[i]
            # Accumulate reward components per environment
            self.current_episode_reward_closeness[i] += info['reward_closeness_to_puck']
            self.current_episode_reward_touch[i] += info['reward_touch_puck']
            self.current_episode_reward_direction[i] += info['reward_puck_direction']

            if dones[i]:  # If episode ends in this environment
                episode_finished = True  # Mark that at least one episode ended

                episode_info = info['episode']
                winner = info['winner']

                # Store completed episode stats
                self.all_rewards.append(episode_info['r'])
                self.all_lengths.append(episode_info['l'])

                # Track game outcome
                if winner == 1:
                    self.wins += 1
                elif winner == 0:
                    self.draws += 1
                else:
                    self.losses += 1

                # Reset accumulated rewards for this environment
                self.current_episode_reward_closeness[i] = 0
                self.current_episode_reward_touch[i] = 0
                self.current_episode_reward_direction[i] = 0

        # Log only if at least one episode finished
        if episode_finished:
            # Ensure every completed episode has a win/loss/draw
            assert len(self.all_rewards) == (self.wins + self.draws + self.losses), \
                f"Mismatch: Episodes recorded ({len(self.all_rewards)}) != Wins ({self.wins}) + Draws ({self.draws}) + Losses ({self.losses})"

            # Compute statistics over completed episodes
            self.logger.record("episode/mean_reward", np.mean(self.all_rewards))
            self.logger.record("episode/mean_length", np.mean(self.all_lengths))
            self.logger.record("episode/draw_rate", self.draws / (self.wins + self.draws + self.losses))
            self.logger.record("episode/loss_rate", self.losses / (self.wins + self.draws + self.losses))
            self.logger.record("episode/win_rate", self.wins / (self.wins + self.draws + self.losses))
            self.logger.record("time/no_total_episodes", len(self.all_rewards))

            # Compute mean reward components over all environments
            self.logger.record("reward/closeness_to_puck", np.mean(self.current_episode_reward_closeness))
            self.logger.record("reward/touch_puck", np.mean(self.current_episode_reward_touch))
            self.logger.record("reward/puck_direction", np.mean(self.current_episode_reward_direction))

        return True



class WinRateCheckpointCallback(BaseCallback):
    def __init__(self, save_path, win_rate_threshold=0.8, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.win_rate_threshold = win_rate_threshold
        self.last_saved_step = 0  # Prevent excessive saving
        self.last_win_rate = 0

    def _on_step(self) -> bool:

        # Get logged statistics
        win_rate = self.logger.name_to_value.get("episode/win_rate", None)
        if win_rate is not None and win_rate >= self.win_rate_threshold and win_rate > self.last_win_rate:
            # TODO: said frequency differently later on and make hydra parameter
            if self.num_timesteps - self.last_saved_step > 100000:  # Avoid frequent saves
                save_file = os.path.join(self.save_path, f"checkpoint_{self.num_timesteps}.zip")
                self.model.save(save_file)
                self.last_saved_step = self.num_timesteps
                if self.verbose > 0:
                    print(f"Checkpoint saved at step {self.num_timesteps} (Win rate: {win_rate:.2f})")
                self.last_win_rate = win_rate

        return True



class CustomEvalCallback(EvalCallback):
    def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        Since hockey_env' success key is named differently we have to overwrite it

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]
        if locals_["done"]:
            winner = info.get("winner")
            if winner==1:
                self._is_success_buffer.append(True)
            else:
                self._is_success_buffer.append(False)


