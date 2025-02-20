import os
from typing import Optional, Union, Dict, Any
from pathlib import Path
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
import json
from src.utils.selfplay_utils import OpponentLog

class CustomTensorboardCallback(BaseCallback):
    def __init__(self, n_envs=1, verbose=0):
        super().__init__(verbose)
        self.n_envs = n_envs
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



class SelfPlayCallback(BaseCallback):
    def __init__(
        self,
        model_pool,
        save_path: str,
        selfplay_env,
        opponent_share_constraint: Optional[float] = 0.5,
        opponent_switch_freq: int = 50000,
        n_envs: int = 1,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.model_pool = model_pool
        self.save_path = save_path
        self.selfplay_env = selfplay_env
        print(f"selfplay_env: {selfplay_env}")
        print(f"self.save_path: {self.save_path}")
        self.opponent_share_constraint = opponent_share_constraint
        self.opponent_switch_freq = opponent_switch_freq
        self.last_switch_step = 0
        self.current_opponent_id = None
        self.win_rates: Dict[str, float] = {}
        self.num_seen_opponents = 0
        self.opponent_log: list[OpponentLog] = [] # Used for json logging
        # Metrics
        self.n_envs = n_envs
        self.all_rewards = []  # Stores completed episode rewards
        self.all_lengths = []  # Stores completed episode lengths
        self.wins = 0
        self.draws = 0
        self.losses = 0
        # Overall match stats
        self.overall_wins = 0
        self.overall_draws = 0
        self.overall_losses = 0
        self.overall_rewards = []

    def _on_step(self) -> bool:
        # add metrics
        self._track_current_model_performance()
        # Check if it's time to evaluate and possibly switch opponents
        if self.num_timesteps - self.last_switch_step >= self.opponent_switch_freq:
            self._evaluate_and_switch_opponent()
            self.last_switch_step = self.num_timesteps
        return True
    
    def _track_current_model_performance(self):
        infos = self.locals['infos']
        dones = self.locals['dones']
        episode_finished = False  # Flag to track if at least one environment finished

        for i in range(min(self.n_envs, len(infos))):
            info = infos[i]

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

        if episode_finished:
            # Ensure every completed episode has a win/loss/draw
            assert len(self.all_rewards) == (self.wins + self.draws + self.losses), \
                f"Mismatch: Episodes recorded ({len(self.all_rewards)}) != Wins ({self.wins}) + Draws ({self.draws}) + Losses ({self.losses})"

            # Compute statistics over completed episodes
            self.logger.record("selfplay/mean_reward", np.mean(self.all_rewards))
            self.logger.record("selfplay/mean_length", np.mean(self.all_lengths))
            self.logger.record("selfplay/draw_rate", self.draws / (self.wins + self.draws + self.losses))
            self.logger.record("selfplay/loss_rate", self.losses / (self.wins + self.draws + self.losses))
            self.logger.record("selfplay/win_rate", self.wins / (self.wins + self.draws + self.losses))


    def _evaluate_and_switch_opponent(self):
        # Get current win rate
        win_rate = self.wins / (self.wins + self.draws + self.losses)
        if win_rate is None:
            raise ValueError("Win rate is None. This should not happen.")
        if self.current_opponent_id is None:
            print("Getting opponent from env")
            current_opponent_ids = self.selfplay_env.get_attr("current_opponent_id")
            assert len(set(current_opponent_ids)) == 1, "All environments should have the same opponent"
            self.current_opponent_id = str(current_opponent_ids[0])
        
        # Store win rate against current opponent
        print(f"self.current_opponent_id: {self.current_opponent_id}")
        self.win_rates[self.current_opponent_id] = win_rate
        print(f"selfplaycallback win_rate: {win_rate}")
        print(f"selfplaycallback self.win_rates: {self.win_rates}")
        # Update elo ratings
        last_relative_strength = self.model_pool.current_relative_strength
        current_relative_strength = self.model_pool._update_relative_strengths(self.win_rates) # Updates current_relative_strength
        opponent_relative_strength = self.model_pool.models[self.current_opponent_id].relative_strength

        print(f"CALLBACK current model's new relative_strength is {current_relative_strength}")
        # Store model only if elo improved and it played against at least 50% of the models in the model pool
        if (current_relative_strength - last_relative_strength >= 0 and 
            len(self.win_rates) >= 2 and 
            len(self.win_rates) / len(self.model_pool.models) > self.opponent_share_constraint and
            self.model_pool.curriculum_phase != "assessment"):
            
            model_path = os.path.join(
                self.save_path, 
                f"model_{self.num_timesteps}.zip"
            )
            self.model.save(model_path)
            self.model_pool.add_model(
                model_path,
                self.num_timesteps,
                self.win_rates.copy()
            )
            self.selfplay_env.env_method("synchronize_model_pools", self.model_pool.models)
            # reset win rates
            self.win_rates = {} # TODO: should I really reset them?
        
        # log prior opponent
        self.opponent_log.append([self.num_timesteps, self.current_opponent_id, win_rate, current_relative_strength, opponent_relative_strength])
        # Select new opponent based on current performance
        new_opponent_id = self.model_pool.select_opponent()

        # new_opponent_id = self.selfplay_env.env_method("select_opponent", win_rate, indices=0)
        print(f"new_opponent_id in callback: {new_opponent_id}")
        # Switch opponent in all environments TODO: set opponents in environments differently
        self.selfplay_env.env_method("set_opponent", new_opponent_id)        
        self.current_opponent_id = new_opponent_id
        
        if self.verbose > 0:
            print(f"Switching to opponent {new_opponent_id} \n (Win rate of current model against {new_opponent_id} so far: {self.win_rates.get(self.current_opponent_id, None)})") # :.2f
        # log and update overall metrics
        self.num_seen_opponents += 1
        self.overall_wins += self.wins
        self.overall_draws += self.draws
        self.overall_losses += self.losses
        self.overall_rewards += self.all_rewards

        self.reset_stats()

        self.logger.record("selfplay/opponent_switches", self.num_seen_opponents)
        
        self.logger.record("selfplay/overall_draw_rate", self.overall_draws / (self.overall_wins + self.overall_draws + self.overall_losses))
        self.logger.record("selfplay/overall_loss_rate", self.overall_losses / (self.overall_wins + self.overall_draws + self.overall_losses))
        self.logger.record("selfplay/overall_win_rate", self.overall_wins / (self.overall_wins + self.overall_draws + self.overall_losses))
        self.logger.record("selfplay/overall_mean_reward", np.mean(self.overall_rewards))
        self.logger.record("selfplay/no_total_episodes", len(self.overall_rewards))
        self.logger.record("selfplay/elo_score", current_relative_strength)
        self.save_logs()

    def reset_stats(self):
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.all_rewards = []
        self.all_lengths = []

    def save_logs(self):
        with open((Path(self.save_path).parent / "selfplay_log.json"), "w") as f:
            json.dump(self.opponent_log, f)
