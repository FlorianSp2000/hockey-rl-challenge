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
        self.episode_rewards = [[] for _ in range(n_envs)]  # Track rewards per env
        self.episode_lengths = [[] for _ in range(n_envs)]  # Track lengths per env
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

            # Compute mean reward components over all environments
            self.logger.record("reward/closeness_to_puck", np.mean(self.current_episode_reward_closeness))
            self.logger.record("reward/touch_puck", np.mean(self.current_episode_reward_touch))
            self.logger.record("reward/puck_direction", np.mean(self.current_episode_reward_direction))

        return True



class CustomEvalCallback(EvalCallback):
    """Wrapper around SB3's EvalCallback to log hyperparameters and evaluation metrics."""
    
    def __init__(self, eval_env, custom_logger, eval_freq, **kwargs):
        super().__init__(eval_env, eval_freq=eval_freq, **kwargs)
        self._custom_logger = custom_logger
        self._wins = 0
        self._draws = 0
        self._losses = 0


    def _on_step(self):
        """Enhanced step method to track rewards during evaluation."""
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            print(f"on_step EvalCallback inside")
            # Sync environments if needed
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see Stable Baselines documentation for details."
                    ) from e

            # Reset tracking variables
            self._wins = 0
            self._draws = 0
            self._losses = 0
            episode_rewards = []

            # Perform evaluation
            eval_results, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn
            )
            print(f"eval_results: {eval_results}")
            print(f"episode_lengths: {episode_lengths}")
            # Process episode rewards
            for reward in eval_results:
                episode_rewards.append(reward)
                if reward == 10:
                    self._wins += 1
                elif reward == 0:
                    self._draws += 1
                elif reward == -10:
                    self._losses += 1

            # Compute rates
            total_episodes = len(eval_results)
            mean_reward = np.mean(eval_results)
            
            # Logging
            if self._custom_logger:
                self._custom_logger.log_scalar("Eval/MeanReward", mean_reward, self.num_timesteps)
                
                if total_episodes > 0:
                    self._custom_logger.log_scalar("Eval/WinRate", self._wins / total_episodes, self.num_timesteps)
                    self._custom_logger.log_scalar("Eval/DrawRate", self._draws / total_episodes, self.num_timesteps)
                    self._custom_logger.log_scalar("Eval/LossRate", self._losses / total_episodes, self.num_timesteps)

            # Standard EvalCallback processing
            self.last_mean_reward = float(mean_reward)
            
            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, episode_reward={mean_reward:.2f}")
                print(f"Wins: {self._wins}/{total_episodes}, Draws: {self._draws}/{total_episodes}, Losses: {self._losses}/{total_episodes}")

            # Rest of the original implementation remains the same
            # (checking for best model, triggering callbacks, etc.)

        return continue_training


class ComprehensiveTrainingCallback(BaseCallback):
    """
    A comprehensive callback to track and log various training metrics for model training.
    
    Tracks:
    - Episode rewards
    - Episode lengths
    - Win/Draw/Loss statistics
    - Mean rewards
    - Episodic performance metrics
    """
    
    def __init__(
        self, 
        verbose: int = 0, 
        custom_logger: Optional[Logger] = None,
        log_dir: Optional[str] = None
    ):
        super().__init__(verbose)
        
        # Tracking variables
        self._episode_rewards = []
        self._episode_lengths = []
        self._wins = 0
        self._draws = 0
        self._losses = 0
        
        # Custom logging
        self._custom_logger = custom_logger
        self._log_dir = log_dir
    
    def _on_step(self) -> bool:
        # Check if an episode has ended
        if self.locals['dones']:
            print(f"self.locals {self.locals}")
            episode_reward = self.locals['infos'][0]['episode']['r']
            self._episode_rewards.append(episode_reward)
            
            # Track episode length
            episode_length = self.locals['infos'][0]['episode']['l']
            self._episode_lengths.append(episode_length)
            
            # Track win/draw/loss based on final reward
            self._wins += 1 if self.locals['infos'][0]['winner'] == 1 else self._wins
            self._draws += 1 if self.locals['infos'][0]['winner'] == 0 else self._draws
            self._losses += 1 if self.locals['infos'][0]['winner'] == -1 else self._losses
            print(f"len(self.locals['infos']) {len(self.locals['infos'])}")
            self._log_metrics()
        
        return True
    
    def _log_metrics(self):
        """Log various training metrics"""
        # Compute running metrics
        if self._episode_rewards:
            self._custom_logger.log_scalar(
                "Training/MeanEpisodeReward", 
                np.mean(self._episode_rewards), 
                self.num_timesteps
            )
            self._custom_logger.log_scalar(
                "Training/MeanEpisodeLength", 
                np.mean(self._episode_lengths), 
                self.num_timesteps
            )
        
        # Log performance statistics
        total_episodes = len(self._episode_rewards)
        print(f"Updated self._episode_rewards: {self._episode_rewards}")
        print(f"Updated self._episode_lengths: {self._episode_lengths}")
        print(f"Total episodes counted: {len(self._episode_rewards)}")
        print(f"self._wins: {self._wins}, self._draws: {self._draws}, self._losses: {self._losses}")
        assert total_episodes == (self._wins + self._draws + self._losses)

        if total_episodes != 0:
            self._custom_logger.log_scalar(
                "Training/WinRate", 
                self._wins / total_episodes, 
                self.num_timesteps
            )

            self._custom_logger.log_scalar(
                "Training/DrawRate", 
                self._draws / total_episodes, 
                self.num_timesteps
            )
            self._custom_logger.log_scalar(
                "Training/LossRate", 
                self._losses / total_episodes, 
                self.num_timesteps
            )
    
    # TODO: seems to be called every step, not at the end of a rollout for TD3
    # def _on_rollout_end(self):
    #     """Called at the end of a rollout (e.g., after multiple episodes)."""
    #     # Log metrics if custom logger is provided
    #     if self._custom_logger:
    #         self._log_metrics()

    
    def _on_training_end(self) -> None:
        """Final logging at the end of training"""
        print(f"_on_training_end EvalCAllback")
        if self._custom_logger:
            # Log final summary
            total_episodes = self._wins + self._draws + self._losses
            print(f"\nTraining Summary:")
            print(f"Total Episodes: {total_episodes}")
            print(f"Wins: {self._wins} ({self._wins/total_episodes*100:.2f}%)")
            print(f"Draws: {self._draws} ({self._draws/total_episodes*100:.2f}%)")
            print(f"Losses: {self._losses} ({self._losses/total_episodes*100:.2f}%)")
            print(f"Mean Episode Reward: {np.mean(self._episode_rewards):.2f}")
            print(f"Mean Episode Length: {np.mean(self._episode_lengths):.2f}")
