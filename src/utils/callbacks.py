from stable_baselines3.common.callbacks import EvalCallback

class CustomEvalCallback(EvalCallback):
    """Wrapper around SB3's EvalCallback to log hyperparameters and evaluation metrics."""
    
    def __init__(self, eval_env, logger, eval_freq, **kwargs):
        super().__init__(eval_env, eval_freq=eval_freq, **kwargs)
        self.logger = logger

    def _on_step(self):
        result = super()._on_step()
        if result:  # Log only if evaluation was successful
            mean_reward = self.last_mean_reward
            self.logger.log_scalar("Eval/MeanReward", mean_reward, self.num_timesteps)
        return result
