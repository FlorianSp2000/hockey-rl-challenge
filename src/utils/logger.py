# src/utils/logger.py
import os
import yaml
import datetime
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.logger import Logger as SB3Logger


class Logger:
    def __init__(self, base_log_dir, config):
        self.run_id = self._generate_run_id(base_log_dir)
        self.base_dir = base_log_dir
        self.log_dir = os.path.join(base_log_dir, self.run_id)
        os.makedirs(self.log_dir, exist_ok=True)

        # Create a TensorBoard writer
        self.writer = SummaryWriter(self.log_dir)

        # Create an SB3 logger that writes to the same directory
        # self.sb3_logger = SB3Logger(folder=self.log_dir, output_formats=['tensorboard', 'stdout'])
        
        # Save HPs to YAML and TensorBoard
        self.save_hyperparams(config)
        self.log_hyperparams(config)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_hyperparams(self, config):
        """Logs all hyperparameters to TensorBoard"""
        for key, value in config.items():
            if isinstance(value, (int, float, str, bool)):
                self.writer.add_text(f"HPs/{key}", str(value), 0)

    def save_hyperparams(self, config):
        """Saves HPs to a file for future reference"""
        with open(os.path.join(self.log_dir, "hyperparams.yaml"), "w") as f:
            yaml.dump(config, f)

    def _generate_run_id(self, base_log_dir):
        """Generates a unique run ID (timestamped) to avoid overwriting logs."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"run_{timestamp}"

    def close(self):
        self.writer.close()
