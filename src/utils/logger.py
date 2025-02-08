# src/utils/logger.py
import os
import torch
import multiprocessing
import datetime
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.logger import Logger as SB3Logger
from omegaconf import DictConfig, OmegaConf

class Logger:
    def __init__(self, log_dir_datetimed, config: DictConfig):
        self.log_dir = log_dir_datetimed
        os.makedirs(self.log_dir, exist_ok=True)

        # Create a TensorBoard writer
        self.writer = SummaryWriter(self.log_dir)

        # Create an SB3 logger that writes to the same directory
        # self.sb3_logger = SB3Logger(folder=self.log_dir, output_formats=['tensorboard', 'stdout'])
        
        # Save HPs to YAML and TensorBoard
        self.save_hyperparams(config)
        self.log_hyperparams(config)

    def log_scalar(self, tag, value, step=None):
        self.writer.add_scalar(tag, value, step)

    def log_hyperparams(self, config):
        """Logs all hyperparameters to TensorBoard"""
        flat_config = OmegaConf.to_container(config, resolve=True)
        for key, value in flat_config.items():
            if isinstance(value, (int, float, str, bool)):
                self.writer.add_text(f"HPs/{key}", str(value), 0)

    def save_hyperparams(self, config):
        """Saves HPs to a file for future reference"""
        with open(os.path.join(self.log_dir, "hyperparams.yaml"), "w") as f:
            OmegaConf.save(config, f)

    # def _generate_run_id(self, base_log_dir):
    #     """Generates a unique run ID (timestamped) to avoid overwriting logs."""
    #     timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     return f"run_{timestamp}"

    def close(self):
        self.writer.close()

def print_system_info():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    
    print(f"Available CPU cores: {num_cpus}")
    print(f"Available GPUs: {num_gpus}")
    
    if num_gpus > 0:
        for i in range(num_gpus):
            print(f"🔹 GPU {i}: {torch.cuda.get_device_name(i)}")
    return num_cpus, num_gpus