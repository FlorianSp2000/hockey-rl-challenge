from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import os

def get_run_dir(checkpoint_path, mode_log_dir):
    """
    Determine the run directory based on checkpoint path
    """
    if checkpoint_path:
        return os.path.dirname(checkpoint_path)
    return mode_log_dir #f"{mode_log_dir}/run_${{now:%Y-%m-%d_%H-%M-%S}}"

# Register the resolver
OmegaConf.register_new_resolver(
    "get_run_dir",
    get_run_dir,
    replace=True  # Allows updating the resolver if it already exists
)

# Create and register the config
def register_configs():
    cs = ConfigStore.instance()
