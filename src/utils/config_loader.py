# src/utils/config_loader.py
import yaml
from stable_baselines3 import SAC, TD3
import os
import glob

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_trained_model(model_path, config):
    """Loads a trained model from file."""
    if config["algorithm"] == "SAC":
        model = SAC.load(model_path)
    elif config["algorithm"] == "TD3":
        model = TD3.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {config['algorithm']}")
    
    return model


def get_latest_run(log_dir):
    """Finds the most recent run directory inside log_dir."""
    run_dirs = sorted(
        glob.glob(os.path.join(log_dir, "run_*")),
        key=os.path.getmtime,  # Sort by modification time (latest first)
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None

def load_latest_model(log_dir, config):
    """Loads the latest trained model from the most recent run."""
    latest_run = get_latest_run(log_dir)
    if latest_run:
        model_path = os.path.join(latest_run, "final_model.zip")
        if os.path.exists(model_path):
            print(f"🔄 Loading latest model from: {model_path}")
            if config["algorithm"] == "SAC":
                return SAC.load(model_path)
            elif config["algorithm"] == "TD3":
                return TD3.load(model_path)
            else:
                raise ValueError(f"Unknown algorithm: {config['algorithm']}")
        else:
            print(f"⚠️ No model found in latest run: {latest_run}")
    else:
        print(f"⚠️ No previous runs found in {log_dir}")
    
    return None  # No model found

# Example Usage
# model = load_latest_model(config["log_dir"], config)
# if model:
#     print("✅ Model successfully loaded!")
# else:
#     print("❌ No previous model found. Starting training from scratch.")
#     run(config)
