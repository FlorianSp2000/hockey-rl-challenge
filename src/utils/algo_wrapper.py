# src/utils/algo_wrapper.py
import importlib
from stable_baselines3 import TD3, SAC

class AlgoWrapper:
    def __init__(self, config):
        self.algorithm = config["algorithm"]
        self.implementation = config["implementation"]
        self.config = config

    def get_model(self, env):
        if self.implementation == "sb3":
            return self._get_sb3_model(env)
        elif self.implementation == "custom":
            return self._get_custom_model(env)
        else:
            raise ValueError(f"Unknown implementation: {self.implementation}")

    def _get_sb3_model(self, env):
        policy = self.config.get("policy", "MlpPolicy")  # Defaults to MlpPolicy
        if self.algorithm == "TD3":
            return TD3(policy, env, verbose=1, batch_size=self.config["batch_size"], learning_rate=self.config["learning_rate"])
        elif self.algorithm == "SAC":
            return SAC(policy, env, verbose=1, batch_size=self.config["batch_size"], learning_rate=self.config["learning_rate"])
        else:
            raise ValueError(f"SB3 does not support algorithm: {self.algorithm}")

    def _get_custom_model(self, env):
        module_name = f"src.custom_algos.{self.algorithm.lower()}"
        class_name = self.algorithm

        module = importlib.import_module(module_name)
        ModelClass = getattr(module, class_name)

        return ModelClass(env, self.config)
