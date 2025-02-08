# src/utils/algo_wrapper.py
import importlib
from stable_baselines3 import TD3, SAC, PPO
import torch.nn as nn

class AlgoWrapper:
    def __init__(self, config):
        self.algorithm = config["algorithm"]["name"]
        self.implementation = config["implementation"]
        self.config = config["algorithm"]["params"]
        self.tensorboard_log = config["tensorboard_log"]
        self.parallelize = config["parallelize"]
        self.activation_fns = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'LeakyReLU': nn.LeakyReLU}
        print(f"self.config: {self.config}")
    def get_model(self, env):
        if self.implementation == "sb3":
            return self._get_sb3_model(env)
        elif self.implementation == "custom":
            return self._get_custom_model(env)
        else:
            raise ValueError(f"Unknown implementation: {self.implementation}")

    def _get_sb3_model(self, env):
        policy = self.config['policy']
        net_arch=self.config['net_arch']
        activation_fn=self.activation_fns[self.config['activation_fn']]
        policy_kwargs = dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
            log_std_init=self.config["log_std_init"],
        )
        print(f"policy_kwargs: {policy_kwargs}")

        if self.algorithm == "TD3":
            return TD3(
                policy,
                env,
                verbose=1,
                batch_size=self.config["batch_size"],
                learning_rate=self.config["learning_rate"],
                tensorboard_log=self.tensorboard_log,
                gamma=self.config["gamma"],
                policy_kwargs=policy_kwargs,
                learning_starts=self.config["learning_starts"],
                gradient_steps=(
                    -1
                    if self.parallelize and not self.config["gradient_steps"]
                    else (
                        self.config["gradient_steps"]
                        if self.config["gradient_steps"]
                        else 1
                    )
                ),  # See 1.6.4 https://stable-baselines3.readthedocs.io/_/downloads/en/master/pdf/
            )
        elif self.algorithm == "SAC":
            return SAC(
                policy,
                env,
                verbose=1,
                tensorboard_log=self.tensorboard_log,
                policy_kwargs=policy_kwargs,
                batch_size=self.config["batch_size"],
                gamma=self.config["gamma"],
                learning_rate=self.config["learning_rate"],
                tau=self.config["tau"],
                buffer_size=self.config["buffer_size"],
                train_freq=tuple(self.config["train_freq"]) if isinstance(self.config["train_freq"], list) else self.config["train_freq"],
                use_sde=self.config["use_sde"],
                sde_sample_freq=self.config["sde_sample_freq"],
                learning_starts=self.config["learning_starts"],
                gradient_steps=(
                    -1
                    if self.parallelize and not self.config["gradient_steps"]
                    else (
                        self.config["gradient_steps"]
                        if self.config["gradient_steps"]
                        else 1
                    )
                ),  # See 1.6.4 https://stable-baselines3.readthedocs.io/_/downloads/en/master/pdf/
            )
        elif self.algorithm == "PPO":
            return PPO(
                policy,
                env,
                verbose=1,
                batch_size=self.config["batch_size"],
                learning_rate=self.config["learning_rate"],
                tensorboard_log=self.tensorboard_log,
                n_steps=self.config["n_steps"],
                policy_kwargs=policy_kwargs,
            )
        else:
            raise ValueError(f"SB3 does not support algorithm: {self.algorithm}")

    def _get_custom_model(self, env):
        module_name = f"src.custom_algos.{self.algorithm.lower()}"
        class_name = self.algorithm

        module = importlib.import_module(module_name)
        ModelClass = getattr(module, class_name)

        return ModelClass(env, self.config)
