# src/utils/algo_wrapper.py
import importlib
from stable_baselines3 import TD3, SAC, PPO
from stable_baselines3.common.buffers import ReplayBuffer
import torch
import torch.nn as nn
from src.custom_sb3.replay_buffer import EREBuffer
from src.custom_sb3.SAC_ERE import SACERE
from src.custom_sb3.SAC_CEPO import SACCEPO
from pink import PinkNoiseDist, PinkActionNoise

class AlgoWrapper:
    activation_fns = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'LeakyReLU': nn.LeakyReLU}
    replay_buffers = {'ERE': EREBuffer}

    def __init__(self, config):
        self.algorithm = config["algorithm"]["name"]
        self.implementation = config["implementation"]
        self.config = config["algorithm"]["params"]
        self.tensorboard_log = config["tensorboard_log"]
        self.parallelize = config["parallelize"]
        
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try :
            self.device=torch.device(config["device"])
        except AssertionError:
            print(f"Device {config['device']} not found, using CPU")
            self.device = torch.device("cpu")
        self.replay_buffer_kwargs = {'ERE': self.config["replay_buffer_kwargs"]}
        print(f"self.replay_buffer_kwargs {self.replay_buffer_kwargs}")
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
            model = TD3(
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
            if self.config['replay_buffer_class'] in ['ERE', 'PER', 'ERE+PER']:
                print("using SACERE Agent")
                model = SACERE(
                    policy,
                    env,
                    verbose=1,
                    tensorboard_log=self.tensorboard_log,
                    replay_buffer_class=self.replay_buffers.get(self.config["replay_buffer_class"], None),
                    replay_buffer_kwargs=self.replay_buffer_kwargs.get(self.config["replay_buffer_class"], None),
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
                    device=self.device,
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
            elif self.config['use_cepo']:
                print("using SAC-CEPO Agent")
                model = SACCEPO(
                    policy,
                    env,
                    verbose=1,
                    tensorboard_log=self.tensorboard_log,
                    replay_buffer_class=self.replay_buffers.get(self.config["replay_buffer_class"], None),
                    replay_buffer_kwargs=self.replay_buffer_kwargs.get(self.config["replay_buffer_class"], None),
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
                    device=self.device,
                    ce_N=self.config["cepo_kwargs"]["ce_N"], # sample number N
                    ce_ne=self.config["cepo_kwargs"]["ce_ne"], # number of elites
                    ce_t=self.config["cepo_kwargs"]["ce_t"], # number of iterations
                    ce_size=self.config["cepo_kwargs"]["ce_size"], # for initializing scale
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
                
            else:
                print("using SAC Agent")
                model = SAC(
                    policy,
                    env,
                    verbose=1,
                    tensorboard_log=self.tensorboard_log,
                    # replay_buffer_class=self.replay_buffers.get([self.config["replay_buffer_class"]], None),
                    # replay_buffer_kwargs=self.replay_buffer_kwargs.get([self.config["replay_buffer_class"]], None),
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
                    device=self.device,
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
            if self.config['action_noise']:
                noise_type = self.config["action_noise"]
                if noise_type == "pink":
                    seq_len = env.get_attr('max_timesteps')[0] 
                    action_dim = env.action_space.shape[-1]
                    # model.action_noise = PinkActionNoise(seq_len, action_dim)
                    model.actor.action_dist = PinkNoiseDist(seq_len, action_dim)
                    print("Applied Pink Action Noise.")
                else:
                    raise ValueError(f"Unknown action noise type: {noise_type}")

        elif self.algorithm == "PPO":
            model = PPO(
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
        
 
        return model
    
    def _get_custom_model(self, env):
        module_name = f"src.custom_algos.{self.algorithm.lower()}"
        class_name = self.algorithm

        module = importlib.import_module(module_name)
        ModelClass = getattr(module, class_name)

        return ModelClass(env, self.config)
