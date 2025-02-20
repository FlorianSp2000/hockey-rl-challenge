from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from torch.nn import functional as F

# from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3 import SAC
# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20

from torch import nn
from torch.distributions import Normal
from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution, TanhBijector
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.torch_layers import (
    create_mlp,
)
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule


class DecoupledActor(BasePolicy):
    """
    Actor network (policy) for SAC decoupling mean and std deviation

    """

    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )
        print("Decoupled Actor is used")
        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        action_dim = get_action_dim(self.action_space)
        self.action_dim = action_dim
        # latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        # self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
        self.mean_network = nn.Sequential(*create_mlp(features_dim, -1, net_arch, activation_fn),
                                          nn.Linear(last_layer_dim, action_dim))
        self.std_network = nn.Sequential(*create_mlp(features_dim, -1, net_arch, activation_fn),
                                         nn.Linear(last_layer_dim, action_dim))

        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)  # type: ignore[assignment]

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using gSDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).

        :return:
        """
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(self, obs: PyTorchObs) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs, self.features_extractor)
        # latent_pi = self.latent_pi(features)

        mean_actions = self.mean_network(features)
        log_std = self.std_network(features)

        # if self.use_sde:
            # return mean_actions, self.log_std, dict(latent_sde=mean_latent)
        # Unstructured exploration (Original implementation)
        # mean_actions = self.mu_out(mean_latent)
        # log_std = self.log_std_out(std_latent)
        # log_std = self.log_std(latent_pi)  # type: ignore[operator]
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: PyTorchObs) -> tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)


class SACCEPOPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    """

    actor: DecoupledActor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> DecoupledActor:
        print("SACCEPOPolicy is used")
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return DecoupledActor(**actor_kwargs).to(self.device)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.mean_network.optimizer = self.optimizer_class(
            self.actor.mean_network.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )
        self.actor.std_network.optimizer = self.optimizer_class(
            self.actor.std_network.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )
        # self.actor.optimizer = self.optimizer_class(
        #     self.actor.parameters(),
        #     lr=lr_schedule(1),  # type: ignore[call-arg]
        #     **self.optimizer_kwargs,
        # )

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic.named_parameters() if "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = list(self.critic.parameters())

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

MlpPolicy = SACCEPOPolicy

class SACCEPO(SAC):
    """
    Soft Actor-Critic (SAC) with Cross-Entropy Policy Optimization

    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: SACCEPOPolicy
    actor: DecoupledActor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
            self,
            policy: Union[str, type[SACPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            buffer_size: int = 1_000_000,  # 1e6
            learning_starts: int = 100,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, tuple[int, str]] = 1,
            gradient_steps: int = 1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[type[ReplayBuffer]] = None,
            replay_buffer_kwargs: Optional[dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            ent_coef: Union[str, float] = "auto",
            target_update_interval: int = 1,
            target_entropy: Union[str, float] = "auto",
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            use_sde_at_warmup: bool = False,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            # add arguments for CEM:
            ce_N: int = 100, # sample number N
            ce_ne: int = 5, # number of elites
            ce_t : int = 10, # number of iterations
            ce_size: float = 0.05, # elite fraction
        ):
            super().__init__(
                policy=policy,
                env=env,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                learning_starts=learning_starts,
                batch_size=batch_size,
                tau=tau,
                gamma=gamma,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                action_noise=action_noise,
                replay_buffer_class=replay_buffer_class,
                replay_buffer_kwargs=replay_buffer_kwargs,
                optimize_memory_usage=optimize_memory_usage,
                ent_coef=ent_coef,
                target_update_interval=target_update_interval,
                target_entropy=target_entropy,
                use_sde=use_sde,
                sde_sample_freq=sde_sample_freq,
                use_sde_at_warmup=use_sde_at_warmup,
                stats_window_size=stats_window_size,
                tensorboard_log=tensorboard_log,
                policy_kwargs=policy_kwargs,
                verbose=verbose,
                seed=seed,
                device=device,
                _init_setup_model=_init_setup_model,
            )
            print("SACCEPO is used")

            # Add Cross-Entropy Method parameters
            self.ce_N = ce_N # sample number N
            self.ce_ne = ce_ne # number of elites
            self.ce_t = ce_t # number of iterations
            self.ce_size = ce_size # elite fraction
            

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        
        optimizers = [self.actor.mean_network.optimizer, self.actor.std_network.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_mean_losses, actor_std_losses, critic_losses = [], [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # CROSS ENTROPY METHOD UPDATE:
            observations = replay_data.observations
            with th.no_grad():
                mean_actions, log_std, _ = self.actor.get_action_dist_params(observations)
                location, static_log_std = mean_actions.detach(), log_std.detach() # .exp().detach()
                # scale = th.zeros(batch_size, self.actor.action_dim).fill_(self.ce_size)
                scale = th.full_like(location, self.ce_size).detach()

                original_shape = self.ce_N, batch_size
                compressed_shape = self.ce_N * batch_size
                observation_reshape = observations.expand(self.ce_N, self.batch_size, observations.shape[-1]).reshape(compressed_shape, observations.shape[-1])
                # expanded_observations = observations.expand(self.ce_N, batch_size, -1).reshape(self.ce_N * batch_size, -1)

                # search for optimal mean
                for t in range(self.ce_t):
                    # Bound normal distribution with 3 stddev
                    scale = th.clamp(scale, min=1e-6)

                    upper_bound = location + scale * 3
                    lower_bound = location - scale * 3
                    samples = th.clamp(Normal(location, scale).sample((self.ce_N,)), lower_bound, upper_bound)

                    # Evaluate samples 
                    mean = samples
                    
                    # actions, log_prob = self.actor.action_dist.log_prob_from_params(mean, static_log_std)

                    # actions_from_params internally updates the distribution (calls self.proba_distribution(mean_actions, log_std))
                    actions = self.actor.action_dist.actions_from_params(mean, static_log_std)

                    # print(f"self.actor.action_dist.distribution {self.actor.action_dist.distribution}")
                    # gaussian_actions = self.actor.action_dist.gaussian_actions
                    # if gaussian_actions is None:
                    #     gaussian_actions = TanhBijector.inverse(actions)
                    actions_raw = TanhBijector.inverse(actions)
                    log_prob = Normal(mean, static_log_std.exp()).log_prob(actions_raw) - th.log(1 - actions.pow(2) + self.actor.action_dist.epsilon)
                    log_prob = log_prob.sum(-1, keepdim=True)
                    
                    # log_prob = self.actor.action_dist.distribution.log_prob(gaussian_actions)
                    # print(f"gaussian_actions.shape {gaussian_actions.shape}")
                    # print(f"log_prob.shape {log_prob.shape}")
                    # print(f"actions.shape {actions.shape}")
                    # crucial change for CEM: set dim=-1
                    # log_prob -= th.sum(th.log(1 - actions**2 + self.actor.action_dist.epsilon), dim=-1, keepdim=True)

                    action_reshape = actions.reshape(compressed_shape, self.actor.action_dim)
                    # Compute Q values
                    q_values = th.cat(self.critic(observation_reshape, action_reshape), dim=1)
                    min_q_values, _ = th.min(q_values, dim=1, keepdim=True)
                    min_q_values = min_q_values.reshape(original_shape[0], original_shape[1], 1)

                    loss = (ent_coef * log_prob - min_q_values).mean(dim=1).detach()
                    # print(f"loss.shape {loss.shape}")

                    _, indices = th.topk(loss, k=self.ce_ne, largest=False, dim=0)
                    # print(f"indices.shape {indices.shape}")
                    elite_samples = th.index_select(samples, dim=0, index=indices.flatten())
                    # print(f"elite_samples.shape {elite_samples.shape}")
                    location = elite_samples.mean(dim=0)
                    scale = elite_samples.std(dim=0)
                target_mean = location
                assert target_mean.shape == static_log_std.shape

            # Train Policy Mean network based on target_mean
            predicted_mean = self.actor.mean_network(observations)
            actor_mean_loss = F.mse_loss(predicted_mean, target_mean)
            self.actor.mean_network.optimizer.zero_grad()
            actor_mean_loss.backward()
            self.actor.mean_network.optimizer.step()

            # Train Policy Std Network
            actions, log_prob = self.actor.action_log_prob(observations)
            q_values_pi = th.cat(self.critic(observations, actions), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_std_loss = (ent_coef * log_prob - min_qf_pi).mean()
            
            self.actor.std_network.optimizer.zero_grad()
            actor_std_loss.backward()
            self.actor.std_network.optimizer.step()
            # CEM METHOD END

            # Compute actor loss (DEFAULT SB3)
            # q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            # min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            # actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_mean_losses.append(actor_mean_loss.item())
            actor_std_losses.append(actor_std_loss.item())

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_mean_loss", np.mean(actor_mean_losses))
        self.logger.record("train/actor_std_loss", np.mean(actor_std_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

