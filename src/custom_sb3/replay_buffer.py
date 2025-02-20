import numpy as np
from typing import Optional
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import ReplayBuffer

from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
)

class EREBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device, n_envs=1, 
                 optimize_memory_usage=False, handle_timeout_termination=True, 
                 eta0=0.996, etaT=1, total_timesteps=1_000_000, cmin=5000, alpha=0.6, use_per=False, beta0=0.4, betaT=1):
        super().__init__(buffer_size, observation_space, action_space, device, 
                         n_envs, optimize_memory_usage, handle_timeout_termination)
        print(f"EREBuffer Init with: use_per={use_per}, alpha={alpha}, eta0={eta0}, etaT={etaT}, cmin={cmin}, total_timesteps={total_timesteps}")
        self.eta0 = eta0
        self.etaT = etaT
        self.cmin = cmin 
        self.total_timesteps = total_timesteps
        self.max_eps_length = 250
        self.alpha = alpha  # Priority exponent (PER)
        self.priorities = np.zeros((buffer_size, self.n_envs), dtype=np.float32)  # Priority storage: one priority per transition per env
        # self.priorities = np.zeros(buffer_size, dtype=np.float32)  # Priority storage
        self.use_per = use_per
        self.beta0 = beta0
        self.betaT = betaT
        print(f"self.n_envs {self.n_envs}")
        assert self.buffer_size * self.n_envs > self.cmin, "Buffer size too small for cmin"

    def get_eta(self, current_timestep):
        """Compute annealed eta using linear schedule."""
        eta = self.eta0 + (self.etaT - self.eta0) * (current_timestep / self.total_timesteps)
        return min(self.etaT, max(self.eta0, eta))  # Clamp within [etaT, eta0]

    def get_beta(self, current_timestep):
        """Compute beta for PER using linear schedule"""
        beta = self.beta0 + (self.betaT - self.beta0) * (current_timestep / self.total_timesteps)
        return min(self.betaT, max(self.beta0, beta))  # Clamp within [0, 1]

    def add(self, *args, **kwargs):
        """Add a new experience to the buffer with a priority based on TD error."""
        index = self.pos # Parent add method increments self.pos so save before
        super().add(*args, **kwargs)
        # self.priorities[index] = max(self.priorities) if max(self.priorities) !=0 else 1   # New samples receive high priority
        self.priorities[index, :] = np.max(self.priorities) if np.max(self.priorities) != 0 else 1

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None, env_indices: Optional[np.ndarray] = None) -> ReplayBufferSamples:
        """
        Get samples from the replay buffer with specified indices.
        
        Args:
            batch_inds: Indices in the buffer to sample
            env: Optional VecNormalize environment for normalization
            env_indices: Optional array of environment indices (used for PER)
                        If None, randomly sample environment indices
        """
        # If env_indices not provided (non-PER case), sample randomly
        if env_indices is None:
            env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        
        # the rest is as in original implementation:
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


    def sample(self, batch_size: int, env: Optional[VecNormalize] = None, k: int = 0, K: int = 0, current_timestep: int = 0):
        N = self.size() * self.n_envs

        if self.use_per:
            samples, batch_inds, weights, env_indices = self._sample_recent(batch_size, N, k, K, current_timestep, env=env)
            return samples, batch_inds, weights, env_indices
        else:
            batch_inds = self._sample_recent(batch_size, N, k, K, current_timestep, env=env)
            return self._get_samples(batch_inds, env=env), batch_inds, None, None  # Return indices too (but no weights and env_indices)


    def _sample_recent(self, batch_size: int, N: int, k: int, K: int, current_timestep: int, env: Optional[VecNormalize] = None):
        # Emphasizing Recent Experience (ERE)
        eta = self.get_eta(current_timestep)  # Use annealed eta
        ck = max(int(N * (eta ** (k * self.max_eps_length / K ))), self.cmin) // self.n_envs  # Make sure that recency bias comes into effect when we have collected c_k transitions in total
        # print(f"ck: {ck}, N: {N}, k: {k}, K: {K}, max_eps_length: {self.max_eps_length}")
        # Handle wrap around case
        if self.full and self.pos - ck < 0:
            recent_indices = np.concatenate([
                np.arange(self.buffer_size - (ck - self.pos), self.buffer_size), 
                np.arange(0, self.pos)
            ])

            assert len(recent_indices) == ck, "Not sampling correct number of indices"
        else:
            recent_indices = np.arange(max(0, self.pos - ck), self.pos)
        
        # Prioritized Experience Replay (PER)
        if self.use_per:
            all_priorities = self.priorities[recent_indices].flatten()
            probs = all_priorities ** self.alpha
            # probs = self.priorities[recent_indices] ** self.alpha
            probs /= np.sum(probs)  # Normalize

            # Sample indices and corresponding env indices
            num_transitions = len(probs)
            assert num_transitions == len(recent_indices) * self.n_envs, "Probs and recent indices do not match"

            flat_indices = np.random.choice(num_transitions, size=batch_size, p=probs)
            buffer_indices = recent_indices[flat_indices // self.n_envs]
            env_indices = flat_indices % self.n_envs

            # Compute importance sampling weights
            beta = self.get_beta(current_timestep)

            # print(f"samples.shape {samples.shape}")
            weights  = (num_transitions * probs[flat_indices]) ** (-beta)
            # normalize weights

            weights /= weights.max() 
            weights  = np.array(weights, dtype=np.float32) 
            
            samples = self._get_samples(buffer_indices, env=env, env_indices=env_indices)

            return samples, buffer_indices, weights, env_indices
        
        else:
            return np.random.choice(recent_indices, size=batch_size)
        
