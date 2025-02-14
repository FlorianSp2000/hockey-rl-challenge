from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
from typing import Optional
from stable_baselines3.common.vec_env import VecNormalize

class EREBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device, n_envs=1, 
                 optimize_memory_usage=False, handle_timeout_termination=True, 
                 eta0=0.996, etaT=1, total_timesteps=1_000_000, cmin=5000):
        super().__init__(buffer_size, observation_space, action_space, device, 
                         n_envs, optimize_memory_usage, handle_timeout_termination)
        print(f"Using EREBuffer with: eta0={eta0}, etaT={etaT}, cmin={cmin}, total_timesteps={total_timesteps}")
        self.eta0 = eta0
        self.etaT = etaT
        self.cmin = cmin
        self.total_timesteps = total_timesteps
        self.max_eps_length = 250


    def get_eta(self, current_timestep):
        """Compute annealed eta using linear schedule."""
        eta = self.eta0 + (self.etaT - self.eta0) * (current_timestep / self.total_timesteps)
        return min(self.etaT, max(self.eta0, eta))  # Clamp within [etaT, eta0]

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None, k: int = 0, K: int = 0, current_timestep: int = 0):
        upper_bound = self.size() #self.pos if self.full else self.pos
        batch_inds = self._sample_recent(batch_size, upper_bound, k, K, current_timestep)
        return self._get_samples(batch_inds, env=env)

    def _sample_recent(self, batch_size: int, upper_bound: int, k: int, K: int, current_timestep: int):
        eta = self.get_eta(current_timestep)  # Use annealed eta
        # max(c_k, c_min) with c_k=upper_bound * eta^(batch_size/1000) with upper_bound being replay buffer size
        ck = max(int(upper_bound * (eta ** (k * self.max_eps_length / K ))), self.cmin) 
        # print(f"ck: {ck}, upper_bound: {upper_bound}, k: {k}, K: {K}, max_eps_length: {self.max_eps_length}")
        # Handle wrap around case
        if self.full and self.pos - ck < 0:
            recent_indices = np.concatenate([
                np.arange(self.buffer_size - (ck - self.pos), self.buffer_size), 
                np.arange(0, self.pos)
            ])

            assert len(recent_indices) == ck, "Not sampling correct number of indices"
        else:
            recent_indices = np.arange(max(0, self.pos - ck), self.pos)
        
        return np.random.choice(recent_indices, size=batch_size)