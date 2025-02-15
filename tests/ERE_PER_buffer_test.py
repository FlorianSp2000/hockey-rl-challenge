from dotenv import load_dotenv
load_dotenv()
import numpy as np
import pytest
from stable_baselines3.common.buffers import ReplayBuffer
from src.custom_sb3.replay_buffer import EREBuffer 
import hockey.hockey_env as h_env
from scipy import stats
import numpy as np


def test_fifo_behavior(buffer_size=1000):
    """Verify that the buffer correctly follows FIFO behavior."""
    env = h_env.HockeyEnv()

    buffer = EREBuffer(
        buffer_size=buffer_size, device="cpu", cmin=500, eta0=0.996, etaT=1, 
        total_timesteps=1_000_000, use_per=True, alpha=0.6, 
        observation_space=env.observation_space, action_space=env.action_space
    )

    # Fill buffer beyond its capacity (twice)
    num_total_samples = buffer_size * 2
    for i in range(num_total_samples):
        buffer.add(
            np.array([i] * env.observation_space.shape[0]), 
            np.array([i+1] * env.observation_space.shape[0]), 
            np.array([0] * env.action_space.shape[0]), 
            np.array([0]), 
            np.array([False]), 
            [{}]
        )

    # Check that the oldest entries are removed
    stored_data = np.array([obs[0] for obs in buffer.observations[:,0,:]])  # Extract stored observations
    expected_start = num_total_samples - buffer_size  # FIFO means we should keep the latest buffer_size elements
    assert np.all(stored_data == np.arange(expected_start, num_total_samples)), "FIFO property violated!"

    # Ensure the buffer position is at zero after full wrap-around
    assert buffer.pos == 0, "Buffer position should wrap around to 0 after full rotation."


@pytest.mark.parametrize("n_envs, recent_size, batch_size, alpha", [
    (4, 10, 5, 0.6),   # Standard test case
    (3, 15, 7, 0.8),   # Different env and buffer sizes
    (5, 20, 10, 1.0),  # Edge case with alpha=1
])
def test_sampling_indices(n_envs, recent_size, batch_size, alpha):
    # Create mock priorities
    np.random.seed(42)  # For reproducibility
    priorities = np.random.rand(recent_size, n_envs)

    # Select recent indices (assuming they are the last `recent_size` indices in a buffer)
    recent_indices = np.arange(recent_size)

    # Compute probabilities
    all_priorities = priorities[recent_indices].flatten()
    probs = all_priorities ** alpha
    probs /= np.sum(probs)  # Normalize

    # Check shape consistency
    assert len(probs) == recent_size * n_envs, "Probs and recent indices do not match"

    # Sample flat indices
    flat_indices = np.random.choice(len(probs), size=batch_size, p=probs)

    # Recover buffer and env indices
    buffer_indices = recent_indices[flat_indices // n_envs]
    env_indices = flat_indices % n_envs

    # Assertions to verify correctness
    assert buffer_indices.shape == (batch_size,), "buffer_indices has incorrect shape"
    assert env_indices.shape == (batch_size,), "env_indices has incorrect shape"
    assert np.all(buffer_indices >= 0) and np.all(buffer_indices < recent_size), "Invalid buffer indices"
    assert np.all(env_indices >= 0) and np.all(env_indices < n_envs), "Invalid env indices"


@pytest.mark.parametrize("buffer_size, total_timesteps, cmin", [
    (1000, 10000, 500),   # Standard test case
])
def test_eta_schedule(buffer_size, total_timesteps, cmin):
    # """Ensure eta follows the expected schedule."""
    env = h_env.HockeyEnv()
    buffer = EREBuffer(buffer_size=buffer_size, total_timesteps=total_timesteps, cmin=cmin, use_per=True, observation_space=env.observation_space, action_space=env.action_space, device='cpu')
    eta_values = [buffer.get_eta(t) for t in range(0, total_timesteps+1, 500)]
    assert eta_values[0] == pytest.approx(buffer.eta0, rel=1e-2)
    assert eta_values[-1] == pytest.approx(buffer.etaT, rel=1e-2)

@pytest.mark.parametrize("buffer_size, total_timesteps, cmin", [
    (1000, 10000, 500),   # Standard test case
])
def test_beta_schedule(buffer_size, total_timesteps, cmin):
    """Ensure beta follows the expected schedule for PER."""
    env = h_env.HockeyEnv()
    buffer = EREBuffer(buffer_size=buffer_size, total_timesteps=10000, cmin=cmin, use_per=True, observation_space=env.observation_space, action_space=env.action_space, device='cpu')
    beta_values = [buffer.get_beta(t) for t in range(0, total_timesteps+1, 500)]
    assert beta_values[0] == pytest.approx(buffer.beta0, rel=1e-2)
    assert beta_values[-1] == pytest.approx(buffer.betaT, rel=1e-2)

@pytest.mark.parametrize("buffer_size, batch_size, K", [
    (1000, 100, 10),   
    (1000, 100, 30),
    (1000, 256, 50),
])
def test_ere_sampling(buffer_size, batch_size, K):
    """Verify that ERE samples more recent experiences as k increases."""
    env = h_env.HockeyEnv()
    buffer = EREBuffer(buffer_size=buffer_size, observation_space=env.observation_space, action_space=env.action_space, device='cpu', cmin=500, eta0=0.996, etaT=1, total_timesteps=1_000_000)
    for i in range(buffer_size):
        # Write the timestep when observation is added as observation itself to track sample recency
        buffer.add(np.array([i]*env.observation_space.shape[0]), np.array([i+1]*env.observation_space.shape[0]), np.array([0]*env.action_space.shape[0]), np.array([0]), np.array([False]), [{}])

    k_values = np.sort(np.random.choice(K,3,replace=False))
    sampled_indices = [buffer.sample(batch_size, k=k, K=K, current_timestep=0)[0].observations.numpy() for k in k_values] # leaving current_timestep at 0 fixes eta
    assert np.mean(sampled_indices[2]) > np.mean(sampled_indices[0])  # Later k should sample more recent indices

@pytest.mark.parametrize("buffer_size, batch_size, K", [
    (1000, 100, 10),   
    (1000, 300, 30),
    (1000, 500, 50),
])
def test_per_sampling(buffer_size, batch_size, K):
    """Verify that PER samples high-priority experiences more often."""
    env = h_env.HockeyEnv()
    buffer = EREBuffer(buffer_size=buffer_size, device='cpu', cmin=500, eta0=0.996, etaT=1, total_timesteps=1_000_000, use_per=True, alpha=0.6, observation_space=env.observation_space, action_space=env.action_space)
    for i in range(buffer_size):
        buffer.add(np.array([i]*env.observation_space.shape[0]), np.array([i+1]*env.observation_space.shape[0]), np.array([0]*env.action_space.shape[0]), np.array([0]), np.array([False]), [{}])
        buffer.priorities[i, 0] = (i + 1) ** buffer.alpha  # Set priorities increasing

    sampled_indices = buffer.sample(batch_size, current_timestep=0, k=0, K=K)[1] # set k=0 for c_k=buffer_size
    assert np.median(sampled_indices) > buffer_size // 2  # More recent indices should be sampled more due to higher priority
    statistic, p_value = stats.kstest(sampled_indices, 'uniform')
    assert p_value < 0.05  # Reject null hypothesis of uniformity

if __name__ == "__main__":
    pytest.main([__file__])
