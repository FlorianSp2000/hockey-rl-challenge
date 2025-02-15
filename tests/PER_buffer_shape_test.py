import numpy as np
import pytest

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

if __name__ == "__main__":
    pytest.main([__file__])
