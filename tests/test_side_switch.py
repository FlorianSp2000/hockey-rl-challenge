import numpy as np
import pytest
from src.utils.env_wrapper import HockeySB3Wrapper
import hockey.hockey_env as h_env

@pytest.fixture
def env():
    """Create a fresh environment for each test"""
    base_env = h_env.HockeyEnv()
    return HockeySB3Wrapper(base_env, rank=0, opponent_type="weak", change_sides=False) # Turn off as we are manually testing

def test_side_switching_occurs(env):
    """Test that side switching actually occurs over multiple resets"""
    wrapped_env = env
    wrapped_env.change_sides = True # Turn on side switching

    # Collect sides over multiple resets
    sides = []
    for _ in range(25):
        obs, _ = wrapped_env.reset()
        sides.append(wrapped_env.playing_as_agent2)
    
    # Verify both sides are represented
    assert True in sides, "Never played as agent 2"
    assert False in sides, "Never played as agent 1"

def test_opponent_consistency(env):
    """Test that opponent behaves consistently regardless of side"""
    wrapped_env = env
    
    # Test opponent observation storage
    for side in [True, False]:
        wrapped_env.playing_as_agent2 = side
        obs, _ = wrapped_env.reset()
        
        if side:
            # When playing as agent 2, main obs should match player 2's obs
            assert np.array_equal(wrapped_env.env.obs_agent_two(), obs), \
                "Opponent observation incorrect when playing as agent 2"
        else:
            # When playing as agent 1, opponent_obs should be from agent 2 perspective
            assert np.array_equal(wrapped_env.obs_opponent, wrapped_env.env.obs_agent_two()), \
                "Opponent observation incorrect when playing as agent 1"
            
def test_reward_consistency(env):
    """Play a full episode and check reward consistency."""
    wrapped_env = env
    wrapped_env.playing_as_agent2 = True  # Play as agent 2 for this test
    obs, _ = wrapped_env.reset()

    total_reward = 0
    done = False
    while not done:
        action = np.random.uniform(-1, 1, size=(4,))  # Random valid action
        obs, reward, done, truncated, info = wrapped_env.step(action)

        assert isinstance(reward, (int, float)), f"Invalid reward type: {type(reward)}"
        assert not np.isnan(reward), "Reward is NaN"
        total_reward = reward
    info_2 = wrapped_env.env.get_info_agent_two()
    assert info['winner'] == info_2['winner'], "Winner is not consistent"


if __name__ == "__main__":
    test_opponent_consistency(env())