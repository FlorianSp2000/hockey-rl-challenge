from gymnasium.envs.registration import register
# Reregister the environment as old registry is invalid (laserhockey instead of hockey module)
register(
    id='Hockey-v1',
    entry_point='hockey.hockey_env:HockeyEnv',
    kwargs={'mode': 0}
)