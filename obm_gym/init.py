from gym.envs.registration import register

register(
    id='obm-v0',
    entry_point='omb_gym.envs:ObmEnv'
)