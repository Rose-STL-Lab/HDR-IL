from gym.envs.registration import register

register(
    id='bimanual_task-v0',
    entry_point='bimanual.envs:BimanualEnv',
)
