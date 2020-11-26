import numpy as np

import pfrl
from rsarl.envs import make_env

class SerialVectorEnv(pfrl.envs.SerialVectorEnv):
    """VectorEnv where each env is run sequentially.

    The purpose of this VectorEnv is to help debugging. For speed, you should
    use MultiprocessVectorEnv if possible.

    Args:
        env_fns (list of gym.Env): List of gym.Env.
    """

    def __init__(self, envs):
        self.envs = envs
        self.last_obs = [None] * self.num_envs


def make_serial_vector_env(env, n_env, base_seed, test):
    process_seeds = np.arange(n_env) + base_seed * n_env
    return SerialVectorEnv(
        [make_env(env, process_seeds[idx], test) for idx in range(n_env)]
    )
