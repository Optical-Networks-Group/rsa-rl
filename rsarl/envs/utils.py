

import copy

def make_env(org_env, seed, test):
    # Use different random seeds for train and test envs
    env_seed = 2 ** 31 - 1 - seed if test else seed
    # copy
    env = copy.deepcopy(org_env)
    env.seed(env_seed)
    return env
