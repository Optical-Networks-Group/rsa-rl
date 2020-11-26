import signal
import warnings
from multiprocessing import Pipe, Process

import numpy as np
from torch.distributions.utils import lazy_property

import pfrl
import functools
from rsarl.envs import make_env


def worker(remote, env_fn):
    # Ignore CTRL+C in the worker process
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                ob, reward, done, info = env.step(data)
                remote.send((ob, reward, done, info))
            elif cmd == "reset":
                ob = env.reset()
                remote.send(ob)
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "seed":
                remote.send(env.seed(data))
            else:
                raise NotImplementedError
    finally:
        env.close()


class MultiprocessVectorEnv(pfrl.envs.MultiprocessVectorEnv):

    def __init__(self, env_fns):
        if np.__version__ == "1.16.0":
            warnings.warn(
                """
NumPy 1.16.0 can cause severe memory leak in pfrl.envs.MultiprocessVectorEnv.
We recommend using other versions of NumPy.
See https://github.com/numpy/numpy/issues/12793 for details.
"""
            )  # NOQA

        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(target=worker, args=(work_remote, env_fn))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)
        ]
        for p in self.ps:
            p.start()
        self.last_obs = [None] * self.num_envs
        self.closed = False

    def __del__(self):
        if not self.closed:
            self.close()

    @lazy_property
    def spec(self):
        raise NotImplementedError


def make_multiprocess_vector_env(env, n_env, base_seed, test):
    process_seeds = np.arange(n_env) + base_seed * n_env
    return MultiprocessVectorEnv(
        [
            functools.partial(make_env, env, process_seeds[idx], test)
            for idx in range(n_env)
        ]
    )