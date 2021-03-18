

import numpy as np
from typing import NamedTuple
from collections import defaultdict

from rsarl.utils import list_to_str
from rsarl.data import Experience

def create_experience(req_id: int, obs, act, is_success: bool, reward: float) -> NamedTuple:
    exp = Experience(
        request_id = req_id,
        # request info
        source = obs.request.source,
        destination = obs.request.destination,
        bandwidth = obs.request.bandwidth,
        duration = obs.request.duration,
        # action info
        path = None if act is None else list_to_str(act.path),
        slot_index = None if act is None else act.slot_idx,
        n_slot = None if act is None else act.n_slot,
        # result
        is_success = is_success,
        reward = reward,
        # pre-state
        network = obs.net.dump_json(),
        slot_utilization = obs.net.resource_util()
    )
    return exp


def evaluation(env, agent, n_requests: int) -> tuple:
    """
    """
    logs = []
    obs = env.last_obs
    for req_id in range(n_requests):
        # Get action from observation
        act = agent.act(obs)
        # Do action and get next state
        next_obs, reward, done, info = env.step(act)
        # Store log
        exp = create_experience(req_id, obs, act, info["is_success"], reward)
        logs.append(exp)
        # Store next state
        if done:
            obs = env.reset()
        else:
            obs = next_obs

    return logs


def batch_evaluation(vec_env, agent, n_requests: int) -> tuple:
    """
    """
    experience_lists = defaultdict(lambda: [])
    obss = vec_env.last_obs
    # Generate requests
    for req_id in range(n_requests):
        # Get action from observation
        acts = agent.batch_act(obss)
        # Do action and get next state
        _, rewards, dones, infos = vec_env.step(acts)
        # Store log
        for i, (act, info, obs, rw) in enumerate(zip(acts, infos, obss, rewards)):
            exp = create_experience(req_id, obs, act, info["is_success"], rw)
            experience_lists[i].append(exp)

        # reset
        not_end = np.logical_not(dones)
        obss = vec_env.reset(not_end)

    return experience_lists


def warming_up(env, agent, n_requests: int):
    """
    """
    obs = env.last_obs
    for _ in range(n_requests):
        # Get action from observation
        act = agent.act(obs)
        # Do action and get next state
        obs, _, done, _ = env.step(act)
        # Store next state
        if done:
            obs = env.reset()


def batch_warming_up(vec_env, agent, n_requests: int):
    """
    """
    obss = vec_env.last_obs
    for _ in range(n_requests):
        # Get action from observation
        acts = agent.batch_act(obss)
        # Do action and get next state
        obss, _, dones, _ = vec_env.step(acts)
        # create mask to reset
        not_end = np.logical_not(dones)
        obss = vec_env.reset(not_end)


def summary(experiences):
    """
    """
    # bp
    n_requests = len(experiences)
    n_blocking = np.sum([0 if x.is_success else 1 for x in experiences])
    block_prob = n_blocking / n_requests * 100
    # others
    avg_util = np.average([x.slot_utilization for x in experiences])
    total_reward = np.sum([x.reward for x in experiences])
    return block_prob, avg_util, total_reward


def batch_summary(experiences):
    """
    """
    blocking_probs = []
    avg_utils = []
    total_rewards = []
    # calc performance
    for env_id, exps in experiences.items():
        # bp
        n_requests = len(exps)
        n_blocking = np.sum([0 if x.is_success else 1 for x in exps])
        block_prob = n_blocking / n_requests * 100
        blocking_probs.append(block_prob)
        # other
        avg_utils.append(np.average([x.slot_utilization for x in exps]))
        total_rewards.append(np.sum([x.reward for x in exps]))

    return blocking_probs, avg_utils, total_rewards


def train(vec_env, agent, train_steps):
    obses = vec_env.last_obs
    resets = [False for _ in range(len(obses))]

    for _ in range(train_steps):
        acts = agent.batch_act(obses)
        obses, rews, dones, infos = vec_env.step(acts)
        agent.batch_observe(obses, rews, dones, resets)
        # Make mask(not_end). 0 if done/reset, 1 if pass
        end = np.logical_or(resets, dones)
        not_end = np.logical_not(end)
        obses = vec_env.reset(not_end)


def train_eval_loop(
    vec_env, 
    agent, 
    evaluator,
    n_loop: int, 
    train_loop: int, 
):
    vec_env.reset()
    for _ in range(n_loop):
        train(vec_env, agent, train_loop)
        evaluator(agent)


class Evaluator():

    def __init__(
        self,
        test_env,
        warming_up_steps=3000,
        evalutate_steps=10000,
        logger=None,
    ):
        self.env = test_env
        self.warming_up_steps = warming_up_steps
        self.evalutate_steps = evalutate_steps
        self.logger = logger


    def evaluate(self, agent):
        self.env.reset()
        # eval
        batch_warming_up(self.env, agent, n_requests=self.warming_up_steps)
        experiences = batch_evaluation(self.env, agent, n_requests=self.evalutate_steps)
        # calc metrics
        blocking_probs, avg_utils, total_rewards = batch_summary(experiences)
        # logger
        if self.logger is not None:
            self.logger(agent, experiences, blocking_probs, avg_utils, total_rewards)

        return np.average(blocking_probs)


    def __call__(self, agent):
        if hasattr(agent, "drl"):
            with agent.drl.eval_mode():
                self.evaluate(agent)
        else:
            self.evaluate(agent)


