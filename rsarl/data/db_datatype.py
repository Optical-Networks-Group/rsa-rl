

import datetime
from typing import NamedTuple


class DBExperiment(NamedTuple):
    # primary key
    experiment_name: str
    # exp settings
    environment_name: str
    network_name: str
    requester_name: str
    agent_name: str
    hyper_parameters: str
    #
    created_at: datetime.datetime


class DBEvaluation(NamedTuple):
    # primary key(exp_name, seed, batch)
    experiment_name: str
    env_id: int
    batch: int
    # results
    blocking_prob: float
    slot_utilization: float
    total_reward: float


class DBExperience(NamedTuple):
    # primary key(exp_name, request_id)
    experiment_name: str
    request_id: int
    # elements: experience
    source: int
    destination: int
    bandwidth: int
    duration: float
    path: str
    slot_index: int
    n_slot: int
    is_success: bool
    reward: int
    network: str
    slot_utilization: float

