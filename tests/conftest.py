

import pytest
from rsarl.networks import SingleFiberNetwork
from rsarl.envs import DeepRMSAEnv
from rsarl.requester import UniformRequester

n_slot = 10
n_nodes = 14

@pytest.fixture
def net():
    _net = SingleFiberNetwork("nsf", n_slot=n_slot, is_weight=True)
    return _net


@pytest.fixture
def requester():
    _requester = UniformRequester(n_nodes, 10, 12)
    return _requester


@pytest.fixture
def env():
    net = SingleFiberNetwork("nsf", n_slot=n_slot, is_weight=True)
    requester = UniformRequester(n_nodes, 10, 12)
    _env = DeepRMSAEnv(net, requester)
    return _env


