
import pytest
from rsarl.algorithms import Routing

def test_dump_and_load(net):
    # Assign path
    s, d = 0, 3
    start_idx = 0
    req_n_slot = 3
    duration = 2.0
    path = Routing.shortest_path(net, s, d)
    # pre-status
    pre_slot_dict = net.slot
    pre_time_dict = net.time
    # Assign path
    net.assign_path(path, start_idx, req_n_slot, duration)
    # dump & load
    data = net.dump_json()
    net.load_json(data)
    # status
    slot_dict = net.slot
    time_dict = net.time
    # check 
    for e in net.G.edges():
        assert pre_slot_dict[e] == slot_dict[e]
        assert pre_time_dict[e] == time_dict[e]


