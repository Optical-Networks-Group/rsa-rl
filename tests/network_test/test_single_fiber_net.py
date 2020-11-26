
import pytest
import random
import networkx as nx

from rsarl.utils import path_to_edges
from rsarl.algorithms import Routing
from rsarl.networks import SingleFiberNetwork


def check_is_weight(net, is_weight = False):
    w_dict = nx.get_edge_attributes(net.G, "weight")
    if is_weight:
        assert len(w_dict) != 0, f"network {net.name} does not have weight attr"
    else:
        assert len(w_dict) == 0, f"network {net.name} must not have weight attr"


def test_build_network():
    """
    Test Network.build_network()
    is_weight is available to NationalScienceFoundation only
    """    
    n_slot = 10

    # check is_weight flag works well
    net = SingleFiberNetwork("nsf", n_slot=n_slot, is_weight=False)
    check_is_weight(net, False)
    net = SingleFiberNetwork("nsf", n_slot=n_slot, is_weight=True)
    check_is_weight(net, True)


def test_distance(net):
    # Gen samples
    for _ in range(10):
        s, d = random.randint(0, net.n_nodes - 1), random.randint(0, net.n_nodes - 1)
        path = Routing.shortest_path(net, s, d)

        distance = net.distance(path)
        ans_distance = nx.dijkstra_path_length(net.G, s, d, weight="weight")
        assert distance == ans_distance



def test_init_graph_slot(net):
    acc = [1 for _ in range(net.n_slot)]

    # Test initial state
    slot_dict = net.slot
    for v in slot_dict.values():
        assert v == acc, f"Failure to initialize slot table: {v} != {acc}"

    # Assume occupied network
    sample_slot = [0,0,0,0,0,0,0,0,0,0]
    for e in net.G.edges():
        for s in range(net.n_slot):
            slot_dict[e][s] = sample_slot[s]

    # Test init_network method
    net.init_graph()
    slot_dict = net.slot
    # check init
    for v in slot_dict.values():
        assert v == acc, f"Failure to initialize slot table: {v} != {acc}"


def test_init_graph_time(net):
    acc = [1 for _ in range(net.n_slot)]

    # Test initial state
    acc = [0 for _ in range(net.n_slot)]
    time_dict = nx.get_edge_attributes(net.G, name='time')
    for v in time_dict.values():
        assert v == acc, f"Failure to initialize time: {v} != {acc}"
    
    # Assume occupied network
    sample_time = [2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0]
    for e in net.G.edges():
        for s in range(net.n_slot):
            time_dict[e][s] = sample_time[s]
    
    # Test init_network method
    net.init_graph()
    time_dict = nx.get_edge_attributes(net.G, name='time')
    for v in time_dict.values():
        assert v == acc, f"Failure to initialize time: {v} != {acc}"


def test_spend_time(net):
    # Gen samples
    s, d = 0, 3
    start_idx = 0
    req_n_slot = 3
    duration = 2.0
    path = Routing.shortest_path(net, s, d)
    # Assign path
    net.assign_path(path, start_idx, req_n_slot, duration)

    # Spend time
    net.spend_time(1.2)

    slot_dict = net.slot
    time_dict = net.time

    # ans
    ans_slot = [0,0,0,1,1,1,1,1,1,1]
    ans_time = [0.8,0.8,0.8,0,0,0,0,0,0,0]

    init_slot = [1,1,1,1,1,1,1,1,1,1]
    init_time = [0,0,0,0,0,0,0,0,0,0]

    path_edges = path_to_edges(path)

    # check edges which are not on the path
    for e in net.G.edges():
        if e in path_edges:
            assert slot_dict[e] == ans_slot, f"Failure to assign path in slot: {slot_dict[e]} != {ans_slot}"
            assert time_dict[e] == ans_time, f"Failure to assign path in time: {time_dict[e]} != {ans_time}"
        else:
            assert slot_dict[e] == init_slot, f"Assign wrong path in slot: {slot_dict[e]} != {init_slot}"
            assert time_dict[e] == init_time, f"Assign wrong path in time: {time_dict[e]} != {init_time}"


def test_path_slot(net):
    # Gen samples
    s, d = 0, 3
    path = Routing.shortest_path(net, s, d)
    path_edges = path_to_edges(path)
    # print(path_edges) # [(0, 1), (1, 3)]
    slot_dict = net.slot
    # 
    ans_slot_1 = [1,1,1,1,1,0,0,0,0,0]
    ans_slot_2 = [0,0,1,1,1,1,1,0,0,0]
    i = 0
    for e in net.G.edges():
        if e in path_edges:
            i += 1
            if i == 1:
                for s in range(net.n_slot):
                    slot_dict[e][s] = ans_slot_1[s]
            elif i == 2:
                for s in range(net.n_slot):
                    slot_dict[e][s] = ans_slot_2[s]

    path_slot = net.path_slot(path)
    from bitarray import bitarray
    ans = bitarray([0,0,1,1,1,0,0,0,0,0])
    assert ans == path_slot, f"Failure to calculate path-slot: {path_slot} != ~{ans}"


def test_is_assignable(net):
    # Gen samples
    s, d = 0, 3
    start_idx = 0
    req_n_slot = 3
    duration = 2.0
    path = Routing.shortest_path(net, s, d)

    # Assign path
    net.assign_path(path, start_idx, req_n_slot, duration)

    # Test occupied
    assert net.is_assignable(path, start_idx=0, n_req_slot=2) is False, f"Assignable judge is wrong"
    assert net.is_assignable(path, start_idx=1, n_req_slot=2) is False, f"Assignable judge is wrong"
    assert net.is_assignable(path, start_idx=2, n_req_slot=2) is False, f"Assignable judge is wrong"
    # Test available
    assert net.is_assignable(path, start_idx=3, n_req_slot=2) is True, f"Assignable judge is wrong"
    assert net.is_assignable(path, start_idx=4, n_req_slot=2) is True, f"Assignable judge is wrong"
    assert net.is_assignable(path, start_idx=8, n_req_slot=2) is True, f"Assignable judge is wrong"
    # Test exceeds the amount of slots
    assert net.is_assignable(path, start_idx=3, n_req_slot=8) is False, f"Assignable judge is wrong"
    assert net.is_assignable(path, start_idx=4, n_req_slot=20) is False, f"Assignable judge is wrong"
    assert net.is_assignable(path, start_idx=8, n_req_slot=5) is False, f"Assignable judge is wrong"


def check_assign_path(net):
    # Gen samples
    s, d = 0, 3
    start_idx = 0
    req_n_slot = 3
    duration = 2.0
    path = Routing.shortest_path(net, s, d)

    # Assign path
    net.assign_path(path, start_idx, req_n_slot, duration)

    slot_dict = net.slot
    time_dict = net.time

    # ans
    ans_slot = [0,0,0,1,1,1,1,1,1,1]
    ans_time = [2.0,2.0,2.0,0,0,0,0,0,0,0]

    init_slot = [1,1,1,1,1,1,1,1,1,1]
    init_time = [0,0,0,0,0,0,0,0,0,0]

    path_edges = path_to_edges(path)

    # check edges which are not on the path
    for e in net.G.edges():
        if e in path_edges:
            assert slot_dict[e] == ans_slot, f"Failure to assign path in slot: {slot_dict[e]} != {ans_slot}"
            assert time_dict[e] == ans_time, f"Failure to assign path in time: {time_dict[e]} != {ans_time}"
        else:
            assert slot_dict[e] == init_slot, f"Assign wrong path in slot: {slot_dict[e]} != {init_slot}"
            assert time_dict[e] == init_time, f"Assign wrong path in time: {time_dict[e]} != {init_time}"


def test_resource_util(net):
    slot_dict = net.slot
    time_dict = net.time
    # Occupy half slots of all edges
    ans_slot = [1,1,1,1,1,0,0,0,0,0]
    ans_time = [1,1,1,1,1,0,0,0,0,0]
    for e in net.G.edges():
        for s in range(net.n_slot):
            slot_dict[e][s] = ans_slot[s]
            time_dict[e][s] = ans_time[s]

    assert net.resource_util() == 0.5, f"Failure to calculate resource util: {net.resource_util()} != 0.5"

