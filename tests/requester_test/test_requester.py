
import copy
from rsarl.requester import UniformRequester

def test_requester():
    n_nodes = 10

    # Uniform Requester: Test init method
    requester1 = UniformRequester(n_nodes, 10, 12, seed=0)
    requester2 = UniformRequester(n_nodes, 10, 12, seed=0)
    s, d, bandwidth, duration = requester1.request()
    s2, d2, bandwidth2, duration2 = requester2.request()
    assert s == s2
    assert d == d2
    assert bandwidth == bandwidth2
    assert duration == duration2

    # init func test
    requester1 = UniformRequester(n_nodes, 10, 12, seed=0)
    s, d, bandwidth, duration = requester1.request()
    requester1.init()
    s2, d2, bandwidth2, duration2 = requester1.request()
    assert s == s2
    assert d == d2
    assert bandwidth == bandwidth2
    assert duration == duration2

    # interval check
    interval = requester1.time_interval()
    interval2 = requester2.time_interval()
    assert interval == interval2

    # deep copy check
    requester1 = UniformRequester(n_nodes, 10, 12, seed=0)
    requester2 = copy.deepcopy(requester1)
    s, d, bandwidth, duration = requester1.request()
    s2, d2, bandwidth2, duration2 = requester2.request()
    assert s == s2
    assert d == d2
    assert bandwidth == bandwidth2
    assert duration == duration2

