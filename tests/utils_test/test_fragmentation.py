
import pytest
import numpy as np
from bitarray import bitarray
from rsarl.utils.fragmentation.entropy import entropy, _path_based_entropy
from rsarl.utils.fragmentation import is_cut, misalignment


ent_test_data = [
    ([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 0.0),
    ([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 0.0),
    ([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1], 0.34657359027997264),
    ([0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1], 0.6931471805599453),
    ([0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1], 1.0397207708399179),
    ([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1], 1.3862943611198906),
    ([0,0,0,1,1,1,0,1,1,0,1,1,1,0,0,0], 0.8876713552993563),
]
@pytest.mark.parametrize("path_slot, expect", ent_test_data)
def test_entropy(path_slot, expect):
    path_slot = bitarray(path_slot)
    assert entropy(path_slot) == expect


ent_test_data = [
    ([0,0,0,1,1,1,0,1,1,0,1,1,1,0,0,0], 2, [np.inf,np.inf,np.inf,-0.14058379,-0.14058379,np.inf,np.inf,-0.25993019,np.inf,np.inf,-0.14058379,-0.14058379,np.inf,np.inf,np.inf,np.inf]),
    ([0,0,1,1,1,1,0,1,1,0,1,1,1,1,0,0], 2, [np.inf,np.inf,-0.0866434,0.,-0.0866434,np.inf,np.inf,-0.25993019,np.inf,np.inf,-0.0866434 ,0.,-0.0866434,np.inf,np.inf,np.inf]),
    ([0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,1], 2, [np.inf,np.inf,np.inf,np.inf,-0.25993019,np.inf,np.inf,0.03802954,0.21745543,0.29977249,0.33680184,0.33680184,0.29977249,0.21745543,0.03802954,np.inf]),
]
@pytest.mark.parametrize("path_slot, n_req_slot, expect", ent_test_data)
def test_path_based_entropy(path_slot, n_req_slot, expect):
    path_slot = bitarray(path_slot)
    vent = _path_based_entropy(path_slot, n_req_slot)
    assert np.allclose(vent, expect)


is_cut_test_data = [
    ([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 0, 1, False),
    ([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 1, 1, True),
    ([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 14, 1, True),
    ([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 15, 1, False),
    ([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 0, 3, False),
    ([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 3, 3, True),
    ([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 13, 3, False),
]
@pytest.mark.parametrize("path_slot, start_idx, req_n_slot, expect", is_cut_test_data)
def test_is_cut(path_slot, start_idx, req_n_slot, expect):
    path_slot = bitarray(path_slot)
    assert is_cut(path_slot, start_idx, req_n_slot) == expect


misallignment_test_data = [
    # increase
    ([1,1,1,1], [1,1,1,1], 0, 1, 1), 
    ([1,1,1,1], [1,1,1,1], 0, 2, 2), 
    ([1,1,1,1], [1,1,1,1], 0, 3, 3), 
    ([1,1,1,1], [1,1,1,1], 0, 4, 4), 
    # decrease
    ([1,1,1,1], [0,0,0,0], 0, 1, -1), 
    ([1,1,1,1], [0,0,0,0], 0, 2, -2), 
    ([1,1,1,1], [0,0,0,0], 0, 3, -3), 
    ([1,1,1,1], [0,0,0,0], 0, 4, -4), 
    # 
    ([1,1,1,1,1,0,0,0,1,1,1,0], [1,1,0,0,1,1,0,0,1,1,0,0], 2, 1, -1), 
]
@pytest.mark.parametrize("path_slot, neighbor_slot, start_idx, req_n_slot, expect", misallignment_test_data)
def test_misallignment(path_slot, neighbor_slot, start_idx, req_n_slot, expect):
    path_slot = bitarray(path_slot)
    neighbor_slot = bitarray(neighbor_slot)
    assert misalignment(path_slot, neighbor_slot, start_idx, req_n_slot) == expect

