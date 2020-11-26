
import pytest
import numpy as np
from bitarray import bitarray
from rsarl.utils import sort_tuple, onehot_list, path_to_edges, list_to_str, str_to_list, copy_and_assign_slot

sort_tuple_test_data = [
    ((1, 2), (1, 2)),
    ((2, 0), (0, 2)),
    ((-2, 0), (-2, 0)),
    ((0, -2), (-2, 0)),
]
@pytest.mark.parametrize("t, expect", sort_tuple_test_data)
def test_sort_tuple(t, expect):
    assert sort_tuple(t) == expect

test_list_to_str_data = [
    ([1,2], "1,2"),
    ([3,2,4,5], "3,2,4,5"),
]
@pytest.mark.parametrize("l, expect", test_list_to_str_data)
def test_list_to_str(l, expect):
    assert list_to_str(l) == expect


test_str_to_list_data = [
    ("1,2", [1,2]),
    ("3,2,4,5", [3,2,4,5]),
]
@pytest.mark.parametrize("s, expect", test_str_to_list_data)
def test_str_to_list(s, expect):
    assert str_to_list(s) == expect


onehot_list_test_data = [
    (1, [[1]]),
    (2, [[1,0],[0,1]]),
    (3, [[1,0,0],[0,1,0],[0,0,1]]),
]
@pytest.mark.parametrize("n, expect", onehot_list_test_data)
def test_onehot_list(n, expect):
    assert onehot_list(n) == expect


path_to_edges_test_data = [
    ([1, 2, 3], [(1, 2), (2, 3)]),
    ([1, 3, 2], [(1, 3), (2, 3)]),
    ([5, 3, 0], [(3, 5), (0, 3)]),
]
@pytest.mark.parametrize("path, expect", path_to_edges_test_data)
def test_path_to_edges(path, expect):
    assert path_to_edges(path) == expect


assign_test_data = [
    ([1,1,1,1,1,1,1,1,1,1], 0, 3, [0,0,0,1,1,1,1,1,1,1]),
    ([1,1,1,1,1,1,1,1,1,1], 3, 2, [1,1,1,0,0,1,1,1,1,1]),
]
@pytest.mark.parametrize("slot, st_idx, n, expect", assign_test_data)
def test_assign_slot(slot, st_idx, n, expect):
    slot = bitarray(slot)
    o_slot = copy_and_assign_slot(slot, st_idx, n)
    assert o_slot == bitarray(expect)
    assert o_slot != slot

