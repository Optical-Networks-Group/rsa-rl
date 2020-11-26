

import pytest
from rsarl.utils import cal_slot

cal_slot_test_data = [
    # minimum bandwidth`
    (25, 625, 0, 1),
    (25, 1250, 0, 1),
    (25, 2500, 0, 1),
    (25, 3000, 0, 2),
    # maximum bandwidth`
    (100, 625, 0, 2),
    (100, 1250, 0, 3),
    (100, 2500, 0, 4),
    (100, 3000, 0, 8),
]
@pytest.mark.parametrize("bandwidth, path_len, guard, expect", cal_slot_test_data)
def test_cal_slot(bandwidth, path_len, guard, expect):
    assert cal_slot(bandwidth, path_len, guard=guard) == expect

