
from bitarray import bitarray
from rsarl.utils import path_to_edges, k_consecutive_available_slot


def is_cut(slot_vec: bitarray, start_idx: int, req_n_slot: int):
    """Check whether assignment candidates break the spectrum contiguousness or not. 

    Paper: Spectral and Spatial 2D Fragmentation-Aware Routing 
        and Spectrum Assignment Algorithms in Elastic Optical Networks [Invited]
        (https://doi.org/10.1364/JOCN.5.00A100)

    Args:
        slot_vec (bitarray): slot bitarray.
        start_idx  (int): the number of used start index.
        req_n_slot (int): the number of required slots.

    Returns:
        bool: True if cut happens; otherwise False
        
    """
    # calc boundary idx
    _, st_idx1, _ = k_consecutive_available_slot(slot_vec, 1)
    _, st_idx2, _ = k_consecutive_available_slot(slot_vec[::-1], 1)
    n_slot = len(slot_vec)
    st_idx2 = [n_slot - i - 1 for i in st_idx2]
    boundary_idx = st_idx1 + st_idx2
    # check start-idx and end-idx is located at boundary
    end_idx = start_idx + req_n_slot - 1
    # check
    assert (0 <= start_idx) and (end_idx < n_slot)
    
    if start_idx in boundary_idx or end_idx in boundary_idx:
        return False # not-cut
    else:
        return True # cut


def count_cut(net, path: list, start_idx: int, n_req_slot: int):
    """
    Args:

        start_idx  (int): the number of used start index.
        req_n_slot (int): the number of required slots.

    Returns:
        int: the number of cut.

    """
    count = 0
    for e in path_to_edges(path):
        slot = bitarray(net.slot[e])
        if is_cut(slot, start_idx, n_req_slot):
            count += 1
            
    return count

