

import numpy as np
from bitarray import bitarray
from rsarl.utils import path_to_edges, assignable_indices, k_consecutive_available_slot, copy_and_assign_slot


def entropy(path_slot: bitarray) -> float:
    """The Shannon entropy spectrum fragmentation metric

        Paper: https://ieeexplore.ieee.org/document/6647621

        Args:
            path_slot (bitarray): slot bitarray of target path or edge

    """
    # search assignable vector slot greater than 0
    _, _, slot_lens = k_consecutive_available_slot(path_slot, 1)
    n_slot = len(path_slot)
    ent = sum([(l / n_slot) * np.log(n_slot / l) for l in slot_lens])
    return ent


def _path_based_entropy(path_slot: bitarray, n_req_slot: int) -> np.ndarray:
    """The original functions for RSA using Entropy metric
    
        Paper: https://ieeexplore.ieee.org/document/6647621
    
        Args:
            path_slot (bitarray): slot bitarray of target path or edge.
            n_req_slot (int): required number of slot to assign path. 

    """
    target_indices = assignable_indices(path_slot, n_req_slot)
    # calculate entropy after assignment
    each_case_diff_entropy = []
    
    # calc entropy of current path-slot
    base_entropy = entropy(path_slot)

    for assignable_idx in target_indices:
        # temporary path assignment
        temp_path_slot = copy_and_assign_slot(path_slot, assignable_idx, n_req_slot)
        # search assignable slot vector
        aft_entropy = entropy(temp_path_slot)
        each_case_diff_entropy.append(aft_entropy-base_entropy)
    
    # if not assignable, the elements are np.inf
    entropy_vector = np.ones((len(path_slot),)) * np.inf    
    each_case_diff_entropy = np.array(each_case_diff_entropy)
    entropy_vector[target_indices] = each_case_diff_entropy

    return entropy_vector.astype(np.float32)


def path_based_entropy(net, path: list, n_req_slot: int) -> np.ndarray:
    """The original functions for RSA using Entropy metric

        Paper: https://ieeexplore.ieee.org/document/6647621

        Args:
            net: Network.
            path (list): List of node-ids.
            n_req_slot (int): required number of slot to assign path. 
    
    """
    path_slot = net.get_path_slot(path)
    return _path_based_entropy(path_slot, n_req_slot)


def edge_based_entropy(net, path: list, n_req_slot: int) -> np.ndarray:
    """The original functions for RSA using Entropy metric
    
        Paper: https://ieeexplore.ieee.org/document/6647621

        Args:
            net: Network.
            path (list): List of node-ids.
            n_req_slot (int): required number of slot to assign path. 
    
    """
    slot_dict = net.slot
    edges = path_to_edges(path)
    total_entropy = np.zeros(net.n_slot,)
    for e in edges:
        path_slot = bitarray(slot_dict[e])
        path_vec_ent = _path_based_entropy(path_slot, n_req_slot)
        total_entropy += path_vec_ent

    return total_entropy


