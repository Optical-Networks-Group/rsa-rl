
import math
import copy
import numpy as np
from bitarray import bitarray
from collections import defaultdict
from networkx.utils import pairwise


def sort_tuple(t: tuple) -> tuple:
    """Sort tuple. 

    Args:
        t (tuple): tuple

    Returns:
        tuple: sorted tuple

    """
    return tuple(sorted(t))


def list_to_str(l: list) -> str:
    """Convert list to str. 

    Args:
        l (list): to convert path to str-path

    Returns:
        str: list in str type

    Examples:
        >>> list_to_str([1,2,3])
        '1,2,3'

    """
    return ','.join(map(str, l))


def str_to_list(s: str) -> list:
    """Convert str-list to list. 

    Args:
        s (str): to convert str-path to path

    Returns:
        list:

    Examples:
        >>> str_to_list('1,2,3')
        [1, 2, 3]

    """
    return [int(n) for n in s.split(",")]


def bitarray2nparray(b: bitarray) -> np.ndarray:
    """Convert bitarray to np.ndarray. 

    Args:
        b (bitarray): usually slot in bitarray type

    Returns:
        np.ndarray: ndarray consisting of 0 and 1. 

    """
    return np.array(b.tolist()).astype(np.float32)


def get_mean_std(bp_per_batch: defaultdict) -> tuple:
    """Calculate mean and std values from dict. 

    Args:
        bp_per_batch (defaultdict): key is "batch" and value is blocking probabilities(when using several seeds)

    Returns:
        1st (np.ndarray): mean values whose element size is the number of "batch"
        2nd (np.ndarray): std values whose element size is the number of "batch"

    """
    y_mean = []
    y_std = []
    for _, v in bp_per_batch.items():
        y_mean.append(np.mean(v))
        y_std.append(np.std(v))

    return np.array(y_mean), np.array(y_std)


def onehot_list(num: int) -> list:
    """Generate one hot list. 

    Args:
        num (int): the number of elements

    Returns:
        list: 2d-list(num x num)

    """
    return np.eye(num, dtype=np.int8).tolist()


def path_to_edges(path: list) -> list:
    """Convert path to edges. 

    Args:
        path (list): List of node-ids.

    Returns:
        list: list of edges on the args's path.

    Examples:
        >>> path = [node_id1, node_id2, ...]
        >>> print(path_to_edges(path))
        [(node_id1, node_id2), (node_id2, node_id3), ...]
    
    """
    return [ sort_tuple(t) for t in list(pairwise( path )) ]


def copy_and_assign_slot(slot: bitarray, start_idx: int, n_req_slot: int) -> bitarray:
    """Copy slot and assign path to copied slot.

    Args:
        slot (bitarray): slot bitarray of target path or edge
        start_idx (int): start index of slot table. 
        n_req_slot (int): required number of slot to assign path.           

    """
    temp_path_slot = copy.deepcopy(slot)
    temp_path_slot[start_idx: start_idx + n_req_slot] = 0
    return temp_path_slot



