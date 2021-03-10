
import math

def cal_slot(bandwidth: int, path_len: int, guard: int=0) -> int:
    """Calculate the number of required slots. 

    Args:
        bandwidth (int): required bandwidth
        path_len (int): length of required path
        guard (int): slot guard

    Returns:
        int: the number of required slot

    """
    if path_len <= 625:
        n_slot = math.ceil(bandwidth/(4*12.5)) + guard
    elif path_len <= 1250:
        n_slot = math.ceil(bandwidth/(3*12.5)) + guard   
    elif path_len <= 2500:
        n_slot = math.ceil(bandwidth/(2*12.5)) + guard
    else:
        n_slot = math.ceil(bandwidth/(1*12.5)) + guard
    return int(n_slot)


