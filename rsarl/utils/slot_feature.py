

import re
from bitarray import bitarray


def assignable_indices(slot: bitarray, n: int) -> list:
    """Search assignable indices of argment's slot by the number of required slots. 

    Args:
        slot (bitarray): slot bitarray.
        n (int): the number of required slots.

    Returns:
        list: all indices of assignable slot

    """
    req_slot = bitarray([1] * n)
    start_idx = list(slot.itersearch(req_slot))
    return start_idx


def k_consecutive_available_slot(slot: bitarray, k: int) -> tuple:
    """Search vectors of assignable slot whose length is greater than argment n. 

    Args:
        slot (bitarray): slot bitarray.
        k (int): length of vector.

    Returns:
        tuple: each element is: 
            1st (int): the number of vectors
            2nd (list): list of start indices of found vectors
            3rd (list): list of lengths of found vectors
        
    """
    # convert to str
    str_slot = slot.to01()
    base_idx = 0
    reg = re.compile("(1)\\1{%d,}" % (k - 1)) 
    
    start_idx = []
    slot_continue = []
    
    while True:
        m = reg.search(str_slot)
        if not m:
            break
        
        start_idx.append(base_idx + m.start())
        slot_continue.append(m.end()-m.start())
        
        base_idx += m.end()
        str_slot = str_slot[m.end():]
    
    return len(start_idx), start_idx, slot_continue

