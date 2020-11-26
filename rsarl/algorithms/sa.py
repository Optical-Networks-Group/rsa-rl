
import random
import numpy as np
from bitarray import bitarray
from rsarl.utils import assignable_indices
from rsarl.utils.fragmentation import path_based_entropy, edge_based_entropy


class SpectrumAssignment:
    """ Spectrum Assignment Class 
    
    """
    @staticmethod
    def random(slot: bitarray, n_req_slot: int) -> int:
        """Random algorithm searches assignable indices
        Args:
            slot (bitarray): slot bitarray.
            n_req_slot (int): the number of required slots.

        Returns:
            int: index   
            
        """
        slot_indices = assignable_indices(slot, n_req_slot)
        if slot_indices:
            return random.choice(slot_indices)
        else:
            return None


    @staticmethod
    def first_fit(slot: bitarray, n_req_slot: int) -> int:
        """ First-fit algorithm searches assignable indices. 

        Args:
            slot (bitarray): slot bitarray.
            n_req_slot (int): the number of required slots.

        Returns:
            int: index
        
        """
        slot_indices = assignable_indices(slot, n_req_slot)

        if slot_indices:
            return slot_indices[0]
        else:
            return None


    @staticmethod
    def entropy(net, path: list, n_req_slot: int, mode: str="edge") -> int:
        """Original entropy in the paper: 
            https://ieeexplore.ieee.org/document/6647621

        Args:
            net: Network 
            path: list of node id
            n_req_slot (int): the number of required slots.            
            mode (str): calculation mode of entropy, "edge" or "path".

        Returns:
            int: index        

        """
        if mode == "path":
            ent = path_based_entropy(net, path, n_req_slot)
        elif mode == "edge":
            ent = edge_based_entropy(net, path, n_req_slot)
        else:
            raise ValueError("Modes are 'path' or 'edge'")

        min_ent = np.min(ent)
        slot_index = np.argmin(ent)

        if not min_ent == np.inf:
            return int(slot_index)
        else:
            return None

