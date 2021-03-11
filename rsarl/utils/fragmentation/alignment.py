
from bitarray import bitarray
from bitarray.util import count_xor
from rsarl.utils import copy_and_assign_slot, sort_tuple, path_to_edges

def misalignment(target: bitarray, neighbor: bitarray, start_idx: int, req_n_slot: int):
    """Calculate change of misalighment.

    Paper: Spectral and Spatial 2D Fragmentation-Aware Routing 
        and Spectrum Assignment Algorithms in Elastic Optical Networks [Invited]
        (https://doi.org/10.1364/JOCN.5.00A100)

    Args:
        target   (bitarray): slot bitarray.
        neighbor (bitarray): slot bitarray.
        start_idx  (int): the number of used start index.
        req_n_slot (int): the number of required slots.

    Returns:
        int: change number of misalighment

    """
    bef = count_xor(neighbor, target)
    aft = count_xor(neighbor, copy_and_assign_slot(target, start_idx, req_n_slot))
    return aft - bef


def count_misalignment(net, path: list, start_idx: int, req_n_slot: int):
    """Sum up misalignment.

    Paper: Spectral and Spatial 2D Fragmentation-Aware Routing 
        and Spectrum Assignment Algorithms in Elastic Optical Networks [Invited]
        (https://doi.org/10.1364/JOCN.5.00A100)

    Args:
        net  (object): Network object.
        path (list): list of nodes.
        start_idx  (int): the number of used start slot index.
        req_n_slot (int): the number of required slots.

    Returns:
        int: total number of misalighment

    """
    edges = path_to_edges(path)
    
    misalign_change = 0
    for e in edges:
        # search neighbors
        for node in list(e):
            for neighbor, attr in net.G[node].items():
                if sort_tuple((neighbor, node)) in edges:
                    continue
                    
                # calc misalignment
                target_slot = bitarray(net.slot[e])
                neighbor_slot = bitarray(attr["slot"])
                # sum up
                misalign_change += misalignment(target_slot, neighbor_slot, start_idx, req_n_slot)
                
    return misalign_change

    