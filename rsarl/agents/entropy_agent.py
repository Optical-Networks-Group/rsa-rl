
import numpy as np
from rsarl.data import Action
from rsarl.agents import KSPAgent
from rsarl.utils import cal_slot, sort_tuple
from rsarl.utils.fragmentation import edge_based_entropy


class EntropyAgent(KSPAgent):
    """Entropy-based RSA Agent with K-Shortest Path

        Paper: Simulation results of Shannon entropy based 
            flexgrid routing and spectrum assignment on a real network topology
            (https://ieeexplore.ieee.org/document/6647621)

        Args:
            k (int): The number of paths to be considered.

        Attributes:
            k (int): The number of paths to be considered.

    """

    def __init__(self, k):
        super().__init__(k)


    def act(self, observation):
        # get current network
        net = observation.net
        # generate current request
        src, dst, bandwidth, duration = observation.request
        # get pre-calculated k-sp path
        sd_tuple = (src, dst)
        paths = self.path_table[sort_tuple(sd_tuple)]

        # Search KSP-FF
        candidates = []
        for i, path in enumerate(paths):
            path_len = net.distance(path)  # physical length of the path
            n_req_slot = cal_slot(bandwidth, path_len)
            # calc entropy
            ent = edge_based_entropy(net, path, n_req_slot)
            min_ent = np.min(ent)
            slot_index = np.argmin(ent)
            # candidate (k-path, slot-idx, n_req_slot, entropy)
            candidates.append((i, int(slot_index), n_req_slot, min_ent))

        # search the minimum entropy among k-sp
        i_th, start_idx, n_req_slot, _ = min(candidates, key=lambda item:item[3])
        path = paths[i_th]

        act = Action(path, start_idx, n_req_slot, duration)
        return act

