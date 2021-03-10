
import numpy as np
from rsarl.agents.drl_agents import RoutingAgent
from rsarl.utils import cal_slot, sort_tuple, onehot_list, k_consecutive_available_slot


class DeepRMSAv2Agent(RoutingAgent):
    """DeepRMSAv2 Agent

        paper: DeepRMSA: A Deep Reinforcement Learning Framework for Routing, 
                Modulation and Spectrum Assignment in Elastic Optical Networks
                (https://ieeexplore.ieee.org/document/8738827)

        NOTE that original paper uses slot_guard = 1
    """

    def preprocess(self, obs):
        """
        """
        net = obs.net
        src, dst, bandwidth, duration = obs.request
        # get k-sp
        sd_tuple = sort_tuple((src, dst))
        paths = self.path_table[sd_tuple]

        fvec = []
        # Feature 1: onehot of source-destination nodes
        node_onehot = onehot_list(net.n_nodes)
        fvec += node_onehot[src]
        fvec += node_onehot[dst]
        # path information if there is no available path
        # all_zeros = [0 for _ in range(3 + 2)]
        all_nega_ones = [-1 for _ in range(3 + 2)]

        # Feature 2: Path information
        for path in paths:
            # spectrum utilization on the whole path
            path_slot = net.path_slot(path)
            # calc path-length and the required number of slots
            path_len = net.distance(path)
            req_n_slot = cal_slot(bandwidth, path_len)
            # wavelength assignment
            isFound, slot_start_indices, slot_continuous = k_consecutive_available_slot(path_slot, req_n_slot)

            if isFound:
                # normalized slot num is added
                # (required number of FS: 2 <= req_n_slot <= 9)
                fvec.append((req_n_slot - 5.5) / 3.5)
                # slot start idx
                fvec.append(2 * (slot_start_indices[0] - 0.5 * net.n_slot) / net.n_slot)
                # slot continue
                fvec.append((slot_continuous[0] - 8) / 8)
                # total available FS's
                fvec.append(2 * (sum(slot_continuous) - 0.5 * net.n_slot) / net.n_slot)
                # mean size of FS-blocks
                fvec.append((np.mean(slot_continuous) - 4) / 4) 
            else:
                fvec += all_nega_ones
        
        fvec = np.array(fvec, dtype=np.float32)
        return fvec

