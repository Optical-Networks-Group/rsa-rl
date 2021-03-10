
import numpy as np
from rsarl.agents.drl_agents import RoutingAgent


def vectorize(n_nodes: int, node_id: int):
    mp = np.eye(n_nodes, dtype=np.float32)[node_id].reshape(-1, 1, 1)
    return mp


class DeepRMSAv1Agent(RoutingAgent):
    """DeepRMSAv1 Agent

        paper: Deep-RMSA: A Deep-Reinforcement-Learning Routing, 
            Modulation and Spectrum Assignment Agent for Elastic Optical Networks
            (https://doi.org/10.1364/OFC.2018.W4F.2)

    """

    def preprocess(self, obs):
        """
        """
        net = obs.net
        source, destination, bandwidth, duration = obs.request
        # slot table
        whole_slot = np.array(list(net.slot.values()))
        whole_slot = whole_slot.reshape(1, net.n_edges, net.n_slot).astype(np.float32)
        # source, destination, bandwidth map
        smap = np.ones_like(whole_slot) * vectorize(net.n_nodes, source)
        dmap = np.ones_like(whole_slot) * vectorize(net.n_nodes, destination)
        bmap = np.ones_like(whole_slot) * bandwidth
        # concate: (1, ICH, #edges, #slots)
        fvec = np.concatenate([whole_slot, smap, dmap, bmap], axis=0)
        return fvec.astype(np.float32, copy=False)

