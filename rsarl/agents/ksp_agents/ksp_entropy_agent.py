


from rsarl.algorithms import SpectrumAssignment
from rsarl.agents import PrioritizedKSPAgent



class KSP_EntropyAgent(PrioritizedKSPAgent):
    """Entropy-based RSA Agent with K-Shortest Path

        Add priority of k-shortest paths to 
        original entropy agent in the paper: https://ieeexplore.ieee.org/document/6647621

        Args:
            k (int): The number of paths to be considered.

        Attributes:
            k (int): The number of paths to be considered.

    """

    def __init__(self, k: int, mode: str="edge"):
        super().__init__(k)
        # the number of paths to consider
        self.mode = mode

    def assign_spectrum(self, net, path: list, n_req_slot: int) -> int:
        # spectrum utilization on the whole path
        path_slot = net.path_slot(path)
        # search 
        slot_index = SpectrumAssignment.entropy(net, path, n_req_slot, self.mode)
        return slot_index


