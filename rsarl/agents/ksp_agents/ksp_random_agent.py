
from rsarl.algorithms import SpectrumAssignment
from rsarl.agents import PrioritizedKSPAgent

class KSP_RANDOM_Agent(PrioritizedKSPAgent):
    """K-Shortest paths & Random Agent

        Args:
            k (int): The number of paths to be considered.

        Attributes:
            k (int): The number of paths to be considered.
    """

    def __init__(self, k: int):
        super().__init__(k)

    def assign_spectrum(self, net, path: list, n_req_slot: int) -> int:
        # spectrum utilization on the whole path
        path_slot = net.path_slot(path)
        # search 
        slot_index = SpectrumAssignment.random(path_slot, n_req_slot)
        return slot_index
