
from abc import ABCMeta, abstractmethod
from rsarl.data import Action
from rsarl.agents import KSPDRLAgent
from rsarl.utils import cal_slot, sort_tuple
from rsarl.algorithms import SpectrumAssignment

class RoutingAgent(KSPDRLAgent, metaclass=ABCMeta):
    """
    """

    def __init__(self, k: int, drl):
        super().__init__(k, drl)

    @abstractmethod
    def preprocess(self, obs):
        """
        Args:
            obs (~object): Observation.

        Returns:
            fvec (np.array): feature vector
        """
        raise NotImplementedError()

    def map_drlout_to_action(self, obs, out):
        """Mapping RL outputs to KSP

        """
        net = obs.net
        s, d, bandwidth, duration = obs.request
        paths = self.path_table[sort_tuple((s, d))]
        # map
        path = paths[out]

        #required slots
        path_len = net.distance(path)
        n_req_slot = cal_slot(bandwidth, path_len)
        # FF
        path_slot = net.path_slot(path)
        slot_index = SpectrumAssignment.first_fit(path_slot, n_req_slot)
        if slot_index is None:
            return None
        else:
            return Action(path, slot_index, n_req_slot, duration)
        


