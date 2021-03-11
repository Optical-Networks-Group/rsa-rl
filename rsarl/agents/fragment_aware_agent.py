
import numpy as np
from rsarl.data import Action
from rsarl.agents import KSPAgent
from rsarl.utils import cal_slot, sort_tuple, k_consecutive_available_slot
from rsarl.utils.fragmentation import count_cut, count_misalignment
from rsarl.algorithms import SpectrumAssignment

class FragmentAwareAgent(KSPAgent):
    """Fragment-aware Agent with K-Shortest Path

    Paper: Spectral and Spatial 2D Fragmentation-Aware Routing 
        and Spectrum Assignment Algorithms in Elastic Optical Networks [Invited]
        (https://doi.org/10.1364/JOCN.5.00A100)

    Args:
        k (int): The number of paths to be considered.

    Attributes:
        k (int): The number of paths to be considered.

    """

    def __init__(self, k):
        super().__init__(k)

    def min_cut_candidates(self, obs) -> list:
        net = obs.net
        # generate current request
        src, dst, bandwidth, duration = obs.request
        # get pre-calculated k-sp path
        paths = self.path_table[sort_tuple((src, dst))]

        candidates = []
        min_n_cut = np.inf
        for path in paths:
            # calculate candidates of spectrum assignment
            path_slot = net.path_slot(path)
            path_len = net.distance(path)  # physical length of the path
            n_req_slot = cal_slot(bandwidth, path_len)
            num, start_indices, _ = k_consecutive_available_slot(path_slot, n_req_slot)
            
            if num > 0:
                # explore all candidates
                for start_idx in start_indices:
                    n_cut = count_cut(net, path, start_idx, n_req_slot)
                    if n_cut < min_n_cut:
                        min_n_cut = n_cut
                        # empty
                        candidates = []
                        # append  
                        cand = Action(path, start_idx, n_req_slot, duration)
                        candidates.append(cand)
                    elif n_cut == min_n_cut:
                        # append  
                        cand = Action(path, start_idx, n_req_slot, duration)
                        candidates.append(cand)

        return candidates


    def min_misalignment(self, obs, min_cut_candidates: list) -> list:
        candidates = []
        min_misalign_cnt = np.inf
        for cand in min_cut_candidates:
            n_misalign = count_misalignment(obs.net, cand.path, cand.slot_idx, cand.n_slot)
            
            if n_misalign < min_misalign_cnt:
                min_misalign_cnt = n_misalign
                # empty
                candidates = []
                candidates.append(cand)
            elif n_misalign == min_misalign_cnt:
                candidates.append(cand)

        return candidates
    

    def sp_ff(self, obs):
        """Shortest path and first fit

        """
        net = obs.net
        src, dst, bandwidth, duration = obs.request
        # select shortest path
        path = self.path_table[sort_tuple((src, dst))][0]
        path_len = net.distance(path)
        n_req_slot = cal_slot(bandwidth, path_len)
        # target path slot
        path_slot = net.path_slot(path)
        # first fit
        start_idx = SpectrumAssignment.first_fit(path_slot, n_req_slot)
        if start_idx is None:
            return None
        else:
            return Action(path, start_idx, n_req_slot, duration)


    def act(self, obs):

        # 1st step: search min cut candidates
        min_cut_candidates = self.min_cut_candidates(obs)

        if min_cut_candidates == []: # there is no assignable candidates
            return None
        elif len(min_cut_candidates) == 1:
            return min_cut_candidates[0]

        # 2nd step: search min misalignment candidates
        min_misalign_candidates = self.min_misalignment(obs, min_cut_candidates)

        if min_misalign_candidates == []:
            raise ValueError("ERROR")
        elif len(min_misalign_candidates) == 1:
            return min_misalign_candidates[0]

        # 3rd step: SP-FF
        return self.sp_ff(obs)

