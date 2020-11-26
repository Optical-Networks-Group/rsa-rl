
from rsarl.data import Action
from rsarl.utils import cal_slot, sort_tuple
from rsarl.algorithms import SpectrumAssignment, Routing


class Agent(object):

    def __init__(self):
        pass

    def act(self, observation):
        raise NotImplementedError

    def batch_act(self, observations: list):
        return [self.act(o) for o in observations]


class KSPAgent(Agent):

    def __init__(self, k: int):
        """
        Args:
            k (int): The number of paths to be considered. 
        """
        self.k = k
        self.path_table = {}


    def prepare_ksp_table(self, net):
        """Prepare k-shortest path table to shorten exec time. 
            Args:
                net (Network): The target network            
        """
        for s in range(net.n_nodes):
            for d in range(net.n_nodes):
                if s < d:
                    paths = Routing.k_shortest_paths(net, s, d, self.k, is_weight=net.is_weight)
                    self.path_table[(s, d)] = paths


class PrioritizedKSPAgent(KSPAgent):

    def __init__(self, k):
        """
        Args:
            k (int): The number of paths to be considered. 
        """
        super().__init__(k)


    def assign_spectrum(self, net, path: list, n_req_slot: int) -> int:
        raise NotImplementedError


    def act(self, observation):
        # get current network
        net = observation.net
        # generate current request
        src, dst, bandwidth, duration = observation.request

        sd_tuple = (src, dst)
        paths = self.path_table[sort_tuple(sd_tuple)]

        # Search assignable path & slot
        for path in paths:
            # physical length of the path
            path_len = net.distance(path)
            # number of requred slots
            n_req_slot = cal_slot(bandwidth, path_len)

            # spectrum assignment
            slot_idx = self.assign_spectrum(net, path, n_req_slot)

            if slot_idx is not None:
                return Action(path, slot_idx, n_req_slot, duration)

        return None


class KSPDRLAgent(KSPAgent):

    def __init__(self, k: int, drl):
        super().__init__(k)
        self.drl = drl

    def preprocess(self, obs):
        """Convert observation(net, request) to feature vector

        """
        raise NotImplementedError

    def map_drlout_to_action(self, obs, drl_out):
        """Mapping outputs of DRL agent to RSA actions

        """
        raise NotImplementedError

    def observe(self, obs, reward, done, reset):
        self.drl.observe(self.preprocess(obs), reward, done, reset)
    
    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        obs = [self.preprocess(o) for o in batch_obs]
        self.drl.batch_observe(obs, batch_reward, batch_done, batch_reset)

    def act(self, obs):
        act = self.drl.act(self.preprocess(obs))
        return self.map_drlout_to_action(obs, act)

    def batch_act(self, batch_obs):
        obs = [self.preprocess(o) for o in batch_obs]
        drl_outs = self.drl.batch_act(obs)
        acts = [self.map_drlout_to_action(obs, out) for obs, out in zip(batch_obs, drl_outs)]
        return acts

