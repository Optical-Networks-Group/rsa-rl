
import random
import itertools
import numpy as np
from abc import ABCMeta, abstractmethod
from rsarl.utils import sort_tuple

class Requester(metaclass=ABCMeta):
    def __init__(self, n_nodes, seed=0):
        # rand gen
        self._seed = seed
        self.rand_generator = np.random.RandomState(seed)
        # calc all pairs of nodes
        self.n_nodes = n_nodes
        self.nodes = list(range(self.n_nodes))
        _pairs = list(itertools.permutations(self.nodes, 2))
        # remove the same edge e.g. (1, 0) and (0, 1) -> (0, 1)
        self.pairs = list(set([sort_tuple(l) for l in _pairs]))
        self.pair_ids = list(range(len(self.pairs)))


    def init(self):
        self.rand_generator = np.random.RandomState(self._seed)


    def seed(self, s):
        """Set seed.

            Args:
                seed (int): seed to initialize the pseudo-random number generator. 
        """
        self._seed = s
        self.rand_generator = np.random.RandomState(self._seed)


    @abstractmethod
    def request(self):
        raise NotImplementedError

