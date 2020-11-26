
from abc import ABCMeta, abstractmethod

class Topology(metaclass=ABCMeta):
    """Topology Base class

    Attributes:
        name: 
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def weighted_edges(self):
        raise NotImplementedError

    @abstractmethod
    def edges(self):
        raise NotImplementedError

    @abstractmethod
    def nodes_2D_pos(self):
        raise NotImplementedError
