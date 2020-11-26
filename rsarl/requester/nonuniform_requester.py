
import random
import numpy as np
from rsarl.requester import Requester
from rsarl.data import Request

class NonuniformRequester(Requester):
    """Nonuniform Requester class

    The requester generate requests according to nonuniform distribution. 

    Args:
        n_nodes (int): The number of nodes in used network. 
        seed (int): seed to initialize the pseudo-random number generator. 

    Attributes:
        avg_service_time (int): average service time per request
        avg_request_arrival_rate (int): average number of requests per unit time

    """
    def __init__(self, n_nodes, node_select_prob, avg_service_time, avg_request_arrival_rate, seed=0):

        assert len(node_select_prob) == n_nodes

        super().__init__(n_nodes, seed)
        # demand settings
        self.avg_service_time = avg_service_time
        self.avg_request_arrival_rate = avg_request_arrival_rate
        #
        self.node_select_prob = node_select_prob


    def source_destination(self):
        """Generate a random pair of nodes. 

        Returns:
            tuple: a pair of sourde-destination nodes. 

        """
        s = random.choices(self.nodes, weights=self.node_select_prob)[0]
        # delete prob of source node
        copy_node_select_prob = np.copy(self.node_select_prob)
        copy_node_select_prob[s] = 0.
        copy_node_select_prob = copy_node_select_prob / np.sum(copy_node_select_prob)
        d = random.choices(self.nodes, weights=copy_node_select_prob)[0]
        return s, d

    def duration(self) -> float:
        """Generate duration time. 

        Returns:
            float: duration time. 

        """
        return self.rand_generator.exponential(self.avg_service_time)

    def bandwidth(self): 
        """Generate bandwidth. 

        Returns:
            int: bandwidth. 

        """
        return self.rand_generator.randint(25, 101)

    def time_interval(self):
        """Generate time interval between requests

        Returns:
        float: time interval
        
        """
        return self.rand_generator.exponential(1 / self.avg_request_arrival_rate)


    def request(self):
        """Generate request. 

        Returns:
            networkingrl.data.Request: request in namedtuple format. 

        """
        s, d = self.source_destination()
        return Request(s, d, self.bandwidth(), self.duration())

