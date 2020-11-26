

import random
import numpy as np
from rsarl.requester import Requester
from rsarl.data import Request

class UniformRequester(Requester):
    """Uniform Requester class

    The requester generate requests according to uniform distribution. 

    Args:
        n_nodes (int): The number of nodes in used network. 
        seed (int): seed to initialize the pseudo-random number generator. 

    Attributes:
        avg_service_time (int): average service time per request
        avg_request_arrival_rate (int): average number of requests per unit time

    """
    def __init__(self, n_nodes, avg_service_time, avg_request_arrival_rate, seed=0):
        super().__init__(n_nodes, seed)
        # demand settings
        self.avg_service_time = avg_service_time
        self.avg_request_arrival_rate = avg_request_arrival_rate

    def source_destination(self):
        """Generate a random pair of nodes. 

        Returns:
            tuple: a pair of sourde-destination nodes. 

        """
        idx = self.rand_generator.choice(self.pair_ids)
        s, d = self.pairs[idx]
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

