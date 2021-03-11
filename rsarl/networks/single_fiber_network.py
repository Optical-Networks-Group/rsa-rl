

import itertools
import numpy as np
import networkx as nx
from bitarray import bitarray

from rsarl.networks import Network
from rsarl.utils import sort_tuple, path_to_edges


class SingleFiberNetwork(Network):
    """Network class

    Args:
        topology_name (str): network topology name.
        n_slot (int): The number of slots in each edge.
        is_weight (bool): whether to consider distance or not

    Attributes:
        n_slot (int): The number of slots in each edge.

    """

    def __init__(self, topology_name: str, n_slot: int, is_weight: bool):
        super().__init__(topology_name, n_slot, is_weight)
        self.add_attribute_to_graph()


    def add_attribute_to_graph(self): 
        """ Add attributes, 'slot' and 'time', to all edges in graph self.G. 

            Note:
                Slot 1 is available, otherwise occupied.

        """
        # add attr
        slot_dict = dict(zip(self.G.edges(), 
            [[1 for x in range(self.n_slot)] for y in range(self.n_edges)]))
        time_dict = dict(zip(self.G.edges(), 
            [[0 for x in range(self.n_slot)] for y in range(self.n_edges)]))
        # add edge attributes
        nx.set_edge_attributes(self.G, name='slot', values=slot_dict)
        nx.set_edge_attributes(self.G, name='time', values=time_dict)

    def init_graph(self):
        """ Initialize attribute, slot and time.

        """
        self.add_attribute_to_graph()


    @property
    def slot(self):
        return nx.get_edge_attributes(self.G, name='slot')


    @property
    def time(self):
        return nx.get_edge_attributes(self.G, name='time')


    def spend_time(self, period: float):
        """Spend time between requests. 

            Args:
                period (float): Time between requests. 
        
        """
        slot_dict = nx.get_edge_attributes(self.G, name='slot')
        time_dict = nx.get_edge_attributes(self.G, name='time')

        for e in self.G.edges():
            for s in range(self.n_slot):
                if time_dict[e][s] > period:
                    time_dict[e][s] -= period

                elif time_dict[e][s] > 0:
                    time_dict[e][s] = 0
                    slot_dict[e][s] = 1


    def distance(self, path: list) -> int:
        """Calculate distance of the path. 

        Args:
            path (list): List of node-ids.

        Returns:
            int: distance of the target path. 

        """
        edges = path_to_edges(path)
        weight_dict = nx.get_edge_attributes(self.G, name='weight')
        distance = sum([weight_dict[sort_tuple(e)] for e in edges])
        return distance


    def path_slot(self, path: list) -> bitarray:
        """Calculate AND for slot table of edges on the path. 

        Args:
            path (list): List of node-ids.

        Returns:
            bitarray: 

        """
        slot_dict = nx.get_edge_attributes(self.G, name='slot')

        edges = path_to_edges(path)
        path_slot = bitarray([True] * self.n_slot)
        for e in edges:
            path_slot = path_slot & bitarray(slot_dict[e])

        return path_slot


    def adj_path_slot(self, path:list):
        """Get slot of adjacent edges on the path.

        Args:
            path (list): List of node-ids.

        Returns:
            adj_path_slot_list: list of adjacent path slot by bitarray
            
        """
        adj_path_slot_list = []

        edges = path_to_edges(path)
        for base_node in path:
            # search neighbors
            for node_id, attr in self.G[base_node].items():
                e = (node_id, base_node)
                if sort_tuple(e) in edges:
                    continue
                adj_path_slot_list.append(bitarray(attr["slot"]))

        return adj_path_slot_list


    def is_assignable(self, path: list, start_idx: int, n_req_slot: int) -> bool:
        """Check target path is assignable or not. 

        Args:
            path (list): List of node-ids.
            start_idx (int): start index of slot table. 
            n_req_slot (int): required number of slot to assign path. 

        Returns:
            bool: target path is assignable(True) or not(False)

        """
        # exceed the amount of slots
        if start_idx + n_req_slot > self.n_slot:
            return False

        # check whether target slots are already occupied or not
        edges = path_to_edges(path)
        slot_dict = nx.get_edge_attributes(self.G, name='slot')
        for e in edges:
            if 0 in slot_dict[e][start_idx: start_idx + n_req_slot]:
                return False
        
        return True


    def assign_path(self, path: list, start_idx: int, n_req_slot: int, duration: float):
        """Assign path, updating slot table (mark allocated FS' as occupied). 

        Note:
            Slot 1 is available, otherwise occupied.

        Args:
            path (list): List of node-ids.
            start_idx (int): start index of slot table. 
            n_req_slot (int): required number of slot to assign path. 
            duration (float): required duration time to path. 

        Raises:
            ValueError: When target slot is already occupied, return ValueError. 

        """
        edges = path_to_edges(path)
        slot_dict = nx.get_edge_attributes(self.G, name='slot')
        time_dict = nx.get_edge_attributes(self.G, name='time')

        for e in edges:
            if 0 in slot_dict[e][start_idx: start_idx + n_req_slot]:
                raise ValueError(f"Target slot is already occupied. slot[{e}][{start_idx}: {start_idx + n_req_slot}]")
            else: 
                slot_dict[e][start_idx: start_idx + n_req_slot] = itertools.repeat(0, n_req_slot)
                time_dict[e][start_idx: start_idx + n_req_slot] = itertools.repeat(duration, n_req_slot)


    def resource_util(self) -> float:
        """Calculate slot utilization of whole network. 

        Returns:
            float: utilization of slot table in whole network.
        
        """
        slot_dict = nx.get_edge_attributes(self.G, name='slot')
        table = list(slot_dict.values())
        slot_arr = np.array(table)
        return 1.0 - np.sum(slot_arr) / slot_arr.size


