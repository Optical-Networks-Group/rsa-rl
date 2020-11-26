
import networkx as nx
from itertools import islice
from rsarl.networks import Network

def _k_shortest_paths(G, source, destination, k, weight=None):
	return list(
		islice(nx.shortest_simple_paths(G, source, destination, weight=weight), k)
	)

class Routing:
	""" Routing Class 
	
	"""

	@staticmethod
	def shortest_path(net: Network, s: int, d: int, is_weight: bool=True) -> list:
		"""Search the shortest path. 

		Args:
			net (rsarl.networks.Network): Network instance. 
			s (int): id of source node
			d (int): id of destination node
			is_weight (bool): whether weight(physical distance) is consider or not

		Returns:
			list: list of node ids between source and destination nodes.
		
		"""
		if is_weight:
			path = _k_shortest_paths(net.G, s, d, 1, weight="weight")[0]
		else:
			path = _k_shortest_paths(net.G, s, d, 1)[0]

		return path


	@staticmethod
	def k_shortest_paths(net: Network, s: int, d: int, k: int, is_weight: bool=True) -> list:
		""" Search shortest paths up to k-th. 

		Args:
			net (rsarl.networks.Network): Network instance. 
			s (int): id of source node
			d (int): id of destination node
			k (int): the number of paths to search
			is_weight (bool): whether weight(physical distance) is consider or not

		Returns:
			list: list of list of node ids between source and destination nodes.
		
		"""
		if is_weight:
			paths = _k_shortest_paths(net.G, s, d, k, weight="weight")
		else:
			paths = _k_shortest_paths(net.G, s, d, k)

		return paths

