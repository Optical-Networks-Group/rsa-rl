
import json
import networkx as nx
from networkx.readwrite.json_graph import adjacency_graph, adjacency_data

from rsarl.networks.topology_factory import TopologyFactory


class Network(object):
	"""Network Base class

	Args:
		n_slot (int): The number of slots in each edge.
		is_weight (bool): Generate weighted network or not.

	Attributes:
		n_slot (int): The number of slots in each edge.
		is_weight (bool): Generate weighted network or not.
		G (networkx.classes.graph.Graph): Graph object generated by networkx. 

	"""

	def __init__(self, topology_name: str, n_slot: int, is_weight: bool):
		self.n_slot = n_slot
		self.is_weight = is_weight
		# topology_name
		topology = TopologyFactory.create(topology_name)
		self.name = topology.name
		# initlalize graph
		self.build_graph(topology)


	def build_graph(self, topology):
		"""Build network topology in networkX

		"""
		self.G = nx.Graph()
		nodes_pos_dict = topology.nodes_2D_pos()
		# add prop		
		self.n_nodes = len(nodes_pos_dict)
		# add nodes
		self.G.add_nodes_from([i for i in range(self.n_nodes)])

		# add weighted edges or edges
		if self.is_weight:
			self.G.add_weighted_edges_from(topology.weighted_edges())
		else:
			self.G.add_edges_from(topology.edges())

		# add prop
		self.n_edges = len(self.G.edges())
		# add node attributes
		nx.set_node_attributes(self.G, name='position', values=nodes_pos_dict)


	def init_graph(self):
		raise NotImplementedError


	def dump_json(self) -> str:
		"""Dump data in json format

		"""
		return json.dumps(adjacency_data(self.G))


	def load_json(self, data: str):
		"""Load dumped data in json format

			Args:
				data (str): Graph in json format
		"""
		G = adjacency_graph(json.loads(data))


	def plot_topology(self):
		""" Plot network topology. 
		
		"""
		import matplotlib.pyplot as plt
		node_pos = nx.get_node_attributes(self.G, name='position')
		nx.draw_networkx(self.G, pos=node_pos)
		plt.show()

