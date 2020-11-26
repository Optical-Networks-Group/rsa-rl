

from rsarl.networks.topology import Topology

class JANET(Topology): 
	""" U.K. Joint Academic Network (JANET) """

	def __init__(self):
		super().__init__('JANET')

	def weighted_edges(self):
		raise NotImplementedError()

	def edges(self):
		return [\
			(0,1), (0,2),                # 0
			(1,2), (1,3),                # 1
			(2,4),                       # 2
			(3,4), (3,5), #(3,6),        # 3
			(4,6),                       # 4
			(5,6)                        # 5
		]

	def nodes_2D_pos(self):
		return dict([\
			[0, (1.50, 4.00)], # 0
			[1, (1.00, 3.00)], # 1
			[2, (2.00, 3.00)], # 2
			[3, (1.00, 2.00)], # 3
			[4, (2.00, 2.00)], # 4
			[5, (1.00, 1.00)], # 5
			[6, (2.00, 1.00)]  # 6
		])