
from rsarl.networks.topology import Topology

class CLARA(Topology): 
	""" Cooperación Latino Americana de Redes Avanzadas (RedClara) """

	def __init__(self):
		super().__init_('CLARA')
		
	def weighted_edges(self):
		raise NotImplementedError()

	def edges(self):
		return [\
			(0,1), (0,5), (0,8), (0,11),  #  0
			(1,2),                        #  1
			(2,3),                        #  2
			(3,4),                        #  3
			(4,5),                        #  4
			(5,6), (5,7), (5,11),         #  5
			(7,8),                        #  7
			(8,9), (8,11),                #  8
			(9,10), (9,11),               #  9
			(11,12)                       # 11
		]

	def nodes_2D_pos(self):
		return dict([\
			[0, (2.00, 6.00)], #  0
			[1, (1.00, 6.00)], #  1
			[2, (1.00, 4.50)], #  2
			[3, (1.00, 2.50)], #  3
			[4, (1.00, 1.00)], #  4
			[5, (2.00, 1.00)], #  5
			[6, (1.50, 1.70)], #  6
			[7, (3.00, 1.00)], #  7
			[8, (4.00, 1.00)], #  8
			[9, (5.00, 3.50)], #  9
			[10, (5.00, 1.00)], # 10
			[11, (4.00, 6.00)], # 11
			[12, (5.00, 6.00)]  # 12
		])
