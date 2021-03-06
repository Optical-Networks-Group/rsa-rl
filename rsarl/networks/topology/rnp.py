
from rsarl.networks.topology import Topology

class RNP(Topology): 
	""" Rede (Brasileira) Nacional de Pesquisa (Rede Ipê / RNP) """

	def __init__(self):
		super().__init__('RNP')

	def weighted_edges(self):
		raise NotImplementedError()

	def edges(self):
		return [\
			(0,1),                                #  0
			(1,3), (1,4),                         #  1
			(2,4),                                #  2
			(3,4), (3,7), (3,17), (3,19), (3,25), #  3
			(4,6), (4,12),                        #  4
			(5,25),                               #  5
			(6,7),                                #  6
			(7,8), (7,11), (7,18), (7,19),        #  7
			(8,9),                                #  8
			(9,10),                               #  9
			(10,11),                              # 10
			(11,12), (11,13), (11,15),            # 11
			(13,14),                              # 13
			(14,15),                              # 14
			(15,16), (15,19),                     # 15
			(16,17),                              # 16
			(17,18),                              # 17
			(18,19), (18,20), (18,22),            # 18
			(20,21),                              # 20
			(21,22),                              # 21
			(22,23),                              # 22
			(23,24),                              # 23
			(24,25), (24,26),                     # 24
			(26,27)                               # 26
	]

	def nodes_2D_pos(self):
		return dict([\
			[0, (5.00,  3.25)], #  0
			[1, (5.50,  3.75)], #  1
			[2, (8.25,  3.75)], #  2
			[3, (4.00,  5.00)], #  3
			[4, (9.00,  3.00)], #  4
			[5, (3.00,  3.00)], #  5
			[6, (9.00,  4.00)], #  6
			[7, (9.50,  5.00)], #  7
			[8, (10.50, 5.00)], #  8
			[9, (10.50, 3.00)], #  9
			[10, (10.50, 1.00)], # 10
			[11, (9.50,  1.00)], # 11
			[12, (9.00,  2.00)], # 12
			[13, (8.00,  2.00)], # 13
			[14, (7.00,  2.00)], # 14
			[15, (6.00,  2.00)], # 15
			[16, (6.00,  1.00)], # 16
			[17, (4.00,  1.00)], # 17
			[18, (2.00,  1.00)], # 18
			[19, (6.00,  5.50)], # 19
			[20, (1.00,  1.00)], # 20
			[21, (1.00,  2.00)], # 21
			[22, (2.00,  2.00)], # 22
			[23, (2.00,  4.00)], # 23
			[24, (2.00,  5.00)], # 24
			[25, (3.00,  5.00)], # 25
			[26, (1.00,  5.00)], # 26
			[27, (1.00,  4.00)]  # 27
		])
