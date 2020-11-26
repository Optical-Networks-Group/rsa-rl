

from rsarl.networks.topology import Topology

class NSF(Topology): 
	""" U.S. National Science Foundation Network (NSFNET) """

	def __init__(self):
		super().__init__('NSF')


	def weighted_edges(self):
		return [\
			(0,1,1050), (0,2,1500), (0,7,2400),  #  0
			(1,2,600), (1,3,750),                #  1
			(2,5,1800),                          #  2
			(3,4,600), (3,10,1950),              #  3
			(4,5,1200), (4,6,600),               #  4
			(5,9,1050), (5,13,1800),             #  5
			(6,7,750), (6,9,1350),               #  6
			(7,8,750),                           #  7
			(8,9,750), (8,11,300), (8,12,300),   #  8
			(10,11,600), (10,12,750),            #  10
			(11,13,300),                         #  11
			(12,13,150)                          #  12
		]


	def edges(self):
		return [\
			(0,1), (0,2), (0,7),     #  0
			(1,2), (1,3),            #  1
			(2,5),                   #  2
			(3,4), (3,10),           #  3
			(4,5), (4,6),            #  4
			(5,9), (5,13),           #  5
			(6,7), (6,9),            #  6
			(7,8),                   #  7
			(8,9), (8,11), (8,12),   #  8
			(10,11), (10,12),        #  10
			(11,13),                 #  11
			(12,13)                  #  12
		]


	def nodes_2D_pos(self):
		return dict([\
			[0,  (1.00, 0.90)], #  0
			[1,  (0.70, 0.70)], #  1
			[2,  (1.20, 0.50)], #  2
			[3,  (1.50, 0.74)], #  3
			[4,  (2.10, 0.66)], #  4
			[5,  (3.10, 0.45)], #  5
			[6,  (2.95, 0.70)], #  6
			[7,  (3.70, 0.75)], #  7
			[8,  (4.60, 0.80)], #  8
			[9,  (5.80, 0.50)], #  9
			[10, (5.40, 0.90)], # 10
			[11, (6.50, 0.90)], # 11
			[12, (7.30, 0.80)], # 12
			[13, (6.50, 0.60)]  # 13
		])
