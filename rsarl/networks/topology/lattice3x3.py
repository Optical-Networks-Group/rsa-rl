

from rsarl.networks.topology import Topology

class LATTICE3x3(Topology): 
	""" LATTICE3x3 """

	def __init__(self):
		super().__init__('LATTICE 3x3')

	def weighted_edges(self):
		return [\
                  (0, 1, 0),
                  (0, 3, 0),
                  (1, 2, 0),
                  (1, 4, 0),
                  (2, 5, 0),
                  (3, 4, 0),
                  (3, 6, 0),
                  (4, 5, 0),
                  (4, 7, 0),
                  (5, 8, 0),
                  (6, 7, 0),
                  (7, 8, 0)
		]


	def edges(self):
		return [\
                  (0, 1),
                  (0, 3),
                  (1, 2),
                  (1, 4),
                  (2, 5),
                  (3, 4),
                  (3, 6),
                  (4, 5),
                  (4, 7),
                  (5, 8),
                  (6, 7),
                  (7, 8)
            ]


	def nodes_2D_pos(self):
		return dict([\
                  [0,  (0, 0)],
                  [1,  (100, 0)],
                  [2,  (200, 0)],
                  [3,  (0, 100)],
                  [4,  (100, 100)],
                  [5,  (200, 100)],
                  [6,  (0, 200)],
                  [7,  (100, 200)],
                  [8,  (200, 200)]
		])

