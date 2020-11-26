
from rsarl.networks.topology import Topology

class VERIZON(Topology): 
	""" Verizon Photonic Network (VERIZON) """

	def __init__(self):
		super().__init__('VERIZON')

	def weighted_edges(self):
		return [\
                  (0, 2, 141),
                  (0, 6, 651),
                  (1, 2, 197),
                  (1, 6, 584),
                  (2, 3, 931),
                  (2, 5, 460),
                  (2, 12, 1045),
                  (3, 4, 280),
                  (4, 10, 795),
                  (5, 7, 183),
                  (6, 7, 39),
                  (6, 8, 48),
                  (6, 9, 181),
                  (6, 11, 454),
                  (7, 8, 88),
                  (7, 11, 433),
                  (8, 11, 367),
                  (8, 13, 522),
                  (9, 13, 571),
                  (10, 12, 546),
                  (11, 12, 678),
                  (11, 13, 486),
                  (12, 17, 840),
                  (12, 18, 803),
                  (13, 14, 579),
                  (13, 16, 692),
                  (14, 20, 671),
                  (15, 16, 428),
                  (15, 19, 609),
                  (16, 21, 887),
                  (16, 22, 971),
                  (17, 18, 45),
                  (17, 24, 894),
                  (18, 20, 680),
                  (18, 25, 779),
                  (20, 22, 546),
                  (20, 23, 589),
                  (21, 26, 318),
                  (22, 23, 53),
                  (22, 26, 421),
                  (22, 30, 780),
                  (23, 27, 818),
                  (23, 30, 729),
                  (24, 25, 95),
                  (24, 27, 295),
                  (24, 28, 593),
                  (24, 32, 752),
                  (25, 27, 312),
                  (26, 31, 559),
                  (27, 29, 399),
                  (28, 32, 657),
                  (29, 30, 458),
                  (29, 33, 419),
                  (29, 36, 393),
                  (30, 35, 341),
                  (31, 34, 232),
                  (32, 33, 64),
                  (32, 42, 453),
                  (33, 36, 293),
                  (33, 46, 584),
                  (34, 38, 424),
                  (34, 41, 391),
                  (35, 37, 283),
                  (35, 38, 520),
                  (35, 48, 659),
                  (36, 40, 181),
                  (37, 40, 160),
                  (38, 39, 108),
                  (39, 48, 394),
                  (39, 49, 539),
                  (39, 54, 339),
                  (40, 43, 172),
                  (41, 44, 392),
                  (41, 49, 267),
                  (41, 50, 418),
                  (42, 45, 273),
                  (43, 47, 202),
                  (44, 50, 135),
                  (44, 51, 451),
                  (45, 46, 21),
                  (45, 53, 216),
                  (45, 55, 307),
                  (45, 66, 728),
                  (46, 47, 44),
                  (46, 53, 192),
                  (48, 56, 147),
                  (49, 50, 226),
                  (49, 54, 319),
                  (50, 52, 342),
                  (51, 52, 49),
                  (53, 62, 393),
                  (53, 64, 490),
                  (54, 58, 434),
                  (55, 61, 246),
                  (55, 66, 584),
                  (56, 58, 123),
                  (56, 60, 477),
                  (57, 58, 275),
                  (57, 59, 175),
                  (57, 60, 193),
                  (59, 60, 27),
                  (59, 62, 54),
                  (60, 62, 55),
                  (61, 65, 207),
                  (62, 63, 243),
                  (62, 64, 179),
                  (62, 68, 379),
                  (63, 64, 94),
                  (63, 68, 140),
                  (64, 66, 136),
                  (64, 69, 151),
                  (65, 67, 24),
                  (65, 70, 221),
                  (66, 68, 25),
                  (67, 69, 243),
                  (67, 70, 194),
                  (68, 69, 35),
                  (68, 70, 280),
                  (69, 71, 346),
                  (70, 71, 88)
		]


	def edges(self):
		return [\
                  (0, 2),
                  (0, 6),
                  (1, 2),
                  (1, 6),
                  (2, 3),
                  (2, 5),
                  (2, 12),
                  (3, 4),
                  (4, 10),
                  (5, 7),
                  (6, 7),
                  (6, 8),
                  (6, 9),
                  (6, 11),
                  (7, 8),
                  (7, 11),
                  (8, 11),
                  (8, 13),
                  (9, 13),
                  (10, 12),
                  (11, 12),
                  (11, 13),
                  (12, 17),
                  (12, 18),
                  (13, 14),
                  (13, 16),
                  (14, 20),
                  (15, 16),
                  (15, 19),
                  (16, 21),
                  (16, 22),
                  (17, 18),
                  (17, 24),
                  (18, 20),
                  (18, 25),
                  (20, 22),
                  (20, 23),
                  (21, 26),
                  (22, 23),
                  (22, 26),
                  (22, 30),
                  (23, 27),
                  (23, 30),
                  (24, 25),
                  (24, 27),
                  (24, 28),
                  (24, 32),
                  (25, 27),
                  (26, 31),
                  (27, 29),
                  (28, 32),
                  (29, 30),
                  (29, 33),
                  (29, 36),
                  (30, 35),
                  (31, 34),
                  (32, 33),
                  (32, 42),
                  (33, 36),
                  (33, 46),
                  (34, 38),
                  (34, 41),
                  (35, 37),
                  (35, 38),
                  (35, 48),
                  (36, 40),
                  (37, 40),
                  (38, 39),
                  (39, 48),
                  (39, 49),
                  (39, 54),
                  (40, 43),
                  (41, 44),
                  (41, 49),
                  (41, 50),
                  (42, 45),
                  (43, 47),
                  (44, 50),
                  (44, 51),
                  (45, 46),
                  (45, 53),
                  (45, 55),
                  (45, 66),
                  (46, 47),
                  (46, 53),
                  (48, 56),
                  (49, 50),
                  (49, 54),
                  (50, 52),
                  (51, 52),
                  (53, 62),
                  (53, 64),
                  (54, 58),
                  (55, 61),
                  (55, 66),
                  (56, 58),
                  (56, 60),
                  (57, 58),
                  (57, 59),
                  (57, 60),
                  (59, 60),
                  (59, 62),
                  (60, 62),
                  (61, 65),
                  (62, 63),
                  (62, 64),
                  (62, 68),
                  (63, 64),
                  (63, 68),
                  (64, 66),
                  (64, 69),
                  (65, 67),
                  (65, 70),
                  (66, 68),
                  (67, 69),
                  (67, 70),
                  (68, 69),
                  (68, 70),
                  (69, 71),
                  (70, 71)
            ]


	def nodes_2D_pos(self):
		return dict([\
                  [0,  (186, 68)],
                  [1,  (212, 244)],
                  [2,  (280, 264)],
                  [3,  (85, 108)],
                  [4,  (217, 222)],
                  [5,  (191, 119)],
                  [6,  (163, 58)],
                  [7,  (57, 278)],
                  [8,  (295, 17)],
                  [9,  (253, 269)],
                  [10,  (44, 33)],
                  [11,  (161, 32)],
                  [12,  (233, 195)],
                  [13,  (285, 221)],
                  [14,  (122, 146)],
                  [15,  (55, 217)],
                  [16,  (225, 144)],
                  [17,  (278, 291)],
                  [18,  (183, 92)],
                  [19,  (268, 216)],
                  [20,  (10, 261)],
                  [21,  (182, 80)],
                  [22,  (227, 248)],
                  [23,  (54, 109)],
                  [24,  (83, 130)],
                  [25,  (159, 52)],
                  [26,  (121, 278)],
                  [27,  (206, 202)],
                  [28,  (182, 276)],
                  [29,  (185, 174)],
                  [30,  (66, 162)],
                  [31,  (119, 169)],
                  [32,  (217, 51)],
                  [33,  (200, 216)],
                  [34,  (147, 228)],
                  [35,  (214, 16)],
                  [36,  (134, 137)],
                  [37,  (49, 247)],
                  [38,  (93, 77)],
                  [39,  (230, 288)],
                  [40,  (273, 23)],
                  [41,  (144, 222)],
                  [42,  (264, 12)],
                  [43,  (93, 174)],
                  [44,  (195, 251)],
                  [45,  (274, 264)],
                  [46,  (120, 74)],
                  [47,  (190, 124)],
                  [48,  (178, 15)],
                  [49,  (214, 245)],
                  [50,  (257, 64)],
                  [51,  (43, 230)],
                  [52,  (79, 202)],
                  [53,  (154, 150)],
                  [54,  (210, 289)],
                  [55,  (251, 50)],
                  [56,  (142, 20)],
                  [57,  (162, 106)],
                  [58,  (138, 16)],
                  [59,  (275, 264)],
                  [60,  (86, 135)],
                  [61,  (10, 69)],
                  [62,  (176, 175)],
                  [63,  (19, 216)],
                  [64,  (292, 196)],
                  [65,  (196, 48)],
                  [66,  (235, 128)],
                  [67,  (132, 57)],
                  [68,  (246, 170)],
                  [69,  (115, 75)],
                  [70,  (82, 204)],
                  [71,  (164, 252)]
		])

