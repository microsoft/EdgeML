// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#define L3_N 1
#define L3_H 30
#define L3_W 40
#define L3_CIN 32
#define L3_CTEMP 64
#define L3_HF 3
#define L3_WF 3
#define L3_COUT 32
#define L3_HOUT 30
#define L3_WOUT 40
#define L3_HPADL 1
#define L3_HPADR 1
#define L3_WPADL 1
#define L3_WPADR 1
#define L3_HSTRIDE 1
#define L3_WSTRIDE 1

// The convention followed for filter weights is F[groups][filter_height][filter_width][in_channels][out_channels]
static Q15_T L3_F1[1 * 1 * 1 * L3_CIN * L3_CTEMP] = {504, 5114, 3073, -1727, 7803, -1299, 1964, 2140, 1155, 2521, 5219, 3015, -4478, 1186, 4289, 484, 459, 5970, 13288, 3159, 2830, 1372, 1843, -1240, -11433, 606, -6874, -4899, 2614, -1934, -116, -111, 7942, 2243, 1667, -494, -2771, 5660, 6938, -2929, -5624, 2051, 6341, -12790, -2583, 456, 2800, -1552, -690, -3531, -8129, 1779, -1456, -9233, -2510, 493, 24909, -2170, 6211, 4866, -13039, 1175, 9882, -6153, 3372, 271, -2863, 5663, -3951, 5095, -5566, -945, -2490, 7448, 6321, -5743, 3651, 6080, -9896, -8068, -2331, -6298, -1337, 1488, 5084, -1812, -6749, 327, -421, -9820, 12959, -2016, 18335, -18433, -1594, -8203, -3345, -3488, -13493, 8360, -6038, -5966, -1178, -10701, 4881, -4234, 3370, -5922, 6426, 433, -1932, -2025, 3521, 1646, -8772, -4038, 2702, 857, -4061, -187, 15369, 5314, -2209, -1363, 2840, -10474, -3300, 8135, 4703, 5477, 6952, 3037, -1624, 9746, -3724, -3280, -5676, 5615, -2444, -593, 4623, -1293, 1947, 6377, 3375, 5693, 4750, -8350, 524, -2016, 5715, -1634, 2146, 65, 3311, 7315, 1583, -4030, 6201, -2386, -4564, 8680, 2514, -2443, -4639, 163, 14759, 507, 3294, -2340, -8731, 4858, 7365, 6537, -1207, 3755, 5147, 4110, 4847, 5354, -1993, 2349, 1715, 5501, 11470, 116, 1931, 10377, 2416, 9852, -6382, -817, 1526, -8565, -4399, -4025, -8305, -7257, 2201, -8888, 5344, -7015, 1080, -5339, -666, -636, -6618, -3182, 909, -7178, 1694, 4904, -3711, 187, 499, -7287, -100, -10060, 4487, 4742, 1629, 5093, 1144, 1213, 4721, -3311, 8735, 5538, -10444, -1373, 3701, 750, -1440, 6532, 1642, 2458, 1216, 3900, 7129, -1459, 2884, -11142, -4569, -1904, -4268, -5813, -2549, -9439, -11614, 1693, 1775, -4738, -1397, -6477, 988, -6998, -3050, 7425, -566, -11195, -2530, -3366, 1182, 5417, -8537, -647, 5180, -5671, 3719, 7830, -3706, -6503, -4692, -4560, 5824, 3473, 7903, 687, -6689, 1586, 4135, -1516, -10421, -5082, 1456, 7447, 5469, 2877, -5675, 7618, -259, 217, 12015, 5866, -3104, -3991, 7993, 6314, -1205, 1591, 11398, 4637, 4936, 4234, -2763, 5157, 3680, -313, 3505, 10878, 3449, -5475, 6498, -5776, 979, -4719, 1458, 561, 1787, 7879, 2542, 4573, -5316, 3449, 1756, -803, 11224, -3525, 7188, 9065, -2615, -3724, 4870, -7968, -3340, -6163, -7281, -13102, 9489, 3021, -5504, -4560, -5939, -1612, 4647, -4596, -338, -3164, -7166, -5044, -5992, -539, -4901, 701, 2385, 797, -1703, 437, -6259, -28, 164, 5890, -1757, -13734, -3938, -3774, 1690, 2336, 2397, -3839, 4268, 3220, 2073, -1316, -6322, 10112, 2499, 4944, 547, 1381, -3528, -5801, -6587, 4727, 2015, -1487, -5547, 1981, 487, 6395, 7415, 3291, -3525, -4230, -9071, -1244, 1215, -1072, 9911, 6955, 3493, -2842, 12697, -747, -4332, 9105, 3142, -7182, 6433, -1748, -6535, 4271, -1192, -4228, 758, 13321, 5422, 4607, 14835, 6045, 4554, 3762, -6786, -6177, -103, -3050, -4789, -986, 16217, -7705, -9526, -2530, 3008, 3277, -6926, 3392, 4587, 6352, -1450, -3698, 5766, 2125, 3164, 557, -372, 1812, 2924, 5435, -2190, -2268, 9632, 811, 2468, -3084, -1389, -5733, 2518, -13800, 734, 816, -1558, -1719, -5479, 2986, 467, 1753, 4455, -2689, 1389, -1888, -5421, 248, 10364, -2344, -9845, 10714, 5298, 9829, 2608, 3473, 9318, 3235, 102, -7302, -4311, 1873, 3264, -421, 11139, -2313, 3631, -3155, 5572, -2039, -510, 6566, 11208, -678, -3069, -1402, 4657, -437, -2701, -4583, 323, 2396, 4049, -35, 5845, -7450, 890, 6019, 549, 4553, 3124, 4155, -5888, -195, -2602, -6697, -3382, 9272, 4397, -7984, 1631, 3196, -2859, -1696, -8236, -760, -1047, 1878, 9534, -2700, 773, -8236, 6161, 7310, -5448, 6684, 4298, -2156, 4647, 6990, 3211, 2295, -5114, -7741, -12415, 5047, 1452, 2889, -1145, -340, 7452, 2924, -1589, -6874, 278, -5936, 1035, 12981, 735, -3603, 3528, 5947, -1363, -2242, -3209, 8727, 4976, -9717, 11493, -3653, -1302, 7029, -12120, 1577, -6148, -12411, 8196, 2327, 935, -3240, 4949, -881, 68, -6146, 3684, 9828, 4767, -975, 639, -5324, -5317, -6376, -1472, 1070, 5166, -2915, -1203, -336, -6923, 1020, -2838, -1153, 3935, 1527, 547, 13045, -4747, 4595, -7203, -6599, -8090, -5231, -1755, -1874, -8080, 9451, -1089, -7897, -1891, 2298, -653, 5711, 13085, -25, 918, 6524, 5484, -7481, 5240, -5075, -2314, -2419, -234, 39, 1839, 12437, 10035, -1100, -980, -2186, 4624, 8032, -7570, -1511, -6072, -6392, 3115, 3531, 3746, -3535, 3818, 1290, 1712, 697, 6804, -6606, -10735, -7377, 5120, -152, 1530, 3659, 2327, -6759, -6970, -1898, 1747, 2607, -86, 6581, 2741, -7841, 1567, 7011, -7457, -5076, 12005, 3489, -5182, -5698, 10308, 476, 4815, 1250, 5014, -344, 805, 2325, -1744, 2670, 2400, 2742, 5213, -5953, 11061, 1004, 10184, -1013, -1397, -3257, 5067, 6504, -3815, 1017, 7068, 3923, -538, -2173, -1314, -145, -4267, 5011, -3719, -1720, -3261, 1234, 4672, -4101, 4626, 1783, 2351, 6150, -7857, 10454, -1877, -6632, -5411, -9573, 1219, 731, -8828, 6347, -9976, 11986, -1446, -3680, 3421, 2318, -5295, 6570, -11729, -4596, 1948, 1848, 2738, 10619, 5941, 1149, -1155, 10489, -470, 7689, -3530, -4340, 384, -9585, -1032, 4822, -2286, -3651, -3893, -4488, 893, 7635, -8882, 2568, -8627, -4738, 382, 2650, 3364, -6259, -5171, -7697, 7788, 1464, 1234, 616, -6515, 1625, -448, -2661, 4436, 4193, 7276, -6953, -7586, 9962, 5546, 3173, 3597, -287, -6889, 11183, 5548, 533, -1655, -80, -393, 354, -14021, -3468, 949, -8984, -744, 4430, 5288, -2753, 10668, 5808, -273, -1288, 992, -4711, 5411, 1059, 11023, 729, 475, 1149, -12456, -3191, 4257, -95, -6731, 3291, -3922, -725, 2739, 2648, 4675, -3565, 9151, 2389, -4139, -4944, -1807, -2201, -5125, 6344, 1306, -7417, 3294, -2250, -2292, -3399, 3605, 1602, 285, -7415, 1455, -1014, 12184, 9404, 4091, -3096, -6136, -1195, -7596, 862, -494, 6259, 8179, 539, -3070, 382, 6024, 8902, 9060, 12435, 5408, 4110, 690, -4178, -1472, 1184, 4869, 6350, -807, -1306, -395, 461, -2755, -4732, 1911, -558, -1073, -4919, -796, -5236, -1566, 6461, 29, 7529, 2098, 331, -498, 1976, 1238, -6849, 124, 17890, 4120, 2832, 9448, -1812, -648, -5653, -3909, -9823, 194, -2260, 1437, -8088, 6693, 1882, -6982, 1745, -5054, -3332, 10247, -2418, 11258, 4320, -6774, -2213, 9713, 21631, -5465, 10485, 11173, -1494, -3543, 4705, -9664, -13686, 6989, 4054, 7333, 2441, 479, 3049, -8920, 7328, -529, 585, 3472, -5078, 18320, 2654, -1343, 6929, 1213, -7824, -493, -8792, 8692, -2237, -2776, 1383, 7201, -3060, 10555, 8085, -4344, 2169, -2651, 4200, 3320, 3779, 9513, 5605, 1337, 345, -13375, 2614, 3030, 8357, -3029, 5656, 7918, 7187, -6752, -2624, 1211, 6723, 1560, -4609, 4005, 1028, -1979, -5032, 2641, -1581, 8011, -5489, 3448, -525, 13685, -1364, 2277, 5701, 13825, 726, -589, -3213, 10310, 4470, -55, 9364, 3378, 5622, -3853, -6369, 3702, -3406, 11111, 894, -11235, 847, -3034, -1806, -6188, -8071, -2102, -1417, 4970, 4026, -4995, -9177, -268, -13224, 1077, 9281, 630, 5526, -1301, 125, 1655, 2872, 2888, -3175, 8947, 10971, 5530, -2973, -6664, 5638, 1133, -6640, -2631, 948, 11367, 1416, 5782, 5167, -5729, -8349, 4314, 8734, 2926, 8881, -3323, 8063, -330, 10591, 7456, 2164, 6517, 337, -755, -12599, 8510, 4266, -6888, 313, 73, -224, -5541, 4029, 715, 3181, -2869, 6255, 2114, -2082, 7649, 3639, 1180, 120, -1270, 3005, 3134, 3664, -2094, -6782, 4113, 6596, 3227, -447, 273, -783, 1703, 6094, -7658, 275, -6189, 6037, -5415, -9574, 442, -4047, 67, -4803, -2912, -7901, -5705, 6325, -115, -1441, 5228, 4655, -7669, -6316, -6839, 7081, -1024, -6698, 1084, -786, 5761, 1119, 4540, -5946, 8058, -9393, -7771, 495, -3514, -2900, 8961, -8525, -3095, -1002, 14149, 5156, 4504, -3501, -7096, -15462, -314, -9625, 536, 1496, -2497, -2692, 2314, -9436, -3485, -2305, -2180, -2794, -1544, 8287, -6133, -4333, 613, 4970, -6070, 3554, -4045, -6082, -2761, 3206, -2267, -11458, 5422, 3749, 1570, -6142, -3094, 297, 8246, 2564, -12468, 3513, -760, -8149, -2539, -3785, 12290, 10556, 275, 3226, 17281, 12760, -9354, 1032, 1453, -1219, 5570, -1122, -161, -4833, 18066, 843, -6806, 379, 9297, 6103, -538, 1703, -2956, -3112, -3454, -5464, 5263, 4237, 3806, 4081, -3143, 3223, -4523, 6825, 4613, 2578, 707, 3078, 3494, 2137, 10149, -788, 1987, -1211, 657, -301, 2090, -3495, -8241, -1931, 1118, 4666, 2003, 976, -1366, 12261, -161, 15945, 11052, 7386, 7956, -534, 4012, 12308, 11848, -519, 8466, -154, 10, 1950, 4060, 4179, -2852, 2894, -831, -3269, -5217, 5351, -9497, -3057, 789, -7883, 1768, 6465, -5966, 999, 735, 2403, 4637, 12557, 5989, -139, -4703, 3450, -1475, -1513, 1922, 575, 5202, -2016, -6795, 246, 3020, -424, 20, -3650, -2454, 3767, -824, 6484, 2328, 8857, -11193, -1135, -5150, -3668, -162, -450, 1304, 7791, -5759, -5359, 1203, 2996, -841, 8594, 1456, -3517, 5606, -746, -1057, 6468, -7774, -7690, 824, 337, 4946, -490, -7695, -1778, 4981, -1519, 654, -4614, 12829, -1635, 3805, -9046, -3935, 6886, 6189, 1211, -3974, 4323, 5068, -182, -8403, -2466, -6632, -12664, 8913, 537, -709, 8210, 1546, -120, 2781, 6278, 3045, 4583, -2242, 11193, 840, -8316, -6222, -6815, 7300, 3536, -4289, -1572, 1383, 1734, -6657, 1156, -4257, -5069, -3482, 98, -5383, -1387, 13057, 7082, 10831, 301, 2841, -6115, 2805, -748, 3573, -5875, 746, 515, 9313, -9669, -3221, -1759, -10753, 5624, -58, -3082, 4285, 3922, -674, 5483, 2135, -5258, 9282, 8738, 3117, -5885, -963, 2427, -265, 1941, 4206, -3401, -2213, 1008, -4364, -5770, 4408, 10351, -687, 2979, 1420, -4038, 2891, 1123, 10249, -7093, -3436, 545, 3447, 11119, -1791, 6615, -3452, 2590, 3562, -836, 6002, -6880, -1340, -2861, -7353, 6597, 8550, -276, -4992, 2155, 13168, 4132, 8845, -2285, -1036, 11549, -400, 2320, 6549, 3335, 16496, 9310, -7778, 224, -3316, 2061, -6268, -12301, 2484, -1373, 2787, -910, -100, -3015, -6724, -6502, 2311, 7882, -4, 2031, 100, 4743, -1564, 2010, 3522, 2373, 8215, 167, 529, -7653, -10687, 1825, -4440, 8395, -2507, -947, 5266, -4689, 6364, 1164, -1858, -13578, -7739, 7910, -2194, -110, -1731, 3662, -9317, -3730, -4403, -9321, 886, 1054, 10851, 902, -1399, 1373, -9421, 1298, -2924, 1286, 8235, -7515, 1772, -1406, -4612, -2655, -2912, 425, -5416, 265, -7001, 1993, -2174, 2059, 8481, 2249, -685, 6028, -2119, -4751, -1391, 5238, -2551, 4724, 6303, -6004, -9406, 8810, -3019, 409, 8773, -1542, 10112, 3254, 6524, 1711, -1266, -6322, -1116, -5086, -882, 1913, 1719, -2974, -523, 8618, -949, -6698, 5561, 1358, 9289, -11570, 589, -1309, 8602, 1541, -1347, 3313, -2877, 766, -2607, -1158, 11218, 1068, 11039, 3500, -21467, 10931, 116, 211, -7907, 3475, -5051, -4779, -4494, 2498, -1538, 784, 7236, 1940, -331, 1469, 4899, 4312, 2102, -5138, 3618, -3752, 3946, 5912, -5109, 4772, 1275, -5402, 3645, 5561, -8297, 1131, 3887, -2949, -6109, -1926, -6661, 1949, 5693, -2329, 4573, -8537, 6484, 5050, -4374, 8329, -11994, 575, -806, 4962, -11095, 2239, 2074, 4940, -7071, -9430, 4909, 2184, -5, 685, -5030, -2376, 13885, 219, 530, 3981, 5682, -700, 2504, -165, 2247, 1878, 9088, 1644, 969, -5582, 3858, -2177, -3984, -2044, 434, 10780, -452, 1916, 295, 2087, -9680, 14221, 26, 1054, -10203, 3995, 6390, -1193, -222, -1912, 4580, -5121, -1532, -6141, 4790, 2196, 10140, 4110, 3073, 3414, -1700, 2154, -2723, -3734, -770, 3844, -973, 3867, -74, 9403, 4014, -2318, 2166, -5641, 629, 5854, -7782, 2186, -1029, 481, 2662, -1348, 5084, -1142, 430, -6571, -5973, -3534, 7860, -12376, 8945, -4540, -5666, 1977, -6002, -1101, -2551, 1760, -762, -1616, 6111, 551, -2078, -3344, -6985, 3493, -8738, -3568, -546, 1099, 7716, -4255, 819, 3792, -4521, -3097, 875, 1322, 3146, 9352, 12641, -752, 6184, -7525, -3417, -1377, 6801, -2699, 14970, -7929, -316, -4723, 693, 830, 8662, 259, 6666, 4525, -3501, -13454, -5721, 261, 10215, -5508, 5720, 3262, -5819, 3751, 7993, 517, 476, -11564, -1191, 7338, 15, 6790, -2158, 3662, -6144, -534, 11513, 2206, -5003, -5961, 4785, 7776, -6833, -2166, -5818, 1519, -1711, 532, 5466, -2610, 6868, -462, 3952, -3285, 8218, -2673, 1026, -3687, -3581, 311, -682, -3382, 2062, -2285, 3306, -216, 5308, 2752, -871, 10186, -1202, -2814, -398, 9399, 8011, -3725, -2158, -3981, -1451, -3247, 4469, 5897, 2178, 4589, 10767, -1454, -628, 8156, 4270, -4129, 1755, -1424, 4073, 4027, -678, 1412, -9379, -7440, 4126, 2832, 1000, -4482, 1772, -14259, 3964, -7633, 4240, -1144, 2114, 2364, 119, -825, -2499, -1601, 1795, 1279, 10663, -4387, -769, -1333, 2538, -1520, -2623, -7190, -1289, 788, 4319, 3804, -6488, 9712, -49, 387, 1113, -169, -11314, -2015, -6143, -4608, 8127, -1904, -3842, 1291, -2669, -61, 1218, -2331, -1929, -1775, 850, -10091, -15068, -11685, 4929, 982, -1141, -5689, -5688, -5537, 404, -2761, 4905, 7279, 4947, -2467, 1747, 8528, -4498, 3391, -1952, -3153, 1419, 2391, 6117, 2069, 5108, 6057, -2598, -459, -2164, -1574, -8070, 7242, 3264, 5674, 1540, -601, 7101, 6762, 4042, -2720, -6381, -498, 9896, 390, -2068, -4925, 2406, 4365, 16008, -3651, -3089, -1177, 6852, 8941, 1710, 9233, 1370, 2315, 2289, 1560, 3685, 2656, -6108, -16678, -19643, 6294, -2213, 5352, 5194, 6240, 8227, -363, -1643, 818, 5363, -4287, -3800, 1336, -3368, 1137, -4927, -1607, 4602, 6582, 3538, -2419, -2391, -13636, -7163, -5186, 998, -969, 11, 6338, -3959, 12594, -5225, -10273, 11489, -2167, 1715, -10311, 6135, -1543, 338, -5609, -9959, 4537, -3034, -3380, -754, -562, 8593, 8044, 3638, 465, 3269, 8908, -1769, 3647, 1886, 4258, -7438, -10853, 869, -912, 5138, -4064, 3411, -1161, -3223, -3636, 3881, -4411, -3491, -1355};
static Q15_T L3_F2[L3_CTEMP * L3_HF * L3_WF * 1 * 1] = {-470, 6804, 1699, -9045, 14019, 3314, 2289, -1293, 908, -2098, -6720, -4535, 11120, 4501, -2052, -1027, 10925, -84, -5612, -8672, -6952, 7224, 10612, 3853, 7556, -3030, 4669, -16, 9746, 1876, 5156, -13825, 7967, 1187, -10520, 3177, 3669, -9314, 2400, 2620, -8747, 17537, 4062, -630, -302, 15931, -2281, -1593, -14716, -904, -3653, -2575, 4882, -3528, -4007, -1170, -6067, -7260, -2049, -6992, -968, 754, -3932, 6297, 5354, 195, -2057, -8466, 12846, -5334, -6226, 259, 4252, 5047, 6174, 719, 7097, 1624, -348, -944, 2963, -4022, -4441, -2749, -1205, -3080, 1417, 18558, 12312, 17921, -3562, -11481, 6672, 3444, -9315, 1470, -8715, -1433, 2235, 14709, 1018, -376, -10170, 3950, 1723, -609, -589, 148, -5315, 2247, -8844, 12452, -6224, -778, 3723, -1106, -7705, -3189, -569, -1754, 20836, -8630, -1656, 32, 3102, -2429, -61, -1774, -433, -3005, 17063, -6201, 3984, 1162, 2368, 3527, -178, 4047, 1310, 2858, 8350, 6931, 12039, 5040, 6454, 5156, 7992, 1018, -1378, 2744, -9888, -2333, -9601, 2413, -2231, 2271, 1627, 4363, -3308, 1898, 14875, -10331, 13696, 8179, 4586, 1340, -686, 2973, -823, -285, -20, 1379, -3284, 2125, -12189, 12201, -1319, -5259, -5499, -1196, 18397, 734, -5648, 10483, -9036, -3863, 3687, -5788, -6864, -3044, -2258, 2264, 7798, 13566, 528, -105, -3466, 2272, 1821, 11481, -1226, 4965, 11381, -1369, 3171, 3685, 542, 2489, -2625, 388, 540, 16351, 416, 1209, -3908, -552, 7457, -7864, 2922, 7148, 7849, 4382, 1868, 12904, 4209, 3595, 15913, 1289, 139, -1972, -1665, -2193, -1238, -2586, 12286, 4541, 4675, -957, -2153, 138, 6244, 5626, 5307, 831, -1284, 1258, -2932, 18496, 673, 1261, 2625, 1373, 27433, -1604, -19352, 24354, -592, -20279, 21915, -1015, -16276, 7992, 17248, 3794, -1206, -9219, 5692, 577, -9640, 1996, 3870, -6681, -1298, -7172, 21498, -10507, -4080, -789, -2458, 2603, -928, 490, -4327, 7120, -3847, -5488, 17066, -5964, -1070, 1509, -2529, -4382, -5030, 1398, 13435, 6424, 3805, 1076, 3237, 6090, 62, -3150, 19332, -410, 1309, 5464, 1425, 8367, 3038, -718, 15152, 1440, 2989, 2624, 294, 16456, 2686, 10555, 7195, -28728, -13, 12271, 2912, 6219, -85, -3740, -1467, -1459, -4872, -1570, 1458, 21077, -2343, -211, -8871, 4042, 4803, -5369, 7374, 8919, 330, 3857, 3891, -202, 3472, 9421, -11493, -577, 14430, 5616, 9733, 5810, 2777, -442, 14759, 2404, -1271, -1609, -541, 4097, 4202, 10457, 5272, 7209, -1897, 6119, 4398, 2007, 8192, -479, -23, -4067, 1465, 7692, 869, 11481, 11548, 1668, -3893, -228, 2844, 27642, -486, 1242, -2475, 1316, 795, 5474, 12640, 7939, 4355, -7584, 3783, 2110, 5541, 3234, -5117, -631, -6164, -4058, 4429, -4575, -7976, -2099, -8338, 1077, 2046, -4722, 318, 2766, 16661, -1574, 1278, 594, -695, -4423, 4368, 14727, 4428, 3261, 7142, 1826, 1105, 6508, -7124, -6861, 8006, 7329, -6401, -2042, -7404, -7209, 2028, 2541, 1997, 305, 11090, -276, -5597, -7968, -5936, -6799, 2514, -10025, -6300, 27989, -13371, -1017, -707, -5292, 1236, 1357, 3364, 2449, 13574, 11266, 1343, -3518, 2628, 4295, -5198, 59, -2119, -3864, -331, 4564, 13015, 7024, -783, -1281, 21594, -12026, -4472, 13128, -7049, -1472, 4974, -918, -915, -4707, -13145, 5042, -6847, -3654, 16485, 2934, 10937, 4600, -1536, 12493, -1779, 1136, -6802, -662, 3552, -3652, 13413, 3439, -732, -793, 8179, 3339, 2091, 6337, 12133, 865, 13190, 7815, -1052, 7909, 3168, -1426, 6514, -1784, -9900, -6998, 3227, 4494, -1778, 13952, 5420, 6684, 4690, -1494, 4810, 4073, 10862, -1593, 2755, 6586, -8346, -2739, 999, 1111, 15394, -5010, 2481, 645, 1965, 201, 1829, -2748, -2535, -884, -5454, 2656, 8391, 11443, 9413, 1224, -9075, 3642, 13484, -7760, 6231, 1225, -4021, 1775, -4867, -9210, -1715, 10032, 11779, -5658, 1681, -989, -61, 4608, 11194, 6976, 6652, -16877, 5064, 6140, 1580, 3058};
static Q15_T L3_F3[1 * 1 * 1 * L3_CTEMP * L3_COUT] = {4556, 6633, -3868, 6042, -217, -3724, -4834, -4939, -739, -286, -4399, -2318, 4856, -9341, -5154, -4640, 8552, -668, -1605, -4736, -1269, 3813, 5191, 2074, 365, -2655, -3691, -990, 7475, -10269, -7, 8847, 3685, -4338, -9404, 6244, -3710, 10938, 1023, 671, 1848, 3448, -273, -2550, -4136, -1996, -2631, -3258, -3160, -4444, -4632, 2062, 4507, -2947, -2518, -553, 8773, -6108, -3995, -4654, -3499, 5006, 12486, -4588, -4807, 986, -106, -4007, -3484, -2970, 9375, -10530, 617, -506, 3663, 4161, 5794, -5469, -5417, 1606, -1414, -9374, 490, -2256, -2836, 477, -1191, -879, 2289, -3863, -2798, -489, 1278, 5521, -8430, -1204, 9006, -3564, 623, -8316, 5168, -1527, -4821, 8701, -5916, 6324, 1411, -1764, -4602, -1395, 8088, 3975, -4175, 4979, 4625, 8366, 3900, 5706, -6023, -868, 6342, 6022, 1339, 4412, -4203, -4432, 6318, -8895, 11321, -3186, -1176, 1494, 239, 3259, -6933, -5543, 2887, 4612, 983, -6801, 2473, 1659, 1030, 568, 330, 975, 6345, -6406, -2522, 5297, 7980, 6565, -4671, -310, 1891, -6756, 2196, -12092, -1406, -681, 2342, 1649, 2083, -3306, 4919, 8808, 2533, 6782, -3612, 5356, -1358, 1366, -4053, 6391, 3708, 4570, -2171, -174, -105, -575, -2903, 4593, -3694, 75, 4930, 45, 4929, 2246, 5637, 11349, 2605, -527, -9733, 3306, -3676, 5198, 5346, 10391, -1575, -3797, -2453, -6710, -5511, -1857, -336, -2545, 2411, -1699, -4718, 196, -1754, 1298, -4257, -2882, 4146, 433, 1626, 2162, 1361, 1676, -1620, 1658, -3933, -643, 1514, -1686, -4464, -192, 979, 5432, 205, -4589, -1086, -2945, -3819, -2483, 4112, 2057, 925, 3789, 2807, 1674, 4480, -1735, 1874, -1140, 8952, -4678, 2600, -3387, 344, -5256, 2554, -2187, -6374, -3898, 5196, -664, 582, -7682, -646, -681, -9593, 6398, -6509, -1734, 2391, 3125, 1402, 5113, -561, 4114, 2953, 14081, 11423, 121, -11121, 3766, -2531, -9245, -7684, -3505, -8581, 10366, -1514, -5613, 8690, 2613, -11790, 9137, -3263, -2811, 6491, -9268, 11175, -4664, 4875, 8962, -12118, 8963, 10848, 3346, -11062, -5070, 1069, -1050, -7899, -2926, -1067, -7185, -2674, 5546, 4082, -298, 4674, 1667, 2778, -285, -480, 3545, -1029, -2451, 4507, -7579, -1266, 3609, 3828, -2564, -4819, -2594, -617, 3299, 3678, -1063, 6842, 9383, 5448, 10056, 6162, -5264, -4339, -9767, -249, -16768, 3619, 2299, -2484, -390, 1326, -6777, -1349, 7412, -6331, -3211, -4148, -1518, 4153, 10637, -2264, 11559, -185, -2768, 3826, -1283, 821, 7357, 823, -3839, -1221, 5458, -1908, 6558, -10267, 1437, -6558, 4336, 3538, -4686, 6301, 5258, 6747, 5833, 29, 1232, -1412, -2783, -5435, 6242, -4904, -5558, -346, 313, -932, 6133, 3766, -4570, -892, -1299, 589, -6906, 1495, -2325, -2727, -968, -3341, 1141, -2484, -3238, 3369, 1646, 2892, -24, 2966, 614, 860, 7760, -5725, -1236, 915, -311, -1100, 5797, -5903, -193, 5873, -2604, -1182, 7375, -11641, 6202, 2777, -4427, -2668, 5868, 1330, 6355, -5957, 6190, -6216, 5053, -874, -5223, 1401, 2909, 3926, 1406, -5300, -3606, -759, -6079, 466, -14301, 4782, 5346, -4531, 523, -6534, 4355, 15, 4126, -8285, -5799, 4941, 6912, 5310, -2925, -2329, -749, -233, -1209, -1728, -788, -2539, 2517, -4202, -1436, -7268, -6489, 4308, 3195, 2750, -233, -4396, -2452, -4293, -264, -8732, -2685, 666, 4076, 9575, -9584, -2115, 2802, 4239, 1071, 7885, -5470, 1901, -5984, 9983, 238, -6825, 1522, -3306, -1941, -2177, 376, 3508, 7976, 1049, -1776, -6366, 1515, -1580, -4169, 7116, -1968, 1354, 5728, -2600, 5789, 1581, 7239, -1526, 5672, -6812, 32, 4406, -4553, -1938, -5659, 4229, -4135, 3127, 2857, -8830, -5503, 1838, -5754, 1672, 8474, -718, 2723, 2023, -6633, -2402, -87, -3824, 4265, 2299, -1097, 4041, 2073, 4874, -6664, 5402, 1604, 2140, -4336, -1313, 3696, -262, -3454, 3617, 6577, -321, 5779, 4157, 2934, -833, -3180, -3536, 312, 5278, -652, 7123, -2721, -2902, -3339, 6664, 342, -2964, 361, -2071, 8797, 147, -5475, 370, 4273, -382, 3712, -739, -2685, 941, -8245, 9327, -7239, 3222, 2881, -5526, -4990, 11665, -7083, 641, 6945, 394, 7737, 6937, -4226, -1493, 192, 5029, 3916, 9605, -5391, -211, 3703, -2383, -2698, 6156, 4917, -3181, 3285, 11300, 988, 3405, 1574, 1011, -2862, 4471, 861, -7177, -7228, -565, -4675, -365, 496, 1991, -5512, 6360, -10604, 2557, -3586, -3696, 275, -4479, -765, 3388, 1357, -7524, 1956, -5158, 2320, -810, -184, 3855, 368, -9223, 5414, -855, 667, 6143, 7111, -1177, 7975, -217, 3215, -4228, -2424, 848, -2068, -3065, 4471, 1290, -5116, 4951, -903, -3568, 1133, -917, -4366, 2504, -761, -407, -3461, 3913, -5172, -173, 9111, 2796, 1322, -937, -2826, -2613, 3471, -1031, -1618, 1218, 6024, -1210, -517, 3235, 2323, -2479, -3835, -3975, -5499, 1762, 1276, -420, 748, 2130, -10219, -1772, 357, -4118, 10579, -7335, -1952, 5704, -135, -12728, -11076, 8492, -18, -5096, 1277, -1622, -8693, 9263, -2054, 481, -1516, 2551, -1302, -7092, 6361, -5585, -6942, 4289, 1675, -6889, 9106, -10973, -7617, 2087, 4180, -1563, -541, 1116, -87, 9131, 4005, 2446, 1326, -448, -1993, 825, -5389, 8578, 4085, 4355, 5178, 3286, 3162, 6612, 6062, -120, -9497, 2257, 5539, -8981, -9743, -8671, -813, 7759, -2539, -493, 1414, -3083, -5339, 14332, 1132, 421, 50, -913, 7419, -4777, 135, -9656, 2190, 705, -8365, 4848, 7401, -11711, -752, -3308, 3089, -3176, 4553, 6541, 5008, -6258, 5647, 7616, -149, 1484, 7723, 804, 8768, -4597, -3634, -2155, 2961, 795, -5482, 1143, -2787, 10399, -5982, -6510, 7304, -834, -1897, 5693, -3079, -564, 6838, 7119, -10407, 1658, -6399, -2065, -620, 3648, 10765, 248, -2593, 2232, -946, 3298, 2544, -5230, -1887, 6589, 4378, -3643, 1254, -1522, 2868, -4466, -6671, 9265, 3215, -7334, 1803, 10, -7210, -1031, 3643, 4486, 4809, -1886, 1743, -1861, 7959, -5246, -3654, 3565, -6351, 3546, -5104, 2334, 504, -7563, 3442, 6362, -3035, 1814, -1932, -2565, -707, -4187, -1972, 3535, 6149, 2956, -421, 178, 8526, 4381, 9434, -5892, -7132, -3366, 2426, -6750, -1079, -2874, -19413, 2113, -4103, -1405, 1013, 899, -549, 4704, -1244, -15412, 46, -1394, -252, -2390, 2640, 7995, 3104, -5476, -1109, 2374, 8044, -484, 2761, 3164, 2655, -6442, -3853, 11142, 1293, -1871, -7033, -11880, 6343, -4224, -10138, -658, -2321, -1869, 3044, 3829, 3909, 481, 3379, -1062, 7934, 2013, -3145, -2963, -5053, 1649, 7455, -10444, 2414, -11113, 2387, 3612, 3452, -2247, -1319, 3206, -15117, 1962, -1748, -1602, -8178, 1303, 736, -1612, 7621, -913, 5695, -1430, 7238, 4202, -2280, 786, -3859, 2349, -6437, 3200, 5523, -3057, 55, 119, 4069, 528, -937, 1638, -4731, 4881, -1982, -7605, -889, 1588, 6602, -6185, -5176, -4091, 2701, 2600, 2529, 5420, -936, 2451, -132, 1881, 2979, -4140, 3765, -204, -3146, 2758, 409, 2339, -466, 3743, 2426, -1292, 258, -954, -454, -3851, -2021, 7798, 2715, 4233, -4983, 8637, 2496, -2579, 3369, -4331, -7819, -1047, 3989, -2433, 4570, 6032, 2676, 5406, 3699, 1610, 3093, 1601, 285, 1394, 2182, -2497, -2966, -8942, -1318, 4779, -5910, -6665, -4338, -445, -10120, 7770, 7642, -5434, 7315, -280, 4173, -2022, 9375, -1380, 2425, -2098, 5544, -4025, 4967, 8433, 5163, -1191, 3299, 1245, 3142, 4348, -3664, 6685, -2964, -500, -12909, -4890, 54, 6825, -6188, 2047, -3386, -7644, 732, 720, -6267, 2043, -3698, 2444, 1398, -10246, 9154, -5266, 3473, -631, 6480, 3803, 1908, 5129, -2282, 9293, -9651, 166, -3612, 1345, -3501, -2596, -6659, 10630, -10383, 4, 524, -3116, -1835, -4080, 2616, 1476, 6390, 2437, -760, -2511, -4377, -4662, 1721, -10502, -3560, 2597, -11382, 4864, 9140, 1652, 12179, -7038, -5007, 1470, 13494, -3422, -3084, 2427, 7785, -10900, -12132, 8367, 287, 3045, -3749, 715, 2394, 423, 4711, -3314, -2867, -5833, 7260, -4937, 3477, -3961, -731, -921, 5163, 865, -6045, -5699, 708, -7667, 2152, 8510, -5508, -579, 1224, -5255, -1675, -10698, -769, 1672, 2480, -924, -7299, 2301, 6410, -995, -5812, 6691, -787, -2309, 6986, -9315, 3912, -1753, 7037, -6732, 1242, -58, 186, 4455, 4104, 3237, -10093, 4587, -4289, -5903, -319, 3493, -7214, 985, -2292, 3024, -3078, 5228, 1264, 2411, -6965, 4120, -9282, 7747, -3202, -1729, 849, -2247, -156, -6484, -4169, -4031, -7752, 305, -6459, 6370, -1786, -5733, 9591, -3653, 2799, 211, -5769, -12901, 8836, -3271, -13288, -344, -2600, -6507, 1341, 1141, -1904, -8551, 3296, 394, -5812, 4795, -4470, 1983, 1218, 6505, -430, -3272, -12635, 7240, -2967, 1665, -7641, 5827, 965, -2402, -1725, 419, -846, 2389, -5074, -4145, -4692, -6274, 6305, 9529, -1535, -3504, -2861, 4774, 6900, -246, 694, -10203, -6566, -1169, -2116, -2346, 1994, 11801, -9070, -1576, 2999, 9442, 10633, 4107, -5566, 10544, 3176, -2681, -7172, 495, 972, -2312, 2053, -650, 5141, -2245, -3192, -14202, 7028, -4688, -10300, -3358, -1812, -812, 5919, -815, 5768, 7385, -1632, 6503, -5861, 1589, 4564, -2459, -6178, 7600, 7825, -2819, 2206, 493, 757, 2851, 1639, 1678, 1602, -1002, -5279, -5856, -2374, 694, 391, 5770, 7432, -6847, 4019, -10148, 243, -4024, 7595, -14185, 1192, 1870, 3, 3809, -1083, 549, 2609, -10810, -449, -395, -3862, 836, 5730, 7619, -300, -3865, 3632, 1122, 3701, 150, -9445, -2084, -3247, 1040, 3075, -32, -2359, 137, 4285, 190, 9786, -103, 2021, 4093, 4473, 7620, 5456, -4749, -10851, -9906, -5158, 6867, -3866, -10108, -4924, 5153, -1630, -3722, 1911, 3372, -6727, -15945, 12553, -676, 9273, -9604, -16154, 3598, 1312, -4244, -6487, 8431, 1314, 8389, 2713, -16680, 8424, 2244, 2517, -5666, -5842, -1805, 11231, -3369, -1882, 9819, 2265, -11318, 14708, -8636, -2961, 5086, -4759, 1530, -3005, -5852, 9111, 8675, 2007, 3333, -3960, 7669, -6115, -1661, -4131, 1481, 8342, -6777, -1080, -1231, 3927, 5746, 1220, -630, 1870, 9536, -848, 1438, -1283, 9875, 5655, -13313, 2841, 395, 3764, -3524, 7957, 4107, 431, -10675, -3633, -422, -1292, -10988, 1160, -5554, -1529, 3294, 6136, -615, -12220, 612, 8760, 17514, -1550, -1669, 3519, -4213, -2458, 774, 3821, 5710, 13, -2651, 2505, 5563, 5904, -3377, -1605, -3606, -3260, -48, -882, -1922, 1019, 5584, -4611, 4084, 1764, 87, 5789, -1865, -10725, -580, 425, -10889, -5478, 1676, 965, 6341, 5240, -7143, 8471, 5247, -6003, -2146, -5260, -5682, 6110, -4019, 3194, -4641, -1739, 3084, -1549, -5364, -6174, 3327, -582, -1191, -2824, -3373, 234, -9209, -2694, 243, -506, -9437, 5505, -4768, -556, 1621, -3816, 1149, -2877, -3023, 639, -1425, 3706, 4632, 2616, -116, -9705, 3361, -1766, -744, 12246, -6797, 4944, -7749, 1629, -5106, -716, -2640, -7116, 8351, 2342, -3555, 970, 671, 7120, 916, 11055, 1007, -1563, 8169, -3702, 1730, 1105, -708, 804, 3859, 1161, -326, 2064, -837, 4238, 3689, 6432, -936, 2848, -4223, -662, -1892, -2434, -9763, -6628, 4220, 6238, 879, -7672, -4061, -3897, 5611, 1109, 1481, -5167, -8339, -4162, 7602, -2384, -12254, -2825, 3173, -2050, 1129, -7893, -1543, 1838, -5832, -7292, 1371, -6552, 7902, -1838, -1491, -1940, 6811, -2982, 2808, -6387, -2838, 8703, -2819, 4988, 4447, 3393, -3672, -1238, 2479, 3770, -779, 1893, 2490, 4047, 5001, 3018, 9730, -9240, -1683, -5500, 4148, -3526, -7818, -7697, -8695, 7270, -8600, -9976, 5841, -3734, 3455, 2884, -771, 584, -5096, -4877, 3464, -6303, 4786, -3361, 6577, -3939, 9709, -2871, -5221, -602, -14668, -3126, -1708, -7856, -500, 3277, 5099, 7104, -1201, 2376, -1381, -3196, -507, 311, -4839, -701, 123, -547, -3119, -2358, 482, 4577, 1750, 1005, -3982, 107, 414, -4822, -11793, 2570, -4708, -8157, 7317, -1272, 1853, 2958, -4505, 5642, -6442, 3037, 2362, -960, -2072, 6143, -3053, 2573, 4062, -361, -9877, -4710, 4530, -4584, -3711, 7886, 3855, 1996, 10132, 9349, -3224, 2062, -753, 3055, -3568, 2304, 518, 2679, -2972, 9191, -1122, 4507, -43, -5025, 5229, -5792, 3400, -813, -72, -1412, 6132, 3303, -822, -7334, -376, -2795, -8504, -3569, 1246, -831, -8613, 2336, -9359, -2793, 3058, -2823, 1697, 2218, 3357, 4648, -13778, 4850, -6216, 4983, 2644, -22007, 1468, 4789, 16887, -2224, -422, -4092, 8269, -5366, -12954, 4575, 8061, -4642, -5033, 4302, 1541, 3173, -12740, -2146, 1455, -4434, -8333, 9806, -7854, 7894, 1395, -1215, -8662, -1144, 6437, -223, -2656, 5234, 4637, -11429, -1264, -442, -2217, 7509, -536, 2986, 9041, -6093, -2114, 4840, -1550, 1546, -2419, 5268, -4408, 3324, -575, -3361, 1866, -4723, 10538, 3812, -936, 1483, 4690, -479, -5707, 5270, -2175, -6757, 2409, -2894, 2166, 1509, -11341, -3752, -577, 178, -2488, -9250, -298, -1482, -6538, -7989, -9023, -5625, 4329, -3406, -3896, -1212, 6885, -6233, -798, -2122, -7747, 3725, -3547, -2748, 251, 5698, 1338, 479, 2075, -4018, -2370, -578, 1726, 7786, 7511, -3969, 1803, -1399, 8115, -3824, 979, 480, 2227, -7575, 4558, -1425, 3369, 7762, 465, 1677, -250, -5639, 7557, 5107, -4734, 741, -5982, -7874, 19, -3197, 1354, -4568, 470, 6376, 4663, 1210, 1922, 1471, -2887, 5015, -5161, 2857, -20, -1223, 11635, -4767, 8102, -9043, -3125, -3885, 1804, -5357, -2263, 749, 3681, -5376, 7698, -10574, -2889, -7248, -4461, 11117, -2653, -5273, -1097, 1441, -1555, 1001, -3963, 5504, 7659, -8176, -2470, 40, 307, -5287, -6219, 4035, -530, 1451, -3389, 2859, -2006, -1258, -514, -6130, -1552, -2210, -1206, -6321, 3558, 210, -1104, 496, -2177, 668, 27, 3881, -2349, 3083, 4008, 4755, 2784, 356, -8130, -2289, 601, 947, -25, 1389, 245, -2066, 4527, -3696, 2571, -4251, -3109, 402, 5479, -3704, -5819, -3589, 8689, -2782, 3897, -2755, -486, -8785, 5031, 15607, 2872, 837, -10124, 6803, -6160, -5926, 8331, 8400, 1292, 8248, -2200, -2867, 13097, -881, -7215, 7114, -5237, -12993, 9821, 3226, 9606, -2645};
static Q15_T L3_W1[L3_CTEMP] = {23351, 10719, 16641, 15287, 15313, 11463, 18717, 18123, 16075, 13698, 17570, 28167, 17899, 24954, 17968, 16541, 14169, 12598, 13568, 16380, 14977, 16673, 13201, 23789, 16111, 18479, 13431, 20426, 21793, 8276, 19708, 22992, 21886, 16867, 17921, 15242, 17329, 17567, 17035, 21442, 13335, 17290, 10515, 15052, 10012, 20957, 10384, 15388, 21464, 11135, 18428, 22716, 14017, 17626, 22372, 16421, 8319, 18218, 15674, 24798, 19830, 10049, 20302, 10940};
static Q15_T L3_W2[L3_CTEMP] = {6078, 7719, 7780, 7678, 7201, 8900, 5490, 10472, 8428, 2473, 5417, 7851, 6243, 6450, 7217, 6365, 10252, 13521, 3870, 9403, 4453, 7664, 3669, 7103, 2868, 5948, 4096, 4643, 2093, 13350, 5371, 5743, 5006, 3546, 5318, 1738, 8582, 4195, 2258, 5403, 2105, 3772, 5500, 5033, 5780, 5425, 5160, 6519, 7993, 5224, 4336, 7120, 3144, 4715, 4362, 2878, 3013, 2837, 4940, 5112, 3683, 20766, 7555, 3819};
static Q15_T L3_W3[L3_COUT] = {17565, 11580, 14172, 15725, 11620, 7797, 12109, 13988, 10424, 13945, 15558, 17286, 16769, 12900, 12247, 12806, 11378, 13371, 7518, 15413, 11025, 13109, 13483, 14576, 15212, 12232, 15196, 11980, 12800, 14811, 12202, 14395};
static Q15_T L3_B1[L3_CTEMP] = {-10000, 4703, -8602, -9349, -7126, 8206, 6925, 1587, -9630, -10456, 10057, 5996, -9223, 17990, -16047, 4229, 9702, -3895, 19114, 10068, 13321, 225, 4227, 2590, -15185, -9682, -20966, -6573, 16655, 203, 5027, 3808, 906, 19824, -54, 13770, 15053, 11120, 17622, -2965, -7019, 8882, 9450, 6644, 19462, 8372, 9267, 20151, -11371, 7401, -10461, -439, 14839, -11669, -9924, 556, -8481, 161, 512, -750, -15775, -14727, 18714, 22331};
static Q15_T L3_B2[L3_CTEMP] = {-3676, -1883, -4783, -4072, -2670, 4148, 14691, 233, -5759, -1761, 11077, -4915, 6004, -3487, -2505, -4699, -913, -2065, -5422, 6797, 3421, -4933, -12516, -6756, -12546, -2024, -7977, -6730, -12846, -1047, -480, -2074, -3252, -16790, -10752, -1471, -849, -2286, -7054, -5071, -9765, -5146, -6768, -6827, 14490, -10137, -6491, 12084, 3750, 11846, -11938, -1572, 4296, 4817, -6052, -7220, -14973, -3879, -7394, -993, -6123, -265, 760, -2376};
static Q15_T L3_B3[L3_COUT] = {1435, 12683, 9284, 574, -2084, 10627, 7682, 13911, 14246, -11129, 10839, -10833, 461, -9132, 3616, 13422, -17028, -12983, 1833, -4272, 1769, -216, -4537, 952, 3744, 9196, -2217, -2692, 6046, 5404, -11374, 2812};

static SCALE_T L3_D1 = 5;
static SCALE_T L3_D2 = 4;
static SCALE_T L3_D3 = 6;
static Q31_T L3_Limit1 = 805306368L;
static Q31_T L3_Limit2 = 100663296L;

#ifdef SHIFT
  static L_SCALE_T L3_ShRU1 = 10; //1024
  static L_SCALE_T L3_ShRB1 = 3;  //8
  static L_SCALE_T L3_ShRX1 = 15; //32768L
  static L_SCALE_T L3_ShRU2 = 10; //1024
  static L_SCALE_T L3_ShRB2 = 3;  //8
  static L_SCALE_T L3_ShRX2 = 12; //4096
  static L_SCALE_T L3_ShRU3 = 9;  //512
  static L_SCALE_T L3_ShRB3 = 3;  //8
  static L_SCALE_T L3_ShRW3 = 13; //8192
  static L_SCALE_T L3_ShLU1 = 0;  //1
  static L_SCALE_T L3_ShLB1 = 0;  //1
  static L_SCALE_T L3_ShLX1 = 0;  //1
  static L_SCALE_T L3_ShLU2 = 0;  //1
  static L_SCALE_T L3_ShLB2 = 0;  //1
  static L_SCALE_T L3_ShLX2 = 0;  //1
  static L_SCALE_T L3_ShLU3 = 0;  //1
  static L_SCALE_T L3_ShLB3 = 0;  //1
  static L_SCALE_T L3_ShLW3 = 0;  //1
#else
  static L_SCALE_T L3_ShRU1 = 1024;
  static L_SCALE_T L3_ShRB1 = 8;
  static L_SCALE_T L3_ShRX1 = 32768L;
  static L_SCALE_T L3_ShRU2 = 1024;
  static L_SCALE_T L3_ShRB2 = 8;
  static L_SCALE_T L3_ShRX2 = 4096;
  static L_SCALE_T L3_ShRU3 = 512;
  static L_SCALE_T L3_ShRB3 = 8;
  static L_SCALE_T L3_ShRW3 = 8192;
  static L_SCALE_T L3_ShLU1 = 1;
  static L_SCALE_T L3_ShLB1 = 1;
  static L_SCALE_T L3_ShLX1 = 1;
  static L_SCALE_T L3_ShLU2 = 1;
  static L_SCALE_T L3_ShLB2 = 1;
  static L_SCALE_T L3_ShLX2 = 1;
  static L_SCALE_T L3_ShLU3 = 1;
  static L_SCALE_T L3_ShLB3 = 1;
  static L_SCALE_T L3_ShLW3 = 1;
#endif
