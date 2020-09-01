// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#define D3NW_HF 1
#define D3NW_WF 1
#define D3NW_CF 1
#define D3NW_COUT 1
#define D3NW_HPADL 0
#define D3NW_HPADR 0
#define D3NW_WPADL 0
#define D3NW_WPADR 0
#define D3NW_HSTRIDE 1
#define D3NW_WSTRIDE 1
#define D3NW_HDILATION 1
#define D3NW_WDILATION 1
#define D3NW_G 96

#define D3CW_HF 3
#define D3CW_WF 3
#define D3CW_CF 96
#define D3CW_COUT 2
#define D3CW_HPADL 1
#define D3CW_HPADR 1
#define D3CW_WPADL 1
#define D3CW_WPADR 1
#define D3CW_HSTRIDE 1
#define D3CW_WSTRIDE 1
#define D3CW_HDILATION 1
#define D3CW_WDILATION 1
#define D3CW_G 1

#define D3LW_HF 3
#define D3LW_WF 3
#define D3LW_CF 96
#define D3LW_COUT 4
#define D3LW_HPADL 1
#define D3LW_HPADR 1
#define D3LW_WPADL 1
#define D3LW_WPADR 1
#define D3LW_HSTRIDE 1
#define D3LW_WSTRIDE 1
#define D3LW_HDILATION 1
#define D3LW_WDILATION 1
#define D3LW_G 1

// The convention followed for filter weights is F[groups][filter_height][filter_width][in_channels][out_channels]
static const Q15_T D3NW[D3NW_G * D3NW_HF * D3NW_WF * D3NW_CF * D3NW_COUT] = {30610, 31397, 31582, 30561, 30710, 31511, 31506, 30974, 30461, 30744, 30616, 31167, 30539, 30422, 31149, 30556, 31281, 31240, 31168, 30562, 30962, 31635, 31164, 30740, 31250, 30988, 30496, 31655, 30771, 30816, 30637, 30947, 31016, 31228, 30639, 30947, 30659, 30772, 30942, 31036, 31194, 30579, 31900, 30629, 30810, 31015, 30882, 30656, 30698, 31557, 30687, 30814, 30836, 30960, 31068, 30658, 30912, 31225, 32062, 30491, 30792, 30827, 30837, 30763, 30876, 31096, 30786, 30570, 31248, 30670, 30580, 30785, 30877, 30600, 30730, 31507, 30810, 31193, 30942, 31089, 30458, 30704, 30772, 30733, 31177, 30881, 30834, 30860, 30932, 31657, 31093, 30725, 30515, 30433, 30935, 30778};
static const Q15_T D3CW[D3CW_G * D3CW_HF * D3CW_WF * D3CW_CF * D3CW_COUT] = {3359, -3464, -1002, 2325, -9947, 9501, -10040, 7736, 145, 770, 8390, -8815, 17142, -15914, 1085, 1196, 8185, -10905, -910, -328, -5054, 2791, 8931, -9066, 5450, -5573, 1381, -2534, 4833, -2110, -1768, 590, -8089, 9996, 5677, -5203, 9178, -7560, -4731, 3995, 811, -1580, 11807, -13797, 2012, -1700, -5652, 6289, 2528, -1885, -5056, 7983, -5285, 5707, 8700, -8639, -1344, 3827, -9245, 11154, -350, -1582, -3176, 2350, 886, 335, 10693, -12611, 11488, -10112, 3370, -2662, 2654, -762, -2241, 4123, 1632, 339, -8737, 8475, 1437, -2087, 5221, -5056, 4656, -5002, -2540, 3450, -660, -470, -791, -1000, -4499, 6189, 3546, -3906, -1820, 1801, 10587, -10737, -3506, 3472, -7393, 6338, -6048, 4108, -2083, 1189, -7035, 6619, 1533, -2121, -3146, 2915, -14953, 16638, -9205, 11828, -7948, 6531, -11893, 10778, -1860, 1626, 9822, -9813, 10092, -11196, -13704, 13837, -6977, 7215, 4992, -5571, -5241, 2573, 807, -1803, 1185, 1548, 3353, -2379, 4332, -5435, 1734, -202, -1270, 2209, -2658, 4660, -12240, 12096, -3397, 3338, 8718, -8419, -5490, 6880, 3435, -2379, -4217, 3274, -1610, 1542, -9280, 8337, 427, -960, 2794, -912, 2473, -1407, 6290, -7023, -1252, -712, 7474, -10046, 6812, -6311, 3060, -3298, -4303, 4508, 4171, -5482, 2092, 294, 4766, -5719, -2478, 1879, -9733, 11299, 1965, -1971, -13397, 14961, -6417, 8700, 2109, -3637, 5870, -3271, 16741, -16974, 3004, -3666, 10109, -11163, -17586, 16648, -4680, 5800, 4542, -5343, 1833, -3782, 4334, -6287, -730, -701, 2560, -1616, -8619, 10572, -10000, 11528, 2252, -4354, 7243, -5408, 2028, -1262, 5123, -5696, 5887, -6650, 12804, -12916, -954, 3421, -1680, 2328, -2480, 3990, 4265, -6359, -987, 2147, -11272, 11150, -3333, 5434, 3204, -1208, 3816, -3616, 11364, -12543, 7674, -10479, 3233, -4517, -591, 567, -3092, 1409, 5155, -3325, -3282, 3534, 6831, -6480, -5723, 3176, 14797, -15478, 1748, -1901, -2338, 117, -13351, 12775, 6391, -6397, -5100, 4749, 6205, -6599, 6393, -6935, -3883, 6884, -11625, 9145, -1365, 887, -10536, 12370, -1472, 317, 6423, -4797, 204, -1015, -6977, 5508, -8130, 7822, -3080, 2957, 12184, -14759, -11313, 12137, 1386, -3063, 4941, -3874, -7942, 8456, -10487, 9806, 4913, -5138, -3619, 3110, 271, -225, -3649, 2764, 2495, -2128, -3719, 3017, 3795, -3564, 8928, -8014, -9560, 8359, -5010, 4540, 3893, -4262, -1129, 296, -12599, 10482, -7120, 8793, -9093, 7492, 3884, -5237, 1613, -3991, -7659, 9240, 1148, -2085, 11947, -12759, 2775, -3880, -1862, 2321, -2812, 453, 8118, -9071, -6617, 4303, 5795, -4994, 3352, -4595, -106, 155, -8998, 8102, -3075, 4436, 4947, -6948, -2011, 4893, 2756, -3632, -668, 3534, 2223, -3116, 6243, -5404, 4504, -5508, -2997, 3564, 5937, -7379, 354, 1853, -746, -1687, 1583, -985, 4387, -3391, 160, 655, 5218, -4297, -6779, 5034, -4852, 5397, -8735, 7613, 1670, -1362, 3335, -1930, -7597, 6021, 39, 539, 4335, -5538, -2065, 719, 2590, -4077, 446, -349, -4993, 4265, 3952, -2071, -2963, 3399, 3956, -2905, 3555, -3469, 1449, -3352, -3832, 1920, 3384, -5089, -1711, -352, 8625, -8426, 4979, -5228, -6561, 4610, 4222, -5639, -5184, 4516, 2841, -3584, 265, -1641, 5915, -6076, -4005, 4742, 2285, -1768, -2202, 2608, 222, 350, 2179, -2619, -192, 519, 1140, -757, 1065, -3687, 921, -505, 10967, -11432, -5361, 6108, -4218, 4638, 2973, -2134, -4543, 4531, 116, -1375, -48, 453, -2339, 2201, 6853, -7379, -3544, 3559, -4509, 2785, 2062, -1534, -5484, 4895, 2191, -768, 6634, -8706, -883, 2434, 2999, -2741, -5595, 5873, 2103, -1898, 2610, -3358, 157, -807, -2571, 3452, -1264, 4087, -7192, 5691, -1147, 1779, 4516, -4186, -1373, 3111, -6853, 7674, -3341, 3376, -5255, 4924, -4072, 4185, -7208, 10341, 3814, -2639, 4973, -2157, 2275, -825, 700, -2621, -3489, 3282, -658, -934, -573, -450, -525, 103, 1704, -2867, -2494, 2877, -304, 1182, 2575, -792, -3774, 3914, 10074, -10542, -12707, 10751, 2433, -3030, -235, 155, 17320, -14186, 20625, -20351, 2653, -2115, -1043, 1416, -579, 529, -18772, 16329, 14385, -14393, -516, 1652, 234, -1981, 695, -1272, 478, -2011, 7406, -7185, -2521, 3257, 3974, -3205, 5666, -5599, 2116, -1558, 11658, -10554, 844, -984, -5712, 3315, 3874, -1828, 7140, -7830, -3421, 4449, -960, 3547, -4328, 4670, -8596, 10060, 1707, -2248, 11608, -10897, -2044, 2964, 7662, -6124, 3568, -1564, 4545, -6951, 2706, -1713, -7393, 8723, -2511, 3672, -1882, 891, 1078, -729, 5346, -4617, -4731, 2006, 1284, -313, 3565, -925, -5436, 6087, 13004, -11769, -210, 260, -5003, 6329, -6392, 3893, -10641, 8492, -3124, 4123, 1801, -1732, -15655, 13141, 1017, -1945, -4812, 6929, 9802, -8531, -7622, 9640, 4939, -3431, -5359, 6448, 523, 961, -7858, 5048, 9888, -10130, 5161, -6017, -11634, 10144, 3234, -3451, 4296, -5988, 7340, -5666, -3337, 1645, -4939, 3121, 3834, -4454, 6333, -5793, 9251, -8898, -1571, 1990, -1631, -973, -18382, 20655, -12810, 14404, 9637, -8503, -10987, 9869, 2942, -2429, 2587, -1149, 277, -39, -3490, 5116, -1808, 3825, 1527, 743, 16223, -17980, 5843, -7797, -669, 1693, 10234, -8779, 11124, -11197, 6243, -4379, -1724, -1109, -2599, 2812, 465, 729, 1246, 482, -2034, 4443, 638, -1160, 13284, -12721, -30253, 31045, 7929, -9694, -7264, 5427, -20841, 21777, 16313, -15471, -6219, 5169, -3352, 1590, 730, 55, -1188, 3148, -5527, 5014, -6717, 8469, 893, -677, 10718, -11663, 2474, -1682, -8312, 5902, -4137, 2402, -4220, 5099, -4865, 5598, 2696, -2093, 17378, -16896, -1741, 4715, 2561, -1094, 6216, -5524, -2747, 2351, 6542, -5831, 9030, -7371, 11981, -10223, -16502, 17713, -18051, 18271, -684, 1455, -7796, 10591, 1206, -1946, 1488, -2714, -3099, 4343, 1596, -1450, -12292, 12985, 17629, -17707, 592, 1031, 5870, -7426, 7906, -6981, 12971, -14154, 6956, -7422, -4715, 3733, -10446, 9679, 424, -576, -5623, 4236, 8290, -7538, 13809, -13575, -14200, 14338, -16695, 15424, 6448, -5769, 5226, -6817, 16602, -14924, -6749, 6132, 8295, -5887, -387, -1130, -7269, 6629, -6009, 7302, 6598, -6635, -8132, 9495, 9150, -6771, 10610, -9797, -3534, 3350, -15615, 16233, 1256, -449, 4025, -1508, 5895, -3374, 12642, -12857, -674, -277, 10573, -8039, 9028, -8662, -774, 1427, -2308, 1174, -8134, 7884, 2136, -967, 338, -1652, -10044, 10238, 13276, -12550, -7221, 9866, 10729, -8232, 2733, -3638, 16290, -16267, -17622, 18365, -5494, 5093, -14733, 14684, -6223, 5421, 6091, -6595, 17980, -17377, -7359, 8835, 5483, -7022, -1704, 2741, 15736, -14278, -5176, 4196, -13504, 11440, 10405, -9977, -2723, 787, -2745, 2767, 6662, -6866, 2630, -2256, 4167, -4787, 8979, -8939, -3425, 4583, -1814, 2641, -2347, 1507, 1056, -2678, -6923, 9174, 1945, 151, -3913, 4569, 2227, -2077, -609, 964, -5739, 4777, -2990, 3232, -277, 3214, -3116, 3755, -556, -788, 5045, -5103, 11136, -13049, 3092, -2727, -359, 2070, 13847, -14004, -4265, 2776, -6227, 5466, -1534, 1363, 3225, -2889, 5592, -5009, -8771, 8320, -6064, 3841, 1055, 443, -5901, 5167, 2152, -1540, 2292, -2692, 3217, -2262, -10304, 9119, 1709, -3186, -1183, -1457, 3252, -3307, 6368, -7734, 1863, -3955, 840, -359, -5130, 4716, -7239, 8494, 374, 602, 312, 1313, -3331, 5428, 502, 1426, -825, 1646, 6929, -6096, 748, -442, -14266, 15191, -6260, 5031, -10432, 10923, -840, -602, 56, 152, -3469, 3626, 6134, -7603, -13402, 10842, -2433, 1110, 9138, -10887, -7385, 6438, -5338, 3925, 6756, -6266, 141, 762, -1497, 632, -237, 1904, 6610, -6133, 4173, -2901, -1526, 379, -486, 1994, -1274, 786, -1718, 2875, -5453, 6946, 2460, -2765, -11309, 9312, -53, 287, -1470, -608, 8560, -7554, 2748, -5029, -8042, 9516, -685, 1657, 5213, -5880, -4289, 3473, -4699, 4326, -1434, 362, 8427, -10417, -8683, 9862, 2629, -2979, -7480, 9702, 1527, -2721, -1405, 1593, -1465, 1207, -1575, 2177, 659, -2013, 11140, -10397, 1037, -4022, 1178, -961, 8921, -10493, 9022, -9426, -2667, 3378, 2192, -3093, -3189, 5491, -8145, 7092, 2592, -889, -5072, 4578, -3490, 3610, 97, -1111, -7914, 9343, 1699, -1977, -7895, 8324, 608, -1363, 167, -1474, -5342, 6257, -11245, 8413, 9214, -9261, -3882, 3597, 667, 750, 1056, -1118, -6049, 7757, -2453, 2447, 3886, -4062, 4337, -4603, 6969, -5609, -1078, 2847, -494, 2525, -2388, 2716, 4573, -4692, 7080, -9136, -6968, 4491, -627, -1090, -5502, 6039, 7500, -6755, 1892, -3030, 3999, -4801, -3661, 5720, -1878, 2015, -463, -1241, 8099, -9396, 6379, -6299, 762, -1637, 5008, -4876, -13169, 15150, -7053, 6320, 1853, -2333, -4373, 5676, -4199, 4109, -2018, 1299, 2989, -3082, 1737, 472, -5848, 5308, 21377, -19374, 1856, 234, -1151, -484, -7010, 6444, 7914, -9490, -13106, 11201, -4200, 5092, 6340, -7316, 10438, -11399, 13344, -12924, 2428, -1688, -3884, 3972, 7369, -9236, 4687, -3346, 5002, -2909, -600, -130, 8433, -6881, 1678, -2625, -5837, 5995, 8911, -8869, -14186, 13246, -7911, 7296, 3934, -5539, 2242, -1826, -4311, 2521, 11505, -13068, 2652, -2281, 12513, -12967, 1110, -531, 3241, -4351, -4933, 7036, -5016, 4421, -2459, 251, -5204, 4437, -6714, 7742, -1320, 2832, -9531, 8513, 8826, -8798, 15523, -14676, 11996, -9758, 4856, -4950, 2401, -5116, 7366, -8158, -13051, 12001, 3367, -3223, -8404, 10125, -3280, 6089, -6766, 5810, 1641, -1062, -2188, 1920, -2081, 345, -2830, 4417, 1411, -1696, -2915, 3144, -9107, 7282, -7814, 4836, 5253, -4942, -3950, 3550, -1996, 3601, 1545, -2751, 6352, -5278, 5905, -4046, -1869, 2926, 4195, -1962, -3779, 4280, 6988, -6312, -511, 305, 11340, -11310, -7369, 5758, -4859, 6145, -1535, -609, 2805, -2840, -3231, 2926, 4295, -6151, -4595, 7195, -2162, 1725, 10436, -9255, 1645, -4615, 13442, -11217, 1116, -4046, 10982, -11053, -5784, 4324, -2113, 2503, -8037, 8157, -2960, 3872, -5547, 4793, -3883, 4824, 6873, -9208, -1504, 3017, -7054, 5313, 10948, -11431, 5653, -5255, 11327, -8151, -305, -424, -3143, 3539, -7916, 5256, -379, 1579, -464, 825, 6200, -6624, 4351, -5339, -289, 193, -3633, 3578, -9324, 11122, 925, -1094, 224, 1087, -7488, 7266, 1277, -3450, 10175, -10666, 6403, -7962, 16885, -15074, 6397, -5803, -5703, 5500, 2343, -3872, -4367, 4746, 4474, -3149, -1713, 602, -5310, 4913, -15754, 13235, -5457, 6247, -3878, 3167, 71, -378, 970, -279, -14549, 12201, -11773, 12845, -2142, 4180, 265, -259, -3838, 3709, 3987, -4422, -1625, 524, -5539, 5844, 5770, -7917, -4370, 3437, 8648, -8526, 1, -1436, -752, -28, 734, 157, 5958, -6469, -2265, 293, 2126, -745, 20920, -20478, -1266, 2331, -5399, 4604, -1788, -507, 1116, -4110, -428, 1366, 5361, -5067, 5659, -5574, -1221, 638, 3938, -3066, -7915, 7988, 43, -1773, -7156, 6378, 3740, -4535, 1734, -364, -9504, 7794, 7338, -6009, -10305, 10314, 5537, -4493, -11071, 11966, -4741, 4958, 3484, -2278, 1740, -3513, -2374, 1687, 5395, -4043, 2104, -2674, -2254, 1448, 3127, -3772, 12675, -13011, 80, 693, 16693, -16593, 3810, -4302, -6910, 9813, 3735, -4778, 9065, -9896, -1169, 2511, 1074, -1260, 12561, -12468, -4983, 3667, 7015, -6743, 4921, -4894, 4399, -4125, -67, 160, -5746, 7295, -912, 2662, 3285, -813, 10490, -8591, -1997, 4008, 5633, -5490, 578, -357, 2398, -2988, 3231, -3832, -12511, 11556, 5941, -6682, -5404, 6410, 416, -733, -469, -1441, -10016, 12164, -6382, 7944, -4418, 4749, 6774, -8440, -6781, 5012, -2089, 3587, 6380, -6963, 4412, -5992, 7142, -4829, -5868, 3074, 192, -2446, -1815, 1448, 455, 725, -11535, 12646, -1981, 1429, 11538, -12751, -1798, 1642, 211, -533, -2607, 2747, -11538, 12375, -12751, 12416, -6288, 5286, -234, -681, 7392, -8420, -2369, 937, 3056, -531, -4242, 2476, -4605, 3540, 4222, -6526, -6808, 9112, 5044, -5756, -5324, 2584, -286, 2418, 2276, -4118};
static const Q15_T D3LW[D3LW_G * D3LW_HF * D3LW_WF * D3LW_CF * D3LW_COUT] = {679, -1047, 2680, 2446, 150, 255, 1081, -2515, 1015, -267, -2387, -5350, -377, 3485, -2879, -196, -2367, -796, -250, -1344, -500, -80, 475, -1090, 683, 1582, -128, -4086, 1436, 2112, 1090, 692, -537, 894, -1230, -1232, -2327, 1460, 624, 1929, 1446, -3292, -2913, 2649, -2022, 6229, 1551, 1335, -2748, -2369, 498, 516, -1392, -1135, -1697, -253, -2257, -101, -58, -3301, 2273, -1546, -1975, -1948, 1185, 1781, 1916, 1254, -2214, -4662, 2322, 3111, -1283, -2408, 332, -1442, -1783, -632, 4613, 4089, 3179, -444, -2417, -67, 5477, 571, 375, 2487, -1028, 475, 168, 186, 1290, -4606, 2494, 2754, 2484, 1201, -2132, 447, -2328, -934, -1189, 605, -334, -1114, -4570, -3259, 268, -1883, 2254, -160, 741, 2368, 671, 2150, -1516, -1907, 3133, 1904, -2348, 2846, 1845, 516, -1458, 4319, 1219, -408, -393, -4068, -1797, -349, 4301, -394, -2414, -4030, 1857, 514, -1894, -3231, -1352, -4714, 284, 1014, 3158, -1547, 14, -643, -3584, 170, -311, -4487, -1166, -439, 169, -1005, -2121, -740, -1169, 2085, -32, -447, 2946, 1851, -323, 3571, 104, -2586, -1422, -1008, -1697, -538, 2298, 269, 1583, -1390, -1272, 556, 175, -3726, -1752, -1906, -2174, -1663, 3708, 2576, 1341, 2105, 516, 789, 1905, 2050, -4193, 1209, -478, -47, -1554, -3131, 1455, 1058, -1247, -1771, -2429, 491, 32, -4636, 2992, 1848, -1699, 1728, 436, -1007, 1908, -725, 3085, 2233, 1614, 857, -3253, -1379, -2163, -911, 1982, -751, 57, -425, 1610, 3084, -3080, 2708, 668, 2168, 2562, 8128, -126, 213, 88, -2580, 2260, 1483, 1804, -565, -666, -2110, -3338, -4620, -837, 4562, 2772, 3464, -2421, -3244, 2341, 1431, 380, 810, -811, -4011, -929, 903, -3041, 4688, -3537, -3144, 113, 360, -2737, -4774, 1725, 2651, -1740, -2972, 117, 2337, -5269, -1789, 854, -2543, -1524, -3879, 83, 117, 1368, -1294, -2275, 2602, 1122, 989, -1279, 3355, -918, -2839, -2858, 1705, -3490, 439, -758, 6329, 2890, 1508, -4595, -3494, -588, -1104, -889, -2923, -311, -4427, 456, 2902, 1471, -902, -903, -72, 3568, -373, -2290, -1425, -158, -1672, -3872, 4763, -1644, -2380, 3558, 3639, -1736, -1925, 3149, -295, -44, 3264, -1727, 189, -2642, 193, -2714, -1597, -2062, 1917, -1903, 1774, -322, -684, 450, 2008, -2757, -4172, 2521, -2751, -1066, 1305, -1018, -1772, 2518, 1254, 592, 1970, 1041, 1358, 1281, 231, -3102, -2518, -4142, 278, 2773, 877, -245, 270, -193, 1683, 75, -1558, 2706, 1972, -1006, -103, -1745, -1952, -653, -811, 314, -1883, 2980, 2299, -150, -3783, 1108, 8976, -3910, -8641, 1719, -4511, 517, -4953, 4040, 3289, 1902, 465, 2710, 5858, 3492, -3588, 5409, -5401, 3512, 2777, -1331, -3804, 4206, 2416, -6117, 2033, 1012, -2281, -36, -901, 784, -3427, 2795, -4773, -3729, -3880, -3365, 1847, -1609, 2340, -4437, 5427, -131, -1188, 3103, 562, -2442, -1663, 4456, -3957, -894, 1451, 1535, -6827, 1242, -809, 1587, 1683, -2936, -2887, -5761, -7466, 7701, 1104, -149, -7725, 4179, 5875, 1660, -8469, 945, -2285, 5342, -3943, -318, 2763, -7581, -2286, -9476, -2459, -5829, 5954, -3603, 3000, -8005, -2916, -1748, 1458, 1731, 6124, 7827, 1497, 3970, 4954, 4711, 3507, -523, 218, -393, 3291, 4813, -463, -2101, -2666, -3906, -4480, 2792, 5752, -6882, 4287, 4602, 2220, 2695, -628, 1440, 2820, -3626, -5494, 2373, 3984, 262, 1304, 4965, 357, -7104, -9360, -2427, 3871, -9122, 616, 2322, -5322, 1616, 2514, -766, -740, 5239, -6967, 1619, 3926, -2420, 429, -2693, -2268, -463, -2923, 4471, -1127, -1965, -174, 2122, -3006, -1186, 3424, -2769, 582, -1788, 879, 3001, 1384, -2806, -2177, -297, -3523, 2234, 4160, -370, -1352, -3550, 6611, 153, 60, -5336, 70, -5448, -3165, 289, 677, -4406, -10723, -3887, 4203, 7075, 2221, -1985, -3839, -1896, 53, 1864, 1311, 4171, 4108, 2834, -2831, 858, 2801, -1023, -3337, -3131, -2496, 2652, 2175, 1280, 358, 4128, -1481, 1394, -582, -3379, -403, 5548, 2921, -4497, 2810, -5288, 187, -2498, -2781, -1274, -600, -6372, 8556, 6, 2251, 5995, -6027, 688, 4984, -3958, 4496, -1343, -962, 3737, -520, 1307, -696, 1641, 7780, -418, -6544, 2756, -5102, -2306, 5438, 0, 2687, -5080, -2512, -324, 2709, 3206, -5755, -3159, -9549, -5008, -243, 2427, 2596, -6136, -767, -5153, -277, 1323, -673, 645, -2902, -1217, -5643, 563, 9124, -7188, -2856, 963, -860, -1380, -6528, 502, 5731, 2314, 765, 3413, -171, 568, -2653, -1048, 413, 211, -2303, 5140, -4063, -5074, 4835, -760, 4289, 3563, 1461, 3080, -2544, 1877, -2686, -6281, 5976, 3827, -4027, 4236, -1733, -17, -3861, 4091, 5227, 6444, -4322, -1071, -2231, -3749, -3280, 399, 68, -789, -825, -3223, 3505, 66, -221, -7081, -7897, 1846, 3039, -1292, -4914, -1433, -250, -383, -7724, 3249, 6222, -2027, 2106, -222, 6694, -2218, 1654, 1332, -5082, 583, -7294, -1365, 6052, -1165, -1632, 5009, -18, 5045, 6795, -7507, -3938, 1497, -3008, 5076, 2076, 2006, 602, 6028, 1018, -4310, 3951, 74, -658, -812, 979, -94, 1900, -3342, 1629, -4673, -4794, 3380, -4367, -3365, -2568, 411, 0, 1891, 290, 414, 286, 588, -1525, -1867, -2836, -1553, -2071, -1389, 2749, 513, -72, 423, 1684, -4159, -3489, 864, -623, 319, -300, -546, -4279, 3257, 4076, -896, 679, -481, -1977, -3756, -1678, -262, -3038, 78, -630, 656, 2029, -928, -798, -1560, 2206, -286, 724, 252, 1448, 1404, -2801, -2723, -467, -1832, -165, 431, -841, 2130, -238, 236, -2011, -4392, 934, -3476, -2375, 1610, -4435, -374, 1828, -1172, -513, -454, 1040, -140, -1896, 1711, 1401, -1133, -930, 2589, 1534, -945, -1457, -2526, -2943, -917, 526, -3594, -1569, 628, -42, -1841, -1602, -238, 2202, 1443, -96, -3375, 307, 634, 2545, -379, 3843, -334, -1069, -2535, 5315, -406, -2855, 1985, -3580, 2209, 1216, 5, 3103, 217, 1363, 3944, 298, 3216, 446, 375, 2698, -627, -2751, 1477, -2895, 1793, 2615, -2137, -3497, 284, 2380, -2342, -1482, 2530, 369, 1132, -2035, 1447, -103, -1136, -256, -1588, 670, -2604, -2458, -4823, -3579, 323, -2576, -1911, 1892, -2969, -2478, -364, 106, 3063, 1500, 1177, 1489, -340, 1377, -614, -1716, 1282, 3999, 2658, 268, -2164, 4024, -1701, -1701, 402, 2572, 301, -548, -949, 1482, -2052, -4710, -185, -37, -2458, -4889, -2554, -3135, 3228, 5010, 911, -245, -1012, -1297, -429, 2605, 1003, 2787, -42, -2644, -708, 3492, 1466, 879, -633, 69, -164, 877, 512, -138, -1722, 3731, 493, -1752, -623, 189, 2290, 3243, 2270, -2946, -3056, 1029, -164, -2111, -867, -2234, 1544, 215, 845, 2503, 1317, -1982, -241, 1419, -1424, 549, -1338, 97, 1473, -2916, 273, 2387, -3983, 524, -3206, -4023, 1098, -1987, -242, 4259, -4686, -2542, -493, -550, 782, 4296, -308, -1394, 865, -2750, -526, 1077, 1034, 819, 365, 1147, 533, -118, 819, -273, 185, -908, 128, -3359, -1898, -402, 456, 2581, 1595, -1293, -2000, -5215, -3824, 2040, 345, -1193, -42, -4999, -607, -1176, -2333, 2550, -1818, -119, -1545, 856, -1780, 1718, 1741, -1954, 2667, 2961, 5518, -2113, 2701, 1654, -1408, -718, 3840, -1317, -177, 5056, -4621, -4965, -753, -1303, 2777, -3677, 2063, 1037, 835, -1236, 1864, 1957, 963, 1318, 1680, -1366, 453, 479, -177, -4306, 1155, 3023, 1144, -1192, -411, 1508, 1952, 477, -967, 406, 92, -519, 146, 3000, 1023, 1288, -242, -2412, 1467, 139, -240, 339, 3402, -248, 1195, -377, -2470, 2022, 566, -79, -1358, -2803, -2161, 435, -920, 1282, -581, -1688, 409, -1332, 441, 485, 425, -847, -944, -705, 1765, -933, -273, 2691, -1005, 3063, 393, 508, -681, 1536, 1282, 2321, 3687, 1624, 1131, 703, -2508, 2339, -1283, -4205, 3859, -339, 2394, 3323, 3532, -1280, 3307, 3240, -5113, 3125, 2157, 3414, -234, -5755, 1437, -467, 62, -1913, 3722, 3214, -728, -6458, -1494, -1551, -3033, -1242, -3583, -666, 1913, 1573, -2880, 1770, -899, -3119, -536, -648, 1181, 484, -2602, -1027, -4697, -2426, 527, -936, -8124, -7525, 555, -2127, 2019, -5042, -1322, -1132, -2130, 4711, 1033, 1570, -4118, 2458, 4215, 4528, 1263, -1208, -5004, -8104, 927, 2398, 5043, 2657, 2110, -2387, -2617, -655, 6051, -27, -1472, 2404, 57, 67, 1075, 3208, -566, 680, 3344, 2616, 506, 2291, 796, 2956, -51, 5324, 342, 3151, -3301, 1268, -693, -347, -3115, 5267, 1183, 340, 2995, 2585, 1133, 1981, -679, 3301, 799, 118, -4990, -2505, 106, 547, -3668, 8529, -389, -1223, 3421, -5574, -139, 3187, 2260, -11274, -3561, -8916, -668, -3524, 1055, 2474, -1739, -2136, -1982, -1255, -496, -5169, 2786, -1044, -5247, -7782, 3616, 1329, -676, -396, 1344, 2744, -5965, 4560, -4110, 812, 4085, -4005, 2651, 2026, -2511, 543, 4071, -1166, -3332, 1767, 2105, 2623, 4608, -4688, 561, 1811, 570, 1557, -1136, -2153, -6269, 744, -3257, -2855, 5059, 1283, 1416, 1651, -544, -3845, 4814, 2979, 161, 2909, 122, 4347, 124, -6678, 1245, 1549, 139, -3700, -1482, 4020, 1133, -2823, 2564, -561, -203, -4345, 903, -3717, -1027, -2438, 3717, -268, 1733, 577, 522, 5237, -3303, -321, 1663, 271, 1127, 3293, -2029, -1367, 88, -1620, 2092, 4451, 9231, -3276, -4156, 1894, 3082, -30, 3630, 2609, 2147, 2172, -388, -4036, -1477, 2135, 166, 3254, 3567, -1146, -1485, -4306, 3545, -2339, -839, 1299, -2480, -3359, 416, 35, 1826, 3918, -3619, -833, -2193, -2686, -1858, -1694, 353, 5685, -661, -478, 1653, 2182, -3756, -3224, 71, -108, -502, -3471, -3071, 9316, 5202, 2513, -2293, 1147, -1272, -99, -3617, 3109, -3529, -5678, 1291, -2294, -2190, 3569, 1697, -2044, 966, 1632, -4746, 2781, -931, -4790, -2044, 4915, 553, -4968, 1456, -8722, 3824, 3294, 1292, -4768, 441, -3837, 985, -6465, -998, -2013, 716, 365, -2687, -3263, -1214, -2224, -2644, -559, -775, -3453, -1009, 1511, -5699, 326, -1242, -4868, -3245, -2486, 2748, 1616, 2739, -618, -909, 193, 939, -342, -1056, -4907, 1185, 13373, -450, -1660, 254, 3423, 3112, 1492, 3037, 1119, -2842, -1095, 215, 4403, 4421, 2708, -2687, 3014, 3008, -1300, 2673, -3090, 14, 2831, 3530, -2524, 3949, 3205, 2469, -3497, -3969, -5026, -3366, -1924, -1784, -182, -2235, -2057, 1841, -724, -20766, 8595, -4923, -5871, 5841, -17690, 4074, -2878, 8094, -3636, 3056, 2862, -2492, 6980, 5849, -5521, 4674, -991, 6028, 11713, -4836, -15390, 7105, 738, -4264, -17960, -3381, -6947, -3760, -254, 1445, -1268, -6927, 12788, -1, -2173, -697, 1499, -2394, 5758, -19019, 4993, 8728, -574, -6268, 1909, 2094, 5426, -4423, -6552, -2575, -1195, -3430, -22488, 3044, 2794, -5406, 2836, -5191, -3992, -17452, 10608, 11280, 5304, -1003, 16704, 11827, 6329, 14058, 20799, 1686, -3367, 1422, 1319, 1197, 5960, -15050, 7926, -8569, 2866, 25037, 3629, -8130, 1373, 8467, 20619, 5346, 6835, 8333, -3225, 6423, 1883, 14880, -21688, 5989, 4972, -11874, 13730, 2633, 10519, 1685, -5814, -3819, -1196, -28820, 9005, 4792, 12514, -1523, 12553, 10177, 4776, 323, 6564, -2411, -1597, -2434, 2842, -1993, -87, -5756, 18437, 2639, -5537, -11342, -12162, -6278, 1068, -7535, -14079, -108, -7271, 4013, 3855, 3728, 8404, 15146, 326, -5577, -643, -6730, -12738, 2569, -1903, 7108, 1967, 6270, 2598, 5884, 4553, 9755, 10304, 12219, 17636, -6580, 810, -17504, 2350, 11712, 11649, -3189, 6889, 5563, -2392, 24617, 17140, 6627, 3345, 4761, -11063, -4000, 3914, -4969, 6170, -13602, -7162, 10973, -3164, 1450, -5896, -9714, 7997, 7323, -5486, -2215, -14160, -2488, -1829, -10463, 1720, -66, 3885, 3805, -27572, -3358, 32, -4972, 6390, -1966, 1234, -10272, -627, -4153, -6395, 12068, 9730, 4955, -1325, -8961, 6404, 8090, 6166, -8501, -9364, -3673, 8913, -12705, 7413, -4882, -4613, -8674, -13048, -4135, -3905, 14952, 1543, 1069, 14130, -30148, -5319, -9532, -7000, 6423, 2118, 3680, -103, 2753, 778, 1018, -8780, 7007, 5111, -3531, 6987, -10422, 2252, -8306, -4113, -3331, -4166, 4333, -3075, 3379, -101, -8846, -1132, 13866, 5301, -13124, 1212, -13623, -3934, 4991, 2438, -3596, 3639, 436, -1329, -17075, 308, -14732, -14695, -782, 4559, -3253, -9220, 1592, -899, 5980, 240, 1207, 12526, -4113, -6155, -2076, 6504, -11320, -10987, -1251, 458, -8927, -563, -6746, 3418, 10455, 12147, 19595, 8996, -1269, -9812, 4889, -3590, 4918, -5992, 18188, -1494, 2479, -1482, 3253, -6181, 5026, -4252, 4283, 11057, 6667, 5586, -4430, 386, 2608, 966, -8738, -1595, 1789, 320, 1036, -6575, -3227, 4875, 4773, -991, 330, 1539, -13779, -1414, 10801, 11565, 4459, -5574, 2187, 5067, 940, -12211, 880, -7067, 273, 12849, -4062, 1248, -16470, -2033, 11870, 8477, 15730, 6459, -16992, -10235, -13201, 11881, 10980, 3947, 5739, -10011, 1772, -10710, -4891, -6696, -4487, 1051, 1950, 20, 4378, 95, -6676, 17703, 4340, 2795, 16241, 5323, -3219, -527, 18, -2645, -63, -1654, -5513, 448, -4680, -550, 148, 1211, 1708, -1451, -3400, -795, -1039, 351, -2591, -2803, -554, -3599, 5222, 4571, 2488, 5689, -1154, -7403, 1566, 1820, -1827, -4623, 5553, 2472, -1533, -1234, -491, -1953, 381, 1049, 767, 1890, -1726, -397, -5458, -1806, 3065, -639, 2515, 1283, 2021, 3917, 911, 747, 1714, -3916, -375, -1815, 3481, -7077, 3115, 27, 40, -241, -4034, -2387, 5751, 2024, 687, -349, 1671, 1314, 5771, 3791, 2181, -1595, 1314, 21, 3156, -2784, 1359, 2135, -6004, -536, -2885, 2202, -4876, 5023, -5810, -763, -2390, 4041, -113, -14, -3969, 2321, 1942, 2012, -581, -4370, 303, 4466, -4506, 955, -4785, 580, 602, 508, -2043, 569, 4202, 4694, 1775, 3534, 217, 4617, 2565, 437, -47, 722, -1530, -3862, 2856, -3804, -1100, -1464, 1919, 1703, 248, 803, 223, -2346, -2240, -55, -1106, -1198, -1641, -4971, -1975, 4297, 333, -1718, -3364, -3892, -1501, -1029, 2482, -1496, -355, -3890, 1623, -1660, -1527, 122, 846, 2059, 1873, 3345, -1460, 7212, -2780, -1564, -3820, -7343, 4097, 1170, 1372, 6955, 3137, -1939, -6082, -3088, -130, 2286, 427, -634, 2749, 4719, -4599, 4109, -3922, -3281, -3556, 2732, -726, -3158, 1185, 450, 3110, 647, 843, 167, 2808, -1257, 739, -1468, 1758, 5512, 1492, -8326, -1292, 2265, 5713, -239, 250, 4366, -2364, -446, -2554, -3894, -1227, -6855, -48, -3869, -367, 1866, 2564, 4585, 4218, -1552, -2805, 3116, 2604, 2017, -3038, -3994, -5188, -1763, -1341, -1073, -175, -5754, -425, 2130, -2977, 170, -2263, 2934, 431, -707, 91, -1931, -2459, -583, -1753, -4429, 197, -67, -262, 3970, 4519, 2078, 913, -2892, -3741, 2100, -3447, -631, 2831, -5393, 2264, 2848, -2659, 432, -2315, 1855, 674, -4827, 961, 279, 410, 561, -2950, -3860, -5073, 2041, -1902, -2557, -477, 856, 15, -3382, -1840, 745, 2485, 731, -1126, 1143, 1, -4620, 5038, 5585, -2193, -598, -2159, 1762, -4308, -384, 655, 1394, 763, 2673, 3278, 761, -4690, -4614, 1509, 3849, 3599, -282, -432, -682, -3760, -3970, -1446, -337, 3576, 272, -2471, -2489, 1953, 1555, 1558, 2658, 181, -501, -1761, -934, -1054, 663, -1545, 130, 1932, 5736, 3255, -3202, 916, 1546, 6424, -1945, 3236, 1579, 155, 2598, 2422, 3240, 2329, -5700, 1744, -4334, 3881, 1772, 3199, 3564, 939, -1898, 2367, 1452, -8643, 2946, -4336, -236, 4034, -1304, 778, 248, 1662, 404, 3747, -867, -2713, 426, -1299, 714, 2530, 2544, 318, 390, 2616, 414, 1029, 1710, -2699, -470, -3682, 17, 332, -3952, 727, 2325, 4096, 1004, 1817, 1398, -2066, -2896, 1394, -164, 1847, -2681, -353, -611, 486, 1452, 3713, 680, 2816, 473, 1813, 2510, -3125, 1317, 1199, 112, -4059, -2405, 1230, 845, -1700, -148, -1877, 204, -1336, -475, -2005, -3325, -877, 1214, -2698, -2398, 760, -6593, -850, -4870, -2249, -1532, -2461, -1771, -154, 422, -1958, -94, -902, -4566, 588, 764, -1109, 1107, -1081, 5157, -494, -485, -376, -221, -4770, 1476, 2378, 122, 2124, 169, -1122, -2776, -1580, 2059, -348, -120, 1644, 1083, -754, -231, 2023, -2942, -334, -4, 2506, 456, 2838, -271, -1460, 1823, 1177, -1577, 2322, 760, 2201, 1776, -1268, 1411, -2564, 73, -1381, 2126, 1991, 467, -816, 364, -208, -389, 552, 165, -1117, 70, 792, 1308, 2498, 334, -427, 2510, 674, 272, -1216, -2750, 36, 1127, -1470, -1305, -2893, 2034, 257, 1121, 2034, -951, -2373, -5062, -2002, 892, 407, 6186, 408, 59, -2291, -1503, 4042, 1585, -1894, 1583, 1529, -333, -356, -1611, 1871, 1160, -1037, 4047, -911, 1279, 2356, -100, -661, 1650, -540, -1345, 281, -1161, -292, 367, -1049, 1559, -1272, 1815, 792, 418, 743, -1032, 867, 42, 639, 2035, 326, 901, 1979, 2015, -933, 1435, -1875, 2800, -2009, 1290, -1025, -2675, 1293, 2742, -1815, 1330, 3027, 2656, 280, -1575, 853, 809, -465, 3354, 171, 696, -3354, -995, 868, -1317, 330, 2152, 3490, 1246, 1755, 1522, -1131, 252, -1053, 1994, 426, 1199, 1488, -2619, 868, 39, 63, 1684, -1761, 3034, -357, -1724, -3471, -58, -164, 2525, 2721, -55, 2330, -1773, 1940, -962, -1517, 1803, 1697, 2579, 3684, -2222, 2325, -622, 797, 885, 1687, 2728, -714, 1938, 48, 2782, -561, 2414, -239, -1847, -1100, -1559, 1485, 1201, -1128, 3127, -2441, -546, 844, 547, -1370, 1886, -840, 2053, -406, -1159, -1748, -412, -748, -2205, 1362, -2881, 354, 547, -2496, 228, -71, -4430, -1284, 3913, 3022, -1683, 2208, -316, -1013, -1971, -1570, 4614, -2990, -2118, 947, -2157, -1555, -2999, 2425, -3004, 3237, 3282, -223, 2943, 511, 173, -3705, -1268, 2114, 1999, -2083, -1184, 78, -2962, -433, -5015, -3592, -1149, -236, 451, -1516, 3841, 1351, 2431, -415, -4325, -555, 2020, 1995, -2296, -3240, -3088, -30, -252, 1719, -3266, -3412, -3309, 3502, -222, -1032, -108, 835, 185, 341, -2331, 3253, 1145, -1914, -715, 686, 1603, 6, 1054, -2476, 2636, 1933, -937, -1036, -2847, -940, 1236, 1134, 937, -1723, -330, 1910, 1297, -1410, -656, -35, 2291, 323, 3487, -2257, 1653, 5026, 3135, -2683, -10542, -2968, -2961, 5445, 3030, 5970, 334, -1068, 1908, 2695, 81, 3648, -7812, 7626, -3272, -4549, 7952, 1787, 5600, -5098, -4884, 1979, -340, 1214, -8290, 1189, -495, 3993, 273, -976, -1583, -3639, -251, -1870, -1597, 656, 1160, -3559, 161, -7697, -4380, 3562, 830, -3565, 6808, 2519, 3941, -3628, 1796, -840, 2210, 2100, 5648, -1547, 1251, 130, 3091, 37, 4180, -4468, 3991, 5065, -1417, -2428, 4056, 6761, 1633, -2465, -1243, 1756, -2684, 4764, 344, 197, 3003, -4702, -694, -4340, 1621, 3746, -4272, -3470, 3131, -2595, 821, -1446, -6396, -120, -3044, 3514, -5818, -3088, -1701, 1377, -3312, 403, -2541, -2397, 4145, -1419, 4152, -109, -1391, -2364, 5120, -1202, 1439, 3043, 24, 5075, -787, 33, 6031, 1605, 3906, -2880, 4644, -1294, 48, -1553, -3948, 896, 122, 3294, 7642, -5336, 5858, -3046, -4025, 154, -837, -3265, -5753, -42, 2061, 1137, 2067, -3701, -509, -623, -1280, -2729, -3159, 1720, 3702, 5882, 1982, 4442, -6288, 4673, 4482, 1816, 1277, 1522, 1808, -6410, -4309, 681, 5519, -1883, 4742, 1289, 641, 7838, -11241, -2880, -2372, -874, -4796, -2558, 2758, -6228, -2301, -6583, 51, 4694, 6854, 5707, 2460, 1210, -1722, 2404, -1738, -2105, 8031, -2027, 1816, -4792, -6701, 418, 4251, 2489, 5175, 1076, 2412, 1214, -6862, 1881, 974, 1828, -714, -6147, -4825, -889, -1845, 2256, -115, 1807, 2067, 5265, 7924, -6373, 3077, -2576, 3315, -4100, 1049, -3315, -3050, -6537, -3708, 503, -372, 1319, 1320, -3279, 9332, -5672, -9836, -6193, -5241, -335, -4887, 1596, 209, -1455, -3097, 4415, -1324, -1024, 6548, -2735, 2502, -1165, -3496, 215, -285, 1808, -1394, 3550, 640, -2924, -2398, -5004, 2603, -3366, 2289, -7028, 1615, -4349, -1611, 5583, 653, -5802, 610, -2413, 82, -3948, 1182, -2872, 683, -1590, -1147, -4878, -4699, -3187, 40, 2347, 689, -4005, 87, -319, -1576, 965, 5705, -4069, -4661, 5655, 918, -3511, -3188, 0, 1043, 1826, 891, -2440, 1122, -3420, -6135, 6247, -2214, 3720, -6080, 8981, -7541, 268, -1581, 7775, -1234, 2074, 1370, 7045, -13060, -3080, -3325, 591, 1617, 3776, -1019, -9187, -4530, -4780, -348, -2165, 778, 841, 7829, -1022, -1613, 2703, -624, 1131, 4725, 2789, 1072, 1370, -1704, -2475, -2744, -1437, 8329, 2828, 1076, -532, 7306, -1828, 3526, -1832, 2512, 2905, -1753, 5716, -6356, -5236, -5012, 1702, 11755, 5784, 1773, 1436, -1352, 2033, -4269, 1631, -3956, -443, 2159, 311, 369, 1384, 2688, 3583, 4912, 492, 2748, 1996, 1353, -2865, 2697, 1757, 2554, 2700, 777, -4269, -2985, -2868, -1339, -311, -2015, -1843, -1990, -1611, 1077, -412, -297, 779, -1344, -291, -3531, 784, 2314, -1500, 331, 1180, 1986, 2706, 3921, 655, -1205, 6445, 3203, -1153, 2235, 51, 17, 1155, -2527, -392, -1354, 823, 1541, -4198, -2316, -2732, -736, -527, -317, -338, 1712, 900, 1222, 435, 220, -504, 456, -511, -2950, 460, 282, -2248, 2302, -2471, 2867, 905, 3039, 419, -1423, 1606, 2305, 2959, 926, -2233, 3385, -262, 375, -1298, 5446, 2870, 2721, 223, 4861, -1503, 296, -1151, 1688, -3773, -1717, 146, -2491, 459, -2248, -2398, -3272, -1717, -4256, -520, 4418, 558, 2248, -2137, -1520, -3165, 667, -794, 1764, 1017, -1830, 45, -75, -951, -2749, -918, 103, 2703, -826, 201, 435, 3125, 1335, 956, 2141, -218, -1323, 2205, 1189, 107, 3428, 1262, 2769, -1281, 4800, 1515, 2960, 1756, 2211, 522, -5405, 443, 273, -1344, -1845, -838, -1812, 2644, -1460, -340, -1831, 5580, 4257, 1571, 2200, 1747, 2838, 354, 2003, -985, 423, 952, 589, 2041, 2276, 2922, 2815, 2749, 2178, 1382, 613, -2425, -2588, -928, 640, 1121, 1430, 1483, 1583, -1459, -2456, -1556, -443, 1780, -1650, -2520, -2954, 112, 1435, 4328, 3650, 1576, 2443, 2376, 1215, 2725, 768, -3777, 720, 1915, 404, 2482, 2690, 5600, -1907, 2313, 2908, -415, -1518, -2200, -328, 769, -2690, -1019, -2661, -1890, 487, 1957, 6243, 3446, 6291, -274, 3190, 447, 20, -1145, -39, -1864, 2035, -4014, -2407, 2081, 1789, 1296, 3419, -2996, -1887, -1364, -1949, 2304, -3456, 1881, 434, -607, 1001, 435, -2902, -2387, -250, -2212, 180, -1071, -555, 2529, 1073, -2371, -5862, -4816, -1812, 1093, 3037, 1026, 3629, -121, 2230, -1270, -577, 515, 4463, 3284, 1582, 843, -976, -1817, -2130, -4123, 109, 621, 411, 2462, 2023, 927, -1481, -1473, 523, 1107, 189, 1990, -1770, 2657, -2148, -29, 1178, -2031, 1410, 2690, 3631, -2110, 471, -1053, -666, -1423, -229, -288, -508, -5466, -3083, -2094, -1465, 2539, -308, 721, -5985, -746, 1253, -1861, -4145, -1955, -582, 836, -3961, 1138, -1270, 2530, 1013, 2783, 1029, 1410, -1580, 1158, 2140, 208, 1151, 1902, 5064, 834, 1385, 3605, -334, -1119, 3477, -808, -1227, 2752, -2453, 1044, 1186, -1276, 1689, 1542, -87, 1575, 2920, 1939, 3983, -257, 3651, 2016, -1544, -4706, -3740, -4513, -2284, 2183, 3518, 223, 1318, 3050, 513, 1743, -538, -87, -1857, 1334, -1080, 2833, 144, -767, 1041, -560, 4396, 463, 965, -3162, 196, -1015, 601};
static const Q15_T D3CB[D3CW_COUT] = {29336, -29299};
static const Q15_T D3LB[D3LW_COUT] = {-1909, 9320, -24331, 32138};

static const SCALE_T D3_ScaleIn = -12;
static const SCALE_T D3_ScaleOut = 8;

#ifdef SHIFT
  static const SCALE_T D3NW_Scinput = 5;     //32
  static const SCALE_T D3NW_Scoutput = 5;    //32
  static const SCALE_T D3NW_Demote = 0 + 0;  //1 * 1
  static const SCALE_T D3CW_Scinput = 4;     //16
  static const SCALE_T D3CW_Scoutput = 5;    //32
  static const SCALE_T D3CW_Demote = 0 + 10; //1 * 1024
  static const SCALE_T D3CW_Scten = 0;       //1
  static const SCALE_T D3CW_Scvec = 2;       //4
  static const SCALE_T D3CW_Scret = 0;       //1
  static const SCALE_T D3LW_Scinput = 4;     //16
  static const SCALE_T D3LW_Scoutput = 5;    //32
  static const SCALE_T D3LW_Demote = 0 + 10; //1 * 1024
  static const SCALE_T D3LW_Scten = 0;       //1
  static const SCALE_T D3LW_Scvec = 5;       //32
  static const SCALE_T D3LW_Scret = 0;       //1
#else
  static const SCALE_T D3NW_Scinput = 32;
  static const SCALE_T D3NW_Scoutput = 32;
  static const SCALE_T D3NW_Demote = 1 * 1;
  static const SCALE_T D3CW_Scinput = 16;
  static const SCALE_T D3CW_Scoutput = 32;
  static const SCALE_T D3CW_Demote = 1 * 1024;
  static const SCALE_T D3CW_Scten = 1;
  static const SCALE_T D3CW_Scvec = 4;
  static const SCALE_T D3CW_Scret = 1;
  static const SCALE_T D3LW_Scinput = 16;
  static const SCALE_T D3LW_Scoutput = 32;
  static const SCALE_T D3LW_Demote = 1 * 1024;
  static const SCALE_T D3LW_Scten = 1;
  static const SCALE_T D3LW_Scvec = 32;
  static const SCALE_T D3LW_Scret = 1;
#endif
