// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#define D1NW_HF 1
#define D1NW_WF 1
#define D1NW_CF 1
#define D1NW_COUT 1
#define D1NW_HPADL 0
#define D1NW_HPADR 0
#define D1NW_WPADL 0
#define D1NW_WPADR 0
#define D1NW_HSTRIDE 1
#define D1NW_WSTRIDE 1
#define D1NW_HDILATION 1
#define D1NW_WDILATION 1
#define D1NW_G 32

#define D1CW_HF 3
#define D1CW_WF 3
#define D1CW_CF 32
#define D1CW_COUT 4
#define D1CW_HPADL 1
#define D1CW_HPADR 1
#define D1CW_WPADL 1
#define D1CW_WPADR 1
#define D1CW_HSTRIDE 1
#define D1CW_WSTRIDE 1
#define D1CW_HDILATION 1
#define D1CW_WDILATION 1
#define D1CW_G 1

#define D1LW_HF 3
#define D1LW_WF 3
#define D1LW_CF 32
#define D1LW_COUT 4
#define D1LW_HPADL 1
#define D1LW_HPADR 1
#define D1LW_WPADL 1
#define D1LW_WPADR 1
#define D1LW_HSTRIDE 1
#define D1LW_WSTRIDE 1
#define D1LW_HDILATION 1
#define D1LW_WDILATION 1
#define D1LW_G 1

// The convention followed for filter weights is F[groups][filter_height][filter_width][in_channels][out_channels]
static const Q15_T D1NW[D1NW_G * D1NW_HF * D1NW_WF * D1NW_CF * D1NW_COUT] = {17535, 17352, 17956, 17307, 17205, 17988, 18002, 18113, 18312, 17818, 17557, 18689, 18284, 18188, 18018, 16417, 18101, 18331, 18360, 18326, 17915, 17984, 18129, 18152, 17982, 17928, 17893, 17043, 18076, 17900, 18290, 18160};
static const Q15_T D1CW[D1CW_G * D1CW_HF * D1CW_WF * D1CW_CF * D1CW_COUT] = {1500, -2113, 2663, -595, -629, 1139, 752, -897, 671, -323, 1422, -472, 578, 66, -1989, 784, 190, -1387, 1818, -1588, -1041, -1931, 921, 1296, -2334, -1639, -1111, 4261, 155, -1122, -472, 651, 73, -1979, -1858, 2972, -1671, -1094, -965, 3640, -764, 892, -3216, 3404, -235, 1332, 2284, -2327, -1208, -166, -1946, 4269, 1491, 379, -1500, 526, -882, -3167, 1159, 3781, -2656, 156, 3252, -1045, 835, 1319, -2335, 820, -755, -2508, -2647, 4888, -84, -1582, 18, 3014, -950, 965, 2779, -2441, -524, 611, -2115, 1676, -705, -1594, 1192, 161, 749, -136, 1268, -1114, 1264, 616, 413, -2605, 2548, -1049, 1286, -1821, 767, 487, -275, -1598, -1196, 783, 455, 1904, 1653, -19, -1166, 1049, 694, -1610, -502, 2998, 344, 3159, -1115, -2542, -860, -2909, -1910, 5532, 553, 4185, 2480, -6450, 1304, -2362, -1413, 2218, 496, 2431, -2700, 585, 2300, -651, 458, -691, -127, -583, -1401, 1072, 1234, -2285, -1560, 3702, 474, -946, 25, 1950, -1684, -284, 834, -654, 540, -484, 42, -1146, 468, 1038, -1974, 1350, 793, -352, -161, 692, -883, -635, -6247, 8750, -1991, -246, 285, 1984, -696, 690, 2648, -2968, 1299, 385, -444, 63, -1232, -1202, 90, 3561, -4740, -1586, 4993, 2705, 933, -733, -3351, 2708, -305, -2523, 741, 1740, -592, 223, 630, 1044, -441, 455, 2238, -2062, 275, -167, -1229, 1916, 1429, -1805, 1555, 107, 467, 1267, 1739, -2439, 838, -1194, -2232, 2917, 417, -1987, 2147, 230, 1594, 1762, 580, -3070, 853, -1176, -3588, 3293, 247, -869, 2505, -2816, 2209, -3841, -1722, 3655, 238, 51, 887, -491, -1320, -1239, 1429, 214, 438, 2604, -886, -2796, 861, -1957, -256, 513, -1036, 2228, -1378, 2, 879, -632, 442, 782, -6, 108, -689, 356, -932, -791, -1760, 2181, 133, -410, 1367, -1109, -1484, -887, 410, 505, -243, -609, 417, 369, 159, 209, -1820, 524, -926, -55, 443, 1131, 547, 2023, -2701, -100, -1785, -1360, -13, 2540, -886, -47, -502, 1427, 769, 1491, -260, -1536, -1046, -1106, -44, 2333, -1942, 1396, 2746, -2567, 304, -691, -1009, -428, -350, -696, -59, 764, -730, -884, 225, 94, -1732, -372, 1779, 160, -259, -172, -959, 1331, 93, -1582, -103, 855, 341, -752, 1410, -404, 1090, 213, 54, -189, 736, -1932, -268, -461, 2462, 631, 667, -2086, 190, 408, 44, 1514, 701, -1733, -1388, 2623, -118, -594, -399, 1213, 714, 1419, -480, -1270, -446, -2121, -89, 3110, -209, 1462, 143, -1065, 395, -4562, -1028, 5998, -367, 1168, -934, -332, -245, -2107, -159, 3034, 840, 862, 107, -2165, 1770, 1180, -672, -1266, -1508, -1121, -864, 4750, 117, -419, 2743, -146, 428, 372, -1926, 2180, 2004, -2705, 480, -166, 79, -164, -268, 783, -748, 92, -2156, 3006, -2204, 1457, -1707, 3644, -114, -133, -2809, 3103, 622, -828, -602, 126, -898, -858, -2372, 3754, -3010, -2380, 991, 6053, 2805, 1761, 61, -3418, 483, -2265, 1422, 217, 2815, -168, 990, -3931, 195, -781, 3307, -2506, 1203, 1074, 1060, -2357, -2107, 1933, 245, -158, -1536, 54, 1261, 1401, 942, 171, -297, -1298, 498, -1202, 1372, -2971, 1224, 1445, -1426, -1486, 637, -1375, 1643, -1596, 34, 711, -2293, 2241, 1284, -3057, -2246, 4282, -67, 3786, -500, -3035, 91, -1015, -42, -250, -1644, 1446, 1545, -560, 932, -3065, 5137, -3434, -334, 1486, 1096, -1999, -2113, -874, -3568, 4933, 1211, -626, 599, -2574, 1016, 969, 398, -2332, -2126, -2804, 560, 4010, -87, -969, 3425, -884, 1256, 1807, -474, -1730, 2342, -2249, 2494, -2597, 65, 2489, -609, -621, -1132, -5824, -2679, 8401, -5757, -1409, -1628, 6563, 2242, 1685, -5105, 1272, 568, -1315, 468, -1686, 678, -1282, -897, 1785, -8053, -6945, -4835, 19856, 3076, 703, -994, -2083, 412, -3254, 1585, 1924, 970, 837, 840, -2478, 2330, -4, 2511, -3816, 84, 1499, -3480, 993, -781, -1449, -1725, 3839, 573, -794, 1589, -1556, 1933, 655, 961, -3800, -653, 1182, -2829, 3975, 1528, 4042, -668, -3969, -137, 1489, -2299, -410, 890, 215, -909, -119, 2357, -6260, -1932, 7171, -3401, -71, -1205, 4564, 1595, -3015, 4993, -2637, -2755, 2879, 3214, -2582, -324, -2933, -1585, 3027, -708, 768, -15, -85, 25, -381, -938, 1642, 475, -238, -1155, -486, 412, 169, -1111, -157, -698, -1314, -382, 515, -457, -313, 1210, 1279, 743, 65, 18, -1169, -278, -1115, -1600, 4050, 783, -27, 149, -462, 831, -466, -858, -362, -1903, 163, -299, 2437, 685, 830, -1024, 710, 41, 75, -1269, 1391, 58, 1104, -106, 381, -1752, -466, 33, 2721, 1948, 2174, 642, -2863, -679, -761, 961, 416, 1377, -1323, 699, -1496, -545, -2377, -217, 1987, 220, -91, -962, 698, -60, 879, -490, 588, -1936, 893, -539, 1660, 877, 793, 681, -1219, 390, 604, -597, -1983, 607, -124, -601, -1678, -300, -480, -396, 1218, 574, 274, -1992, 630, 1695, -1103, 332, -271, -1272, 1186, -1564, 2973, 227, -1422, 34, 1748, -2182, 2096, 867, -1246, 1193, 67, -1061, 667, -556, 1998, 733, -1202, 834, 833, -129, -933, -330, -827, -384, 1481, 179, -310, -216, 147, 217, -1471, 258, 1394, 1029, -1207, 808, 874, -515, -1685, -45, 798, 316, -1788, 165, 751, -166, -1001, -516, -196, -582, 3012, -1755, -166, 3333, -2251, 603, -1112, -339, 309, -611, -402, -708, -540, 416, 1792, -2122, 207, -1314, 2777, 337, 1742, -2595, 1909, 1557, -485, -1073, -1268, 689, 892, -448, -1248, 3860, -1504, 241, -1349, -3683, -1940, 408, 5891, 2487, -772, -164, -967, 1248, -1014, -224, 480, 1965, 1320, 90, -1313, -460, 710, 80, -1655, -702, 1287, -1168, 791, 608, -114, 410, -149, -76, -1570, 314, 269, -432, 397, -1279, -35, 699, -1896, -701, 298, 666, 1720, -1218, -2289, -366, 395, 32, -1818, 1934, -1751, 382, -1855, 1217, -336, -228, -422, -805, 429, 296, 369, 209, 1794, 122, -2534, -417, -42, 510, 93, -66, 447, -768, -153, 1025, -428, 609, -1219, 507, -2594, 774, 183, 229, -1041, 12, 589, 124, -1092, -596, 405, -323, 381, 431, -884, -311, 321, -1703, 569, 2552, 601, 1041, -4342, -558, 271, -877, 1229, -949, 1624, 306, 18, -872, 162, -176, 2140, -691, 117, -1391, 1989, 1416, 2311, -937, -3499, -203, 1021, -881, -1156, 1651, -554, 235, -2198, -3339, -1070, -245, 3422, 103, -259, -1190, -633, 193, 890, -894, -1112, 2557, 2199, -426, -4110, 45, 1385, 1044, -650, -682, 1934, -1668, -812, 562, -1064, 75, 510, -444, -1915, 44, 1025, 198, 587, -938, 292, 832, -1303, -819, 843, 519, 2657, 420, -3649, 1887, 142, 1352, -3377, 1229, 3265, 183, -5290, 749, 335, -671, 1012, -917, 1173, -200, 229, 223, 208, -914, 175, -19, -99, -798, 1745, 296, 958, 159, 570, 593, -1612, -94, 1553, 90, -2554, 403, 2466, -49, -981, 47, 81, -1396, -1103, -577, 2196, 919, -205, 8, -985, 1290, 852, -811, -851, 3082, -1429, -286, -1330, -94, 108, -572, -538, -949, 363, -847, -94, -178, 1057, -306, 695, -252, 1600, -1711, 909, 2113, 543, -802, -1188, 31, -1256, -410, 1902, 2265, -624, 828, -1905, -1480, -672, -247, 3927, 1320, 605, 59, -1549, 1410, -904, -58, 707, 106, 16, -1808, -286, -676, 1094, 571, -2013, -178, 860, -1149, -165, 951, -347, 1039, -1583, 293, -754, 279, -641, 206, 1656, -1711, -1406, 530, -30, -656, -309, -290, 901, -624, 16, 1277, -856, -96, -521, 945, -771, 1079, -636};
static const Q15_T D1LW[D1LW_G * D1LW_HF * D1LW_WF * D1LW_CF * D1LW_COUT] = {977, -3489, 289, -106, -1087, -10268, -212, 332, -30, -6885, -1949, -2152, 1193, -6468, -1305, -1527, -1094, 1598, -644, 95, 513, 1186, -398, -185, -965, 1073, -1354, -2346, 1159, -5933, 1850, 651, -7661, -4900, -1908, -2466, -5210, -169, 565, 854, 5108, -1527, -7015, -6146, -1937, 8055, -1392, -721, 1216, 1379, -6376, -6854, 2495, 2538, 2032, 2600, -1786, -5740, -1851, -1492, 320, 2434, 2618, 3362, 1755, -2065, -3695, -3890, 3031, -903, -700, -423, -3660, -1206, 452, 719, -1343, 2158, 5191, 5589, 1430, -50, -1422, -948, -411, -4278, 3420, 3566, -799, -67, 2266, 2288, -4849, 1930, 321, -156, -1915, -3175, 4515, 4206, -453, 2513, 1243, 863, -74, -1999, -2721, -2855, 4140, 970, 887, 878, 1521, 1141, 845, 1560, 321, 2897, -1027, -862, -1802, -3581, -1162, -980, 2088, 1687, 1607, 2550, -15, -2695, 1000, 422, 72, -1343, 2410, 2344, 61, -76, 1299, 1818, -1223, -521, -715, -527, -3049, -1117, -1981, -1760, -1922, -1975, -2217, 205, 3174, 1363, 4671, 3494, 420, -5960, 182, -298, 6988, -3945, -3924, -6200, 3018, 3004, -888, -843, -1553, -5457, -7988, -9805, 1809, -5841, -706, -2282, -4278, 3764, -4141, -3996, -3300, 2449, -503, 681, 3708, -1117, -43, 726, 2722, -5608, 12245, 8046, -791, -3311, -1357, -1010, -3539, 21, -99, 77, 1496, 6764, 5184, 4617, 1648, 4766, 6414, 6473, -452, 692, 836, 1117, -2989, 862, 1021, 212, 1674, 10713, 1094, -460, 626, -1491, -1059, -2454, 483, -9748, 745, 1198, 302, 6676, -800, -124, 3091, 1208, 130, -1072, -4184, 4705, 43, 261, -638, -418, 1797, 1417, -1463, -5169, 659, 684, -816, 1183, -1327, -1176, 781, 3964, 1642, 1879, 458, -1539, 1026, 614, -78, -1792, -1162, -1171, 392, -1046, 967, 658, -990, -797, -1712, -1900, 1308, -974, -313, -1058, -548, 826, 1775, 1599, -402, 81, 208, -252, 738, -249, 1373, 396, -243, -1292, -898, -1159, 1424, 521, -1421, -1572, 190, -2447, -3314, -4215, 122, 102, -4365, -3993, 958, -112, -1311, -1772, 19, -49, 1402, 757, -42, -514, -1778, -2133, -2247, 1847, 2806, 4756, 1417, 235, -2188, -1819, 196, 588, -53, -337, 562, -253, 1527, 2052, -1357, -2668, 760, 904, -111, 33, -1042, -1462, -1033, -2284, -368, -380, 462, -561, 660, 1023, 97, 1332, 1136, 577, -265, -882, 1571, 1321, -1117, -919, 1716, 841, 182, -86, -2558, -2393, 420, -3493, -2036, -1423, 124, -939, -79, -162, -1023, 2032, 754, 1546, 192, -2782, -1612, -1048, -935, 531, -349, 448, 905, 7705, -465, -282, -336, 3173, 829, 1421, -1436, -3771, -3119, -2159, -1835, -2595, 902, -166, -397, 231, -375, 111, -3223, -4113, -3854, -3094, 469, -1702, -3675, -3223, -1052, -4456, 241, -1078, -2513, 12086, -3832, -4776, -216, -3254, 89, 687, -3136, -2368, -5710, -5212, -708, -1225, 1172, 1845, -4188, 6799, 307, 962, -1023, -3710, -1254, -681, -3948, -2365, 1095, 1899, -1250, 264, 10395, 11664, -1846, 2956, 1138, 969, 3479, 3640, 560, 736, 2030, 6286, -4909, -4348, 45, -4319, 2217, 3963, 897, -726, -1087, -1835, 673, 10173, 877, -76, 4066, -5877, -1959, -1174, 260, -3535, 1120, 1238, -1497, -4374, -862, -1056, -4214, 896, 3600, 1860, 1235, -6693, 257, -72, 27, -4428, -218, 361, -1465, 1282, -1265, -983, 3537, -5161, -6683, -5942, 1188, -9740, 2700, 2962, -2158, 1742, 3415, 3827, 1744, 10676, -1626, -1818, 2122, -7996, 2260, 2496, -3973, 173, 1745, 1560, 7458, -1440, -990, -1187, -3185, 1355, -1034, -403, -2646, 10921, -6478, -5887, 3114, -244, 2324, 1983, -9428, -1291, 80, -214, -6864, 7006, -7568, -7717, 986, 2479, -3611, -4918, -6484, 9454, -4772, -5174, 4255, 3018, 1247, -592, 8869, -1675, 8897, 9303, 16836, -3921, -2191, -584, -5335, -8404, 1518, 2633, -4987, -804, 32631, 29179, 1651, 14866, 3749, 3642, -15019, -3453, -2114, -1374, -4466, 4377, 1242, 986, -4222, 2987, 15870, 17170, -8585, 5859, 3182, 1070, 8939, 2428, -3375, -4626, -9878, -7260, -3076, -2371, 3305, 1760, 2267, 1759, 3880, 7604, 687, 942, -2571, -3017, -2586, -3726, 739, -6250, -2038, -2297, -6487, -2224, -1240, -1557, 3883, -4824, -1291, -1741, -3935, -5471, -421, -343, 9040, 700, -430, 86, -3397, 10012, -2374, -3683, 1128, -150, 776, 1096, 149, 709, 436, 476, 1360, 491, -1487, -1405, 775, 1128, -630, -825, 464, 1238, 62, -1081, 283, -1296, 0, 944, 142, -266, -948, -629, -305, 710, -411, -397, -501, -1078, -830, -1369, -787, -1465, -1414, -887, 167, -2817, -3320, -3730, 2202, 819, -4728, -5133, -1895, 759, 1284, 1361, 1352, -893, -190, -242, 451, 1418, -2472, -2286, -1994, -4443, 7201, 9729, 324, 2903, -3065, -2863, 1289, -981, 477, 492, -467, 522, 2349, 2710, -978, -398, -112, 742, 779, 334, -1137, -1566, -88, -1082, -2422, -2248, -167, -786, -1320, -563, 0, 266, 631, 311, 1312, 1353, 832, 241, 212, 72, 2170, 1050, -2068, 1332, -654, -862, -504, 1999, -1654, -1303, 421, 2411, -1529, -1134, 246, -2637, -1561, -475, 769, 1362, 2121, 1535, -1184, 1940, -1028, -493, -1152, 556, -2076, -2310, 379, -1379, 1037, 921, -720, -608, -2871, -2098, 30, 1789, 203, 22, 550, 418, -76, 252, 29, 1518, 84, 656, 483, -1756, -1370, -1618, -213, -413, 1066, 213, 1119, -2195, 515, 1632, -126, 1260, -937, -1224, -765, -889, -3768, -4212, 551, 4290, 691, 573, 561, -885, -1980, -1956, -20, 1078, 295, 1297, 431, 483, -2425, -2091, -1533, 1941, 5103, 6378, -1304, 654, 1334, 1069, 901, -478, 620, 102, 749, -1399, -86, 53, -58, 596, -2821, -2539, -1269, 1280, 1045, 892, -13, -285, -731, -1516, -868, 662, -1972, -1201, 136, 1959, 1853, 2055, 317, 50, -1974, -1255, -1096, 1604, 1691, -130, -215, 12, 674, 8, -278, -252, -900, 54, 1309, -271, -703, 980, 398, -1350, -4153, -3831, 629, 1697, 1202, 1100, -172, 3037, 878, 1414, 1855, -3668, -181, -940, 79, -510, 2208, 1863, 238, 306, 64, 204, -1189, -1719, -557, -207, -1344, 626, -1603, -675, 1377, -1193, 86, 756, -1358, -1944, -1519, -1622, -391, -997, 333, 252, 2125, -102, 2659, 3001, -40, -1906, -1856, -2102, 541, 2732, -5434, -5523, -372, 4479, 1820, 287, -780, 875, -318, -200, -647, -552, 488, 956, -418, 1074, -1871, -1078, -926, 4813, 10464, 9270, -97, -3841, 604, 808, 297, -1591, 441, -7, -520, -657, 1243, 1371, 682, 2284, -3336, -3352, 1161, -466, 1716, 414, 1773, 750, -1877, -3419, 911, -2186, -2269, -2997, 901, -3, 435, 278, -872, -1194, -1757, -1027, 2137, -1612, 1077, 1018, 1134, 800, 966, 270, 491, 541, 911, 1374, -1910, 3541, -1980, -885, 2035, 1327, -1514, -2565, -601, -1908, 345, 597, -1090, -2792, -176, -1403, -439, 617, -1212, -1468, 69, 209, 1539, 950, 139, -276, -1190, -1900, 476, 302, -285, -338, 387, -173, -946, -1369, 219, 1506, 1072, 1878, -42, -1153, -828, -1552, 258, -1573, -285, -424, 42, 826, 2740, 3141, -883, -1063, -2372, -2145, 823, 1562, -1903, -1796, -296, 1858, -2305, -1948, 312, -1501, 128, 288, -280, 178, 638, 401, 97, -592, -1763, -1671, 1673, 2954, 5723, 7564, -263, -1619, -540, -803, 384, 139, -477, -226, 279, -1324, 93, 39, -607, 820, -1567, -1423, -702, 317, 869, 132, -312, -1044, -2600, -2169, -600, 151, -2204, -1397, -784, 36, 133, 104, 30, 79, -816, -731, 339, -150, 1522, 99, -469, 368, 724, 812, -198, -443, -1320, -1168, -901, -2018, -1792, -1225, 156, 1819, -2083, -1545, 263, 1233, 309, -9, -362, -209, -1409, -1242};
static const Q15_T D1CB[D1CW_COUT] = {-27363, 16659, 23430, -8836};
static const Q15_T D1LB[D1LW_COUT] = {-1066, -513, -16947, -554};

static const SCALE_T D1_ScaleIn = -12;
static const SCALE_T D1_ScaleOut = 8;

#ifdef SHIFT
  static const SCALE_T D1NW_Scinput = 5;    //32
  static const SCALE_T D1NW_Scoutput = 5;   //32
  static const SCALE_T D1NW_Demote = 0 + 0; //1 * 1
  static const SCALE_T D1CW_Scinput = 3;    //8
  static const SCALE_T D1CW_Scoutput = 4;   //16
  static const SCALE_T D1CW_Demote = 0 + 9; //1 * 512
  static const SCALE_T D1CB_Scten = 0;      //1
  static const SCALE_T D1CB_Scvec = 4;      //16
  static const SCALE_T D1CB_Scret = 0;      //1
  static const SCALE_T D1LW_Scinput = 3;    //8
  static const SCALE_T D1LW_Scoutput = 4;   //16
  static const SCALE_T D1LW_Demote = 0 + 9; //1 * 512
  static const SCALE_T D1LB_Scten = 0;      //1
  static const SCALE_T D1LB_Scvec = 2;      //4
  static const SCALE_T D1LB_Scret = 0;      //1
#else
  static const SCALE_T D1NW_Scinput = 32;
  static const SCALE_T D1NW_Scoutput = 32;
  static const SCALE_T D1NW_Demote = 1 * 1;
  static const SCALE_T D1CW_Scinput = 8;
  static const SCALE_T D1CW_Scoutput = 16;
  static const SCALE_T D1CW_Demote = 1 * 512;
  static const SCALE_T D1CB_Scten = 1;
  static const SCALE_T D1CB_Scvec = 16;
  static const SCALE_T D1CB_Scret = 1;
  static const SCALE_T D1LW_Scinput = 8;
  static const SCALE_T D1LW_Scoutput = 16;
  static const SCALE_T D1LW_Demote = 1 * 512;
  static const SCALE_T D1LB_Scten = 1;
  static const SCALE_T D1LB_Scvec = 4;
  static const SCALE_T D1LB_Scret = 1;
#endif
