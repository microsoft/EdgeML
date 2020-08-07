// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#define HIDDEN_DIM2 16

static Q15_T W2[HIDDEN_DIM2 * HIDDEN_DIM1] = {1835, -3126, -14237, 4985, -1347, 7074, -6448, 275, -4192, 5931, 1187, -7992, 5103, 6864, 18060, -3637, 260, 8033, 3160, 8031, -11654, 14665, -22339, 16368, -2909, -4356, 1694, -1672, -213, -3178, 3717, -931, -6350, -7699, -3400, 15514, 1173, 2156, 4166, -5117, 5012, 11540, -7619, 90, -15822, 13664, -6676, 1348, 881, -5523, 1284, 4358, 5227, 101, -2394, 4758, 758, 2002, 3134, -2798, -4642, 2636, 773, 1362, -143, -18780, -3305, -952, 9521, -3012, 10131, -1876, 3278, 14043, -1048, -8288, 1225, -15709, 250, 3682, -1527, -3164, -3262, -1510, 2927, 4283, -163, -6697, 1275, 3118, 586, 65, -9362, 4817, -245, -1272, 2986, 3655, 1548, 9138, 3492, -1687, 2950, 6374, 395, 7241, 85, -314, 4319, -5034, 542, 453, -665, 8921, -223, 10181, 369, 7132, 8221, 4444, 652, 133, 321, -2676, -63, 3744, -892, 2592, -9161, -8010, 8393, -3003, -1809, -6670, -5769, 18160, -8538, 10958, -1955, 2358, 868, 1696, -14311, -4181, -3918, -6106, -4863, -8615, -5898, -2883, 641, 1184, -2210, -2213, -3600, -5243, 2180, -6331, -1584, -1421, 1080, -8809, 2520, -1503, 5509, -5251, 25143, -1267, 196, -7925, -387, 2577, 3863, -9509, -2379, 1003, -4778, 6318, -2105, 693, 14982, 10812, 11626, -6825, -2102, -17220, -1687, -1066, 7712, 10961, -11514, 1992, 294, -8972, 280, 272, 1126, -4082, -6613, 4506, 1349, 5011, 623, 396, -15225, -6343, -1282, -1190, -3729, -1960, 4287, -4289, -7496, 3398, 3190, -5227, -3508, 239, 377, -119, 5213, -2487, -5596, 298, -297, 2531, -1444, 180, -2888, -4566, -11549, -11068, -36, 30387, 712, 611, 5320, 9124, 91, 160, 1889, -4331, 824, 12732, -4041, -8097, -10903, -734, -235, -13075, 1161, -8497, 1611, -597, 4955, 1458};
static Q15_T U2[HIDDEN_DIM2 * HIDDEN_DIM2] = {5878, -3142, 2156, -2088, 6655, -1340, -3003, 7941, -5094, 321, 949, 991, -6984, -764, 1713, -4477, -1879, -4178, 2395, -3978, 18248, -6918, -6705, -3696, 8651, -12, 4751, -1983, -2493, 346, -2092, -11876, 1730, 1902, 4550, -5509, 5159, 290, -9765, -5314, -1348, -2042, 1813, 2617, 1532, 93, 2533, -319, -5427, 1988, 3170, 5137, 1070, 4591, 1565, 10522, 3503, -6722, -662, 1758, -3105, 2246, 3846, -67, -1187, 2353, -206, 438, 5982, -8633, -2162, -4508, -5034, -946, -4709, -156, -4340, -6318, 177, 1829, -3009, 1985, -2660, -6235, 9105, 8597, 7955, -2257, 4643, -4210, -2402, 4672, 12362, 552, -6638, 3199, -3265, 5633, 589, -1947, -11, -9986, 3035, 4165, -2439, 1194, -10328, 1443, 10922, -218, -109, 3537, -6462, -3940, -352, -8303, 10169, -3791, -1032, 8136, -36, -1318, 1818, -5371, -1379, -889, -14297, -4058, -3759, -4810, -3565, 176, 7923, -4217, 234, -2557, 7731, -1973, 4379, 1820, 6735, -1100, -2989, -5924, -156, 2191, 363, 3164, -290, 4237, -5107, 6872, -1721, 9821, -5871, 1960, 6738, -5244, -3255, 1000, 1689, 1374, 3211, 3093, 417, -10634, 3552, -7298, 582, 6234, -1901, -9353, 7113, -3675, 4685, 4219, -355, 4592, -5164, -3164, -5567, -5629, -1995, 6872, 372, -402, -4728, 8011, 167, 4463, 1120, 896, 5411, 2797, -393, -713, -524, -5793, -11153, 7569, -12352, -6842, -1647, -554, 3246, -139, -3845, 1166, -1253, 3944, 1824, -865, 9148, -3638, -2461, 4083, -3486, -1558, -3614, -2778, 3504, 7106, 5145, 1298, -2397, 1202, -8591, -621, 13, 2639, -1245, 10044, -7856, -5035, 4156, 3502, 2943, -7892, 4048, 6256, -3389, 1188, 723, -4049, 7687, -6484, -7357, 4, 5371, 2189, -4633, 588, -1156, 1148, -5204, 4708};
static Q15_T Bg2[HIDDEN_DIM2] = {19214, 24258, 28014, 15949, -1013, 13811, 8543, 21295, 17343, 16986, 21293, 29165, 15931, 23696, 14925, 23654};
static Q15_T Bh2[HIDDEN_DIM2] = {10647, 13378, 9304, 7233, 13814, 17639, 4013, 9945, 12177, 19303, 12202, 530, 26468, 12106, 13442, 15580};
static Q15_T sigmoid_zeta2 = 16384;
static Q15_T sigmoid_nu2 = 23611;

static SCALE_T input2 = 0;
static SCALE_T meanScale2 = 0;
static SCALE_T meanSub2 = 0;
static SCALE_T stdDevScale2 = 0;
static SCALE_T normFeaturesHDStdDev2 = 0;
static SCALE_T H1W2 = 4;
static SCALE_T H2W2 = 0;
static SCALE_T H1U2 = 4;
static SCALE_T H2U2 = 0;
static Q15_T div2 = 0;
static Q15_T add2 = 0;
static Q15_T sigmoidLimit2 = 0;
static SCALE_T sigmoidScaleIn2 = 0;
static SCALE_T sigmoidScaleOut2 = 0;
static SCALE_T tanhScaleIn2 = 0;
static SCALE_T tanhScaleOut2 = 0;
static Q15_T qOne2 = 16384;

#ifdef SHIFT
  static SCALE_T WScale2 = 6; //64
  static SCALE_T normFeaturesMVW2 = 6; //64
  static SCALE_T UScale2 = 6; //64
  static SCALE_T hiddenStateMVU2 = 5; //32
  static SCALE_T mV1AddMV22 = 0; //1
  static SCALE_T mV2AddMV12 = 0; //1
  static SCALE_T mV1AddMV2Out2 = 0; //1
  static SCALE_T pC1AddBg2 = 0; //1
  static SCALE_T BgScale2 = 1; //2
  static SCALE_T pC1AddBgOut2 = 1; //2
  static SCALE_T pC1AddBh2 = 0; //1
  static SCALE_T BhScale2 = 1; //2
  static SCALE_T pC1AddBhOut2 = 1; //2
  static SCALE_T gateHDHiddenState2 = 7; //128
  static SCALE_T hiddenStateHDGate2 = 7; //128
  static SCALE_T qOneScale2 = 0; //1
  static SCALE_T qOneSubGate2 = 0; //1
  static SCALE_T qOneSubGateOut2 = 0; //1
  static SCALE_T sigmoidZetaScale2 = 7; //128
  static SCALE_T sigmoidZetaMulQOneSubGate2 = 5; //32
  static SCALE_T sigmoidNuScale2 = 15; //32767
  static SCALE_T sigmoidNuAddQOneSubGate2 = 0; //1
  static SCALE_T sigmoidNuAddQOneSubGateOut2 = 0; //1
  static SCALE_T sigmoidNuAddQOneSubGateHDUpdate2 = 7; //128
  static SCALE_T updateHDSigmoidNuAddQOneSubGate2 = 7; //128
  static SCALE_T pC3AddPC12 = 0; //1
  static SCALE_T pC1AddPC32 = 0; //1
  static SCALE_T hiddenStateOut2 = 0; //1
#else
  static SCALE_T WScale2 = 64;
  static SCALE_T normFeaturesMVW2 = 64;
  static SCALE_T UScale2 = 64;
  static SCALE_T hiddenStateMVU2 = 32;
  static SCALE_T mV1AddMV22 = 1;
  static SCALE_T mV2AddMV12 = 1;
  static SCALE_T mV1AddMV2Out2 = 1;
  static SCALE_T pC1AddBg2 = 1;
  static SCALE_T BgScale2 = 2;
  static SCALE_T pC1AddBgOut2 = 2;
  static SCALE_T pC1AddBh2 = 1;
  static SCALE_T BhScale2 = 2;
  static SCALE_T pC1AddBhOut2 = 2;
  static SCALE_T gateHDHiddenState2 = 128;
  static SCALE_T hiddenStateHDGate2 = 128;
  static SCALE_T qOneScale2 = 1;
  static SCALE_T qOneSubGate2 = 1;
  static SCALE_T qOneSubGateOut2 = 1;
  static SCALE_T sigmoidZetaScale2 = 128;
  static SCALE_T sigmoidZetaMulQOneSubGate2 = 32;
  static SCALE_T sigmoidNuScale2 = 32767;
  static SCALE_T sigmoidNuAddQOneSubGate2 = 1;
  static SCALE_T sigmoidNuAddQOneSubGateOut2 = 1;
  static SCALE_T sigmoidNuAddQOneSubGateHDUpdate2 = 128;
  static SCALE_T updateHDSigmoidNuAddQOneSubGate2 = 128;
  static SCALE_T pC3AddPC12 = 1;
  static SCALE_T pC1AddPC32 = 1;
  static SCALE_T hiddenStateOut2 = 1;
#endif
