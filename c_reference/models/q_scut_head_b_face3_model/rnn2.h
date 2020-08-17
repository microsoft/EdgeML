// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stddef.h>

#define HIDDEN_DIM2 16

static Q15_T W2[HIDDEN_DIM2 * HIDDEN_DIM1] = {120, -5408, 2905, -3441, -18938, 1489, -5292, -316, 1969, 27, -634, -1548, -3066, 2418, -888, 866, -961, 1444, -2982, 5195, -4169, -910, -6384, 448, 3204, -4888, 4974, 7903, 2774, -10645, -2450, 763, -16, 1882, 510, 2973, 1351, 146, -4170, -148, 34, 1193, -3152, -5999, 846, 6448, 146, -1987, 609, 1274, 282, 3532, 6475, 448, -2388, -1291, -1813, -1579, -1117, -2617, -32, 3016, 523, 1854, 1325, -2534, 751, -819, -9898, 5173, -2999, -1861, -2972, -1389, -280, 3091, -2104, 4780, -1596, 5438, -221, 3695, -282, -6735, 3579, -702, -3604, -2141, -552, -2127, -1581, -105, 1663, -5296, 2430, -4344, -4206, 4878, 463, 4316, -7123, 4020, -6988, -2252, 2538, -3692, -8660, 5130, -1490, -3390, -6651, -14228, 278, -3550, 1585, -2319, 10046, 436, 1321, 201, -1373, -911, -1198, -2291, -114, -3555, 926, 5458, 2669, -5352, 3657, -2083, 7560, 315, 8572, 3453, -169, 1832, 4078, -475, 3966, 8392, 1974, 4785, 369, 180, -263, 1730, 7334, 36, 1833, 388, -2317, 284, 859, -2146, -1195, -293, 220, 633, -1754, 1955, 454, -32, 2648, 601, -4791, 237, -2069, 7501, -3785, -915, -569, -2034, 2776, -8690, 127, -890, 872, 4231, -2725, 928, 3747, -1962, 2018, -1178, -2337, 3229, 1097, -398, 1451, 971, -287, 555, -1751, -906, 3294, -2858, -235, 1438, 3333, -961, 740, 13799, 2427, 7232, -1498, 770, -938, -1041, 781, 3980, -2583, -5906, -1850, 812, 2000, 599, 3186, -4586, -2086, -530, 864, -6245, 584, -2058, 203, 2801, -3946, -433, 5326, -104, -986, 1113, 439, -391, -552, 679, 801, 668, 324, 331, 1022, -7172, -3101, -1883, 3834, -306, -1710, 1599, 1996, -1265, -3205, 2972, 373, -5260};
static Q15_T U2[HIDDEN_DIM2 * HIDDEN_DIM2] = {2385, 5088, -8431, -695, -1198, 4920, 921, 5795, -8081, 1596, -446, 5425, 10706, -1348, -1275, 8920, -4879, 209, -4874, -1702, -3244, 113, 5666, 10089, -10704, -7761, 2607, 2935, -8406, 1173, -2764, -9690, 3217, 7233, 10384, -4746, -4219, -3177, -5824, 8324, 2961, -1993, -2376, -5661, 8159, -6088, 7771, -327, 8970, 2786, 751, 8212, -6814, -8653, -1350, -6021, 2771, -4029, -946, 1638, -6584, -5500, 18168, -11109, -4772, 5329, 2339, -6164, 6110, -4927, 752, -8203, -1400, 4041, -2777, 1407, -5983, 3585, 14410, -4844, 418, 682, -5309, -9053, 4528, 8214, 2789, 4126, -507, -4109, 1981, 2049, 7690, -4342, -1652, 3974, 1294, 4444, 605, 1221, 6261, 586, 3662, 2677, -4285, -233, -2952, 6949, -2048, 3561, -1002, 376, 1519, -2722, -10423, 2804, -3858, -3697, 1777, 1079, 6108, 147, -4725, -14398, -552, -3018, 6785, -9834, -5344, -5884, 3909, 4493, -859, -968, 5892, 9958, 5727, -384, -2875, 5743, -1436, 6389, 1489, 5012, 9404, 2323, 2464, 5070, 803, 4635, 874, -3507, -912, 11216, 591, -13298, 663, 1746, -11753, -1433, 1156, -3477, -2005, 6659, 5314, -2335, -398, -155, 1287, -4303, 11403, 4318, -2252, -1887, 9474, 3531, 1709, -4018, 3321, -2452, -4616, -2332, -1670, 11247, 3038, 9565, -3292, 9016, 4623, -2050, 1234, 4220, -2914, 3011, -2142, 6173, -3062, -6964, -1808, -1287, 5454, 4833, 3004, -7116, 3417, 4353, -6907, -1056, -3181, -3316, 1165, -968, -5462, -917, -1175, 3611, -2774, 4826, 1065, -1642, -7313, 11362, 2185, -674, -2764, 8294, -9559, -12199, -4856, -2930, -4445, 867, 4515, 10447, -2404, -4813, 6990, -6853, 7684, 1564, -382, 7705, 5397, 2536, -6299, -2752, -1170, 8295, -6008, 5566, -751, -6185, 244, -596, 3321, 11359};
static Q15_T Bg2[HIDDEN_DIM2] = {16637, 13727, 11984, 10236, 10297, 14007, 10937, 8005, 9837, 11567, 9368, 7635, 12688, 8691, 8062, 7896};
static Q15_T Bh2[HIDDEN_DIM2] = {1077, 5694, 10715, 5704, 913, 8669, 15997, 7459, 23544, 5861, -7641, 3603, 3149, 8170, 15308, 8582};

static Q15_FastGRNN_Params RNN2_PARAMS = {
  .mean = NULL,
  .stdDev = NULL,
  .W = W2,
  .U = U2,
  .Bg = Bg2,
  .Bh = Bh2,
  .sigmoid_zeta = 16384,
  .sigmoid_nu = 23611
};

static Q15_T preComp21[HIDDEN_DIM2] = {};
static Q15_T preComp22[HIDDEN_DIM2] = {};
static Q15_T preComp23[HIDDEN_DIM2] = {};
static Q15_T normFeatures2[HIDDEN_DIM1] = {};

static Q15_FastGRNN_Buffers RNN2_BUFFERS = {
  .preComp1 = preComp21,
  .preComp2 = preComp22,
  .preComp3 = preComp23,
  .normFeatures = normFeatures2
};

#ifdef SHIFT
  static Q15_FastGRNN_Scales RNN2_SCALES = {
    .input = 0,
    .mean = 0,
    .meanSub = 0,
    .stdDev = 0,
    .normFeaturesHDStdDev = 0,
    .W = 5, //32
    .normFeaturesMVW = 5, //32
    .H1W = 4,
    .H2W = 0,
    .U = 6, //64
    .hiddenStateMVU = 5, //32
    .H1U = 4,
    .H2U = 0,
    .mV1AddMV2 = 0, //1
    .mV2AddMV1 = 0, //1
    .mV1AddMV2Out = 1, //2
    .mV1AddMV2Demote = 0, //1
    .pC1AddBg = 0, //1
    .Bg = 0, //1
    .pC1AddBgOut = 0, //1
    .pC1AddBgDemote = 2, //4
    .sigmoidLimit = 0,
    .sigmoidScaleIn = 0,
    .sigmoidScaleOut = 0,
    .pC1AddBh = 0, //1
    .Bh = 1, //2
    .pC1AddBhOut = 0, //1
    .pC1AddBhDemote = 2, //4
    .tanhScaleIn = 0,
    .tanhScaleOut = 0,
    .gateHDHiddenState = 7, //128
    .hiddenStateHDGate = 7, //128
    .qOneScale = 0, //1
    .qOneSubGate = 0, //1
    .qOneSubGateOut = 0, //1
    .sigmoidZeta = 7, //128
    .sigmoidZetaMulQOneSubGate = 6, //64
    .sigmoidNu = 15, //32767
    .sigmoidNuAddQOneSubGate = 0, //1
    .sigmoidNuAddQOneSubGateOut = 0, //1
    .sigmoidNuAddQOneSubGateHDUpdate = 6, //64
    .updateHDSigmoidNuAddQOneSubGate = 7, //128
    .pC3AddPC1 = 0, //1
    .pC1AddPC3 = 1, //2
    .hiddenStateOut = 0, //1
    .hiddenStateDemote = 0, //1
    .div = 0,
    .add = 0,
    .qOne = 16384,
    .useTableSigmoid = 1,
    .useTableTanH = 1
  };
  static SCALE_T ShR2 = 8; //256
  static SCALE_T ShL2 = 0; //1
#else
  static Q15_FastGRNN_Scales RNN2_SCALES = {
    .input = 0,
    .mean = 0,
    .meanSub = 0,
    .stdDev = 0,
    .normFeaturesHDStdDev = 0,
    .W = 32,
    .normFeaturesMVW = 32,
    .H1W = 4,
    .H2W = 0,
    .U = 64,
    .hiddenStateMVU = 32,
    .H1U = 4,
    .H2U = 0,
    .mV1AddMV2 = 1,
    .mV2AddMV1 = 1,
    .mV1AddMV2Out = 2,
    .mV1AddMV2Demote = 1,
    .pC1AddBg = 1,
    .Bg = 1,
    .pC1AddBgOut = 1,
    .pC1AddBgDemote = 4,
    .sigmoidLimit = 0,
    .sigmoidScaleIn = 0,
    .sigmoidScaleOut = 0,
    .pC1AddBh = 1,
    .Bh = 2,
    .pC1AddBhOut = 1,
    .pC1AddBhDemote = 4,
    .tanhScaleIn = 0,
    .tanhScaleOut = 0,
    .gateHDHiddenState = 128,
    .hiddenStateHDGate = 128,
    .qOneScale = 1,
    .qOneSubGate = 1,
    .qOneSubGateOut = 1,
    .sigmoidZeta = 128,
    .sigmoidZetaMulQOneSubGate = 64,
    .sigmoidNu = 32767,
    .sigmoidNuAddQOneSubGate = 1,
    .sigmoidNuAddQOneSubGateOut = 1,
    .sigmoidNuAddQOneSubGateHDUpdate = 64,
    .updateHDSigmoidNuAddQOneSubGate = 128,
    .pC3AddPC1 = 1,
    .pC1AddPC3 = 2,
    .hiddenStateOut = 1,
    .hiddenStateDemote = 1,
    .div = 0,
    .add = 0,
    .qOne = 16384,
    .useTableSigmoid = 1,
    .useTableTanH = 1
  };
  static SCALE_T ShR2 = 256;
  static SCALE_T ShL2 = 1;
#endif
