// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stddef.h>

#define INPUT_CHANNELS 4
#define PATCH_DIM 8
#define HIDDEN_DIM1 16

static Q15_T W1[HIDDEN_DIM1 * INPUT_CHANNELS] = {74, 634, -22699, -681, 1102, 1030, -923, 643, 10602, 12476, -284, 1802, -503, -340, 16683, 7306, -3742, 4009, -1148, -2550, 10346, -4327, 4024, 4086, 1120, 372, -602, 780, 3707, 2253, -3725, 703, 25782, -13055, 8276, 845, 239, 194, -328, 225, 260, -21953, 13647, 1565, -6569, -1564, 4197, -6682, 766, -2046, 6929, 635, 811, 2295, -4524, -501, 15, 17854, 18155, 2634, -22334, 8200, -6066, -15159};
static Q15_T U1[HIDDEN_DIM1 * HIDDEN_DIM1] = {-9863, -10249, 646, 20150, -6835, -1261, -1799, 1766, 1251, 4031, 6619, 5072, 10211, -4936, 1065, -3089, 7670, -6194, -535, -173, -829, 3050, -3237, -6306, 3552, 12013, 3251, -4109, 1613, 4363, 10403, 6135, -12259, -4983, -10532, 4722, -8412, -3133, -274, -9892, -5084, -7171, 17764, 4084, 618, -5500, -6067, 6591, -2054, -6937, 683, 1815, -1552, -2407, -4253, -4837, 2395, -2793, -1015, 528, -2720, 4324, -117, 3708, -2838, -2918, -4334, 1033, 474, -3624, -6693, -5181, -7553, -3510, -759, 5799, -1299, -4149, 115, 2895, -3166, -308, -8859, 7030, 2061, 4769, 4483, -90, 5251, -3046, 5131, -9355, -3446, -2082, 3889, -6492, -1614, -1362, -3236, -1716, 12718, 3036, 6061, -2043, 9131, 7725, -3353, -4785, -15047, -938, -2153, 6729, 1770, 418, 15944, 5121, 4642, 2745, -261, 2826, -806, 888, -11461, -4041, -3489, 417, -4463, -6322, 4940, -5968, 1007, -7293, 4687, -2712, -3003, -10115, -3490, -2168, -6090, 1640, -4643, -2197, 5001, 12574, -6419, -15804, -2186, -5838, -836, 2752, -2266, -4208, 1902, 9333, -3764, -7264, -3308, 94, -1693, 1939, 13975, -894, 5011, -10094, 7000, 3140, 2914, 883, -658, -958, -8022, -6932, -1731, 11750, 9106, 1270, 2486, 5578, 2061, -9292, -2403, -1666, 3932, -2214, 3913, 11346, -5309, 5486, 4616, 7743, 2280, -7924, 570, -764, -1515, -7588, 1645, 248, 8822, -7393, -7046, -148, 3120, -9766, -2972, -2802, 4851, 323, -1486, 8521, -9280, 8011, -2300, -1260, 608, -5751, -3042, -7512, -7713, -549, 3889, 1379, -3411, -7533, 15521, -7099, -693, -20071, 918, -4456, 130, -3815, 275, -3556, 2380, -8762, -12403, 3313, 2201, 1385, -262, -3208, 1718, 2757, 1855, 5532, -4948, 15, 13199, 5749, -952, -7651, 1703, 7685, -1948, 2321};
static Q15_T Bg1[HIDDEN_DIM1] = {-338, -3203, 6803, 2320, 17390, 13977, 5211, 7074, 1766, 3903, 2250, -120, 10203, 4521, 4815, -4989};
static Q15_T Bh1[HIDDEN_DIM1] = {15794, -277, 8519, 6835, 22333, -4807, 7724, -1502, -2091, 27088, 9474, -3292, 18416, 8642, -4160, 12199};

static Q7xQ15_FastGRNN_Params RNN1_PARAMS = {
  .mean = NULL,
  .stdDev = NULL,
  .W = W1,
  .U = U1,
  .Bg = Bg1,
  .Bh = Bh1,
  .sigmoid_zeta = 16384,
  .sigmoid_nu = 23611
};

static Q15_T preComp11[HIDDEN_DIM1] = {};
static Q15_T preComp12[HIDDEN_DIM1] = {};
static Q15_T preComp13[HIDDEN_DIM1] = {};
static Q7_T normFeatures1[INPUT_CHANNELS] = {};

static Q7xQ15_FastGRNN_Buffers RNN1_BUFFERS = {
  .preComp1 = preComp11,
  .preComp2 = preComp12,
  .preComp3 = preComp13,
  .normFeatures = normFeatures1
};

#ifdef SHIFT
  static Q15_FastGRNN_Scales RNN1_SCALES = {
    .input = 0,
    .mean = 0,
    .meanSub = 0,
    .stdDev = 0,
    .normFeaturesHDStdDev = 0,
    .W = 3, //8
    .normFeaturesMVW = 2, //4
    .H1W = 2,
    .H2W = 0,
    .U = 6, //64
    .hiddenStateMVU = 5, //32
    .H1U = 4,
    .H2U = 0,
    .mV1AddMV2 = 0, //1
    .mV2AddMV1 = 2, //4
    .mV1AddMV2Out = 0, //1
    .mV1AddMV2Demote = 0, //1
    .pC1AddBg = 0, //1
    .Bg = 3, //8
    .pC1AddBgOut = 0, //1
    .pC1AddBgDemote = 0, //1
    .sigmoidLimit = 0,
    .sigmoidScaleIn = 0,
    .sigmoidScaleOut = 0,
    .pC1AddBh = 0, //1
    .Bh = 4, //16
    .pC1AddBhOut = 0, //1
    .pC1AddBhDemote = 0, //1
    .tanhScaleIn = 0,
    .tanhScaleOut = 0,
    .gateHDHiddenState = 6, //64
    .hiddenStateHDGate = 7, //128
    .qOneScale = 0, //1
    .qOneSubGate = 0, //1
    .qOneSubGateOut = 0, //1
    .sigmoidZeta = 7, //128
    .sigmoidZetaMulQOneSubGate = 7, //128
    .sigmoidNu = 15, //32767
    .sigmoidNuAddQOneSubGate = 0, //1
    .sigmoidNuAddQOneSubGateOut = 0, //1
    .sigmoidNuAddQOneSubGateHDUpdate = 7, //128
    .updateHDSigmoidNuAddQOneSubGate = 7, //128
    .pC3AddPC1 = 1, //2
    .pC1AddPC3 = 0, //1
    .hiddenStateOut = 0, //1
    .hiddenStateDemote = 0, //1
    .div = 0,
    .add = 0,
    .qOne = 16384,
    .useTableSigmoid = 1,
    .useTableTanH = 1
  };
  static SCALE_T ShR1 = 0; //1
  static SCALE_T ShL1 = 1; //2
#else
  static Q15_FastGRNN_Scales RNN1_SCALES = {
    .input = 0,
    .mean = 0,
    .meanSub = 0,
    .stdDev = 0,
    .normFeaturesHDStdDev = 0,
    .W = 8,
    .normFeaturesMVW = 4,
    .H1W = 2,
    .H2W = 0,
    .U = 64,
    .hiddenStateMVU = 32,
    .H1U = 4,
    .H2U = 0,
    .mV1AddMV2 = 1,
    .mV2AddMV1 = 4,
    .mV1AddMV2Out = 1,
    .mV1AddMV2Demote = 1,
    .pC1AddBg = 1,
    .Bg = 8,
    .pC1AddBgOut = 1,
    .pC1AddBgDemote = 1,
    .sigmoidLimit = 0,
    .sigmoidScaleIn = 0,
    .sigmoidScaleOut = 0,
    .pC1AddBh = 1,
    .Bh = 16,
    .pC1AddBhOut = 1,
    .pC1AddBhDemote = 1,
    .tanhScaleIn = 0,
    .tanhScaleOut = 0,
    .gateHDHiddenState = 64,
    .hiddenStateHDGate = 128,
    .qOneScale = 1,
    .qOneSubGate = 1,
    .qOneSubGateOut = 1,
    .sigmoidZeta = 128,
    .sigmoidZetaMulQOneSubGate = 128,
    .sigmoidNu = 32767,
    .sigmoidNuAddQOneSubGate = 1,
    .sigmoidNuAddQOneSubGateOut = 1,
    .sigmoidNuAddQOneSubGateHDUpdate = 128,
    .updateHDSigmoidNuAddQOneSubGate = 128,
    .pC3AddPC1 = 2,
    .pC1AddPC3 = 1,
    .hiddenStateOut = 1,
    .hiddenStateDemote = 1,
    .div = 0,
    .add = 0,
    .qOne = 16384,
    .useTableSigmoid = 1,
    .useTableTanH = 1
  };
  static SCALE_T ShR1 = 1;
  static SCALE_T ShL1 = 2;
#endif
