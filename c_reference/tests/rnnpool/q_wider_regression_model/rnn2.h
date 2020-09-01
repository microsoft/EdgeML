// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#define HIDDEN_DIM2 8

static const Q15_T W2[HIDDEN_DIM2 * HIDDEN_DIM1] = {-849, -1535, 10036, -3592, -286, -6119, -5622, -3760, 359, -6106, -339, -1294, -220, -4337, -2852, -841, -9841, -1977, 745, -996, -1397, -1678, -861, -712, 5701, -5419, -3624, 0, 439, -9575, -3738, 396, 7390, -1214, 1684, 1441, -1650, 13070, 2595, 1405, -4589, -5064, -1926, 2806, 3409, -12783, -284, 3339, -3958, 77, 2312, -1717, -19971, -55, -672, -1476, 2759, -4657, 2028, -3686, -192, -5647, -5103, -3669};
static const Q15_T U2[HIDDEN_DIM2 * HIDDEN_DIM2] = {8755, -905, -2432, -2758, -3162, -4223, 181, 312, 2010, 9661, 1688, 2240, 5634, 5383, -1084, 996, -3641, -1874, 1328, 1795, 15468, -15323, 863, -3599, -912, -327, -1492, 6117, -1188, -2615, 1111, -866, 5998, 4034, 4122, 6542, 704, 19957, -4613, 2397, -2311, -3909, 769, -6010, -1738, 2042, 4177, -1213, -388, -354, -176, 710, 483, -578, 3342, -916, -1570, -5116, 9988, 283, 3409, -318, 4059, 8633};
static const Q15_T Bg2[HIDDEN_DIM2] = {-5410, -15414, -13002, -12121, -18930, -17922, -8692, -12150};
static const Q15_T Bh2[HIDDEN_DIM2] = {21417, 6457, 6421, 8970, 6601, 836, 3060, 8468};

static Q15_FastGRNN_Params rnn2_params = {
  .mean = NULL,
  .stdDev = NULL,
  .W = W2,
  .U = U2,
  .Bg = Bg2,
  .Bh = Bh2,
  .sigmoid_zeta = 32520,
  .sigmoid_nu = 16387
};

static Q15_T preComp21[HIDDEN_DIM2] = {};
static Q15_T preComp22[HIDDEN_DIM2] = {};
static Q15_T preComp23[HIDDEN_DIM2] = {};
static Q15_T normFeatures2[HIDDEN_DIM1] = {};

static Q15_FastGRNN_Buffers rnn2_buffers = {
  .preComp1 = preComp21,
  .preComp2 = preComp22,
  .preComp3 = preComp23,
  .normFeatures = normFeatures2
};

#ifdef SHIFT
  static Q15_FastGRNN_Scales rnn2_scales = {
    .input = 0,
    .mean = 0,
    .meanSub = 0,
    .stdDev = 0,
    .normFeaturesHDStdDev = 0,
    .W = 6, //64
    .normFeaturesMVW = 5, //32
    .H1W = 3,
    .H2W = 0,
    .U = 6, //64
    .hiddenStateMVU = 6, //64
    .H1U = 3,
    .H2U = 0,
    .mV1AddMV2 = 1, //2
    .mV2AddMV1 = 0, //1
    .mV1AddMV2Out = 0, //1
    .mV1AddMV2Demote = 0, //1
    .pC1AddBg = 0, //1
    .Bg = 1, //2
    .pC1AddBgOut = 0, //1
    .pC1AddBgDemote = 0, //1
    .sigmoidLimit = 8192,
    .sigmoidScaleIn = 13, //8192
    .sigmoidScaleOut = 14, //16384
    .pC1AddBh = 0, //1
    .Bh = 1, //2
    .pC1AddBhOut = 0, //1
    .pC1AddBhDemote = 0, //1
    .tanhScaleIn = 13, //8192
    .tanhScaleOut = 13, //8192
    .gateHDHiddenState = 6, //64
    .hiddenStateHDGate = 7, //128
    .qOneScale = 0, //1
    .qOneSubGate = 0, //1
    .qOneSubGateOut = 0, //1
    .sigmoidZeta = 7, //128
    .sigmoidZetaMulQOneSubGate = 8, //256
    .sigmoidNu = 7, //128
    .sigmoidNuAddQOneSubGate = 0, //1
    .sigmoidNuAddQOneSubGateOut = 0, //1
    .sigmoidNuAddQOneSubGateHDUpdate = 6, //64
    .updateHDSigmoidNuAddQOneSubGate = 7, //128
    .pC3AddPC1 = 1, //2
    .pC1AddPC3 = 0, //1
    .hiddenStateOut = 0, //1
    .hiddenStateDemote = 0, //1
    .div = 2,
    .add = 4096,
    .qOne = 16384,
    .useTableSigmoid = 0,
    .useTableTanH = 0
  };
  static SCALE_T ShR2 = 0; //1
  static SCALE_T ShL2 = 0; //1
#else
  static Q15_FastGRNN_Scales rnn2_scales = {
    .input = 0,
    .mean = 0,
    .meanSub = 0,
    .stdDev = 0,
    .normFeaturesHDStdDev = 0,
    .W = 64,
    .normFeaturesMVW = 32,
    .H1W = 8,
    .H2W = 0,
    .U = 64,
    .hiddenStateMVU = 64,
    .H1U = 8,
    .H2U = 0,
    .mV1AddMV2 = 2,
    .mV2AddMV1 = 1,
    .mV1AddMV2Out = 1,
    .mV1AddMV2Demote = 1,
    .pC1AddBg = 1,
    .Bg = 2,
    .pC1AddBgOut = 1,
    .pC1AddBgDemote = 1,
    .sigmoidLimit = 8192,
    .sigmoidScaleIn = 13, //8192
    .sigmoidScaleOut = 14, //16384
    .pC1AddBh = 1,
    .Bh = 2,
    .pC1AddBhOut = 1,
    .pC1AddBhDemote = 1,
    .tanhScaleIn = 13, //8192
    .tanhScaleOut = 13, //8192
    .gateHDHiddenState = 64,
    .hiddenStateHDGate = 128,
    .qOneScale = 1,
    .qOneSubGate = 1,
    .qOneSubGateOut = 1,
    .sigmoidZeta = 128,
    .sigmoidZetaMulQOneSubGate = 256,
    .sigmoidNu = 128,
    .sigmoidNuAddQOneSubGate = 1,
    .sigmoidNuAddQOneSubGateOut = 1,
    .sigmoidNuAddQOneSubGateHDUpdate = 64,
    .updateHDSigmoidNuAddQOneSubGate = 128,
    .pC3AddPC1 = 2,
    .pC1AddPC3 = 1,
    .hiddenStateOut = 1,
    .hiddenStateDemote = 1,
    .div = 2,
    .add = 4096,
    .qOne = 16384,
    .useTableSigmoid = 0,
    .useTableTanH = 0
  };
  static SCALE_T ShR2 = 1;
  static SCALE_T ShL2 = 1;
#endif
