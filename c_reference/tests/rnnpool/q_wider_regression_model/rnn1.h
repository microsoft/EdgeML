// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#define INPUT_CHANNELS 4
#define PATCH_DIM 8
#define HIDDEN_DIM1 8

static const Q15_T W1[HIDDEN_DIM1 * INPUT_CHANNELS] = {7069, -10389, 1562, -1992, 3262, -37, -1143, -995, 5513, -17035, -14615, -6636, 4733, -403, 4106, -1104, -2707, -1287, -18128, -1832, -10108, -137, 2064, 1207, 5233, 226, 831, -1909, 4489, -1099, 2845, -1261};
static const Q15_T U1[HIDDEN_DIM1 * HIDDEN_DIM1] = {15238, -1371, -930, -310, 3195, -4774, -434, 16, -4080, -2624, -10159, 3353, -2368, 5477, 4946, 3484, -18972, 23200, -4141, 10395, -20747, -4430, 11025, 10337, -1467, 5474, -3772, -409, -7005, 2161, 4571, 5800, 3401, 7390, 1400, 2437, 5303, 829, 1986, 2855, 12650, -3378, 1952, 426, -2543, 18282, -2558, 549, -910, 5065, -7026, 5921, -1008, 1428, -1212, 5397, -1587, 7849, -4936, 4664, -11563, 3197, 4943, 561};
static const Q15_T Bg1[HIDDEN_DIM1] = {-18777, -9518, 4055, -7309, 8584, -17257, -5280, -7933};
static const Q15_T Bh1[HIDDEN_DIM1] = {9658, 19740, -10057, 19114, 17227, 12226, 19080, 15855};

static Q15_FastGRNN_Params rnn1_params = {
  .mean = NULL,
  .stdDev = NULL,
  .W = W1,
  .U = U1,
  .Bg = Bg1,
  .Bh = Bh1,
  .sigmoid_zeta = 32522,
  .sigmoid_nu = 30111
};

static Q15_T preComp11[HIDDEN_DIM1] = {};
static Q15_T preComp12[HIDDEN_DIM1] = {};
static Q15_T preComp13[HIDDEN_DIM1] = {};
static Q15_T normFeatures1[HIDDEN_DIM1] = {};

static Q15_FastGRNN_Buffers rnn1_buffers = {
  .preComp1 = preComp11,
  .preComp2 = preComp12,
  .preComp3 = preComp13,
  .normFeatures = normFeatures1
};

#ifdef SHIFT
  static Q15_FastGRNN_Scales rnn1_scales = {
    .input = 0,
    .mean = 0,
    .meanSub = 0,
    .stdDev = 0,
    .normFeaturesHDStdDev = 0,
    .W = 7, //128
    .normFeaturesMVW = 6, //64
    .H1W = 2,
    .H2W = 0,
    .U = 7, //128
    .hiddenStateMVU = 6, //64
    .H1U = 3,
    .H2U = 0,
    .mV1AddMV2 = 0, //1
    .mV2AddMV1 = 2, //4
    .mV1AddMV2Out = 0, //1
    .mV1AddMV2Demote = 0, //1
    .pC1AddBg = 0, //1
    .Bg = 3, //8
    .pC1AddBgOut = 0, //1
    .pC1AddBgDemote = 0, //1
    .sigmoidLimit = 2048,
    .sigmoidScaleIn = 11, //2048
    .sigmoidScaleOut = 14, //16384
    .pC1AddBh = 0, //1
    .Bh = 4, //16
    .pC1AddBhOut = 0, //1
    .pC1AddBhDemote = 0, //1
    .tanhScaleIn = 11, //2048
    .tanhScaleOut = 11, //2048
    .gateHDHiddenState = 7, //128
    .hiddenStateHDGate = 7, //128
    .qOneScale = 0, //1
    .qOneSubGate = 0, //1
    .qOneSubGateOut = 0, //1
    .sigmoidZeta = 7, //128
    .sigmoidZetaMulQOneSubGate = 8, //256
    .sigmoidNu = 8, //256
    .sigmoidNuAddQOneSubGate = 0, //1
    .sigmoidNuAddQOneSubGateOut = 0, //1
    .sigmoidNuAddQOneSubGateHDUpdate = 5, //32
    .updateHDSigmoidNuAddQOneSubGate = 6, //64
    .pC3AddPC1 = 0, //1
    .pC1AddPC3 = 0, //1
    .hiddenStateOut = 0, //1
    .hiddenStateDemote = 0, //1
    .div = 2,
    .add = 1024,
    .qOne = 16384,
    .useTableSigmoid = 0,
    .useTableTanH = 0
  };
  static SCALE_T ShR1 = 0; //1
  static SCALE_T ShL1 = 0; //1
#else
  static Q15_FastGRNN_Scales rnn1_scales = {
    .input = 0,
    .mean = 0,
    .meanSub = 0,
    .stdDev = 0,
    .normFeaturesHDStdDev = 0,
    .W = 128,
    .normFeaturesMVW = 64,
    .H1W = 4,
    .H2W = 0,
    .U = 128,
    .hiddenStateMVU = 64,
    .H1U = 8,
    .H2U = 0,
    .mV1AddMV2 = 1,
    .mV2AddMV1 = 4,
    .mV1AddMV2Out = 1,
    .mV1AddMV2Demote = 1,
    .pC1AddBg = 1,
    .Bg = 8,
    .pC1AddBgOut = 1,
    .pC1AddBgDemote = 1,
    .sigmoidLimit = 2048,
    .sigmoidScaleIn = 11, //2048
    .sigmoidScaleOut = 14, //16384
    .pC1AddBh = 1,
    .Bh = 16,
    .pC1AddBhOut = 1,
    .pC1AddBhDemote = 1,
    .tanhScaleIn = 11, //2048
    .tanhScaleOut = 11, //2048
    .gateHDHiddenState = 128,
    .hiddenStateHDGate = 128,
    .qOneScale = 1,
    .qOneSubGate = 1,
    .qOneSubGateOut = 1,
    .sigmoidZeta = 128,
    .sigmoidZetaMulQOneSubGate = 256,
    .sigmoidNu = 256,
    .sigmoidNuAddQOneSubGate = 1,
    .sigmoidNuAddQOneSubGateOut = 1,
    .sigmoidNuAddQOneSubGateHDUpdate = 32,
    .updateHDSigmoidNuAddQOneSubGate = 64,
    .pC3AddPC1 = 1,
    .pC1AddPC3 = 1,
    .hiddenStateOut = 1,
    .hiddenStateDemote = 1,
    .div = 2,
    .add = 1024,
    .qOne = 16384,
    .useTableSigmoid = 0,
    .useTableTanH = 0
  };
  static SCALE_T ShR1 = 1;
  static SCALE_T ShL1 = 1;
#endif
