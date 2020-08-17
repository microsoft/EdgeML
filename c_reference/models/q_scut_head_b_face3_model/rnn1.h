// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stddef.h>

#define INPUT_CHANNELS 4
#define PATCH_DIM 16
#define HIDDEN_DIM1 16

static Q15_T W1[HIDDEN_DIM1 * INPUT_CHANNELS] = {-210, -27032, 937, -6793, 9238, -3708, -2870, -15450, 1693, -7075, 9292, 302, -1432, -1211, -612, -185, -5023, 1175, -763, 758, 8099, 18648, -4062, 5946, 6542, 1563, -1099, -7114, 7901, -6661, -15763, 7786, -2979, 1201, -1364, -92, -13909, 2198, 15012, 673, 831, 24951, 11791, 2190, -1082, -2085, -801, 2566, -6389, 2501, 17032, -5762, 5209, 9138, 1910, 6332, 13552, 19706, -21165, -4891, 786, -785, 3316, 3694};
static Q15_T U1[HIDDEN_DIM1 * HIDDEN_DIM1] = {841, -3789, 4454, 3313, 495, 7664, -657, -3435, -3434, -4923, 5736, 813, -6536, -729, 754, -6324, -2418, -294, 4579, 3084, 8789, -1581, 12486, 7319, -2633, -4038, -7735, 3878, -2780, -2481, -6100, -1248, 1089, -3500, 2929, 2391, 3396, -8252, -2666, -2253, -963, 4325, -2722, -1713, -2070, 2290, -486, 5260, -3464, 2331, -10406, 6309, -5772, -7402, -2588, 9720, 6231, -5734, 12433, 6362, 10873, -9696, -13642, 7913, -1601, 2921, -2302, 301, 9349, -3409, 8471, 3433, 3665, 4659, -3986, 899, 5703, -6712, 4658, -4057, -6602, 1364, -13508, 102, 1827, -2073, 2370, -2021, -3586, 762, -3622, -2389, 4823, -4418, -2712, 12432, 954, -6890, -2337, 25, -5916, 2526, 2699, 435, 1529, -1426, 736, -1803, -197, 263, 3973, 7616, -4785, 3107, -8398, -5166, -4100, -7253, -3428, -946, -850, 19672, 11830, -2774, 12868, -3371, -9447, 775, -3206, 6290, -10846, 3328, -3361, -7770, 2185, 14050, 3768, -14568, -10714, 4033, -10967, 6556, 9255, -353, 1052, 9663, 155, 3933, -6407, 4606, 885, 14749, -2975, -7595, -4977, 4985, -12251, -4804, 6561, -4964, 5487, -3399, -4954, -2248, 5101, -6428, -3209, 5551, 1428, -6807, 2044, 4679, -4810, 325, 643, 3233, -200, -11002, 15891, 2341, -493, 15769, 3857, -13337, 469, 924, -16842, 4629, -4384, 7545, 10928, -12818, -1464, -1916, -4218, -3071, 1957, -4323, -2810, -7872, 5747, 8444, 1589, -2352, 12448, -761, -5322, 4112, 4300, 5905, -884, 1268, -3597, -1771, -5134, -1702, 1258, 5289, 2632, -2722, 2274, 7073, 4580, -3012, 11680, -13309, 7875, 2523, 2965, -2956, 67, -7231, 4618, 21517, -8240, -742, 12009, -1877, -2298, -7166, 14579, -9547, 3149, -1351, 47, 3528, -9995, 9711, 5413, -2713, -7528, 1654, 1471, 6530, -1836, 9415};
static Q15_T Bg1[HIDDEN_DIM1] = {-2138, 10826, 6140, 25064, 12847, 8407, 20211, 12381, 23684, 11153, -5968, 21482, 14153, 13973, 9733, 26886};
static Q15_T Bh1[HIDDEN_DIM1] = {20314, 3646, 8073, 5051, 1522, -5865, 2808, 11369, 9208, -118, -13912, 865, -4372, -8318, -5080, 4996};

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
    .Bh = 3, //8
    .pC1AddBhOut = 0, //1
    .pC1AddBhDemote = 0, //1
    .tanhScaleIn = 0,
    .tanhScaleOut = 0,
    .gateHDHiddenState = 7, //128
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
    .pC3AddPC1 = 0, //1
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
  static SCALE_T ShL1 = 0; //1
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
    .Bh = 8,
    .pC1AddBhOut = 1,
    .pC1AddBhDemote = 1,
    .tanhScaleIn = 0,
    .tanhScaleOut = 0,
    .gateHDHiddenState = 128,
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
    .pC3AddPC1 = 1,
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
  static SCALE_T ShL1 = 1;
#endif
