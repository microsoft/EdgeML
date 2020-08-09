// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#define HIDDEN_DIM2 8

static Q15_T W2[HIDDEN_DIM2 * HIDDEN_DIM1] = {-849, -1535, 10036, -3592, -286, -6119, -5622, -3760, 359, -6106, -339, -1294, -220, -4337, -2852, -841, -9841, -1977, 745, -996, -1397, -1678, -861, -712, 5701, -5419, -3624, 0, 439, -9575, -3738, 396, 7390, -1214, 1684, 1441, -1650, 13070, 2595, 1405, -4589, -5064, -1926, 2806, 3409, -12783, -284, 3339, -3958, 77, 2312, -1717, -19971, -55, -672, -1476, 2759, -4657, 2028, -3686, -192, -5647, -5103, -3669};
static Q15_T U2[HIDDEN_DIM2 * HIDDEN_DIM2] = {8755, -905, -2432, -2758, -3162, -4223, 181, 312, 2010, 9661, 1688, 2240, 5634, 5383, -1084, 996, -3641, -1874, 1328, 1795, 15468, -15323, 863, -3599, -912, -327, -1492, 6117, -1188, -2615, 1111, -866, 5998, 4034, 4122, 6542, 704, 19957, -4613, 2397, -2311, -3909, 769, -6010, -1738, 2042, 4177, -1213, -388, -354, -176, 710, 483, -578, 3342, -916, -1570, -5116, 9988, 283, 3409, -318, 4059, 8633};
static Q15_T Bg2[HIDDEN_DIM2] = {-5410, -15414, -13002, -12121, -18930, -17922, -8692, -12150};
static Q15_T Bh2[HIDDEN_DIM2] = {21417, 6457, 6421, 8970, 6601, 836, 3060, 8468};
static Q15_T sigmoid_zeta2 = 32520;
static Q15_T sigmoid_nu2 = 16387;

static SCALE_T input2 = 0;
static SCALE_T meanScale2 = 0;
static SCALE_T meanSub2 = 0;
static SCALE_T stdDevScale2 = 0;
static SCALE_T normFeaturesHDStdDev2 = 0;
static SCALE_T H1W2 = 3;
static SCALE_T H2W2 = 0;
static SCALE_T H1U2 = 3;
static SCALE_T H2U2 = 0;
static Q15_T div2 = 2;
static Q15_T add2 = 4096;
static Q15_T sigmoidLimit2 = 8192;
static SCALE_T sigmoidScaleIn2 = 13; //8192
static SCALE_T sigmoidScaleOut2 = 14; //16384
static SCALE_T tanhScaleIn2 = 13; //8192
static SCALE_T tanhScaleOut2 = 13; //8192
static Q15_T qOne2 = 16384;
static ITER_T useTableSigmoid2 = 0;
static ITER_T useTableTanH2 = 0;

#ifdef SHIFT
  static SCALE_T WScale2 = 6; //64
  static SCALE_T normFeaturesMVW2 = 5; //32
  static SCALE_T UScale2 = 6; //64
  static SCALE_T hiddenStateMVU2 = 6; //64
  static SCALE_T mV1AddMV22 = 1; //2
  static SCALE_T mV2AddMV12 = 0; //1
  static SCALE_T mV1AddMV2Out2 = 0; //1
  static SCALE_T pC1AddBg2 = 0; //1
  static SCALE_T BgScale2 = 1; //2
  static SCALE_T pC1AddBgOut2 = 0; //1
  static SCALE_T pC1AddBh2 = 0; //1
  static SCALE_T BhScale2 = 1; //2
  static SCALE_T pC1AddBhOut2 = 0; //1
  static SCALE_T gateHDHiddenState2 = 6; //64
  static SCALE_T hiddenStateHDGate2 = 7; //128
  static SCALE_T qOneScale2 = 0; //1
  static SCALE_T qOneSubGate2 = 0; //1
  static SCALE_T qOneSubGateOut2 = 0; //1
  static SCALE_T sigmoidZetaScale2 = 7; //128
  static SCALE_T sigmoidZetaMulQOneSubGate2 = 8; //256
  static SCALE_T sigmoidNuScale2 = 7; //128
  static SCALE_T sigmoidNuAddQOneSubGate2 = 0; //1
  static SCALE_T sigmoidNuAddQOneSubGateOut2 = 0; //1
  static SCALE_T sigmoidNuAddQOneSubGateHDUpdate2 = 6; //64
  static SCALE_T updateHDSigmoidNuAddQOneSubGate2 = 7; //128
  static SCALE_T pC3AddPC12 = 1; //2
  static SCALE_T pC1AddPC32 = 0; //1
  static SCALE_T hiddenStateOut2 = 0; //1
#else
  static SCALE_T WScale2 = 64;
  static SCALE_T normFeaturesMVW2 = 32;
  static SCALE_T UScale2 = 64;
  static SCALE_T hiddenStateMVU2 = 64;
  static SCALE_T mV1AddMV22 = 2;
  static SCALE_T mV2AddMV12 = 1;
  static SCALE_T mV1AddMV2Out2 = 1;
  static SCALE_T pC1AddBg2 = 1;
  static SCALE_T BgScale2 = 2;
  static SCALE_T pC1AddBgOut2 = 1;
  static SCALE_T pC1AddBh2 = 1;
  static SCALE_T BhScale2 = 2;
  static SCALE_T pC1AddBhOut2 = 1;
  static SCALE_T gateHDHiddenState2 = 64;
  static SCALE_T hiddenStateHDGate2 = 128;
  static SCALE_T qOneScale2 = 1;
  static SCALE_T qOneSubGate2 = 1;
  static SCALE_T qOneSubGateOut2 = 1;
  static SCALE_T sigmoidZetaScale2 = 128;
  static SCALE_T sigmoidZetaMulQOneSubGate2 = 256;
  static SCALE_T sigmoidNuScale2 = 128;
  static SCALE_T sigmoidNuAddQOneSubGate2 = 1;
  static SCALE_T sigmoidNuAddQOneSubGateOut2 = 1;
  static SCALE_T sigmoidNuAddQOneSubGateHDUpdate2 = 64;
  static SCALE_T updateHDSigmoidNuAddQOneSubGate2 = 128;
  static SCALE_T pC3AddPC12 = 2;
  static SCALE_T pC1AddPC32 = 1;
  static SCALE_T hiddenStateOut2 = 1;
#endif
