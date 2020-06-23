// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#define HIDDEN_DIM2 8

MYINT W2[HIDDEN_DIM2 * HIDDEN_DIM1] = {-849, -1535, 10036, -3592, -286, -6119, -5622, -3760, 359, -6106, -339, -1294, -220, -4337, -2852, -841, -9841, -1977, 745, -996, -1397, -1678, -861, -712, 5701, -5419, -3624, 0, 439, -9575, -3738, 396, 7390, -1214, 1684, 1441, -1650, 13070, 2595, 1405, -4589, -5064, -1926, 2806, 3409, -12783, -284, 3339, -3958, 77, 2312, -1717, -19971, -55, -672, -1476, 2759, -4657, 2028, -3686, -192, -5647, -5103, -3669};
MYINT U2[HIDDEN_DIM2 * HIDDEN_DIM2] = {8755, -905, -2432, -2758, -3162, -4223, 181, 312, 2010, 9661, 1688, 2240, 5634, 5383, -1084, 996, -3641, -1874, 1328, 1795, 15468, -15323, 863, -3599, -912, -327, -1492, 6117, -1188, -2615, 1111, -866, 5998, 4034, 4122, 6542, 704, 19957, -4613, 2397, -2311, -3909, 769, -6010, -1738, 2042, 4177, -1213, -388, -354, -176, 710, 483, -578, 3342, -916, -1570, -5116, 9988, 283, 3409, -318, 4059, 8633};
MYINT Bg2[HIDDEN_DIM2] = {-5410, -15414, -13002, -12121, -18930, -17922, -8692, -12150};
MYINT Bh2[HIDDEN_DIM2] = {21417, 6457, 6421, 8970, 6601, 836, 3060, 8468};
MYINT sigmoid_zeta2 = 32520;
MYINT sigmoid_nu2 = 16387;

static MYSCL input2 = 0;
static MYSCL meanScale2 = 0;
static MYSCL meanSub2 = 0;
static MYSCL stdDevScale2 = 0;
static MYSCL normFeaturesHDStdDev2 = 0;
static MYSCL H1W2 = 3;
static MYSCL H2W2 = 0;
static MYSCL H1U2 = 3;
static MYSCL H2U2 = 0;
static MYINT div2 = 2;
static MYINT add2 = 4096;
static MYINT sigmoidLimit2 = 8192;
static MYSCL sigmoidScaleIn2 = 13; //8192
static MYSCL sigmoidScaleOut2 = 14; //16384
static MYSCL tanhScaleIn2 = 13; //8192
static MYSCL tanhScaleOut2 = 13; //8192
static MYINT qOne2 = 16384;

#ifdef SHIFT
  static MYSCL WScale2 = 6; //64
  static MYSCL normFeaturesMVW2 = 5; //32
  static MYSCL UScale2 = 6; //64
  static MYSCL hiddenStateMVU2 = 6; //64
  static MYSCL mV1AddMV22 = 1; //2
  static MYSCL mV2AddMV12 = 0; //1
  static MYSCL mV1AddMV2Out2 = 0; //1
  static MYSCL pC1AddBg2 = 0; //1
  static MYSCL BgScale2 = 1; //2
  static MYSCL pC1AddBgOut2 = 0; //1
  static MYSCL pC1AddBh2 = 0; //1
  static MYSCL BhScale2 = 1; //2
  static MYSCL pC1AddBhOut2 = 0; //1
  static MYSCL gateHDHiddenState2 = 6; //64
  static MYSCL hiddenStateHDGate2 = 7; //128
  static MYSCL qOneScale2 = 0; //1
  static MYSCL qOneSubGate2 = 0; //1
  static MYSCL qOneSubGateOut2 = 0; //1
  static MYSCL sigmoidZetaScale2 = 7; //128
  static MYSCL sigmoidZetaMulQOneSubGate2 = 8; //256
  static MYSCL sigmoidNuScale2 = 7; //128
  static MYSCL sigmoidNuAddQOneSubGate2 = 0; //1
  static MYSCL sigmoidNuAddQOneSubGateOut2 = 0; //1
  static MYSCL sigmoidNuAddQOneSubGateHDUpdate2 = 6; //64
  static MYSCL updateHDSigmoidNuAddQOneSubGate2 = 7; //128
  static MYSCL pC3AddPC12 = 1; //2
  static MYSCL pC1AddPC32 = 0; //1
  static MYSCL hiddenStateOut2 = 0; //1
#else
  static MYSCL WScale2 = 64;
  static MYSCL normFeaturesMVW2 = 32;
  static MYSCL UScale2 = 64;
  static MYSCL hiddenStateMVU2 = 64;
  static MYSCL mV1AddMV22 = 2;
  static MYSCL mV2AddMV12 = 1;
  static MYSCL mV1AddMV2Out2 = 1;
  static MYSCL pC1AddBg2 = 1;
  static MYSCL BgScale2 = 2;
  static MYSCL pC1AddBgOut2 = 1;
  static MYSCL pC1AddBh2 = 1;
  static MYSCL BhScale2 = 2;
  static MYSCL pC1AddBhOut2 = 1;
  static MYSCL gateHDHiddenState2 = 64;
  static MYSCL hiddenStateHDGate2 = 128;
  static MYSCL qOneScale2 = 1;
  static MYSCL qOneSubGate2 = 1;
  static MYSCL qOneSubGateOut2 = 1;
  static MYSCL sigmoidZetaScale2 = 128;
  static MYSCL sigmoidZetaMulQOneSubGate2 = 256;
  static MYSCL sigmoidNuScale2 = 128;
  static MYSCL sigmoidNuAddQOneSubGate2 = 1;
  static MYSCL sigmoidNuAddQOneSubGateOut2 = 1;
  static MYSCL sigmoidNuAddQOneSubGateHDUpdate2 = 64;
  static MYSCL updateHDSigmoidNuAddQOneSubGate2 = 128;
  static MYSCL pC3AddPC12 = 2;
  static MYSCL pC1AddPC32 = 1;
  static MYSCL hiddenStateOut2 = 1;
#endif
