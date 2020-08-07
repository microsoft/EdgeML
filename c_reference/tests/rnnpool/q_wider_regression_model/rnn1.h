// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#define INPUT_CHANNELS 4
#define PATCH_DIM 8
#define HIDDEN_DIM1 8

static Q15_T W1[HIDDEN_DIM1 * INPUT_CHANNELS] = {7069, -10389, 1562, -1992, 3262, -37, -1143, -995, 5513, -17035, -14615, -6636, 4733, -403, 4106, -1104, -2707, -1287, -18128, -1832, -10108, -137, 2064, 1207, 5233, 226, 831, -1909, 4489, -1099, 2845, -1261};
static Q15_T U1[HIDDEN_DIM1 * HIDDEN_DIM1] = {15238, -1371, -930, -310, 3195, -4774, -434, 16, -4080, -2624, -10159, 3353, -2368, 5477, 4946, 3484, -18972, 23200, -4141, 10395, -20747, -4430, 11025, 10337, -1467, 5474, -3772, -409, -7005, 2161, 4571, 5800, 3401, 7390, 1400, 2437, 5303, 829, 1986, 2855, 12650, -3378, 1952, 426, -2543, 18282, -2558, 549, -910, 5065, -7026, 5921, -1008, 1428, -1212, 5397, -1587, 7849, -4936, 4664, -11563, 3197, 4943, 561};
static Q15_T Bg1[HIDDEN_DIM1] = {-18777, -9518, 4055, -7309, 8584, -17257, -5280, -7933};
static Q15_T Bh1[HIDDEN_DIM1] = {9658, 19740, -10057, 19114, 17227, 12226, 19080, 15855};
static Q15_T sigmoid_zeta1 = 32522;
static Q15_T sigmoid_nu1 = 30111;

static SCALE_T input1 = 0;
static SCALE_T meanScale1 = 0;
static SCALE_T meanSub1 = 0;
static SCALE_T stdDevScale1 = 0;
static SCALE_T normFeaturesHDStdDev1 = 0;
static SCALE_T H1W1 = 2;
static SCALE_T H2W1 = 0;
static SCALE_T H1U1 = 3;
static SCALE_T H2U1 = 0;
static Q15_T div1 = 2;
static Q15_T add1 = 1024;
static Q15_T sigmoidLimit1 = 2048;
static SCALE_T sigmoidScaleIn1 = 11; //2048
static SCALE_T sigmoidScaleOut1 = 14; //16384
static SCALE_T tanhScaleIn1 = 11; //2048
static SCALE_T tanhScaleOut1 = 11; //2048
static Q15_T qOne1 = 16384;

#ifdef SHIFT
  static SCALE_T WScale1 = 7; //128
  static SCALE_T normFeaturesMVW1 = 6; //64
  static SCALE_T UScale1 = 7; //128
  static SCALE_T hiddenStateMVU1 = 6; //64
  static SCALE_T mV1AddMV21 = 0; //1
  static SCALE_T mV2AddMV11 = 2; //4
  static SCALE_T mV1AddMV2Out1 = 0; //1
  static SCALE_T pC1AddBg1 = 0; //1
  static SCALE_T BgScale1 = 3; //8
  static SCALE_T pC1AddBgOut1 = 0; //1
  static SCALE_T pC1AddBh1 = 0; //1
  static SCALE_T BhScale1 = 4; //16
  static SCALE_T pC1AddBhOut1 = 0; //1
  static SCALE_T gateHDHiddenState1 = 7; //128
  static SCALE_T hiddenStateHDGate1 = 7; //128
  static SCALE_T qOneScale1 = 0; //1
  static SCALE_T qOneSubGate1 = 0; //1
  static SCALE_T qOneSubGateOut1 = 0; //1
  static SCALE_T sigmoidZetaScale1 = 7; //128
  static SCALE_T sigmoidZetaMulQOneSubGate1 = 8; //256
  static SCALE_T sigmoidNuScale1 = 8; //256
  static SCALE_T sigmoidNuAddQOneSubGate1 = 0; //1
  static SCALE_T sigmoidNuAddQOneSubGateOut1 = 0; //1
  static SCALE_T sigmoidNuAddQOneSubGateHDUpdate1 = 5; //32
  static SCALE_T updateHDSigmoidNuAddQOneSubGate1 = 6; //64
  static SCALE_T pC3AddPC11 = 0; //1
  static SCALE_T pC1AddPC31 = 0; //1
  static SCALE_T hiddenStateOut1 = 0; //1
#else
  static SCALE_T WScale1 = 128;
  static SCALE_T normFeaturesMVW1 = 64;
  static SCALE_T UScale1 = 128;
  static SCALE_T hiddenStateMVU1 = 64;
  static SCALE_T mV1AddMV21 = 1;
  static SCALE_T mV2AddMV11 = 4;
  static SCALE_T mV1AddMV2Out1 = 1;
  static SCALE_T pC1AddBg1 = 1;
  static SCALE_T BgScale1 = 8;
  static SCALE_T pC1AddBgOut1 = 1;
  static SCALE_T pC1AddBh1 = 1;
  static SCALE_T BhScale1 = 16;
  static SCALE_T pC1AddBhOut1 = 1;
  static SCALE_T gateHDHiddenState1 = 128;
  static SCALE_T hiddenStateHDGate1 = 128;
  static SCALE_T qOneScale1 = 1;
  static SCALE_T qOneSubGate1 = 1;
  static SCALE_T qOneSubGateOut1 = 1;
  static SCALE_T sigmoidZetaScale1 = 128;
  static SCALE_T sigmoidZetaMulQOneSubGate1 = 256;
  static SCALE_T sigmoidNuScale1 = 256;
  static SCALE_T sigmoidNuAddQOneSubGate1 = 1;
  static SCALE_T sigmoidNuAddQOneSubGateOut1 = 1;
  static SCALE_T sigmoidNuAddQOneSubGateHDUpdate1 = 32;
  static SCALE_T updateHDSigmoidNuAddQOneSubGate1 = 64;
  static SCALE_T pC3AddPC11 = 1;
  static SCALE_T pC1AddPC31 = 1;
  static SCALE_T hiddenStateOut1 = 1;
#endif
