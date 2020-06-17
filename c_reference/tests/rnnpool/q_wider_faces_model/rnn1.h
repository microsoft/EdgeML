#define INPUT_CHANNELS 4
#define PATCH_DIM 8
#define HIDDEN_DIM1 8

static MYINT W1[HIDDEN_DIM1 * INPUT_CHANNELS] = {7069, -10389, 1562, -1992, 3262, -37, -1143, -995, 5513, -17035, -14615, -6636, 4733, -403, 4106, -1104, -2707, -1287, -18128, -1832, -10108, -137, 2064, 1207, 5233, 226, 831, -1909, 4489, -1099, 2845, -1261};
static MYINT U1[HIDDEN_DIM1 * HIDDEN_DIM1] = {15238, -1371, -930, -310, 3195, -4774, -434, 16, -4080, -2624, -10159, 3353, -2368, 5477, 4946, 3484, -18972, 23200, -4141, 10395, -20747, -4430, 11025, 10337, -1467, 5474, -3772, -409, -7005, 2161, 4571, 5800, 3401, 7390, 1400, 2437, 5303, 829, 1986, 2855, 12650, -3378, 1952, 426, -2543, 18282, -2558, 549, -910, 5065, -7026, 5921, -1008, 1428, -1212, 5397, -1587, 7849, -4936, 4664, -11563, 3197, 4943, 561};
static MYINT Bg1[HIDDEN_DIM1] = {-18777, -9518, 4055, -7309, 8584, -17257, -5280, -7933};
static MYINT Bh1[HIDDEN_DIM1] = {9658, 19740, -10057, 19114, 17227, 12226, 19080, 15855};
static MYINT sigmoid_zeta1 = 32522;
static MYINT sigmoid_nu1 = 30111;

static MYSCL input1 = 0;
static MYSCL meanScale1 = 0;
static MYSCL meanSub1 = 0;
static MYSCL stdDevScale1 = 0;
static MYSCL normFeaturesHDStdDev1 = 0;
static MYSCL H1W1 = 2;
static MYSCL H2W1 = 0;
static MYSCL H1U1 = 3;
static MYSCL H2U1 = 0;
static MYINT div1 = 2;
static MYINT add1 = 1024;
static MYINT sigmoidLimit1 = 2048;
static MYSCL sigmoidScaleIn1 = 11; //2048
static MYSCL sigmoidScaleOut1 = 14; //16384
static MYSCL tanhScaleIn1 = 11; //2048
static MYSCL tanhScaleOut1 = 11; //2048
static MYINT qOne1 = 16384;

#ifdef SHIFT
	static MYSCL WScale1 = 7; //128
	static MYSCL normFeaturesMVW1 = 6; //64
	static MYSCL UScale1 = 7; //128
	static MYSCL hiddenStateMVU1 = 6; //64
	static MYSCL mV1AddMV21 = 0; //1
	static MYSCL mV2AddMV11 = 2; //4
	static MYSCL mV1AddMV2Out1 = 0; //1
	static MYSCL pC1AddBg1 = 0; //1
	static MYSCL BgScale1 = 3; //8
	static MYSCL pC1AddBgOut1 = 0; //1
	static MYSCL pC1AddBh1 = 0; //1
	static MYSCL BhScale1 = 4; //16
	static MYSCL pC1AddBhOut1 = 0; //1
	static MYSCL gateHDHiddenState1 = 7; //128
	static MYSCL hiddenStateHDGate1 = 7; //128
	static MYSCL qOneScale1 = 0; //1
	static MYSCL qOneSubGate1 = 0; //1
	static MYSCL qOneSubGateOut1 = 0; //1
	static MYSCL sigmoidZetaScale1 = 7; //128
	static MYSCL sigmoidZetaMulQOneSubGate1 = 8; //256
	static MYSCL sigmoidNuScale1 = 8; //256
	static MYSCL sigmoidNuAddQOneSubGate1 = 0; //1
	static MYSCL sigmoidNuAddQOneSubGateOut1 = 0; //1
	static MYSCL sigmoidNuAddQOneSubGateHDUpdate1 = 5; //32
	static MYSCL updateHDSigmoidNuAddQOneSubGate1 = 6; //64
	static MYSCL pC3AddPC11 = 0; //1
	static MYSCL pC1AddPC31 = 0; //1
	static MYSCL hiddenStateOut1 = 0; //1
#else
	static MYSCL WScale1 = 128;
	static MYSCL normFeaturesMVW1 = 64;
	static MYSCL UScale1 = 128;
	static MYSCL hiddenStateMVU1 = 64;
	static MYSCL mV1AddMV21 = 1;
	static MYSCL mV2AddMV11 = 4;
	static MYSCL mV1AddMV2Out1 = 1;
	static MYSCL pC1AddBg1 = 1;
	static MYSCL BgScale1 = 8;
	static MYSCL pC1AddBgOut1 = 1;
	static MYSCL pC1AddBh1 = 1;
	static MYSCL BhScale1 = 16;
	static MYSCL pC1AddBhOut1 = 1;
	static MYSCL gateHDHiddenState1 = 128;
	static MYSCL hiddenStateHDGate1 = 128;
	static MYSCL qOneScale1 = 1;
	static MYSCL qOneSubGate1 = 1;
	static MYSCL qOneSubGateOut1 = 1;
	static MYSCL sigmoidZetaScale1 = 128;
	static MYSCL sigmoidZetaMulQOneSubGate1 = 256;
	static MYSCL sigmoidNuScale1 = 256;
	static MYSCL sigmoidNuAddQOneSubGate1 = 1;
	static MYSCL sigmoidNuAddQOneSubGateOut1 = 1;
	static MYSCL sigmoidNuAddQOneSubGateHDUpdate1 = 32;
	static MYSCL updateHDSigmoidNuAddQOneSubGate1 = 64;
	static MYSCL pC3AddPC11 = 1;
	static MYSCL pC1AddPC31 = 1;
	static MYSCL hiddenStateOut1 = 1;
#endif
