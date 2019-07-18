#include <OledDisplay.h>
#include <Arduino.h>
#include "lstm.h"

#ifdef __cplusplus
extern "C" {
#endif
struct LSTMParams lstmParams_test;
void initLSTM_test();
#ifdef __cplusplus
}
#endif
static float combinedWMatrix[] = {
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
    0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
    0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
    0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
    0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
    0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
    1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
    1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
    1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1,
    1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2,
    1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3,
    1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
    1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5,
 };

static float combinedBMatrix[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16};

void initLSTM_test() {
    lstmParams_test.timeSteps = 8;
    lstmParams_test.featLen = 6;
    lstmParams_test.statesLen = 4;
    lstmParams_test.forgetBias = 1.0;
    lstmParams_test.W = combinedWMatrix;
    lstmParams_test.B = combinedBMatrix;
}

unsigned runLSTMTests(){
	float epsilon7 = 1e-7f;
	float epsilon5 = 1e-5f;
	float epsilon6 = 1e-6f;
	float epsilon4 = 1e-4f;

	unsigned testFailures = 0;
	initLSTM_test();
	unsigned m = 4 * lstmParams_test.statesLen;
	unsigned n = lstmParams_test.statesLen + lstmParams_test.featLen;
	for(int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			if (fabs(lstmParams_test.W[i * n + j] - 0.1*(float)(i + j + 1.0f)) >= epsilon4){
				testFailures |= 2;
			}
		}
	}

	for(int i = 0; i < n; i++)
		if ((fabs(lstmParams_test.B[i] -  (i + 1))) >= epsilon7)
			testFailures |= 4;

	float x[8][6] = {
		{0.00f, 0.10f, 0.20f, 0.30f, 0.40f, 0.50f,},
		{0.10f, 0.20f, 0.30f, 0.40f, 0.50f, 0.60f,},
		{0.20f, 0.30f, 0.40f, 0.50f, 0.60f, 0.70f,},
		{0.30f, 0.40f, 0.50f, 0.60f, 0.70f, 0.80f,},
		{0.40f, 0.50f, 0.60f, 0.70f, 0.80f, 0.90f,},
		{0.50f, 0.60f, 0.70f, 0.80f, 0.90f, 1.00f,},
		{0.60f, 0.70f, 0.80f, 0.90f, 1.00f, 1.10f,},
		{0.70f, 0.80f, 0.90f, 1.00f, 1.10f, 1.20f,},
	};

	float h[4] = {0.1f, 0.3f, 0.5f, 0.7f};
	float dst[lstmParams_test.featLen + lstmParams_test.statesLen];
	combineXH(&lstmParams_test, x[0], h, dst);
	float target[] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.1f, 0.3f, 0.5f, 0.7f};
	for(int i = 0; i < lstmParams_test.featLen + lstmParams_test.statesLen; i++){
		if(fabs(target[i] - dst[i]) >= epsilon7)
			testFailures |= 8;
	}
	float combinedOut[4 * lstmParams_test.statesLen];
	matrixVectorMul(lstmParams_test.W, 4*lstmParams_test.statesLen,
		lstmParams_test.statesLen + lstmParams_test.featLen, 
		dst, combinedOut);	
	float target2[] = {
		2.1600000858f, 2.4700000286f, 2.7800002098f,
		3.0900001526f, 3.4000000954f, 3.7100000381f,
		4.0199999809f, 4.3299999237f, 4.6399998665f,
		4.9499998093f, 5.2600002289f, 5.5699996948f,
		5.8800001144f, 6.1900000572f, 6.5000000000f,
		6.8099999428f
	};
	for(int i = 0; i < 4 * lstmParams_test.statesLen; i++){
		if (fabs(target2[i] - combinedOut[i]) >= epsilon6){
			testFailures |= 16;
		}
	}

	float testVec0[] = {0.96f, 0.01f, 0.01f, 0.02f};
	float testVec1[] = {0.02f, 0.94f, 0.02f, 0.02f};
	float target3[] =  {0.98f, 0.95f, 0.03f, 0.04f};
	vectorVectorAdd(testVec1, testVec0, 4);
	for(int i = 0; i < 4; i++)
		if(fabs(testVec1[i] - target3[i]) >= epsilon7)
			testFailures |= 32;

	float testVec2[] = {-0.2f, 0.2f, 0.0f, 1.2f, -1.2f};
	vsigmoid(testVec2, 5);
	float target4[] = {0.450166f, 0.549834f, 0.5f, 0.76852478f, 0.23147522f};
	for(int i = 0; i < 5; i++){
		if(fabs(testVec2[i] - target4[i]) >= epsilon7)
			testFailures |= 64;
	}

	float testVec3[] = {-2.0f, 0.1f, 0.0f, 1.2f, -1.2f};
	float target5[] = {-0.96402758f, 0.09966799f, 0.00f, 0.83365461f, -0.83365461f};
	vtanh(testVec3, 5);
	for(int i = 0; i < 5; i++){
		if(fabs(testVec3[i] - target5[i]) >= epsilon7)
			testFailures |= 128;
	}

	float result_c_h_o[3 * lstmParams_test.statesLen];
	for(int i = 0; i < 3 * lstmParams_test.statesLen; i++){
		result_c_h_o[i] = 0;
	}
	float target7[] = {
		0.8455291f, 0.94531806f, 0.9820137f, 0.99423402f,
		0.68872638f, 0.73765609f, 0.7539363f, 0.75916194,
		0.99999976f,  1.0f, 1.0f, 1.0f};	

	LSTMStep(&lstmParams_test, (float*)&(x[0]), result_c_h_o, result_c_h_o);
	for(int i =0; i < 3 * lstmParams_test.statesLen; i++){
		if(fabs(result_c_h_o[i] - target7[i]) >= epsilon7){
			testFailures |= 256;
			Serial.printf("%d %2.10f %2.10f\n", i, result_c_h_o[i], target7[i]);
		}
	}
	float target8[] = {
		1.8336372f, 1.94265039f, 1.98141956f, 1.99410193f,
		0.95018068f, 0.95974363f, 0.96269107f, 0.9636085f,
		1.0f, 1.0f, 1.0f, 1.0f};

	LSTMStep(&lstmParams_test, (float*)&(x[1]), result_c_h_o, result_c_h_o);
	for(int i =0; i < 3 * lstmParams_test.statesLen; i++){
		if(fabs(result_c_h_o[i] - target8[i]) >= epsilon5){
			testFailures |= 512;
		}
	}
	float target9[] = {
		7.81787791f, 7.93995698f, 7.98095536f,
		7.99402111f, 0.99999968f, 0.99999975f,
		0.99999977f, 0.99999977f, 1.0f,
		1.0f, 1.0f, 1.0f};
	for(int i = 2; i < 8; i++)
		LSTMStep(&lstmParams_test, (float*)&(x[i]), result_c_h_o, result_c_h_o);
	for(int i =0; i < 3 * lstmParams_test.statesLen; i++){
		if(fabs(result_c_h_o[i] - target9[i]) >= epsilon5){
			testFailures |= 1024;
		}
	}
	return testFailures;
}

void setup(){
    Screen.init();
    Serial.begin(115200);
}

void loop(){
    // print a string to the screen with wrapped = false
    Screen.print("Hello my dude", false);
    delay(1000);
    int a = runLSTMTests(); 
    // print a string to the screen with wrapped = true
	Screen.print("Testing LSTM", false);
	char buff[30];
	sprintf(buff, "Error Code: %d", a);
	Screen.print(1, buff);
	Serial.println(buff);
    delay(1000);
    // Clean up the screen
    Screen.clean();
    delay(1000);
}

int main(){
	setup();
	for(int i = 0; i < 100; i++)
		loop();
}