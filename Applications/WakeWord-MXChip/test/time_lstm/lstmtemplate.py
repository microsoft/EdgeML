def getFile(inputDim, kernel, bias, x, h_final):
	hiddenDim = int(kernel.shape[0] / 4)
	inputDim_ = int(kernel.shape[1] - hiddenDim)
	assert inputDim_ == inputDim
	kernelStr = ''
	for i in range(kernel.shape[0]):
		for j in range(kernel.shape[1]):
			val = kernel[i][j]
			kernelStr += '%2.5f, ' % val
		kernelStr += '\n'
	
	biasStr = ''
	for i in range(bias.shape[0]):
		val = bias[i]
		biasStr  += '%2.5f, ' % val

	xStr = ''
	for i in range(x.shape[0]):
		val  = x[i]
		xStr += '%2.5f, ' % val
	timeSteps = 10
	template = 	'''
/*
 * h_final = %r
 */
#include <OledDisplay.h>
#include "lstm.h"
#include <Arduino.h>

#ifdef __cplusplus
extern "C" {
#endif
struct LSTMParams lstmParams_test;
void initLSTM_test();
#ifdef __cplusplus
}
#endif
static float combinedWMatrix[] = {
	%s
};

static float combinedBMatrix[] = {
	%s
};

void initLSTM_test() {
    lstmParams_test.timeSteps = %d;
    lstmParams_test.featLen = %d;
    lstmParams_test.statesLen = %d;
    lstmParams_test.forgetBias = 1.0;
    // W is [4 * statesLen, (featLen + staetsLen)] flattened (i.e. row major)
    // (first row, second row, .. .. 4 * statesLen-th row)
    lstmParams_test.W = combinedWMatrix;
    // 4 * statesLen
    lstmParams_test.B = combinedBMatrix;
}

void setup(){
    Screen.init();
    Serial.begin(115200);
}

void loop(){
    // print a string to the screen with wrapped = false
    Screen.print("Hello my dude", false);
    delay(1000);
	initLSTM_test();

	float x[] = {%s};
	float result_c_h_o[3 * lstmParams_test.statesLen];
	for(int i = 0; i < 3 * lstmParams_test.statesLen; i++){
		result_c_h_o[i] = 0;
	}

	LSTMStep(&lstmParams_test, x, result_c_h_o, result_c_h_o);
    for (int i = 0; i <  lstmParams_test.statesLen; i++) {
        int j = i + lstmParams_test.statesLen;
        float val = result_c_h_o[j];
    }
	unsigned long StartTime = millis();
	for (int i = 0; i < 100; i++)
		LSTMStep(&lstmParams_test, x, result_c_h_o, result_c_h_o);
	unsigned long CurrentTime = millis();
	unsigned long ElapsedTime = CurrentTime - StartTime;
	Serial.print("Time taken for 100 runs (ms): ");
	Serial.println(ElapsedTime);
    Serial.print("Inp :"); Serial.print(lstmParams_test.featLen);
    Serial.print(" Hid: "); Serial.println(lstmParams_test.statesLen);

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
''' % (h_final, kernelStr, biasStr, timeSteps, inputDim, hiddenDim, xStr)
	return template