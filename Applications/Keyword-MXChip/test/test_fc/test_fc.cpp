#include <OledDisplay.h>
#include <Arduino.h>
#include "fc.h"

#ifdef __cplusplus
extern "C" {
#endif 
struct FCParams fcParams_test;
extern void initFC_test();
#ifdef __cplusplus
}
#endif

static float W[] = {
        0.496714,-0.138264,0.647689,1.523030,
        -0.234153,-0.234137,1.579213,0.767435,
        -0.469474,0.542560,-0.463418,-0.465730,
};
static float B[] = {0.241962,-1.913280,-1.724918,};
static unsigned inputDim = 4;
static unsigned outputDim = 3;

void initFC_test(){
    fcParams_test.W = W;
    fcParams_test.B = B;
    fcParams_test.inputDim = inputDim;
    fcParams_test.outputDim = outputDim;
}

unsigned runFCTests(){
	float epsilon7 = 1e-4f;
	float vec_0[] = {-0.562288,-1.012831,0.314247,-0.908024,};
	float vec_1[] = {-1.412304,1.465649,-0.225776,0.067528,};
	float vec_2[] = {-1.424748,-0.544383,0.110923,-1.150994,};
	float vec_3[] = {0.375698,-0.600639,-0.291694,-0.601707,};
	float vec_4[] = {1.852278,-0.013497,-1.057711,0.822545,};
	float res_0[] = {-1.076709,-1.745063,-1.733194,};
	float res_1[] = {-0.705581,-2.230472,-0.193496,};
	float res_2[] = {-2.071616,-2.160353,-0.866747,};
	float res_3[] = {-0.593720,-2.783037,-1.811772,};
	float res_4[] = {1.731574,-3.382938,-2.494760,};
	unsigned testFailures = 0;
	unsigned __inpDim = 4;
	unsigned __outputDim = 3;
	unsigned nonLinearity = 0;

	initFC_test();

	float result[__outputDim];
	FCInference(&fcParams_test, vec_0, result, nonLinearity);
	for(int i = 0; i < __outputDim; i++)
		if ((fabs(res_0[i] - result[i])) > epsilon7)
			testFailures |= 2;

	FCInference(&fcParams_test, vec_1, result, nonLinearity);
	for(int i = 0; i < __outputDim; i++)
		if ((fabs(res_1[i] - result[i])) > epsilon7)
			testFailures |= 4;
	
	FCInference(&fcParams_test, vec_2, result, nonLinearity);
	for(int i = 0; i < __outputDim; i++)
		if ((fabs(res_2[i] - result[i])) > epsilon7)
			testFailures |= 8;
	
	FCInference(&fcParams_test, vec_3, result, nonLinearity);
	for(int i = 0; i < __outputDim; i++)
		if ((fabs(res_3[i] - result[i])) > epsilon7)
			testFailures |= 16;
	
	FCInference(&fcParams_test, vec_4, result, nonLinearity);
	for(int i = 0; i < __outputDim; i++)
		if ((fabs(res_4[i] - result[i])) > epsilon7)
			testFailures |= 32;
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
    int a = runFCTests();
    // print a string to the screen with wrapped = true
	char buf[100];
    sprintf(buf, "Testing FC\nError Code: %d", a);
    Screen.print(buf);
	Serial.println(buf);
    delay(3000);
    // Clean up the screen
    Screen.clean();
    delay(1000);
}

int main(){
	setup();
	delay(500);
	for(int i = 0; i < 100; i++)
		loop();
}