#include "fastrnn.h"
#include <Arduino.h>
#include <OledDisplay.h>


#ifdef __cplusplus
extern "C" {
#endif
struct FastRNNParams fastrnnParams_test;
void initFastRNN_test();
#ifdef __cplusplus
}
#endif
static float combinedWMatrix[] = {
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
    0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
 };

static float combinedBMatrix[] = {1, 2, 3, 4};


void initFastRNN_test() {
    fastrnnParams_test.timeSteps = 8;
    fastrnnParams_test.featLen = 6;
    fastrnnParams_test.statesLen = 4;
    fastrnnParams_test.W = combinedWMatrix;
    fastrnnParams_test.b = combinedBMatrix;
    fastrnnParams_test.alpha = 0.2;
    fastrnnParams_test.beta = 0.8;
}

unsigned verifyOutput(float *result, float *expected, unsigned length){
    unsigned errorCount = 0;
    for (int i = 0; i < length; i++){
        if(abs(result[i] - expected[i]) > 0.00001)
            errorCount += 1;
    }
    return errorCount;
}

unsigned testFastRNN(){
    unsigned errorCode = 0;
    initFastRNN_test();
    int statesLen = fastrnnParams_test.statesLen;
    int featLen = fastrnnParams_test.featLen;

    float x[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    float input_h[] = {0.0, 0.0, 0.0, 0.0};
    float hx[statesLen + featLen];
    float hx_expected[] = {0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    combineHX(&fastrnnParams_test, input_h, x, hx);
    if (verifyOutput(hx, hx_expected, statesLen + featLen))
        errorCode |= 1;
    float result_h[statesLen];
    FastRNNStep(&fastrnnParams_test, x, input_h, result_h);
    float result_h_expected[] = {0.18798266, 0.19625871, 0.19886951, 0.19966154};
    if (verifyOutput(result_h, result_h_expected, statesLen))
        errorCode |= 2;

    float xx[] = {
        0.0,0.01,0.02,0.03,0.04,0.05,
        0.06,0.07,0.08,0.09,0.1,0.11,
        0.12,0.13,0.14,0.15,0.16,0.17,
        0.18,0.19,0.2,0.21,0.22,0.23,
        0.24,0.25,0.26,0.27,0.28,0.29,
        0.3,0.31,0.32,0.33,0.34,0.35,
        0.36,0.37,0.38,0.39,0.4,0.41,
        0.42,0.43,0.44,0.45,0.46,0.47
    };
    FastRNNInference(&fastrnnParams_test, xx,  result_h);
    float result_h_expected2[] = {0.77809181, 0.81525842, 0.82698329, 0.83059075};
    if (verifyOutput(result_h, result_h_expected2, statesLen))
        errorCode |= 3;
    return errorCode;
}

void setup(){
    Serial.begin(115200);
    Screen.init();
    delay(500);
    Serial.println("Ready");
}

void loop(){
    unsigned errorCode = testFastRNN();
    Serial.printf("FastRNN error Code: %d\n", errorCode);
    delay(400);
}


int main(){
    setup();
    for(int i=0; i < 100; i++)
        loop();
}
