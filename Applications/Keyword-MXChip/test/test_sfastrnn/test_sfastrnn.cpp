#include <Arduino.h>
#include <OledDisplay.h>
#include "sfastrnn.h"

#define PRECISION 4

#ifdef __cplusplus
extern "C" {
#endif
struct FastRNNParams fastrnnParams_test0;
struct FastRNNParams fastrnnParams_test1;
void initFastRNN_test0();
void initFastRNN_test1();
#ifdef __cplusplus
}
#endif
static float combinedWMatrix0[] = {
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
    0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
 };

static float combinedBMatrix0[] = {1, 2, 3, 4};

static float combinedWMatrix1[] = {
    0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
    0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
 };

static float combinedBMatrix1[] = {0.1, 0.2, 0.3, 0.4};

void initFastRNN_test0() {
    fastrnnParams_test0.timeSteps = 4;
    fastrnnParams_test0.featLen = 6;
    fastrnnParams_test0.statesLen = 4;
    fastrnnParams_test0.W = combinedWMatrix0;
    fastrnnParams_test0.b = combinedBMatrix0;
    fastrnnParams_test0.alpha = 0.2;
    fastrnnParams_test0.beta = 0.8;
}

void initFastRNN_test1() {
    fastrnnParams_test1.timeSteps = 3;
    fastrnnParams_test1.featLen = 4;
    fastrnnParams_test1.statesLen = 4;
    fastrnnParams_test1.W = combinedWMatrix1;
    fastrnnParams_test1.b = combinedBMatrix1;
    fastrnnParams_test1.alpha = 0.3;
    fastrnnParams_test1.beta = 0.7;
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
    initFastRNN_test0();
    initFastRNN_test1();
    unsigned errorCode = 0;
    struct SFastRNNParams2 sparams;
    unsigned statesLen0 = fastrnnParams_test0.statesLen;
    unsigned timeSteps1 = fastrnnParams_test1.timeSteps;
    unsigned statesLen1 = fastrnnParams_test1.statesLen;
    float h0container[statesLen0 * timeSteps1];
    memset(h0container, 0, (statesLen0 * timeSteps1) * sizeof(float));
    /*for(int i = 0; i < statesLen0 * timeSteps1; i++)*/
        /*printf("%d %1.1f\n", i, h0container[i]);*/
    /*printf(">>1\n\n");*/
    initSFastRNN2(&sparams, &fastrnnParams_test0, &fastrnnParams_test1,
            h0container);

    float result_h[statesLen1];
    float xx0[] = {
        0.0,0.01,0.02,0.03,0.04,0.05,   0.06,0.07,0.08,0.09,0.1,0.11,
        0.12,0.13,0.14,0.15,0.16,0.17,  0.18,0.19,0.2,0.21,0.22,0.23,
    };
    float xx1[] = {
        0.12,0.13,0.14,0.15,0.16,0.17,  0.18,0.19,0.2,0.21,0.22,0.23,
        0.24,0.25,0.26,0.27,0.28,0.29,  0.3,0.31,0.32,0.33,0.34,0.35,
    };
    float xx2[] = {
        0.24,0.25,0.26,0.27,0.28,0.29,  0.3,0.31,0.32,0.33,0.34,0.35,
        0.36,0.37,0.38,0.39,0.4,0.41,   0.42,0.43,0.44,0.45,0.46,0.47,
    };
    float xx3[] = {
        0.36,0.37,0.38,0.39,0.4,0.41,   0.42,0.43,0.44,0.45,0.46,0.47,
        0.0,0.01,0.02,0.03,0.04,0.05,   0.06,0.07,0.08,0.09,0.1,0.11,
    };
    float xx4[] = {
        0.0,0.01,0.02,0.03,0.04,0.05,   0.06,0.07,0.08,0.09,0.1,0.11,
        0.24,0.25,0.26,0.27,0.28,0.29,  0.3,0.31,0.32,0.33,0.34,0.35,
    };
    float xx5[] = {
        0.12,0.13,0.14,0.15,0.16,0.17,  0.18,0.19,0.2,0.21,0.22,0.23,
        0.36,0.37,0.38,0.39,0.4,0.41,   0.42,0.43,0.44,0.45,0.46,0.47,
    };
    SFastRNNInference2(&sparams, xx0, result_h);
    SFastRNNInference2(&sparams, xx1, result_h);
    SFastRNNInference2(&sparams, xx2, result_h);
    float exp_h0[] ={0.60863139, 0.62427422, 0.58376938, 0.60744043};
    if(verifyOutput(result_h, exp_h0, statesLen1)) {
        errorCode |= 1;
    }
    SFastRNNInference2(&sparams, xx3, result_h);
    float exp_h1[] ={0.60897788, 0.62464093, 0.58391194, 0.60766335};
    if(verifyOutput(result_h, exp_h1, statesLen1)) {
        errorCode |= 2;
    }
    SFastRNNInference2(&sparams, xx4, result_h);
    float exp_h2[] ={0.60866079, 0.62443474, 0.58348852, 0.60734866};
    if(verifyOutput(result_h, exp_h2, statesLen1)) {
        errorCode |= 4;
    }
    SFastRNNInference2(&sparams, xx5, result_h);
    float exp_h3[] ={0.60841268, 0.62415074, 0.5834325, 0.60720676};
    if(verifyOutput(result_h, exp_h3, statesLen1)) {
        errorCode |= 8;
    }
    return errorCode;
}

void setup(){
    Screen.init();
    Serial.begin(115200);
    delay(500);
    Serial.println("Ready");
    delay(500);
}

int main(){
    setup();
    char buffer[30];
    for(int i = 0; i < 100; i++){
        Serial.print("New Loop - ");
        unsigned errorCode = testFastRNN();
        Serial.printf("Error Code: %d\n", errorCode);
        Screen.print(1, "Hello my dude", false);
        sprintf(buffer, "Error code %d", errorCode);
        Screen.print(2, buffer, false);
        delay(1000);
        Screen.clean();
        delay(500);
    }
}
