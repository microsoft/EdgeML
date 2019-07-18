def getTemplate(W0, B0, alpha0, beta0, W1, B1, alpha1, beta1, timeSteps0,
                timeSteps1, x0, x1, expected):
    statesLen0 = len(B0)
    featLen0 = W0.shape[1] - statesLen0
    statesLen1 = len(B1)
    featLen1 = W1.shape[1] - statesLen1
    assert W0.shape[0] == statesLen0
    assert W1.shape[0] == statesLen1
    assert x0.shape[0] == timeSteps0
    assert x0.shape[1] == featLen0
    assert featLen1 == statesLen0

    W0Str = ''
    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            W0Str += '%f, ' % W0[i][j]
        W0Str += '\n'

    B0Str = ''
    for i in range(B0.shape[0]):
        B0Str += '%f,' % B0[i]

    W1Str = ''
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1Str += '%f, ' % W1[i][j]
        W1Str += '\n'

    B1Str = ''
    for i in range(B1.shape[0]):
        B1Str += '%f,' % B1[i]

    xx0Str = ''
    for i in range(x0.shape[0]):
        for j in range(x0.shape[1]):
            xx0Str += '%f, ' % x0[i][j]
        xx0Str += '\n'
    xx1Str = ''
    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            xx1Str += '%f, ' % x1[i][j]
        xx1Str += '\n'

    return    '''
#include "sfastrnn.h"
#include <OledDisplay.h>
#include <Arduino.h>

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
    %s
 };

static float combinedBMatrix0[] = {%s};

static float combinedWMatrix1[] = {
    %s
 };
static float combinedBMatrix1[] = {%s};

void initFastRNN_test0() {
    fastrnnParams_test0.timeSteps = %d;
    fastrnnParams_test0.featLen = %d;
    fastrnnParams_test0.statesLen = %d;
    fastrnnParams_test0.W = combinedWMatrix0;
    fastrnnParams_test0.b = combinedBMatrix0;
    fastrnnParams_test0.alpha = %f;
    fastrnnParams_test0.beta = %f;}

void initFastRNN_test1() {
    fastrnnParams_test1.timeSteps = %d;
    fastrnnParams_test1.featLen = %d;
    fastrnnParams_test1.statesLen = %d;
    fastrnnParams_test1.W = combinedWMatrix1;
    fastrnnParams_test1.b = combinedBMatrix1;
    fastrnnParams_test1.alpha = %f;
    fastrnnParams_test1.beta = %f;
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
    initSFastRNN2(&sparams, &fastrnnParams_test0, &fastrnnParams_test1,
            h0container);

    float result_h[statesLen1];
    static float xx0[] = {
        %s
    };
    static float xx1[] = {
        %s
    };
    for(int i = 0; i < timeSteps1; i++)
        SFastRNNInference2(&sparams, xx0, result_h);
    unsigned long StartTime = millis();
    for(int i = 0; i < 100; i++)
        SFastRNNInference2(&sparams, xx1, result_h);
    unsigned long CurrentTime = millis();
    unsigned long ElapsedTime = CurrentTime - StartTime;
    Serial.print("Time taken for 100 runs (ms): ");
    Serial.println(ElapsedTime);
    Serial.print("Layer 0 [");
    Serial.print("Inp: "); Serial.print(fastrnnParams_test0.featLen);
    Serial.print(" hDim: "); Serial.print(statesLen0);
    Serial.print(" ts: "); Serial.print(fastrnnParams_test0.timeSteps);
    Serial.println("]");
    Serial.print("Layer 1 [");
    Serial.print("Inp: "); Serial.print(fastrnnParams_test1.featLen);
    Serial.print(" hDim: "); Serial.print(statesLen1);
    Serial.print(" ts: "); Serial.print(fastrnnParams_test1.timeSteps);
    Serial.println("]");
    return errorCode;
}

void setup() {
    Screen.init();
    Serial.begin(115200);
    Serial.println("Ready");
}

void loop() {
    Serial.println("Loop");
    testFastRNN();
    Serial.println();
    delay(1000);

}

int main(){
    setup();
    delay(500);
    for(int i = 0; i < 100; i++)
        loop();
}
''' % (W0Str, B0Str, W1Str, B1Str, timeSteps0, featLen0, statesLen0, alpha0,
       beta0, timeSteps1, featLen1, statesLen1, alpha1, beta1, xx0Str, xx1Str)
