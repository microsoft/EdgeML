def getTemplate(W0, B0, alpha0, beta0, W1, B1, alpha1, beta1, timeSteps0,
                timeSteps1, fcW, fcB, x0, numOutput):
    statesLen0 = len(B0)
    featLen0 = W0.shape[1] - statesLen0
    statesLen1 = len(B1)
    featLen1 = W1.shape[1] - statesLen1
    assert W0.shape[0] == statesLen0
    assert W1.shape[0] == statesLen1
    assert featLen1 == statesLen0
    assert x0.ndim == 1
    assert fcW.shape[0] == numOutput
    assert fcW.shape[1] == statesLen1
    assert fcW.ndim == 2
    assert fcB.ndim == 1
    assert fcB.shape[0] == numOutput

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

    FCWStr = ''
    for i in range(fcW.shape[0]):
        for j in range(fcW.shape[1]):
            FCWStr += '%f, ' % fcW[i][j]
        FCWStr += '\n'

    FCBStr = ''
    for i in range(fcB.shape[0]):
        FCBStr += '%f,' % fcB[i]

    xx0Str = ''
    for i in range(x0.shape[0]):
        xx0Str += '%d, ' % x0[i]
        if i % 10 == 0:
            xx0Str += '\n'

    return    '''
#include <sfastrnn.h>
#include <fc.h>
#include <arm_math.h>

#ifdef __cplusplus
extern "C" {
#endif
struct FastRNNParams fastrnnParams_test0;
struct FastRNNParams fastrnnParams_test1;
struct FCParams fcParams_test;

void initFastRNN_test0();
void initFastRNN_test1();
void initFC_test();

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

static float fcW[] = {%s};
static float fcB[] = {%s};

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

void initFC_test(){
    fcParams_test.W = fcW;
    fcParams_test.B = fcB;
    fcParams_test.inputDim = %d;
    fcParams_test.outputDim = %d;
}

#define TEST_AUDIO_LEN %d
static int16_t test_audio[TEST_AUDIO_LEN] = { %s
};
''' % (W0Str, B0Str, W1Str, B1Str, FCWStr, FCBStr, timeSteps0, featLen0,
       statesLen0, alpha0, beta0, timeSteps1, featLen1, statesLen1, alpha1,
       beta1, statesLen1, numOutput, len(x0), xx0Str)
