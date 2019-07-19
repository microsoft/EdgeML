def getTemplate(W0, B0, alpha0, beta0, W1, B1, alpha1, beta1, timeSteps0,
                timeSteps1, fcW, fcB, numOutput, labelMap, normalize=False,
                mean=None, std=None):
    statesLen0 = len(B0)
    featLen0 = W0.shape[1] - statesLen0
    statesLen1 = len(B1)
    featLen1 = W1.shape[1] - statesLen1
    assert W0.shape[0] == statesLen0
    assert W1.shape[0] == statesLen1
    assert featLen1 == statesLen0
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

    retStr = '''
#include <sfastrnn.h>
#include <fc.h>
#include <arm_math.h>

#ifdef __cplusplus
extern "C" {
#endif
struct FastRNNParams fastrnnParams0;
struct FastRNNParams fastrnnParams1;
struct FCParams fcParams;

void initFastRNN0();
void initFastRNN1();
void initFC();

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

void initFastRNN0() {
    fastrnnParams0.timeSteps = %d;
    fastrnnParams0.featLen = %d;
    fastrnnParams0.statesLen = %d;
    fastrnnParams0.W = combinedWMatrix0;
    fastrnnParams0.b = combinedBMatrix0;
    fastrnnParams0.alpha = %f;
    fastrnnParams0.beta = %f;}

void initFastRNN1() {
    fastrnnParams1.timeSteps = %d;
    fastrnnParams1.featLen = %d;
    fastrnnParams1.statesLen = %d;
    fastrnnParams1.W = combinedWMatrix1;
    fastrnnParams1.b = combinedBMatrix1;
    fastrnnParams1.alpha = %f;
    fastrnnParams1.beta = %f;
}

void initFC(){
    fcParams.W = fcW;
    fcParams.B = fcB;
    fcParams.inputDim = %d;
    fcParams.outputDim = %d;
}

''' % (W0Str, B0Str, W1Str, B1Str, FCWStr, FCBStr, timeSteps0, featLen0,
       statesLen0, alpha0, beta0, timeSteps1, featLen1, statesLen1, alpha1,
       beta1, statesLen1, numOutput)

    if normalize == True:
        meanStr = ''
        for val in mean:
            meanStr += '%f,' % val
        stdStr = ''
        for val in std:
            stdStr += '%f, ' % val
        str2 = '''
float featNormMean[] = {%s};
float featNormStd[] = {%s};
''' % (meanStr, stdStr)
        retStr += str2
    label_inv = {}
    for key in labelMap:
        val = labelMap[key]
        if val in label_inv:
            label_inv[val].append(key)
        else:
            label_inv[val] = [key]
    assert len(label_inv) == numOutput
    for i in range(1, numOutput):
        assert len(label_inv[i]) == 1
    
    label_inv_str = 'const char *labelInvArr[] = {"Noise", '
    for i in range(1, numOutput):
        label_inv_str += '"%s",' % label_inv[i][0];
    label_inv_str += "};\n"
    retStr += label_inv_str;
    return retStr


