// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "lstm.h"
#include <stdio.h> 

extern struct LSTMParams lstmParams;
static unsigned errorCode = -1;
static unsigned timeSteps, featLen, statesLen, forgetBias;
static unsigned WNumRows, WNumCols, BNumElements;
static void combineXH(float *x, float *h, float *dst);
static unsigned getInitErrorCode();


unsigned initializeLSTM(unsigned timeSteps_,
    unsigned featLen_, unsigned statesLen_,
    float forgetBias_){
    initLSTMParams();
    timeSteps = timeSteps_;
    featLen = featLen_;
    statesLen = statesLen_;
    forgetBias = forgetBias_;

    WNumRows = 4 * statesLen;
    WNumCols = featLen + statesLen;
    BNumElements = WNumRows;
    errorCode = getInitErrorCode();
    return errorCode;
}


static unsigned getInitErrorCode(){
    errorCode = 0;
    if ((timeSteps != lstmParams.timeSteps))
        errorCode |= 1;
    if (featLen != lstmParams.featLen)
        errorCode |= 2;
    if (statesLen != lstmParams.statesLen)
        errorCode |= 4;
    if (forgetBias != lstmParams.forgetBias){
        errorCode |= 8;
    }
    return errorCode;
}


static void combineXH(float *x, float *h, float *dst){
    for(int i = 0; i < lstmParams.featLen; i++)
        dst[i] = x[i];
    for(int i = 0; i < lstmParams.statesLen; i++)
        dst[lstmParams.featLen + i] = h[i];
}

void LSTMStep(float *x, float *input_c_h, float *result_c_h_o){
    float *c = &(input_c_h[0*lstmParams.statesLen]);
    float *h = &(input_c_h[1*lstmParams.statesLen]);
    float o[lstmParams.statesLen];

    float combinedOut[4 * (lstmParams.statesLen)];
    float xh[lstmParams.statesLen + lstmParams.featLen];
    combineXH(x, h, xh);
    matrixVectorMul(lstmParams.W, 4*lstmParams.statesLen,
        lstmParams.statesLen + lstmParams.featLen,
        xh, combinedOut);
    vectorVectorAdd(combinedOut, lstmParams.B,
        4 * lstmParams.statesLen);
    // Apply non-linearity
    // i_t
    vsigmoid(&combinedOut[0*lstmParams.statesLen], lstmParams.statesLen);
    // c_cap_t
    vtanh(&combinedOut[1*lstmParams.statesLen], lstmParams.statesLen);
    // f_t (after adding forget bias)
    for(int i = 0; i < lstmParams.statesLen; i++)
        combinedOut[2*lstmParams.statesLen + i] += lstmParams.forgetBias;
    vsigmoid(&combinedOut[2*lstmParams.statesLen], lstmParams.statesLen);
    // o_t
    vsigmoid(&combinedOut[3*lstmParams.statesLen], lstmParams.statesLen);
    
    // update c
    for(int i = 0; i < lstmParams.statesLen; i++){
        //c_t = (f_t + forget_bias)*C_t-1 + i_t*c_cap_t
        c[i] = combinedOut[2*lstmParams.statesLen + i] * c[i];
        c[i] += combinedOut[0*lstmParams.statesLen + i]*combinedOut[1*lstmParams.statesLen + i];
        //o_t
        o[i] = combinedOut[3*lstmParams.statesLen + i];
        //h_t
        h[i] = o[i] * tanh(c[i]);
    }
    // returns c, h, o
    for(int i = 0; i < lstmParams.statesLen; i++){
        result_c_h_o[lstmParams.statesLen * 0 + i] = c[i];
        result_c_h_o[lstmParams.statesLen * 1 + i] = h[i];
        result_c_h_o[lstmParams.statesLen * 2 + i] = o[i];
    }
}


void LSTMInference(float x[][lstmParams.featLen], float* result_c_h_o){
    for(int i = 0; i < 3 * lstmParams.statesLen; i++){
        result_c_h_o[i] = 0;
    }
    for (int t = 0; t < lstmParams.timeSteps; t++){
        LSTMStep((float*)&(x[t]), result_c_h_o, result_c_h_o);
    }
}


unsigned runLSTMTests(){
#ifndef __TEST_LSTM__
    return -1;
#else
    float epsilon7 = 1e-7f;
    float epsilon5 = 1e-5f;
    float epsilon6 = 1e-6f;

    unsigned testFailures = 0;
    int err = initializeLSTM(8, 6, 4, 1.0f);
    if(err != 0)
        testFailures |= 1;
    unsigned m = 4 * lstmParams.statesLen;
    unsigned n = lstmParams.statesLen + lstmParams.featLen;
    for(int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            if (fabs(lstmParams.W[i * n + j] - 0.1*(float)(i + j + 1.0f)) >= epsilon7){
                testFailures |= 2;
            }
        }
    }

    for(int i = 0; i < n; i++)
        if ((fabs(lstmParams.B[i] -  (i + 1))) >= epsilon7)
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
    float dst[lstmParams.featLen + lstmParams.statesLen];
    combineXH(x[0], h, dst);
    float target[] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.1f, 0.3f, 0.5f, 0.7f};
    for(int i = 0; i < lstmParams.featLen + lstmParams.statesLen; i++){
        if(fabs(target[i] - dst[i]) >= epsilon7)
            testFailures |= 8;
    }
    float combinedOut[4 * lstmParams.statesLen];
    matrixVectorMul(lstmParams.W, 4*lstmParams.statesLen,
        lstmParams.statesLen + lstmParams.featLen, 
        dst, combinedOut);  
    float target2[] = {
        2.1600000858f, 2.4700000286f, 2.7800002098f,
        3.0900001526f, 3.4000000954f, 3.7100000381f,
        4.0199999809f, 4.3299999237f, 4.6399998665f,
        4.9499998093f, 5.2600002289f, 5.5699996948f,
        5.8800001144f, 6.1900000572f, 6.5000000000f,
        6.8099999428f
    };
    for(int i = 0; i < 4 * lstmParams.statesLen; i++){
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

    float result_c_h_o[3 * lstmParams.statesLen];
    for(int i = 0; i < 3 * lstmParams.statesLen; i++){
        result_c_h_o[i] = 0;
    }
    float target7[] = {
        0.8455291f, 0.94531806f, 0.9820137f, 0.99423402f,
        0.68872638f, 0.73765609f, 0.7539363f, 0.75916194,
        0.99999976f,  1.0f, 1.0f, 1.0f};    

    LSTMStep((float*)&(x[0]), result_c_h_o, result_c_h_o);
    for(int i =0; i < 3 * lstmParams.statesLen; i++){
        if(fabs(result_c_h_o[i] - target7[i]) >= epsilon7){
            testFailures |= 256;
            printf("%d %2.10f %2.10f\n", i, result_c_h_o[i], target7[i]);
        }
    }
    float target8[] = {
        1.8336372f, 1.94265039f, 1.98141956f, 1.99410193f,
        0.95018068f, 0.95974363f, 0.96269107f, 0.9636085f,
        1.0f, 1.0f, 1.0f, 1.0f};

    LSTMStep((float*)&(x[1]), result_c_h_o, result_c_h_o);
    for(int i =0; i < 3 * lstmParams.statesLen; i++){
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
        LSTMStep((float*)&(x[i]), result_c_h_o, result_c_h_o);
    for(int i =0; i < 3 * lstmParams.statesLen; i++){
        if(fabs(result_c_h_o[i] - target9[i]) >= epsilon5){
            testFailures |= 1024;
        }
    }
    return testFailures;
#endif
}
