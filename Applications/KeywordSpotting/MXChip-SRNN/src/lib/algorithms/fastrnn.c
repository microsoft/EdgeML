#include "fastrnn.h"

void combineHX(const struct FastRNNParams *fastrnnParams, const float *h,
        const float *x, float *dst){
    memcpy(dst, h, fastrnnParams->statesLen * sizeof(float));
    memcpy(&(dst[fastrnnParams->statesLen]), x, fastrnnParams->featLen* sizeof(float));
}

void FastRNNStep(const struct FastRNNParams *fastrnnParams, const float *x,
        const float *input_h, float *result_h){
    unsigned statesLen = fastrnnParams->statesLen;
    unsigned featLen = fastrnnParams->featLen;
    float h[statesLen];
    memcpy(h, input_h, statesLen * sizeof(float));
    float combinedOut[statesLen];
    float hx[statesLen + featLen];
    combineHX(fastrnnParams, h, x, hx);
    // W[h, x]
    matrixVectorMul(fastrnnParams->W, statesLen, statesLen + featLen, hx,
            combinedOut);
    // h_ = h_ + b 
    vectorVectorAdd(combinedOut, fastrnnParams->b, statesLen);
    // Apply non-linearity (currently only sigmoid)
    vsigmoid(combinedOut, statesLen);
    scalarVectorMul(combinedOut, statesLen, fastrnnParams->alpha);
    scalarVectorMul(h, statesLen, fastrnnParams->beta);
    vectorVectorAdd(combinedOut, h, statesLen);
    memcpy(result_h, combinedOut, statesLen * sizeof(float));
}


void FastRNNInference(const struct FastRNNParams *fastrnnParams,
        const float x[], float* result_h){
    for(int i = 0; i < fastrnnParams->statesLen; i++){
        result_h[i] = 0;
    }
    for (int t = 0; t < fastrnnParams->timeSteps; t++){
        FastRNNStep(fastrnnParams, (float*)&(x[t * fastrnnParams->featLen]),
                result_h, result_h);
    }
}


