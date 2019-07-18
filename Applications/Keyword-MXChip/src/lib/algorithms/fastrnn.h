#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include "../utils/helpermath.h"
#include <string.h>

struct FastRNNParams {
    // h~ = W1.h + W2.x + b
    // The projection matrix W1 and W2 concatenated along axis=1, [W1, W2]
    // in row major order. Will be of dimension [n_hid, (n_hid + n_inp)] in
    // numpy.
    float* W;
    // The bias vector. Of dimension [n_hidden]
    float *b;
    // Alpha and beta for FastRNN (sigmoided version)
    float alpha; float beta;
    unsigned timeSteps;
    unsigned featLen;
    unsigned statesLen;
};

// FastRNNParams: Pointer to an instance of FastRNNParams
// x: Input data. Should be of shape [numtime_steps, num_feats] flattened to
//    1-D. That is, the ith time step will be the VECTOR x[i * num_feats]
// result_h: hidden-state(h) stored in this vector.
void FastRNNInference(const struct FastRNNParams *fastrnnParams, const float x[],
        float *result_h);
void FastRNNStep(const struct FastRNNParams *fastrnnParams, const float *x,
        const float *input_h, float *result_h);
void combineHX(const struct FastRNNParams *fastrnnParams, const float *h,
        const float *x, float *dst);

#ifdef __cplusplus
}
#endif
