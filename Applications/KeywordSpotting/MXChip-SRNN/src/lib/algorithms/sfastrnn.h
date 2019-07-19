/*
 * Shallow FastRNN
 * ---------------
 * Currently only a 2 layer network is supported through
 * in the current format, implementing 3 layer is straight
 * forward.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "../utils/helpermath.h"
#include "../algorithms/fastrnn.h"
#include "../utils/circularq.h"


// 2 Layer S-FastRNN
struct SFastRNNParams2 {
    struct FastRNNParams *frnn0;
    struct FastRNNParams *frnn1;
    unsigned timeSteps0;
    unsigned timeSteps1;
    // These parameters are set internally
    unsigned __featLen0;
    unsigned __featLen1;
    // Container for hidden states. Should be of shape
    // [featLen1 * timeSteps1]
    FIFOCircularQ h0q;
    // We need some buffer space to run. I don't want to
    // define these on the stack as that has often caused
    // problems with mbed-os. Don't worry about what these
    // are. h0_buffer has to be statesLen0 long and
    // inp1_buffer has to be statesLen0 * timeSteps1 long.
    float *h0_buffer;
    float *inp1_buffer;
};

// 2 Layer S-FastRNN initialization
// sParams2: An instance of SFastRNNInference2 to be initialized.
// p0: The FastRNNParams for level 0
// p1: The FastRNNParams for level 1
// h0container: A float array of size [numTimeSteps1 * hiddenStates0]
//              To store the intermediate hidden states.
// h0_buffer  :
// inp1_buffer: We need some buffer space to run. I don't want to
//       define these on the stack as that has often caused
//       problems with mbed-os. Don't worry about what these
//       are. h0_buffer has to be statesLen0 long and
//       inp1_buffer has to be statesLen0 * timeSteps1 long.
unsigned initSFastRNN2(struct SFastRNNParams2 *sParams,
        struct FastRNNParams *p0, struct FastRNNParams *p1,
        float *h0container, float *h0_buffer, float *h1_buffer);

// sParams : An instance of SFastRNNInference2 with the parameters
//      initialized through initSFastRNN2.
// x : Input vector of shape [featLen1 * timeSteps1]
// result_h : result hidden state.
void SFastRNNInference2(struct SFastRNNParams2 *sParams, const float *x,
        float *result_h);

#ifdef __cplusplus
}
#endif

