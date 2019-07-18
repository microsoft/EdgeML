#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include "../utils/helpermath.h"

struct LSTMParams {
	float* B;
	unsigned timeSteps;
	unsigned featLen;
	unsigned statesLen;
	float forgetBias;
	float *W; //  [W_i, W_c, W_f, W_o]
};

// lstmParams: Pointer to an instance of LSTMParams.
// x: Input data. Should be of shape [numtime_steps, num_feats] flattened to
// 1-D. That is, the ith time step will be the VECTOR x[i * num_feats]
// result_c_h_o: The cell-state (c), hidden-state(h) and output (o) will be
// stored in this vector.
void LSTMInference(const struct LSTMParams *lstmParams, const float x[],
		float *result_c_h_o);
void LSTMStep(const struct LSTMParams *lstmParams, const float *x,
		const float *input_c_h, float *result_c_h_o);
void combineXH(const struct LSTMParams *lstmParams, const float *x,
		const float *h, float *dst);

#ifdef __cplusplus
}
#endif