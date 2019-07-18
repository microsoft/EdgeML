#include "lstm.h"

void combineXH(const struct LSTMParams *lstmParams, const float *x,
		const float *h, float *dst){
	// TODO: Use memcpy to make this faster
	memcpy(dst, x, lstmParams->featLen * sizeof(float));
	memcpy(&(dst[lstmParams->featLen]), h, lstmParams->statesLen * sizeof(float));
}

void LSTMStep(const struct LSTMParams *lstmParams, const float *x,
		const float *input_c_h, float *result_c_h_o){
	unsigned statesLen = lstmParams->statesLen;

	float c[statesLen];
	float h[statesLen];
	float o[statesLen];
	memcpy(c, &input_c_h[0*statesLen], statesLen * sizeof(float));
	memcpy(h, &input_c_h[1*statesLen], statesLen * sizeof(float));

	float combinedOut[4 * (lstmParams->statesLen)];
	float xh[lstmParams->statesLen + lstmParams->featLen];
	combineXH(lstmParams, x, h, xh);
	matrixVectorMul(lstmParams->W, 4*lstmParams->statesLen,
		lstmParams->statesLen + lstmParams->featLen,
		xh, combinedOut);
	vectorVectorAdd(combinedOut, lstmParams->B,
		4 * lstmParams->statesLen);
	// Apply non-linearity
	// i_t
	vsigmoid(&combinedOut[0*lstmParams->statesLen], lstmParams->statesLen);
	// c_cap_t
	vtanh(&combinedOut[1*lstmParams->statesLen], lstmParams->statesLen);
	// f_t (after adding forget bias)
	for(int i = 0; i < lstmParams->statesLen; i++)
		combinedOut[2*lstmParams->statesLen + i] += lstmParams->forgetBias;
	vsigmoid(&combinedOut[2*lstmParams->statesLen], lstmParams->statesLen);
	// o_t
	vsigmoid(&combinedOut[3*lstmParams->statesLen], lstmParams->statesLen);
	
	// update c
	for(int i = 0; i < lstmParams->statesLen; i++){
		//c_t = (f_t + forget_bias)*C_t-1 + i_t*c_cap_t
		c[i] = combinedOut[2*lstmParams->statesLen + i] * c[i];
		c[i] += combinedOut[0*lstmParams->statesLen +
			i]*combinedOut[1*lstmParams->statesLen + i];
		//o_t
		o[i] = combinedOut[3*lstmParams->statesLen + i];
		//h_t
		h[i] = o[i] * tanh(c[i]);
	}
	// returns c, h, o
	for(int i = 0; i < lstmParams->statesLen; i++){
		result_c_h_o[lstmParams->statesLen * 0 + i] = c[i];
		result_c_h_o[lstmParams->statesLen * 1 + i] = h[i];
		result_c_h_o[lstmParams->statesLen * 2 + i] = o[i];
	}
}


void LSTMInference(const struct LSTMParams *lstmParams, const float x[],
		float* result_c_h_o){
	for(int i = 0; i < 3 * lstmParams->statesLen; i++){
		result_c_h_o[i] = 0;
	}
	for (int t = 0; t < lstmParams->timeSteps; t++){
		LSTMStep(lstmParams, (float*)&(x[t * lstmParams->featLen]), result_c_h_o, result_c_h_o);
	}
}


