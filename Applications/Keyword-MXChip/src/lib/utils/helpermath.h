#pragma once

#include <stdlib.h>
#include <arm_math.h>

#ifdef __cplusplus
	extern "C" {
#endif 

/* Left multiply mxn matrix (max) with n-d vector vec
 * input matrix need to be flattened in row major form.
*/
void matrixVectorMul(const float *mat, const unsigned m,
	const unsigned n, const float *vec, float *dst);
void scalarVectorMul(float32_t *vec, const unsigned length,
	const float32_t scalar);
void vectorVectorAdd(float *dstVec, const float *srcVec,
	const unsigned length);
void vectorVectorHadamard(float *dst, const float *src,
	const unsigned length);
float gaussian(const float *x, const float *y,
	unsigned length, float gamma);
void vsigmoid(float *vec, unsigned length);
void vtanh(float *vec, unsigned length);
void softmax(float *input, size_t input_len);
int argmax(float *, int len);

#ifdef __cplusplus
}
#endif