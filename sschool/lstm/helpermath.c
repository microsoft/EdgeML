// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "helpermath.h"
/* Left multiply mxn matrix (max) with n-d vector vec */
void matrixVectorMul(float *mat, unsigned m, unsigned n,
	float *vec, float *dst){
	for (int i = 0; i < m; i++){
		float dotProd = 0.0f;
		for(int j = 0; j < n; j++){
			dotProd += mat[i*n + j] * vec[j];
		}
		dst[i] = dotProd;
	}
}

void scalarVectorMul(float *vec, unsigned length,
	float scalar) {
	for(unsigned i = 0; i < length; i++) {
		vec[i] = vec[i] * scalar;
	}
}

void vectorVectorAdd(float *dstVec, float *srcVec,
	unsigned length){
	for(unsigned i = 0; i < length; i++)
		dstVec[i] += srcVec[i];
}

void vectorVectorHadamard(float *dst, float *src, unsigned length){
	for(unsigned i = 0; i < length; i++){
		dst[i] = dst[i] * src[i];
	}
}

float gaussian(const float *x, const float *y,
	unsigned length, float gamma) {
	float sumSq = 0.0;
	for(unsigned i = 0; i < length; i++){
		sumSq += (x[i] - y[i])*(x[i] - y[i]);
	}
	sumSq = -1*gamma*gamma*sumSq;
	sumSq = exp(sumSq);
	return sumSq;
}

void vsigmoid(float *vec, unsigned length){
	// Refer to:
	// https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
	for (int i=0; i < length; i++){
		if(vec[i] >= 0){
			float z = exp(-1 * vec[i]);
			vec[i] = 1.0 / (1.0 + z);
		} else {
			float z = exp(vec[i]);
			vec[i] = z / (1.0 + z);
		}	
	}
}


void vtanh(float *vec, unsigned length){
	for (int i=0; i < length; i++){
		vec[i] = tanh(vec[i]);
	}	
}