// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
#pragma once

#include <math.h>

/* Left multiply mxn matrix (max) with n-d vector vec
 * input matrix need to be flattened in row major form.
*/
void matrixVectorMul(float *mat, unsigned m, unsigned n,
	float *vec, float *dst);
void scalarVectorMul(float *vec, unsigned length, float scalar);
void vectorVectorAdd(float *dstVec, float *srcVec, unsigned length);
void vectorVectorHadamard(float *dst, float *src, unsigned length);
float gaussian(const float *x, const float *y,
	unsigned length, float gamma);
void vsigmoid(float *vec, unsigned length);
void vtanh(float *vec, unsigned length);