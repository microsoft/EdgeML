/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * ProtoNN algorithm
 */

#ifndef __PROTONN__
#define __PROTONN__

#include <cstdint>
#include "config.h"
#include <cmath>
#include <float.h>
#include "data.h"

/**
 * ProtoNN predictor 
 * ProtoNN parameters are assumed to be in the data.h file.
 */
class ProtoNNF {
	int8_t errorCode;
	unsigned featDim, ldDim, numPrototypes, numLabels;
	float gamma;
private:
	int8_t getInitErrorCode();
	int8_t denseLDProjection(float* x, float* x_cap);
	float gaussian(const float *x, const float *y, unsigned length, float gamma);
	int8_t scalarVectorMul(float *vec, unsigned length, float scalar);
	int8_t vectorVectorAdd(float *dstVec, float *srcVec, unsigned length);
	int8_t getPrototype(unsigned i, float *prototype);
	int8_t getPrototypeLabel(unsigned i, float *prototypeLabel);
	float getProjectionComponent(unsigned i, unsigned j);
	float rho(float* labelScores, unsigned length);

public:
	ProtoNNF();
	ProtoNNF(unsigned d, unsigned d_cap, unsigned m, unsigned L, float gamma);
	float predict(float *x, unsigned length, int *scores);

	int8_t getErrorCode();
};
#endif // __PROTONN__