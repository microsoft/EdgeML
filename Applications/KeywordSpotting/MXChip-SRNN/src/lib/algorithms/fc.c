#include "fc.h"

void FCInference(const struct FCParams* fcParams, const float x[],
	float* result, unsigned nonLinearity){
	matrixVectorMul(fcParams->W, fcParams->outputDim,
		fcParams->inputDim, x, result);
	vectorVectorAdd(result, fcParams->B, fcParams->outputDim);
}

