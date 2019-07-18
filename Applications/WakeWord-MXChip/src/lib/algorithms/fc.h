#pragma once

#ifdef __cplusplus
extern "C" {
#endif 

#include <math.h>
#include "../utils/helpermath.h"

/* An instance of FCParams needs to be defined else where. This will hold the
 * model matrices. You pass a pointer to this FCParams instance here to perform
 * your computations.
 */

struct FCParams{
	// let x in a vector of size n = input dim
	// And let the hidden dim (or output dim ) be m
	// Then FC would do softmax(Wx) were W is m x n
	float *W;
	float *B;
	unsigned inputDim;
	unsigned outputDim;
};

//
// Set nonLinearity can be used to perform non-linearity on outputs. This
// feature is not implemented as of now and the raw outputs are returned.
// 
// params: The FCParams struct instance containing the parameters for the
//		current layer.
//	x: Input data 
//	result: Float array to hold the FC result.
//	nonLinearity: Choice of non-linearity to use. Currently not implemented.
void FCInference(const struct FCParams* params, const float x[], float* result, 
		unsigned nonLinearity);

#ifdef __cplusplus
}
#endif
