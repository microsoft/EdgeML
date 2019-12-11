// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <cstring>
#include <cmath>

#include "datatypes.h"
#include "predictors.h"
#include "protonn_float_model.h"

#define PROFILE 1

#if  PROFILE
#include "profile.h"
#endif

using namespace std;
using namespace protonn_float;

int protonnFloat(float *X) {
	MYINT ite_idx, ite_val, index;

	float WX[d];
	memset(WX, 0, sizeof(float) * d);

	ite_idx = 0;
	ite_val = 0;
	// Dimensionality reduction
	for (MYINT i = 0; i < D; i++) {
#if  PROFILE
		updateRange(X[i]);
#endif
		float input = X[i];

#if P_SPARSE_W
		index = Widx[ite_idx];
		while (index != 0) {
#if  PROFILE
			updateRange(WX[index - 1]);
			updateRange(Wval[ite_val]);
			updateRange(input);
			updateRange(Wval[ite_val] * input);
			updateRange(WX[index - 1] + Wval[ite_val] * input);
#endif
			WX[index - 1] += Wval[ite_val] * input;
			ite_idx++;
			ite_val++;
			index = Widx[ite_idx];
		}
		ite_idx++;
#else
		for (MYINT j = 0; j < d; j++) {
#if  PROFILE
			updateRange(WX[j]);
			updateRange(W[j][i]);
			updateRange(input);
			updateRange(W[j][i] * input);
			updateRange(WX[j] + W[j][i] * input);
#endif
			WX[j] += W[j][i] * input;
		}
#endif
	}


#if P_NORM == 0
#elif P_NORM == 1
	for (MYINT i = 0; i < d; i++) {
#if  PROFILE
		updateRange(norm[i]);
		updateRange(-norm[i]);
#endif
		WX[i] -= norm[i];
	}
#endif

	float score[c];
	memset(score, 0, sizeof(float) * c);

	for (MYINT i = 0; i < p; i++) {

		// Norm of WX - B
		float v = 0;
		for (MYINT j = 0; j < d; j++) {
#if  PROFILE
			updateRange(WX[j]);
			updateRange(B[j][i]);
			updateRange(WX[j] - B[j][i]);
#endif
			float t = WX[j] - B[j][i];
#if  PROFILE
			updateRange(v);
			updateRange(t);
			updateRange(t);
			updateRange(t * t);
			updateRange(v + t * t);
#endif
			v += t * t;
		}

		// Prediction distribution
#if  PROFILE
		updateRange(g2);
		updateRange(v);
		updateRange(-g2);
		updateRange(-g2 * v);
		updateRange(exp(-g2 * v));

		updateRangeOfExp(g2 * v);
#endif
		float e = exp(-g2 * v);

		for (MYINT j = 0; j < c; j++) {
#if  PROFILE
			updateRange(score[j]);
			updateRange(Z[j][i]);
			updateRange(e);
			updateRange(Z[j][i] * e);
			updateRange(score[j] + Z[j][i] * e);
#endif
			score[j] += Z[j][i] * e;
		}
	}

	// Argmax of score
	float max = score[0];
	MYINT classID = 0;
	for (MYINT i = 1; i < c; i++) {
		if (score[i] > max) {
			max = score[i];
			classID = i;
		}
	}

	return classID;
}
