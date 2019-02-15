// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>

#include "datatypes.h"
#include "predictors.h"
#include "model.h"

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
		float input = X[i];
		/*
		// Read each feature
		while (!Serial.available())
		;

		Serial.readBytes(buff, 8);

		float input = atof(buff);
		*/
		//float input = pgm_read_float_near(&X[i]);

#if P_SPARSE_W
		index = Widx[ite_idx];
		while (index != 0) {
			// HERE
			float w = W_min + ((W_max - W_min) * (Wval[ite_val])) / 128;

			WX[index - 1] += w * input;
			ite_idx++;
			ite_val++;
			index = Widx[ite_idx];
		}
		ite_idx++;
#else
		for (MYINT j = 0; j < d; j++) {
			WX[j] += W[j][i] * input;
		}
#endif
	}


#if P_NORM == 0
#elif P_NORM == 1
	for (MYINT i = 0; i < d; i++) {
		// HERE
		float n = norm_min + ((norm_max - norm_min) * (norm[i])) / 128;
		
		WX[i] -= n;
	}
#endif

	float score[c];
	memset(score, 0, sizeof(float) * c);

	for (MYINT i = 0; i < p; i++) {

		// Norm of WX - B
		float v = 0;
		for (MYINT j = 0; j < d; j++) {
			// HERE
			float b = B_min + ((B_max - B_min) * (B[j][i])) / 128;

			float t = WX[j] - b;
			v += t * t;
		}

		// Prediction distribution
#if  PROFILE
		updateRangeOfExp(g2 * v);
#endif
		float e = exp(-g2 * v);

		for (MYINT j = 0; j < c; j++) {
			// HERE
			float z = Z_min + ((Z_max - Z_min) * (Z[j][i])) / 128;
			
			score[j] += z * e;
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
