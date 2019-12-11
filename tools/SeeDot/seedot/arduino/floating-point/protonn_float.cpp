// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <Arduino.h>

#include "config.h"
#include "predict.h"
#include "model.h"

using namespace model;

int predict() {
	MYINT ite_idx, ite_val, index;

	float WX[d];
	memset(WX, 0, sizeof(float) * d);

	ite_idx = 0;
	ite_val = 0;
	// Dimensionality reduction
	for (MYINT i = 0; i < D; i++) {
		float input = getFloatFeature(i);

#if P_SPARSE_W
		index = ((MYINT) pgm_read_word_near(&Widx[ite_idx]));
		while (index != 0) {
			WX[index - 1] += ((float) pgm_read_float_near(&Wval[ite_val])) * input;
			ite_idx++;
			ite_val++;
			index = ((MYINT) pgm_read_word_near(&Widx[ite_idx]));
		}
		ite_idx++;
#else
		for (MYINT j = 0; j < d; j++)
			WX[j] += ((float) pgm_read_float_near(&W[j][i])) * input;
#endif
	}

#if P_NORM == 0
#elif P_NORM == 1
	for (MYINT i = 0; i < d; i++)
		WX[i] -= ((float) pgm_read_float_near(&norm[i]));
#endif

	float score[c];
	memset(score, 0, sizeof(float) * c);

	for (MYINT i = 0; i < p; i++) {

		// Norm of WX - B
		float v = 0;
		for (MYINT j = 0; j < d; j++) {
			float t = WX[j] - ((float) pgm_read_float_near(&B[j][i]));
			v += t * t;
		}

		// Prediction distribution
		float e = exp(-g2 * v);

		for (MYINT j = 0; j < c; j++)
			score[j] += ((float) pgm_read_float_near(&Z[j][i])) * e;
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
