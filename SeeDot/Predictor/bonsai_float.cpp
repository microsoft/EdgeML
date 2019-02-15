// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>

#include "datatypes.h"
#include "predictors.h"
#include "model.h"

#define TANH 0

using namespace std;
using namespace bonsai_float;

int bonsaiFloat(float *X) {
	MYINT ite_idx, ite_val, index;

	float ZX[d];
	memset(ZX, 0, sizeof(float) * d);

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

#if B_SPARSE_Z
		index = Zidx[ite_idx];
		while (index != 0) {
			// HERE
			float z = Z_min + ((Z_max - Z_min) * (Zval[ite_val])) / 128;

			ZX[index - 1] += z * input;
			ite_idx++;
			ite_val++;
			index = Zidx[ite_idx];
		}
		ite_idx++;
#else
		for (MYINT j = 0; j < d; j++) {
			ZX[j] += Z[j][i] * input;
		}
#endif
	}

	for (MYINT i = 0; i < d; i++) {
		// HERE
		float m = mean_min + ((mean_max - mean_min) * (mean[i])) / 128;

		ZX[i] -= m;
	}

	MYINT currNode = 0;
	float WZX[c], VZX[c], score[c];

	memset(score, 0, sizeof(float) * c);

	while (currNode < internalNodes) {

		memset(WZX, 0, sizeof(float) * c);
		memset(VZX, 0, sizeof(float) * c);

		// Accumulating score at each node
		for (MYINT i = 0; i < d; i++) {
			for (MYINT j = currNode * c; j < (currNode + 1) * c; j++) {
				// HERE
				float w = W_min + ((W_max - W_min) * (W[j][i])) / 128;
				float v = V_min + ((V_max - V_min) * (V[j][i])) / 128;

				WZX[j % c] += w * ZX[i];
				VZX[j % c] += v * ZX[i];
			}
		}

		for (MYINT i = 0; i < c; i++) {
			float t;
			if (VZX[i] > tanh_limit)
				t = tanh_limit;
			else if (VZX[i] < -tanh_limit)
				t = -tanh_limit;
			else
				t = VZX[i];

#if TANH
			score[i] += WZX[i] * tanh(VZX[i]);
#else
			score[i] += WZX[i] * t;
#endif
		}

		// Computing theta value for branching into a child node
		float val = 0;
		for (MYINT i = 0; i < d; i++) {
			// HERE
			float t = T_min + ((T_max - T_min) * (T[currNode][i])) / 128;

			val += t * ZX[i];
		}

		if (val > 0)
			currNode = 2 * currNode + 1;
		else
			currNode = 2 * currNode + 2;
	}

	memset(WZX, 0, sizeof(float) * c);
	memset(VZX, 0, sizeof(float) * c);

	// Accumulating score for the last node
	for (MYINT i = 0; i < d; i++) {
		for (MYINT j = currNode * c; j < (currNode + 1) * c; j++) {
			// HERE
			float w = W_min + ((W_max - W_min) * (W[j][i])) / 128;
			float v = V_min + ((V_max - V_min) * (V[j][i])) / 128;

			WZX[j % c] += w * ZX[i];
			VZX[j % c] += v * ZX[i];
		}
	}

	for (MYINT i = 0; i < c; i++) {
		float t;
		if (VZX[i] > tanh_limit)
			t = tanh_limit;
		else if (VZX[i] < -tanh_limit)
			t = -tanh_limit;
		else
			t = VZX[i];

#if TANH
		score[i] += WZX[i] * tanh(VZX[i]);
#else
		score[i] += WZX[i] * t;
#endif
	}

	MYINT classID;

	// Finding the class ID
	// If binary classification, the sign of the score is used
	// If multiclass classification, argmax of score is used
	if (c <= 2) {
		if (score[0] > 0)
			classID = 1;
		else
			classID = 0;
	}
	else {
		float max = score[0];
		MYINT maxI = 0;
		for (MYINT i = 1; i < c; i++) {
			if (score[i] > max) {
				max = score[i];
				maxI = i;
			}
		}
		classID = maxI;
	}

	return classID;
}
