// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <Arduino.h>

#include "config.h"
#include "predict.h"
#include "model.h"

#define TANH 0

using namespace model;

int predict() {
	MYINT ite_idx, ite_val, index;

	float ZX[d];
	memset(ZX, 0, sizeof(float) * d);

	ite_idx = 0;
	ite_val = 0;
	// Dimensionality reduction
	for (MYINT i = 0; i < D; i++) {
		float input = getFloatFeature(i);

#if B_SPARSE_Z
		index = ((MYINT) pgm_read_word_near(&Zidx[ite_idx]));
		while (index != 0) {
			ZX[index - 1] += ((float) pgm_read_float_near(&Zval[ite_val])) * input;
			ite_idx++;
			ite_val++;
			index = ((MYINT) pgm_read_word_near(&Zidx[ite_idx]));
		}
		ite_idx++;
#else
		for (MYINT j = 0; j < d; j++) {
			ZX[j] += ((float) pgm_read_float_near(&Z[j][i])) * input;
		}
#endif
	}

	for (MYINT i = 0; i < d; i++)
		ZX[i] -= ((float) pgm_read_float_near(&mean[i]));


	MYINT currNode = 0;
	float WZX[c], VZX[c], score[c];

	memset(score, 0, sizeof(float) * c);

	while (currNode < internalNodes) {

		memset(WZX, 0, sizeof(float) * c);
		memset(VZX, 0, sizeof(float) * c);

		// Accumulating score at each node
		for (MYINT i = 0; i < d; i++) {
			for (MYINT j = currNode * c; j < (currNode + 1) * c; j++) {
				WZX[j % c] += ((float) pgm_read_float_near(&W[j][i])) * ZX[i];
				VZX[j % c] += ((float) pgm_read_float_near(&V[j][i])) * ZX[i];
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
		for (MYINT i = 0; i < d; i++)
			val += ((float) pgm_read_float_near(&T[currNode][i])) * ZX[i];

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
			WZX[j % c] += ((float) pgm_read_float_near(&W[j][i])) * ZX[i];
			VZX[j % c] += ((float) pgm_read_float_near(&V[j][i])) * ZX[i];
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
