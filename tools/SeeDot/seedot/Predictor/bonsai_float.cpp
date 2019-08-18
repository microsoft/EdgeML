// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <cstring>

#include "datatypes.h"
#include "predictors.h"
#include "bonsai_float_model.h"

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

#if B_SPARSE_Z
		index = Zidx[ite_idx];
		while (index != 0) {
			ZX[index - 1] += Zval[ite_val] * input;
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

	for (MYINT i = 0; i < d; i++)
		ZX[i] -= mean[i];

	MYINT currNode = 0;
	float WZX[c], VZX[c], score[c];

	memset(score, 0, sizeof(float) * c);

	while (currNode < internalNodes) {

		memset(WZX, 0, sizeof(float) * c);
		memset(VZX, 0, sizeof(float) * c);

		// Accumulating score at each node
		for (MYINT i = 0; i < d; i++) {
			for (MYINT j = currNode * c; j < (currNode + 1) * c; j++) {
				WZX[j % c] += W[j][i] * ZX[i];
				VZX[j % c] += V[j][i] * ZX[i];
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
			val += T[currNode][i] * ZX[i];

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
			WZX[j % c] += W[j][i] * ZX[i];
			VZX[j % c] += V[j][i] * ZX[i];
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
