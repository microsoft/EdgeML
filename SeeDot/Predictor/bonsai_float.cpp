// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>

#include "datatypes.h"
#include "predictors.h"
#include "bonsai_float_model.h"

#define TANH 0
#define PROFILE 1

#if  PROFILE
#include "profile.h"
#endif

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
#if  PROFILE
		updateRange(X[i]);
#endif
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
#if  PROFILE
			updateRange(ZX[index - 1]);
			updateRange(Zval[ite_val]);
			updateRange(input);
			updateRange(Zval[ite_val] * input);
			updateRange(ZX[index - 1] + Zval[ite_val] * input);
#endif
			ZX[index - 1] += Zval[ite_val] * input;
			ite_idx++;
			ite_val++;
			index = Zidx[ite_idx];
		}
		ite_idx++;
#else
		for (MYINT j = 0; j < d; j++) {
#if  PROFILE
			updateRange(ZX[j]);
			updateRange(Z[j][i]);
			updateRange(input);
			updateRange(Z[j][i] * input);
			updateRange(ZX[j] + Z[j][i] * input);
#endif
			ZX[j] += Z[j][i] * input;
		}
#endif
	}

	for (MYINT i = 0; i < d; i++) {
#if  PROFILE
		updateRange(mean[i]);
		updateRange(-mean[i]);
#endif
		ZX[i] -= mean[i];
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
#if  PROFILE
				updateRange(WZX[j % c]);
				updateRange(W[j][i]);
				updateRange(ZX[i]);
				updateRange(W[j][i] * ZX[i]);
				updateRange(WZX[j % c] + W[j][i] * ZX[i]);

				updateRange(VZX[j % c]);
				updateRange(V[j][i]);
				updateRange(ZX[i]);
				updateRange(V[j][i] * ZX[i]);
				updateRange(VZX[j % c] + V[j][i] * ZX[i]);
#endif
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

#if  PROFILE
			updateRange(score[i]);
			updateRange(WZX[i]);
			updateRange(t);
			updateRange(WZX[i] * t);
			updateRange(score[i] + WZX[i] * t);
#endif

#if TANH
			score[i] += WZX[i] * tanh(VZX[i]);
#else
			score[i] += WZX[i] * t;
#endif
		}

		// Computing theta value for branching into a child node
		float val = 0;
		for (MYINT i = 0; i < d; i++) {
#if  PROFILE
			updateRange(val);
			updateRange(T[currNode][i]);
			updateRange(ZX[i]);
			updateRange(T[currNode][i] * ZX[i]);
			updateRange(val + T[currNode][i] * ZX[i]);
#endif
			val += T[currNode][i] * ZX[i];
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
#if  PROFILE
			updateRange(WZX[j % c]);
			updateRange(W[j][i]);
			updateRange(ZX[i]);
			updateRange(W[j][i] * ZX[i]);
			updateRange(WZX[j % c] + W[j][i] * ZX[i]);

			updateRange(VZX[j % c]);
			updateRange(V[j][i]);
			updateRange(ZX[i]);
			updateRange(V[j][i] * ZX[i]);
			updateRange(VZX[j % c] + V[j][i] * ZX[i]);
#endif
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

#if  PROFILE
		updateRange(score[i]);
		updateRange(WZX[i]);
		updateRange(t);
		updateRange(WZX[i] * t);
		updateRange(score[i] + WZX[i] * t);
#endif

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
