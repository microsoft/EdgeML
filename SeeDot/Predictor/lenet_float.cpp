// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>

#include "datatypes.h"
#include "predictors.h"
#include "lenet_float\\testing\\model.h"

using namespace std;
using namespace lenet_float;

void convolution(float *input, const float *weight, const float *bias, float *output, MYUINT H, MYUINT W, MYUINT CI, MYUINT HF, MYUINT WF, MYUINT CO) {
	MYUINT padH = (HF - 1) / 2;
	MYUINT padW = (WF - 1) / 2;

	for (MYUINT h = 0; (h < H); h++) {
		for (MYUINT w = 0; (w < W); w++) {
			for (MYUINT co = 0; (co < CO); co++) {

				float acc = 0;
				for (MYUINT hf = 0; (hf < HF); hf++) {
					for (MYUINT wf = 0; (wf < WF); wf++) {
						for (MYUINT ci = 0; (ci < CI); ci++) {
							float in = (((((h + hf) < padH) || ((h + hf) >= (H + padH))) || (((w + wf) < padW) || ((w + wf) >= (W + padW)))) ? 0 : input[((h + hf) - padH) * W * CI + ((w + wf) - padW) * CI + ci]);
							float we = weight[hf * WF * CI * CO + wf * CI * CO + ci * CO + co];
							acc += (in * we);
						}
					}
				}

				float res = acc + bias[co];
				res = (res > 0) ? res : 0;

				output[h * W * CO + w * CO + co] = res;
			}
		}
	}

	return;
}

void maxpool(float *input, float *output, MYUINT H, MYUINT W, MYUINT C, MYUINT stride) {
	MYUINT HO = H / stride;
	MYUINT WO = W / stride;

	for (MYUINT ho = 0; (ho < HO); ho++) {
		for (MYUINT wo = 0; (wo < WO); wo++) {
			for (MYUINT c = 0; (c < C); c++) {

				float max = input[(stride * ho) * W * C + (stride * wo) * C + c];
				for (MYUINT hs = 0; (hs < stride); hs++) {
					for (MYUINT ws = 0; (ws < stride); ws++) {
						float in = input[((stride * ho) + hs) * W * C + ((stride * wo) + ws) * C + c];
						max = ((in > max) ? in : max);
					}
				}
				output[ho * WO * C + wo * C + c] = max;
			}
		}
	}

	return;
}

void matMul(float *x, const float *y, float *z, const float *bias, MYUINT K, MYUINT J, bool relu) {

	for (MYUINT j = 0; j < J; j++) {
		float acc = 0;
		for (MYUINT k = 0; k < K; k++)
			acc += (x[k] * y[k * J + j]);

		float res = acc + bias[j];

		if (relu)
			res = (res > 0) ? res : 0;

		z[j] = res;
	}

	return;
}

int lenetFloat(float *X) {

	const MYUINT c1InputDim = 32;
	const MYUINT c2InputDim = c1InputDim / 2;
	const MYUINT f1InputDim = c2InputDim / 2;

	const MYUINT kernelSize = 5;

	// channels
	const MYUINT inputChannels = 3;
	const MYUINT c1Channels = 4;
	const MYUINT c2Channels = 12;

	// FC1
	const MYUINT fc1_m = f1InputDim * f1InputDim * c2Channels, fc1_n = 64;
	const MYUINT fc2_m = fc1_n, fc2_n = 10;
	//const MYUINT fc3_m = fc2_n, fc3_n = 10;



	float input[10000];
	float output[10000];

	//float maxpoolOutput1[c2InputDim][c2InputDim][c1Channels];

	//float conv2Output[c2InputDim][c2InputDim][c2Channels];
	//float maxpoolOutput2[f1InputDim][f1InputDim][c2Channels];

	//float fcInput[fc1_m];
	//float fcOutput1[fc2_m];
	//float fcOutput2[fc2_n];
	//float fcOutput3[fc3_n];

	MYUINT tmp2, tmp3, tmp4;

	tmp2 = 0;
	tmp3 = 0;
	tmp4 = 0;
	for (MYUINT i = 0; (i < 3072); i++) {
		input[tmp2 * 32 * 3 + tmp3 * 3 + tmp4] = X[i];
		tmp4 = (tmp4 + 1);
		if ((tmp4 == 3)) {
			tmp4 = 0;
			tmp3 = (tmp3 + 1);
			if ((tmp3 == 32)) {
				tmp3 = 0;
				tmp2 = (tmp2 + 1);
			}
		}
	}

	convolution(input, &Wc1[0][0][0][0], &Bc1[0], output, c1InputDim, c1InputDim, inputChannels, kernelSize, kernelSize, c1Channels);
	maxpool(output, input, c1InputDim, c1InputDim, c1Channels, 2);


	convolution(input, &Wc2[0][0][0][0], &Bc2[0], output, c2InputDim, c2InputDim, c1Channels, kernelSize, kernelSize, c2Channels);
	maxpool(output, input, c2InputDim, c2InputDim, c2Channels, 2);


	// reshape(Hc2P, 1, 1024)
	MYUINT count = 0;
	for (MYUINT i = 0; (i < c2Channels); i++) {
		for (MYUINT j = 0; (j < f1InputDim); j++) {
			for (MYUINT k = 0; (k < f1InputDim); k++) {
				//fcInput[tmp32][tmp33] = maxpoolOutput2[i49][i50][i51];
				output[count] = input[j * c2Channels * f1InputDim + k * c2Channels + i];
				count = (count + 1);
			}
		}
	}

	matMul(output, &Wf1[0][0], input, &Bf1[0], fc1_m, fc1_n, true);
	matMul(input, &Wf2[0][0], output, &Bf2[0], fc2_m, fc2_n, false);
	//matMul(output, &Wf3[0][0], input, &Bf3[0], fc3_m, fc3_n, false);


	MYUINT classID = 0;
	float max = output[0];
	for (MYUINT i = 1; (i < fc2_n); i++) {
		if ((max < output[i])) {
			classID = i;
			max = output[i];
		}
	}

	return classID;
}
