// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <Arduino.h>

#include "config.h"
#include "predict.h"
#include "model.h"

using namespace model;

void convolution(float *input, const int8_t *weight, const int8_t *bias, float *output, MYUINT H, MYUINT W, MYUINT CI, MYUINT HF, MYUINT WF, MYUINT CO, float weight_min, float weight_max, float bias_min, float bias_max) {
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
							
							// HERE
							int8_t we_int = (int8_t) pgm_read_byte_near(&weight[hf * WF * CI * CO + wf * CI * CO + ci * CO + co]);
							float we = weight_min + ((weight_max - weight_min) * (we_int)) / 128;
							
							acc += (in * we);
						}
					}
				}

				// HERE
				float b = bias_min + ((bias_max - bias_min) * ((int8_t) pgm_read_byte_near(&bias[co]))) / 128;

				float res = acc + b;
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

void matMul(float *x, const int8_t *y, float *z, const int8_t *bias, MYUINT K, MYUINT J, bool relu, float y_min, float y_max, float bias_min, float bias_max) {

	for (MYUINT j = 0; j < J; j++) {
		float acc = 0;
		for (MYUINT k = 0; k < K; k++) {
			// HERE
			float yy = y_min + ((y_max - y_min) * ((int8_t) pgm_read_byte_near(&y[k * J + j]))) / 128;

			acc += (x[k] * yy);
		}

		// HERE
		float b = bias_min + ((bias_max - bias_min) * ((int8_t) pgm_read_byte_near(&bias[j]))) / 128;
		
		float res = acc + b;

		if (relu)
			res = (res > 0) ? res : 0;

		z[j] = res;
	}

	return;
}

int predict() {

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



	float input[3072];
	float output[4096];

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
		float x = getFloatFeature(i);
		input[tmp2 * 32 * 3 + tmp3 * 3 + tmp4] = x;
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

	convolution(input, &Wc1[0][0][0][0], &Bc1[0], output, c1InputDim, c1InputDim, inputChannels, kernelSize, kernelSize, c1Channels, Wc1_min, Wc1_max, Bc1_min, Bc1_max);
	maxpool(output, input, c1InputDim, c1InputDim, c1Channels, 2);


	convolution(input, &Wc2[0][0][0][0], &Bc2[0], output, c2InputDim, c2InputDim, c1Channels, kernelSize, kernelSize, c2Channels, Wc2_min, Wc2_max, Bc2_min, Bc2_max);
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

	matMul(output, &Wf1[0][0], input, &Bf1[0], fc1_m, fc1_n, true, Wf1_min, Wf1_max, Bf1_min, Bf1_max);
	matMul(input, &Wf2[0][0], output, &Bf2[0], fc2_m, fc2_n, false, Wf2_min, Wf2_max, Bf2_min, Bf2_max);
	//matMul(&fcOutput2[0][0], &Wf3[0][0], &fcOutput3[0][0], &Bf3[0], fc3_m, fc3_n, false);


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
