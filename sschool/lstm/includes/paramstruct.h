// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
#pragma once

struct LSTMParams {
	float* B;
	unsigned timeSteps;
	unsigned featLen;
	unsigned statesLen;
	float forgetBias;
	float *W; //  [W_i, W_c, W_f, W_o]
};

struct ProtoNNParams {
	float gamma;
	unsigned featDim;
	unsigned ldDim;
	float *ldProjectionMatrix;
	float *prototypeMatrix;
	unsigned numPrototypes;
	float *prototypeLabelMatrix;
	unsigned numLabels;
};