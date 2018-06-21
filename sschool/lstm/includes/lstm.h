// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
#pragma once

#include <math.h>
#include "helpermath.h"
#include "paramstruct.h"

extern struct  LSTMParams lstmParams;
extern void initLSTMParams();

unsigned initializeLSTM(unsigned timeSteps,
    unsigned featLen, unsigned statesLen,
    float forgetBias);
// x is of size [numtime_steps, num_feats]
void LSTMInference(float x[][lstmParams.featLen], float *result_c_h_o);
void LSTMStep(float *x, float *input_c_h, float *result_c_h_o);
unsigned runLSTMTests();