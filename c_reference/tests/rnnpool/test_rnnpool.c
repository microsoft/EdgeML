// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stdio.h>
#include <string.h>

#include "utils.h"
#include "fastgrnn.h"
#include "rnnpool.h"

#include "traces/trace_0_0_input.h"
#include "traces/trace_0_0_output.h"

#include "vww_model/rnn1.h"
#include "vww_model/rnn2.h"

int main() {
  FastGRNN_Params rnn1_params = {
    .mean = NULL,
    .stdDev = NULL,
    .W = W1,
    .U = U1,
    .Bg = Bg1,
    .Bh = Bh1,
    .sigmoid_zeta = sigmoid_zeta1,
    .sigmoid_nu = sigmoid_nu1
  };

  FastGRNN_Params rnn2_params = {
    .mean = NULL,
    .stdDev = NULL,
    .W = W2,
    .U = U2,
    .Bg = Bg2,
    .Bh = Bh2,
    .sigmoid_zeta = sigmoid_zeta2,
    .sigmoid_nu = sigmoid_nu2
  };

  float preComp1[HIDDEN_DIMS1];
  float normFeatures1[INPUT_DIMS];
  memset(preComp1, 0, sizeof(float) * HIDDEN_DIMS1);
  memset(normFeatures1, 0, sizeof(float) * INPUT_DIMS);
  FastGRNN_Buffers rnn1_buffers = {
    .preComp = preComp1,
    .normFeatures = normFeatures1
  };

  float preComp2[HIDDEN_DIMS2];
  float normFeatures2[HIDDEN_DIMS1];
  memset(preComp2, 0, sizeof(float) * HIDDEN_DIMS2);
  memset(normFeatures2, 0, sizeof(float) * HIDDEN_DIMS1);
  FastGRNN_Buffers rnn2_buffers = {
    .preComp = preComp2,
    .normFeatures = normFeatures2
  };

  float output_test[4 * HIDDEN_DIMS2];
  float buffer[HIDDEN_DIMS1 * PATCH_DIM];
  memset(output_test, 0, sizeof(float) * 4 * HIDDEN_DIMS2);
  memset(buffer, 0, sizeof(float) * HIDDEN_DIMS1 * PATCH_DIM);
  rnnpool_block(input, INPUT_DIMS, PATCH_DIM, PATCH_DIM,
    fastgrnn, HIDDEN_DIMS1, (const void*)(&rnn1_params), (void*)(&rnn1_buffers),
    fastgrnn, HIDDEN_DIMS2, (const void*)(&rnn2_params), (void*)(&rnn2_buffers),
    output_test, buffer);

  printf("Error: %f\n", l2squared(output, output_test, 4 * HIDDEN_DIMS2));
}
