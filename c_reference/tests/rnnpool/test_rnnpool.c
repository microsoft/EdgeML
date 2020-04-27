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
    .mean = mean1,
    .stdDev = stdDev1,
    .W = W1,
    .U = U1,
    .Bg = Bg1,
    .Bh = Bh1,
    .zeta = zeta1,
    .nu = nu1
  };

  FastGRNN_Params rnn2_params = {
    .mean = mean2,
    .stdDev = stdDev2,
    .W = W2,
    .U = U2,
    .Bg = Bg2,
    .Bh = Bh2,
    .zeta = zeta2,
    .nu = nu2
  };

  float preComp1[HIDDEN_DIMS1] = { 0.0f };
  float normFeatures1[INPUT_DIMS] = { 0.0f };
  FastGRNN_Buffers rnn1_buffers = {
    .preComp = preComp1,
    .normFeatures = normFeatures1
  };

  float preComp2[HIDDEN_DIMS2] = { 0.0f };
  float normFeatures2[HIDDEN_DIMS1] = { 0.0f };
  FastGRNN_Buffers rnn2_buffers = {
    .preComp = preComp2,
    .normFeatures = normFeatures2
  };

  float output_test[4 * HIDDEN_DIMS2] = { 0.0f };
  float buffer[HIDDEN_DIMS1 * PATCH_DIM];

  rnnpool_block(input, INPUT_DIMS, PATCH_DIM, PATCH_DIM,
    fastgrnn, HIDDEN_DIMS1, (const void*)(&rnn1_params), (void*)(&rnn1_buffers),
    fastgrnn, HIDDEN_DIMS2, (const void*)(&rnn2_params), (void*)(&rnn2_buffers),
    output_test, buffer);

  float error = 0.0f;
  for (unsigned d = 0; d < 4 * HIDDEN_DIMS2; ++d)
    error += (output[d] - output_test[d]) * (output[d] - output_test[d]);
  printf("Error: %f\n", error);
}
