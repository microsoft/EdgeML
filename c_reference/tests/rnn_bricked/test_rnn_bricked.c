// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stdio.h>
#include <stdlib.h>
#include "rnn_bricked.h"
#include "utils.h"

#include "rnn_params.h"
#include "rnn_bricked_io.h"

int main() {

  BrickedFastGRNN_LR_Params bwd_RNN_params = {
    .W1     = B_W1,
    .W2     = B_W2,
    .wRank  = RNN_LOW_RANK,
    .U1     = B_U1,
    .U2     = B_U2,
    .uRank  = RNN_LOW_RANK,
    .Bg     = B_BIAS_GATE,
    .Bh     = B_BIAS_UPDATE,
    .sigmoid_zeta = sigmoid(B_ZETA),
    .sigmoid_nu   = sigmoid(B_NU),
    .block_size_u_from_lr = 100,
    .block_size_u_to_lr = 100,
    .block_size_w_from_lr = 100,
    .block_size_w_to_lr = 100,
  };

  BrickedFastGRNN_LR_Params fwd_RNN_params = {
    .W1     = F_W1,
    .W2     = F_W2,
    .wRank  = RNN_LOW_RANK,
    .U1     = F_U1,
    .U2     = F_U2,
    .uRank  = RNN_LOW_RANK,
    .Bg     = F_BIAS_GATE,
    .Bh     = F_BIAS_UPDATE,
    .sigmoid_zeta = sigmoid(F_ZETA),
    .sigmoid_nu   = sigmoid(F_NU),
    .block_size_u_from_lr = 100,
    .block_size_u_to_lr = 100,
    .block_size_w_from_lr = 100,
    .block_size_w_to_lr = 100,
  };

  float* pred = (float*)malloc(RNN_OUT_TIME * RNN_OUT_FEATURES * sizeof(float));

  forward_bricked_fastgrnn_lr(pred, RNN_OUT_FEATURES >> 1, INPUT,
    RNN_IN_TIME, RNN_IN_FEATURES, FWD_WINDOW, HOP,
    &fwd_RNN_params, 1, 1);

  backward_bricked_fastgrnn_lr(pred + (RNN_OUT_FEATURES >> 1), RNN_OUT_FEATURES >> 1, INPUT,
    RNN_IN_TIME, RNN_IN_FEATURES, BWD_WINDOW, HOP,
    &bwd_RNN_params, 1, 1);
  
  float error = 0;
  float denom = 0;
  for (int t = 0; t < RNN_OUT_TIME; t++) {
    for (int d = 0; d < RNN_OUT_FEATURES; d++) {
      error += ((pred[t * RNN_OUT_FEATURES + d] - OUTPUT[t * RNN_OUT_FEATURES + d]) 
                * (pred[t * RNN_OUT_FEATURES + d] - OUTPUT[t * RNN_OUT_FEATURES + d]));
      denom += OUTPUT[t * RNN_OUT_FEATURES + d] * OUTPUT[t * RNN_OUT_FEATURES + d];
    }
  }
  float avg_error = error / (RNN_OUT_TIME * RNN_OUT_FEATURES);
  float rmse = error / denom;
  
  #ifdef LOOP_UNROLL
    printf("Loop Unrolling Active\n");
  #endif
  printf("Testing Bricked RNNs Bi-Directional\n");
  printf("Agg Squared Error: %f ; MSE: %f ; RMSE: %f\n", error, avg_error, rmse);
  free(pred);
  return 0;
}
