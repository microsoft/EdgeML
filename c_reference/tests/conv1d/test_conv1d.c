// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stdio.h>
#include <stdlib.h>
#include "conv1d.h"
#include "utils.h"

#include "./conv1d_regular/conv_param.h"
#include "./conv1d_depthwise/conv_param_depth.h"
#include "./conv1d_lr/conv_param_lr.h"

// Error Check
void errorCheck(float* pred, float* label, unsigned out_time, int out_features) {
  float error = 0, denom = 0;
  for (unsigned t = 0; t < out_time; t++) {
    for (unsigned d = 0; d < out_features; d++) {
      error += ((pred[t * out_features + d] - label[t * out_features + d]) 
                * (pred[t * out_features + d] - label[t * out_features + d]));
      denom += label[t * out_features + d] * label[t * out_features + d];
    }
  }
  float avg_error = error / (out_time * out_features), rmse = error / denom;
  printf("Agg Squared Error: %f ; MSE: %f ; RMSE: %f\n", error, avg_error, rmse);
}

void conv1d_check() {
  ConvLayers_Params conv_params = {
    .W = CONV1D_CONV_WEIGHT,
    .B = CONV1D_CONV_BIAS,
    .depthwise = 0,
  };
  
  float* pred = (float*)malloc(CONV1D_OUT_TIME * CONV1D_OUT_FEATURES * sizeof(float));
  conv1d(pred, CONV1D_OUT_TIME, CONV1D_OUT_FEATURES, CONV1D_INPUT,
    CONV1D_IN_TIME, CONV1D_IN_FEATURES, CONV1D_PAD, CONV1D_FILT,
    &conv_params, CONV1D_STRIDE, CONV1D_ACT);

  printf("Testing Regular Convolution\n");
  errorCheck(pred, CONV1D_OUTPUT, CONV1D_OUT_TIME, CONV1D_OUT_FEATURES);
  free(pred);
}

void conv1d_parallel_check() {
  ConvLayers_Parallel_Params conv_params = {
    .W = CONV1D_CONV_WEIGHT,
    .B = CONV1D_CONV_BIAS,
    .block_size = 100,
  };
  
  float* pred = (float*)malloc(CONV1D_OUT_TIME * CONV1D_OUT_FEATURES * sizeof(float));
  conv1d_parallel(pred, CONV1D_OUT_TIME, CONV1D_OUT_FEATURES, CONV1D_INPUT,
    CONV1D_IN_TIME, CONV1D_IN_FEATURES, CONV1D_PAD, CONV1D_FILT,
    &conv_params, CONV1D_STRIDE, CONV1D_ACT);

  printf("Testing Parallel Convolution\n");
  errorCheck(pred, CONV1D_OUTPUT, CONV1D_OUT_TIME, CONV1D_OUT_FEATURES);
  free(pred);
}

void conv1d_depth_check() {
  ConvLayers_Params conv_params = {
    .W = CONV1D_DEPTH_CONV_WEIGHT,
    .B = CONV1D_DEPTH_CONV_BIAS,
    .depthwise = 1,
  };
  
  float* pred = (float*)malloc(CONV1D_DEPTH_OUT_TIME * CONV1D_DEPTH_OUT_FEATURES 
                                * sizeof(float));
  conv1d(pred, CONV1D_DEPTH_OUT_TIME, 0, CONV1D_DEPTH_INPUT,
    CONV1D_DEPTH_IN_TIME, CONV1D_DEPTH_IN_FEATURES, CONV1D_DEPTH_PAD, CONV1D_DEPTH_FILT,
    &conv_params, CONV1D_DEPTH_STRIDE, CONV1D_DEPTH_ACT);
  
  printf("Testing Depthwise Convolution\n");
  errorCheck(pred, CONV1D_DEPTH_OUTPUT, 
              CONV1D_DEPTH_OUT_TIME, CONV1D_DEPTH_OUT_FEATURES);
  free(pred);
}

void conv1d_lr_check() {
  ConvLayers_LR_Params conv_params = {
    .W1 = CONV1D_LR_CONV_W1,
    .W2 = CONV1D_LR_CONV_W2,
    .B = CONV1D_LR_CONV_BIAS,
    .rank = CONV1D_LR_LOW_RANK
  };
  
  float* pred = (float*)malloc(CONV1D_LR_OUT_TIME 
                                * CONV1D_LR_OUT_FEATURES * sizeof(float));
  conv1d_lr(pred, CONV1D_LR_OUT_TIME, CONV1D_LR_OUT_FEATURES, CONV1D_LR_INPUT,
    CONV1D_LR_IN_TIME, CONV1D_LR_IN_FEATURES, CONV1D_LR_PAD, CONV1D_LR_FILT,
    &conv_params, CONV1D_LR_STRIDE, CONV1D_LR_ACT);
  
  printf("Testing Low-Rank Convolution\n");
  errorCheck(pred, CONV1D_LR_OUTPUT, CONV1D_LR_OUT_TIME, CONV1D_LR_OUT_FEATURES);
  free(pred);
}

void conv1d_lr_parallel_check() {
  ConvLayers_LR_Parallel_Params conv_params = {
    .W1 = CONV1D_LR_CONV_W1,
    .W2 = CONV1D_LR_CONV_W2,
    .B = CONV1D_LR_CONV_BIAS,
    .rank = CONV1D_LR_LOW_RANK,
    .block_size_to_lr = 100,
    .block_size_from_lr = 100,
  };
  
  float* pred = (float*)malloc(CONV1D_LR_OUT_TIME 
                                * CONV1D_LR_OUT_FEATURES * sizeof(float));
  conv1d_lr_parallel(pred, CONV1D_LR_OUT_TIME, CONV1D_LR_OUT_FEATURES, CONV1D_LR_INPUT,
    CONV1D_LR_IN_TIME, CONV1D_LR_IN_FEATURES, CONV1D_LR_PAD, CONV1D_LR_FILT,
    &conv_params, CONV1D_LR_STRIDE, CONV1D_LR_ACT);
  
  printf("Testing Low-Rank Parallel Convolution\n");
  errorCheck(pred, CONV1D_LR_OUTPUT, CONV1D_LR_OUT_TIME, CONV1D_LR_OUT_FEATURES);
  free(pred);
}

int main() {
  #ifdef LOOP_UNROLL
    printf("Loop Unrolling Active\n");
  #endif
  conv1d_check();
  conv1d_parallel_check();
  conv1d_lr_check();
  conv1d_depth_check();
  conv1d_lr_parallel_check();
  return 0;
}
