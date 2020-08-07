// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stdio.h>
#include <string.h>

#include "quantized_fastgrnn.h"
#include "../rnnpool/q_wider_regression_model/rnn1.h"

// Simple test for comparing Seedot's quantized FastGRNN output with our
// implementation. All values generated from Seedot on Wider Regression dataset.
// By default, all tests run without using bit-shifting operations.
int main() {
  Q15_FastGRNN_Params rnn1_params = {
    .mean = NULL,
    .stdDev = NULL,
    .W = W1,
    .U = U1,
    .Bg = Bg1,
    .Bh = Bh1,
    .sigmoid_zeta = sigmoid_zeta1,
    .sigmoid_nu = sigmoid_nu1
  };

  Q15_T preComp11[HIDDEN_DIM1];
  Q15_T preComp12[HIDDEN_DIM1];
  Q15_T preComp13[HIDDEN_DIM1];
  Q15_T normFeatures1[INPUT_CHANNELS];
  memset(preComp11, 0, sizeof(Q15_T) * HIDDEN_DIM1);
  memset(preComp12, 0, sizeof(Q15_T) * HIDDEN_DIM1);
  memset(preComp13, 0, sizeof(Q15_T) * HIDDEN_DIM1);
  memset(normFeatures1, 0, sizeof(Q15_T) * INPUT_CHANNELS);

  Q15_FastGRNN_Buffers rnn1_buffers = {
    .preComp1 = preComp11,
    .preComp2 = preComp12,
    .preComp3 = preComp13,
    .normFeatures = normFeatures1
  };

  Q15_FastGRNN_Scales rnn1_scales = {
    .input = input1,
    .mean = meanScale1,
    .meanSub = meanSub1,
    .stdDev = stdDevScale1,
    .normFeaturesHDStdDev = normFeaturesHDStdDev1,
    .W = WScale1,
    .normFeaturesMVW = normFeaturesMVW1,
    .H1W = H1W1,
    .H2W = H2W1,
    .U = UScale1,
    .hiddenStateMVU = hiddenStateMVU1,
    .H1U = H1U1,
    .H2U = H2U1,
    .mV1AddMV2 = mV1AddMV21,
    .mV2AddMV1 = mV2AddMV11,
    .mV1AddMV2Out = mV1AddMV2Out1,
    .pC1AddBg = pC1AddBg1,
    .Bg = BgScale1,
    .pC1AddBgOut = pC1AddBgOut1,
    .sigmoidLimit = sigmoidLimit1,
    .sigmoidScaleIn = sigmoidScaleIn1,
    .sigmoidScaleOut = sigmoidScaleOut1,
    .pC1AddBh = pC1AddBh1,
    .Bh = BhScale1,
    .pC1AddBhOut = pC1AddBhOut1,
    .tanhScaleIn = tanhScaleIn1,
    .tanhScaleOut = tanhScaleOut1,
    .gateHDHiddenState = gateHDHiddenState1,
    .hiddenStateHDGate = hiddenStateHDGate1,
    .qOneScale = qOneScale1,
    .qOneSubGate = qOneSubGate1,
    .qOneSubGateOut = qOneSubGateOut1,
    .sigmoidZeta = sigmoidZetaScale1,
    .sigmoidZetaMulQOneSubGate = sigmoidZetaMulQOneSubGate1,
    .sigmoidNu = sigmoidNuScale1,
    .sigmoidNuAddQOneSubGate = sigmoidNuAddQOneSubGate1,
    .sigmoidNuAddQOneSubGateOut = sigmoidNuAddQOneSubGateOut1,
    .sigmoidNuAddQOneSubGateHDUpdate = sigmoidNuAddQOneSubGateHDUpdate1,
    .updateHDSigmoidNuAddQOneSubGate = updateHDSigmoidNuAddQOneSubGate1,
    .pC3AddPC1 = pC3AddPC11,
    .pC1AddPC3 = pC1AddPC31,
    .hiddenStateOut = hiddenStateOut1,
    .div = div1,
    .add = add1,
    .qOne = qOne1
  };

  const Q15_T patch[INPUT_CHANNELS] = {1040, 1919, 4254, 4024};
  const Q15_T expected[HIDDEN_DIM1] = {1423, 7085, -16378, 8209, -12067, 6805, 6475, 6897};

  Q15_T buffer[HIDDEN_DIM1];
  memset(buffer, 0, sizeof(Q15_T) * HIDDEN_DIM1);

  q15_fastgrnn(buffer, HIDDEN_DIM1, patch, INPUT_CHANNELS, 1,
               (const void*)(&rnn1_params), (void*)(&rnn1_buffers),
               (const void*)(&rnn1_scales), 0, 0);

  for (unsigned i = 0; i < HIDDEN_DIM1; i++){
    if (buffer[i] != expected[i]) {
      printf("Output: %d, Expected: %d at Index: %d\n", buffer[i], expected[i], i);
      printf("Quantized FastGRNN Test Failed!\n");
      return -1;
    }
  }

  printf("All Tests Passed!\n");
  return 0;
}
