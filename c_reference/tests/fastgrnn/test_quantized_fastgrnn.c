// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stdio.h>
#include <string.h>

#include "quantized_fastgrnn.h"
#include "../rnnpool/q_wider_regression_model/rnn1.h"

const Q15_T patch[INPUT_CHANNELS] = {1040, 1919, 4254, 4024};
#ifdef SHIFT
  const Q15_T expected[HIDDEN_DIM1] = {1415, 7082, -16378, 8216, -12075, 6805, 6475, 6902};
#else
  const Q15_T expected[HIDDEN_DIM1] = {1423, 7085, -16378, 8209, -12067, 6805, 6475, 6897};
#endif

// Simple test for comparing Seedot's quantized FastGRNN output with our
// implementation. All values generated from Seedot on Wider Regression dataset.
// By default, all tests run without using bit-shifting operations.
int main() {
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
