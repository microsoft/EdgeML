// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <string.h>
#include "quantized_rnnpool.h"

int q_rnnpool_block(const MYINT* const patch, MYITE inputDims, MYITE patchDim,
                    MYITE stride, rnn_t rnn1, MYITE hiddenDims1,
                    const void* rnn1_params, void* rnn1_buffers, rnn_t rnn2,
                    MYITE hiddenDims2, const void* rnn2_params,
                    void* rnn2_buffers, MYINT* const output,
                    MYINT* const buffer) {
  // Clear the output
  memset(output, 0, sizeof(MYINT) * 4 * hiddenDims2);

  // Horizontal pass over each row with RNN1
  memset(buffer, 0, sizeof(MYINT) * hiddenDims1 * patchDim);
  for (MYITE r = 0; r < patchDim; ++r) {
    rnn1(buffer + r * hiddenDims1, hiddenDims1, patch + stride * r * inputDims,
         inputDims, patchDim, rnn1_params, rnn1_buffers, 0, 0);
  }

  // Bidirectional vertical pass over the row summaries
  rnn2(output, hiddenDims2, buffer, hiddenDims1, patchDim, rnn2_params,
       rnn2_buffers, 0, 0);
  rnn2(output + hiddenDims2, hiddenDims2, buffer, hiddenDims1, patchDim,
       rnn2_params, rnn2_buffers, 1, 0);

  // Vertical pass over each column with RNN1
  memset(buffer, 0, sizeof(MYINT) * hiddenDims1 * patchDim);
  for (MYITE c = 0; c < patchDim; ++c) {
    for (MYITE r = 0; r < patchDim; ++r) {
      rnn1(buffer + c * hiddenDims1, hiddenDims1,
           patch + (stride * r + c) * inputDims, inputDims, 1, rnn1_params,
           rnn1_buffers, 0, 0);
    }
  }

  // Bidirectional horizantal pass over the columns summaries
  rnn2(output + 2 * hiddenDims2, hiddenDims2, buffer, hiddenDims1, patchDim,
       rnn2_params, rnn2_buffers, 0, 0);
  rnn2(output + 3 * hiddenDims2, hiddenDims2, buffer, hiddenDims1, patchDim,
       rnn2_params, rnn2_buffers, 1, 0);

  return 0;
}