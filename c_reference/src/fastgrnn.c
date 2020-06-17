// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"
#include "fastgrnn.h"

int fastgrnn_lr(float* const hiddenState, unsigned hiddenDims,
  const float* const input, unsigned inputDims, unsigned steps,
  const void* params, void* buffers, int backward, int normalize) {

  const FastGRNN_LR_Params* tparams = (const FastGRNN_LR_Params*)params;
  FastGRNN_LR_Buffers* tbuffers = (FastGRNN_LR_Buffers*)buffers;

  if (tbuffers->preComp == 0) return ERR_PRECOMP_NOT_INIT;
  if (tbuffers->tempLRW == 0) return ERR_TEMPLRW_NOT_INIT;
  if (tbuffers->tempLRU == 0) return ERR_TEMPLRU_NOT_INIT;
  if (tbuffers->normFeatures == 0) return ERR_NORMFEATURES_NOT_INIT;

  // #steps iterations of the RNN cell starting from hiddenState
  for (unsigned t = 0; t < steps; t++) {
    // Normalize the features
    unsigned offset = backward ? steps - 1 - t : t;
    if (normalize) {
      v_add(1.0f, input + offset * inputDims, -1.0f, tparams->mean + offset * inputDims,
        inputDims, tbuffers->normFeatures);
      v_div(tparams->stdDev + offset * inputDims, tbuffers->normFeatures, inputDims,
        tbuffers->normFeatures);
    }
    else {
      for (unsigned d = 0; d < inputDims; ++d)
        tbuffers->normFeatures[d] = input[offset * inputDims + d];
    }

    // Process the new input and previous hidden state
    matVec(tparams->W1, tbuffers->normFeatures, tparams->wRank, inputDims,
      0.0f, 1.0f, tbuffers->tempLRW);
    matVec(tparams->W2, tbuffers->tempLRW, hiddenDims, tparams->wRank,
      0.0f, 1.0f, tbuffers->preComp);
    matVec(tparams->U1, hiddenState, tparams->uRank, hiddenDims,
      0.0f, 1.0f, tbuffers->tempLRU);
    matVec(tparams->U2, tbuffers->tempLRU, hiddenDims, tparams->uRank,
      1.0f, 1.0f, tbuffers->preComp);

    // Apply the gate to generate the new hidden state
    for (unsigned i = 0; i < hiddenDims; i++) {
      float gate = sigmoid(tbuffers->preComp[i] + tparams->Bg[i]);
      float update = tanh(tbuffers->preComp[i] + tparams->Bh[i]);
      hiddenState[i] = gate * hiddenState[i] + (tparams->sigmoid_zeta * (1.0 - gate) + tparams->sigmoid_nu) * update;
    }
  }
  return 0;
}

int fastgrnn(float* const hiddenState, unsigned hiddenDims,
  const float* const input, unsigned inputDims, unsigned steps,
  const void* params, void* buffers, int backward, int normalize) {

  const FastGRNN_Params* tparams = (const FastGRNN_Params*)params;
  FastGRNN_Buffers* tbuffers = (FastGRNN_Buffers*)buffers;

  if (tbuffers->preComp == 0) return ERR_PRECOMP_NOT_INIT;
  if (tbuffers->normFeatures == 0) return ERR_NORMFEATURES_NOT_INIT;

  for (unsigned t = 0; t < steps; t++) {
    // Normalize the features
    unsigned offset = backward ? steps - 1 - t : t;
    if (normalize) {
      v_add(1.0f, input + offset * inputDims, -1.0f, tparams->mean + offset * inputDims,
        inputDims, tbuffers->normFeatures);
      v_div(tparams->stdDev + offset * inputDims, tbuffers->normFeatures, inputDims,
        tbuffers->normFeatures);
    }
    else {
      for (unsigned d = 0; d < inputDims; ++d)
        tbuffers->normFeatures[d] = input[offset * inputDims + d];
    }

    // Process the new input and previous hidden state
    matVec(tparams->W, tbuffers->normFeatures, hiddenDims, inputDims,
      0.0f, 1.0f, tbuffers->preComp);
    matVec(tparams->U, hiddenState, hiddenDims, hiddenDims,
      1.0f, 1.0f, tbuffers->preComp);


    // Apply the gate to generate the new hidden state
    for (unsigned i = 0; i < hiddenDims; i++) {
      float gate = sigmoid(tbuffers->preComp[i] + tparams->Bg[i]);
      float update = tanh(tbuffers->preComp[i] + tparams->Bh[i]);
      hiddenState[i] = gate * hiddenState[i] + (tparams->sigmoid_zeta * (1.0 - gate) + tparams->sigmoid_nu) * update;
    }
  }
  return 0;
}
