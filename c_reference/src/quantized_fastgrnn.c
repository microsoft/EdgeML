// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "quantized_fastgrnn.h"

int q_fastgrnn_lr(MYINT* const hiddenState, MYITE hiddenDims,
                  const MYINT* const input, MYITE inputDims, MYITE steps,
                  const void* params, void* buffers, int backward,
                  int normalize) {
  const Q_FastGRNN_LR_Params* tparams = (const Q_FastGRNN_LR_Params*)params;
  Q_FastGRNN_LR_Buffers* tbuffers = (Q_FastGRNN_LR_Buffers*)buffers;

  if (tbuffers->preComp == 0) return ERR_PRECOMP_NOT_INIT;
  if (tbuffers->tempLRW == 0) return ERR_TEMPLRW_NOT_INIT;
  if (tbuffers->tempLRU == 0) return ERR_TEMPLRU_NOT_INIT;
  if (tbuffers->normFeatures == 0) return ERR_NORMFEATURES_NOT_INIT;

  // #steps iterations of the RNN cell starting from hiddenState
  for (MYITE t = 0; t < steps; t++) {
    // Normalize the features
    MYITE offset = backward ? steps - 1 - t : t;
    if (normalize) {
      // This diverges from the original implementation because of
      // impracticality of scaled addition beyond 0, 1, and -1 multipliers
      v_q_sub(input + offset * inputDims, tparams->mean + t * inputDims,
              inputDims, tbuffers->normFeatures, scale.input[offset],
              scale.mean[t], scale.normFeatures[t]);
      // Assuming the stdDev values are stored in inverse form
      v_q_hadamard(tparams->stdDev + t * inputDims, tbuffers->normFeatures,
                   inputDims, tbuffers->normFeatures, scale.input[offset],
                   scale.mean[t], scale.normFeatures[t]);
    }
    else {
      for (MYITE d = 0; d < inputDims; ++d)
        tbuffers->normFeatures[d] = input[offset * inputDims + d];
    }

    // Process the new input and previous hidden state
    m_q_mulvec(tparams->W1, tbuffers->normFeatures, tparams->wRank, inputDims,
               0, 1, tbuffers->tempLRW);
    m_q_mulvec(tparams->W2, tbuffers->tempLRW, hiddenDims, tparams->wRank,
               0, 1, tbuffers->preComp);
    m_q_mulvec(tparams->U1, hiddenState, tparams->uRank, hiddenDims,
               0, 1, tbuffers->tempLRU);
    m_q_mulvec(tparams->U2, tbuffers->tempLRU, hiddenDims, tparams->uRank,
               1, 1, tbuffers->preComp);

    // Apply the gate to generate the new hidden state
    for (MYITE i = 0; i < hiddenDims; i++) {
      MYINT gate = sigmoid(tbuffers->preComp[i] + tparams->Bg[i]);
      MYINT update = tanh(tbuffers->preComp[i] + tparams->Bh[i]);
      hiddenState[i] = gate * hiddenState[i] +
                       (tparams->sigmoid_zeta * (1 - gate) + tparams->sigmoid_nu)
                       * update;
    }
  }
  return 0;
}

int q_fastgrnn(MYINT* const hiddenState, MYITE hiddenDims,
               const MYINT* const input, MYITE inputDims, MYITE steps,
               const void* params, void* buffers, int backward, int normalize) {

  const Q_FastGRNN_Params* tparams = (const Q_FastGRNN_Params*)params;
  Q_FastGRNN_Buffers* tbuffers = (Q_FastGRNN_Buffers*)buffers;

  if (tbuffers->preComp == 0) return ERR_PRECOMP_NOT_INIT;
  if (tbuffers->normFeatures == 0) return ERR_NORMFEATURES_NOT_INIT;

  for (MYITE t = 0; t < steps; t++) {
    // Normalize the features
    MYITE offset = backward ? steps - 1 - t : t;
    if (normalize) {
      // This diverges from the original implementation because of
      // impracticality of scaled addition beyond 0, 1, and -1 multipliers
      v_q_sub(input + offset * inputDims, tparams->mean + t * inputDims,
              inputDims, tbuffers->normFeatures, scales->input[offset],
              scales->mean[t], scales->normFeatures[t]);
      // Assuming stdDev values are stored in inverse form
      v_q_hadamard(tparams->stdDev + t * inputDims, tbuffers->normFeatures,
                   inputDims, tbuffers->normFeatures, scales->input[offset],
                   scales->mean[t], scales->normFeatures[t]);
    }
    else {
      for (MYITE d = 0; d < inputDims; ++d)
        tbuffers->normFeatures[d] = input[offset * inputDims + d];
    }

    // Process the new input and previous hidden state
    m_q_mulvec(tparams->W, tbuffers->normFeatures, hiddenDims, inputDims,
               0, 1, tbuffers->preComp);
    m_q_mulvec(tparams->U, hiddenState, hiddenDims, hiddenDims,
               1, 1, tbuffers->preComp);

    // Apply the gate to generate the new hidden state
    for (MYITE i = 0; i < hiddenDims; i++) {
      MYINT gate = sigmoid(tbuffers->preComp[i] + tparams->Bg[i]);
      MYINT update = tanh(tbuffers->preComp[i] + tparams->Bh[i]);
      hiddenState[i] = gate * hiddenState[i] +
                       (tparams->sigmoid_zeta * (1 - gate) + tparams->sigmoid_nu)
                       * update;
    }
  }
  return 0;
}
