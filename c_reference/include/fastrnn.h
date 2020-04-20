// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __RNN_H__
#define __RNN_H__

#include "utils.h"

#define ERR_PRECOMP_NOT_INIT -1
#define ERR_NORMFEATURES_NOT_INIT -4

/**
 * @brief Multi-step updates of a FastRNN cell
 * @param[in]       W            pointer to W
 * @param[in]       U            pointer to U
 * @param[in]       B            pointer to bias for update
 * @param[in]       alpha        weight parameter for current hidden state
 * @param[in]       beta         weight parameter for previous hidden state
 * @param[in]       steps        number of steps of FastRNN cell
 * @param[in,out]   hiddenState  pointer to initial hidden state and output hidden state
 * @param[in]       hiddenDims   dimension of hidden state of the FastRNN cell
 * @param[in]       input        pointer to concatenated input vectors for all steps, size inputDims*steps
 * @param[in]       inputDims    dimension of input vector for each step
 * @param[in]       mean         pointer to mean of input vector for normalization, size inputDims
 * @param[in]       stdDev       pointer to standard dev of input for normalization, size inputDims*steps
 * @param[in,out]   preComp      pointer to buffer space, must be initalized to atleast hiddenDims size
 * @param[in,out]   normFeatures pointer to buffer space, must be initalized to atleast inputDims size
 * @return     The function returns <code>0</code> on success
 *             <code>ERR_PRECOMP_NOT_INIT</code> if preComp not allocated
 *             <code>ERR_NORMFEAT_NOT_INIT</code> if normFeatures not allocated
*/
int FastRNN(const float* const W, const float* const U,
  const float* const B, const float alpha, 
  const float beta, const unsigned steps,
  float* const hiddenState, const unsigned hiddenDims,
  const float* const input, const int inputDims,
  const float* const mean, const float* const stdDev,
  float* preComp, float* normFeatures) {

  if (preComp == 0) return ERR_PRECOMP_NOT_INIT;
  if (normFeatures == 0) return ERR_NORMFEATURES_NOT_INIT;

  for (unsigned t = 0; t < steps; t++) {
    // Normalize the features
    v_add(1.0f, input + t * inputDims, -1.0f, mean + t * inputDims, inputDims, normFeatures);
    v_div(stdDev + t * inputDims, normFeatures, inputDims, normFeatures);

    // Process the new input and previous hidden state
    matVec(W, normFeatures, hiddenDims, inputDims, 0.0f, 1.0f, preComp);
    matVec(U, hiddenState, hiddenDims, hiddenDims, 1.0f, 1.0f, preComp);
    v_add(1.0f, preComp, 1.0f, B, hiddenDims, preComp);

    // Apply the gate to generate the new hidden state
    v_tanh(preComp, hiddenDims, preComp);
    v_add(alpha, preComp, beta, hiddenState, hiddenDims, hiddenState);
  }
  return 0;
}

#endif