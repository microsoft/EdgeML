// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __RNN_H__
#define __RNN_H__

#include "utils.h"

#define ERR_PRECOMP_NOT_INIT -1
#define ERR_TEMPLRW_NOT_INIT -2
#define ERR_TEMPLRU_NOT_INIT -3
#define ERR_NORMFEATURES_NOT_INIT -4
/**
 * @brief Multi-step updates of a FastGRNN cell with low rank W, U(W=W1*W2; U=U1*U2)
 * @param[in]       W1           pointer to first low-rank component of W
 * @param[in]       W2           pointer to second low-rank component of W
 * @param[in]       wRank        rank of W matrix 
 * @param[in]       U1           pointer to first low-rank component of U
 * @param[in]       U2           pointer to second low-rank component of U
 * @param[in]       wRank        rank of U matrix
 * @param[in]       Bg           pointer to bias for sigmoid
 * @param[in]       Bh           pointer to bias for tanh
 * @param[in]       zeta         first weight parameter for update from input from next step
 * @param[in]       nu           second weight parameter for update from input from next step
 * @param[in]       steps        number of steps of FastGRNN cell
 * @param[in,out]   hiddenState  pointer to initial hidden state and output hidden state
 * @param[in]       hiddenDims   dimension of hidden state of the FastGRNN cell
 * @param[in]       input        pointer to concatenated input vectors for all steps, size inputDims*steps 
 * @param[in]       inputDims    dimension of input vector for each step
 * @param[in]       mean         pointer to mean of input vector for normalization, size inputDims
 * @param[in]       stdDev       pointer to standard dev of input for normalization, size inputDims*steps
 * @param[in,out]   preComp      pointer to buffer space, must be initalized to atleast hiddenDims size
 * @param[in,out]   tempLRW      pointer to buffer space, must be initalized to atleast wRank size
 * @param[in,out]   tempLRU      pointer to buffer space, must be initalized to atleast uRank size
 * @param[in,out]   normFeatures pointer to buffer space, must be initalized to atleast inputDims size
 * @return     The function returns <code>0</code> on success
 *             <code>ERR_PRECOMP_NOT_INIT</code> if preComp not allocated
 *             <code>ERR_TEMPLRW_NOT_INIT</code> if tempLRW not allocated
 *             <code>ERR_TEMPLRU_NOT_INIT</code> if tempLRU not allocated
 *             <code>ERR_NORMFEAT_NOT_INIT</code> if normFeatures not allocated
*/
int FastGRNN_LR(const float* const W1, const float* const W2, const int wRank,
  const float* const U1, const float* const U2, const int uRank,
  const float* const Bg, const float* const Bh,
  const float zeta, const float nu, const unsigned steps,
  float* const hiddenState, const unsigned hiddenDims,
  const float* const input, const int inputDims,
  const float* const mean, const float* const stdDev,
  float* preComp, float* tempLRW, float* tempLRU, float* normFeatures) {

  if (preComp == 0) return ERR_PRECOMP_NOT_INIT;
  if (tempLRW == 0) return ERR_TEMPLRW_NOT_INIT;
  if (tempLRU == 0) return ERR_TEMPLRU_NOT_INIT;
  if (normFeatures == 0) return ERR_NORMFEATURES_NOT_INIT;

  // #steps iterations of the RNN cell starting from hiddenState
  for (unsigned t = 0; t < steps; t++) {
    // Normalize the features
    v_add(1.0f, input + t * inputDims, -1.0f, mean + t * inputDims, inputDims, normFeatures);
    v_div(stdDev + t * inputDims, normFeatures, inputDims, normFeatures);

    // Process the new input and previous hidden state
    matVec(W1, normFeatures, wRank, inputDims, 0.0f, 1.0f, tempLRW);
    matVec(W2, tempLRW, hiddenDims, wRank, 0.0f, 1.0f, preComp);
    matVec(U1, hiddenState, uRank, hiddenDims, 0.0f, 1.0f, tempLRU);
    matVec(U2, tempLRU, hiddenDims, uRank, 1.0f, 1.0f, preComp);

    // Apply the gate to generate the new hidden state
    for (int i = 0; i < hiddenDims; i++) {
      float gate = sigmoid(preComp[i] + Bg[i]);
      float update = tanh(preComp[i] + Bh[i]);
      hiddenState[i] = gate * hiddenState[i] + (zeta * (1.0 - gate) + nu) * update;
    }
  }
  return 0;
}


/**
 * @brief Multi-step updates of a FastGRNN cell
 * @param[in]       W            pointer to W
 * @param[in]       U            pointer to U
 * @param[in]       Bg           pointer to bias for sigmoid
 * @param[in]       Bh           pointer to bias for tanh
 * @param[in]       zeta         first weight parameter for update from input from next step
 * @param[in]       nu           second weight parameter for update from input from next step
 * @param[in]       steps        number of steps of FastGRNN cell
 * @param[in,out]   hiddenState  pointer to initial hidden state and output hidden state
 * @param[in]       hiddenDims   dimension of hidden state of the FastGRNN cell
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
int FastGRNN(const float* const W, const float* const U,
  const float* const Bg, const float* const Bh,
  const float zeta, const float nu, const unsigned steps,
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

    // Apply the gate to generate the new hidden state
    for (int i = 0; i < hiddenDims; i++) {
      float gate = sigmoid(preComp[i] + Bg[i]);
      float update = tanh(preComp[i] + Bh[i]);
      hiddenState[i] = gate * hiddenState[i] + (zeta * (1.0 - gate) + nu) * update;
    }
  }
  return 0;
}

#endif