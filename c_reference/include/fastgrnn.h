// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __FASTGRNN_H__
#define __FASRGRNN_H__

#include "utils.h"

#define ERR_PRECOMP_NOT_INIT -1
#define ERR_TEMPLRW_NOT_INIT -2
#define ERR_TEMPLRU_NOT_INIT -3
#define ERR_NORMFEATURES_NOT_INIT -4

/**
 * @brief Model paramters for low-rank FastGRNN
 * @var       mean         pointer to mean of input vector for normalization, size inputDims
 * @var       stdDev       pointer to standard dev of input for normalization, size inputDims*steps
 * @var       W1           pointer to first low-rank component of W
 * @var       W2           pointer to second low-rank component of W
 * @var       wRank        rank of W matrix
 * @var       U1           pointer to first low-rank component of U
 * @var       U2           pointer to second low-rank component of U
 * @var       uRank        rank of U matrix
 * @var       Bg           pointer to bias for sigmoid
 * @var       Bh           pointer to bias for tanh
 * @var       zeta         first weight parameter for update from input from next step
 * @var       nu           second weight parameter for update from input from next step
 */
typedef struct FastGRNN_LR_Params {
  float* mean;
  float* stdDev;
  float* W1; 
  float* W2;
  unsigned wRank;
  float* U1;
  float* U2; 
  unsigned uRank;
  float* Bg; 
  float* Bh;
  float zeta;
  float nu;
} FastGRNN_LR_Params;

/**
* @brief Buffers required for computation of low-rank FastGRNN
* @var   preComp      pointer to buffer space, must be initalized to atleast hiddenDims size
* @var   tempLRW      pointer to buffer space, must be initalized to atleast wRank size
* @var   tempLRU      pointer to buffer space, must be initalized to atleast uRank size
* @var   normFeatures pointer to buffer space, must be initalized to atleast inputDims size
*/
typedef struct FastGRNN_LR_Buffers {
  float* preComp;
  float* tempLRW;
  float* tempLRU;
  float* normFeatures;
} FastGRNN_LR_Buffers;

/**
 * @brief Multi-step updates of a FastGRNN cell with low rank W, U(W=W1*W2; U=U1*U2)
 * @param[in,out]   hiddenState  pointer to initial hidden state and output hidden state
 * @param[in]       hiddenDims   dimension of hidden state of the FastGRNN cell
 * @param[in]       input        pointer to concatenated input vectors for all steps, size inputDims*steps
 * @param[in]       inputDims    dimension of input vector for each step
 * @param[in]       steps        number of steps of FastGRNN cell
 * @param[in]       params       pointer to model parameter
 * @param[in]       buffers      pointer to buffer spaces
 * @return     The function returns <code>0</code> on success
 *             <code>ERR_PRECOMP_NOT_INIT</code> if preComp not allocated
 *             <code>ERR_TEMPLRW_NOT_INIT</code> if tempLRW not allocated
 *             <code>ERR_TEMPLRU_NOT_INIT</code> if tempLRU not allocated
 *             <code>ERR_NORMFEAT_NOT_INIT</code> if normFeatures not allocated
*/
int fastgrnn_lr (float* const hiddenState, const unsigned hiddenDims,
  const float* const input, const int inputDims, const unsigned steps,
  const FastGRNN_LR_Params* params, FastGRNN_LR_Buffers * buffers) {

  if (buffers->preComp == 0) return ERR_PRECOMP_NOT_INIT;
  if (buffers->tempLRW == 0) return ERR_TEMPLRW_NOT_INIT;
  if (buffers->tempLRU == 0) return ERR_TEMPLRU_NOT_INIT;
  if (buffers->normFeatures == 0) return ERR_NORMFEATURES_NOT_INIT;

  // #steps iterations of the RNN cell starting from hiddenState
  for (unsigned t = 0; t < steps; t++) {
    // Normalize the features
    v_add(1.0f, input + t * inputDims, -1.0f, params->mean + t * inputDims,
      inputDims, buffers->normFeatures);
    v_div(params->stdDev + t * inputDims, buffers->normFeatures, inputDims, 
      buffers->normFeatures);

    // Process the new input and previous hidden state
    matVec(params->W1, buffers->normFeatures, params->wRank, inputDims,
      0.0f, 1.0f, buffers->tempLRW);
    matVec(params->W2, buffers->tempLRW, hiddenDims, params->wRank, 
      0.0f, 1.0f, buffers->preComp);
    matVec(params->U1, hiddenState, params->uRank, hiddenDims, 
      0.0f, 1.0f, buffers->tempLRU);
    matVec(params->U2, buffers->tempLRU, hiddenDims, params->uRank, 
      1.0f, 1.0f, buffers->preComp);

    // Apply the gate to generate the new hidden state
    for (unsigned i = 0; i < hiddenDims; i++) {
      float gate = sigmoid(buffers->preComp[i] + params->Bg[i]);
      float update = tanh(buffers->preComp[i] + params->Bh[i]);
      hiddenState[i] = gate * hiddenState[i] + (params->zeta * (1.0 - gate) + params->nu) * update;
    }
  }
  return 0;
}

/**
 * @brief Model paramters for low-rank FastGRNN
 * @var       mean         pointer to mean of input vector for normalization, size inputDims
 * @var       stdDev       pointer to standard dev of input for normalization, size inputDims*steps
 * @var       W            pointer to W matrix
 * @var       U            pointer U matrix
 * @var       Bg           pointer to bias for sigmoid
 * @var       Bh           pointer to bias for tanh
 * @var       zeta         first weight parameter for update from input from next step
 * @var       nu           second weight parameter for update from input from next step
 */
typedef struct FastGRNN_Params {
  float* mean;
  float* stdDev;
  float* W;
  float* U;
  float* Bg;
  float* Bh;
  float zeta;
  float nu;
} FastGRNN_Params;

/**
* @brief Buffers required for computation of FastGRNN
* @var   preComp      pointer to buffer space, must be initalized to atleast hiddenDims size
* @var   normFeatures pointer to buffer space, must be initalized to atleast inputDims size
*/
typedef struct FastGRNN_Buffers {
  float* preComp;
  float* normFeatures;
} FastGRNN_Buffers;

/**
 * @brief Multi-step updates of a FastGRNN cell
 * @param[in,out]   hiddenState  pointer to initial hidden state and output hidden state
 * @param[in]       hiddenDims   dimension of hidden state of the FastGRNN cell
 * @param[in]       input        pointer to concatenated input vectors for all steps, size inputDims*steps
 * @param[in]       inputDims    dimension of input vector for each step
 * @param[in]       steps        number of steps of FastGRNN cell
 * @param[in]       params       pointer to model parameter
 * @param[in]       buffers      pointer to buffer spaces
 * @return     The function returns <code>0</code> on success
 *             <code>ERR_PRECOMP_NOT_INIT</code> if preComp not allocated
 *             <code>ERR_NORMFEAT_NOT_INIT</code> if normFeatures not allocated
*/
int fatgrnn(float* const hiddenState, const unsigned hiddenDims,
  const float* const input, const int inputDims, const unsigned steps,
  const FastGRNN_Params* params, FastGRNN_Buffers* buffers) {

  if (buffers->preComp == 0) return ERR_PRECOMP_NOT_INIT;
  if (buffers->normFeatures == 0) return ERR_NORMFEATURES_NOT_INIT;

  for (unsigned t = 0; t < steps; t++) {
    // Normalize the features
    v_add(1.0f, input + t * inputDims, -1.0f, params->mean + t * inputDims,
      inputDims, buffers->normFeatures);
    v_div(params->stdDev + t * inputDims, buffers->normFeatures, inputDims,
      buffers->normFeatures);

    // Process the new input and previous hidden state
    matVec(params->W, buffers->normFeatures, hiddenDims, inputDims, 
      0.0f, 1.0f, buffers->preComp);
    matVec(params->U, hiddenState, hiddenDims, hiddenDims, 
      1.0f, 1.0f, buffers->preComp);

    // Apply the gate to generate the new hidden state
    for (unsigned i = 0; i < hiddenDims; i++) {
      float gate = sigmoid(buffers->preComp[i] + params->Bg[i]);
      float update = tanh(buffers->preComp[i] + params->Bh[i]);
      hiddenState[i] = gate * hiddenState[i] + (params->zeta * (1.0 - gate) + params->nu) * update;
    }
  }
  return 0;
}

#endif
