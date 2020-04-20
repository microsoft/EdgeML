// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __RNN_H__
#define __RNN_H__

#include "utils.h"

#define ERR_PRECELLSTATE_NOT_INIT -1
#define ERR_FORGETGATE_NOT_INIT -2
#define ERR_INPUTGATE_NOT_INIT -3
#define ERR_OUTPUTGATE_NOT_INIT -4
#define ERR_NORMFEATURES_NOT_INIT -5

/**
 * @brief Multi-step updates of a FastRNN cell
 * @param[in]       W1           pointer to W1
 * @param[in]       U1           pointer to U1
 * @param[in]       W2           pointer to W2
 * @param[in]       U2           pointer to U2
 * @param[in]       W3           pointer to W3
 * @param[in]       U3           pointer to U3
 * @param[in]       W4           pointer to W4
 * @param[in]       U4           pointer to U4
 * @param[in]       Bf           pointer to bias for forget gate
 * @param[in]       Bi           pointer to bias for input gate
 * @param[in]       Bc           pointer to bias for cell state
 * @param[in]       Bo           pointer to bias for output
 * @param[in]       alpha        weight parameter for current hidden state
 * @param[in]       beta         weight parameter for previous hidden state
 * @param[in]       steps        number of steps of FastRNN cell
 * @param[in,out]   cellState    pointer to initial cell state and cell hidden state
 * @param[in,out]   hiddenState  pointer to initial hidden state and output hidden state
 * @param[in]       hiddenDims   dimension of hidden state of the FastRNN cell
 * @param[in]       input        pointer to concatenated input vectors for all steps, size inputDims*steps
 * @param[in]       inputDims    dimension of input vector for each step
 * @param[in]       mean         pointer to mean of input vector for normalization, size inputDims
 * @param[in]       stdDev       pointer to standard dev of input for normalization, size inputDims*steps
 * @param[in,out]   preCellState pointer to intermediate cell state buffer space, must be initalized to atleast hiddenDims size
 * @param[in,out]   forgetGate   pointer to forget gate buffer space, must be initalized to atleast hiddenDims size
 * @param[in,out]   inputGate    pointer to input gate buffer space, must be initalized to atleast hiddenDims size
 * @param[in,out]   outputGate   pointer to output gate buffer space, must be initalized to atleast hiddenDims size
 * @param[in,out]   normFeatures pointer to buffer space, must be initalized to atleast inputDims size
 * @return     The function returns <code>0</code> on success
 *             <code>ERR_PRECOMP_NOT_INIT</code> if preComp not allocated
 *             <code>ERR_NORMFEAT_NOT_INIT</code> if normFeatures not allocated
*/
int LSTM(const float* const W1, const float* const U1, 
  const float* const W2, const float* const U2,
  const float* const W3, const float* const U3,
  const float* const W4, const float* const U4,
  const float* const Bf, const float* const Bi,
  const float* const Bc, const float* const Bo, 
  const unsigned steps, float* const hiddenState, 
  float* cellState, const unsigned hiddenDims, 
  const float* const input, const int inputDims, 
  const float* const mean, const float* const stdDev,  
  float* forgetGate, float* inputGate, float* outputGate,
  float* preCellState, float* normFeatures) {

  if (preCellState == 0) return ERR_PRECELLSTATE_NOT_INIT;
  if (forgetGate == 0) return ERR_FORGETGATE_NOT_INIT;
  if (inputGate == 0) return ERR_INPUTGATE_NOT_INIT;
  if (outputGate == 0) return ERR_OUTPUTGATE_NOT_INIT;
  if (normFeatures == 0) return ERR_NORMFEATURES_NOT_INIT;

  for (unsigned t = 0; t < steps; t++) {
    // Normalize the features
    v_add(1.0f, input + t * inputDims, -1.0f, mean + t * inputDims, inputDims, normFeatures);
    v_div(stdDev + t * inputDims, normFeatures, inputDims, normFeatures);

    // Forget gate
    matVec(W1, normFeatures, hiddenDims, inputDims, 0.0f, 1.0f, forgetGate);
    matVec(U1, hiddenState, hiddenDims, hiddenDims, 1.0f, 1.0f, forgetGate);
    v_add(1.0f, forgetGate, 1.0f, Bf, hiddenDims, forgetGate);
    v_tanh(forgetGate, hiddenDims, forgetGate);

    // Input gate
    matVec(W2, normFeatures, hiddenDims, inputDims, 0.0f, 1.0f, inputGate);
    matVec(U2, hiddenState, hiddenDims, hiddenDims, 1.0f, 1.0f, inputGate);
    v_add(1.0f, inputGate, 1.0f, Bi, hiddenDims, inputGate);
    v_tanh(inputGate, hiddenDims, inputGate);

    // Cell state update
    matVec(W3, normFeatures, hiddenDims, inputDims, 0.0f, 1.0f, preCellState);
    matVec(U3, hiddenState, hiddenDims, hiddenDims, 1.0f, 1.0f, preCellState);
    v_add(1.0f, preCellState, 1.0f, Bc, hiddenDims, preCellState);
    v_sigmoid(preCellState, hiddenDims, preCellState);
    
    // Output gate
    matVec(W4, normFeatures, hiddenDims, inputDims, 0.0f, 1.0f, outputGate);
    matVec(U4, hiddenState, hiddenDims, hiddenDims, 1.0f, 1.0f, outputGate);
    v_add(1.0f, outputGate, 1.0f, Bo, hiddenDims, outputGate);
    v_tanh(outputGate, hiddenDims, outputGate);

    // Apply the gates to generate the new hidden state and cell state
    for (int i = 0; i < hiddenDims; i++) {
      cellState[i] = forgetGate[i] * cellState[i] + inputGate[i] * preCellState[i];
      hiddenState[i] = outputGate[i] * sigmoid(cellState[i]);
    }
  }
  return 0;
}

#endif