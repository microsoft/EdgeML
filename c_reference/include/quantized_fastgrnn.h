// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __QUANTIZED_FASTGRNN_H__
#define __QUANTIZED_FASRGRNN_H__

#define ERR_PRECOMP_NOT_INIT -1
#define ERR_TEMPLRW_NOT_INIT -2
#define ERR_TEMPLRU_NOT_INIT -3
#define ERR_NORMFEATURES_NOT_INIT -4

#include "quantized_utils.h"

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
 * @var       sigmoid_zeta first weight parameter for update from input from next step
 * @var       sigmoid_nu   second weight parameter for update from input from next step
 */
typedef struct Q_FastGRNN_LR_Params {
  MYINT* mean;
  MYINT* stdDev;
  MYINT* W1; 
  MYINT* W2;
  MYITE wRank;
  MYINT* U1;
  MYINT* U2; 
  MYITE uRank;
  MYINT* Bg; 
  MYINT* Bh;
  MYINT sigmoid_zeta;
  MYINT sigmoid_nu;
} Q_FastGRNN_LR_Params;

/**
* @brief Buffers required for computation of low-rank FastGRNN
* @var   preComp      pointer to buffer space, must be initalized to atleast hiddenDims size
* @var   tempLRW      pointer to buffer space, must be initalized to atleast wRank size
* @var   tempLRU      pointer to buffer space, must be initalized to atleast uRank size
* @var   normFeatures pointer to buffer space, must be initalized to atleast inputDims size
*/
typedef struct Q_FastGRNN_LR_Buffers {
  MYINT* preComp;
  MYINT* tempLRW;
  MYINT* tempLRU;
  MYINT* normFeatures;
} Q_FastGRNN_LR_Buffers;

/**
 * @brief Multi-step updates of a FastGRNN cell with low rank W, U(W=W1*W2; U=U1*U2)
 * @param[in,out]   hiddenState  pointer to initial hidden state and output hidden state
 * @param[in]       hiddenDims   dimension of hidden state of the FastGRNN cell
 * @param[in]       input        pointer to concatenated input vectors for all steps, size inputDims*steps
 * @param[in]       inputDims    dimension of input vector for each step
 * @param[in]       steps        number of steps of FastGRNN cell
 * @param[in]       params       pointer to model parameter
 * @param[in]       buffers      pointer to buffer spaces
 * @param[in]       backward     direction of the pass, 0 for forward, 1 for backward
 * @param[in]       normalize    apply mean-var normalization, 0 for no, 1 for yes
 * @return     The function returns <code>0</code> on success
 *             <code>ERR_PRECOMP_NOT_INIT</code> if preComp not allocated
 *             <code>ERR_TEMPLRW_NOT_INIT</code> if tempLRW not allocated
 *             <code>ERR_TEMPLRU_NOT_INIT</code> if tempLRU not allocated
 *             <code>ERR_NORMFEAT_NOT_INIT</code> if normFeatures not allocated
*/
int q_fastgrnn_lr(MYINT* const hiddenState, MYITE hiddenDims,
                  const MYINT* const input, MYITE inputDims, MYITE steps,
                  const void* params, void* buffers, int backward, int normalize);

/**
 * @brief Model paramters for low-rank FastGRNN
 * @var       mean         pointer to mean of input vector for normalization, size inputDims
 * @var       stdDev       pointer to standard dev of input for normalization, size inputDims*steps
 * @var       W            pointer to W matrix
 * @var       U            pointer U matrix
 * @var       Bg           pointer to bias for sigmoid
 * @var       Bh           pointer to bias for tanh
 * @var       sigmoid_zeta first weight parameter for update from input from next step
 * @var       sigmoid_nu   second weight parameter for update from input from next step
 */
typedef struct Q_FastGRNN_Params {
  MYINT* mean;
  MYINT* stdDev;
  MYINT* W;
  MYINT* U;
  MYINT* Bg;
  MYINT* Bh;
  MYINT sigmoid_zeta;
  MYINT sigmoid_nu;
} Q_FastGRNN_Params;

/**
* @brief Buffers required for computation of FastGRNN
* @var   preComp      pointer to buffer space, must be initalized to atleast hiddenDims size
* @var   normFeatures pointer to buffer space, must be initalized to atleast inputDims size
*/
typedef struct Q_FastGRNN_Buffers {
  MYINT* preComp;
  MYINT* normFeatures;
} Q_FastGRNN_Buffers;

/**
 * @brief Multi-step updates of a FastGRNN cell
 * @param[in,out]   hiddenState  pointer to initial hidden state and output hidden state
 * @param[in]       hiddenDims   dimension of hidden state of the FastGRNN cell
 * @param[in]       input        pointer to concatenated input vectors for all steps, size inputDims*steps
 * @param[in]       inputDims    dimension of input vector for each step
 * @param[in]       steps        number of steps of FastGRNN cell
 * @param[in]       params       pointer to model parameter
 * @param[in]       buffers      pointer to buffer spaces
 * @param[in]       backward     direction of the pass, 0 for forward, 1 for backward
 * @param[in]       normalize    apply mean-var normalization, 0 for no, 1 for yes
 * @return     The function returns <code>0</code> on success
 *             <code>ERR_PRECOMP_NOT_INIT</code> if preComp not allocated
 *             <code>ERR_NORMFEAT_NOT_INIT</code> if normFeatures not allocated
*/
int q_fastgrnn(MYINT* const hiddenState, MYITE hiddenDims,
               const MYINT* const input, MYITE inputDims, MYITE steps,
               const void* params, void* buffers, int backward, int normalize);

#endif
