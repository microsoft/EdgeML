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
 * @brief Model scales for different inputs. The naming convention follows
 * two basic rules:
 * 1) If the matrix for which the scale is associated is used only once, name
 * the scale after the matrix.
 * 2) If the matrix for which the scale is associated is used only once, name
 * the scale after the matrix, the operation it is being used under and the
 * matrix it is being operated with.
 */
typedef struct Q_FastGRNN_LR_Scales {
  MYSCL input;
  MYSCL mean;
  MYSCL meanSub;
  MYSCL stdDev;
  MYSCL normFeaturesHDStdDev;
  MYSCL W1;
  MYSCL normFeaturesMVW1;
  MYSCL H1W1;
  MYSCL H2W1;
  MYSCL W2;
  MYSCL tempLRW;
  MYSCL H1W2;
  MYSCL H2W2;
  MYSCL U1;
  MYSCL hiddenStateMVU1;
  MYSCL H1U1;
  MYSCL H2U1;
  MYSCL U2;
  MYSCL tempLRU;
  MYSCL H1U2;
  MYSCL H2U2;
  MYSCL mV2AddMV4;
  MYSCL mV4AddMV2;
  MYSCL mV2AddMV4Out;
  MYSCL pC1AddBg;
  MYSCL Bg;
  MYSCL pC1AddBgOut;
  MYSCL sigmoidScaleIn;
  MYSCL sigmoidScaleOut;
  MYSCL pC1AddBh;
  MYSCL Bh;
  MYSCL pC1AddBhOut;
  MYSCL tanhScaleIn;
  MYSCL tanhScaleOut;
  MYSCL gateHDHiddenState;
  MYSCL hiddenStateHDGate;
  MYSCL qOneScale;
  MYSCL qOneSubGate;
  MYSCL qOneSubGateOut;
  MYSCL sigmoidZeta;
  MYSCL sigmoidZetaMulQOneSubGate;
  MYSCL sigmoidNu;
  MYSCL sigmoidNuAddQOneSubGate;
  MYSCL sigmoidNuAddQOneSubGateOut;
  MYSCL sigmoidNuAddQOneSubGateHDUpdate;
  MYSCL updateHDSigmoidNuAddQOneSubGate;
  MYSCL pC3AddPC1;
  MYSCL pC1AddPC3;
  MYSCL hiddenStateOut;
  MYINT sigmoidLimit;
  MYINT div;
  MYINT add;
  MYINT qOne;
} Q_FastGRNN_LR_Scales;

/**
* @brief Buffers required for computation of low-rank FastGRNN
* @var   preComp1     pointer to buffer space, must be initalized to atleast hiddenDims size
* @var   preComp2     pointer to buffer space, must be initalized to atleast hiddenDims size
* @var   preComp3     pointer to buffer space, must be initalized to atleast hiddenDims size
* @var   tempLRW      pointer to buffer space, must be initalized to atleast wRank size
* @var   tempLRU      pointer to buffer space, must be initalized to atleast uRank size
* @var   normFeatures pointer to buffer space, must be initalized to atleast inputDims size
*/
typedef struct Q_FastGRNN_LR_Buffers {
  MYINT* preComp1;
  MYINT* preComp2;
  MYINT* preComp3;
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
 * @param[in]       scales       pointer to model scales
 * @param[in]       backward     direction of the pass, 0 for forward, 1 for backward
 * @param[in]       normalize    apply mean-var normalization, 0 for no, 1 for yes
 * @return     The function returns <code>0</code> on success
 *             <code>ERR_PRECOMP_NOT_INIT</code> if preComp1 not allocated
 *             <code>ERR_PRECOMP_NOT_INIT</code> if preComp2 not allocated
 *             <code>ERR_PRECOMP_NOT_INIT</code> if preComp3 not allocated
 *             <code>ERR_TEMPLRW_NOT_INIT</code> if tempLRW not allocated
 *             <code>ERR_TEMPLRU_NOT_INIT</code> if tempLRU not allocated
 *             <code>ERR_NORMFEAT_NOT_INIT</code> if normFeatures not allocated
*/
int q_fastgrnn_lr(MYINT* const hiddenState, MYITE hiddenDims,
                  const MYINT* const input, MYITE inputDims, MYITE steps,
                  const void* params, void* buffers, const void* scales,
                  int backward, int normalize);

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
 * @brief Model scales for different inputs. The naming convention follows
 * two basic rules:
 * 1) If the matrix for which the scale is associated is used only once, name
 * the scale after the matrix.
 * 2) If the matrix for which the scale is associated is used only once, name
 * the scale after the matrix, the operation it is being used under and the
 * matrix it is being operated with.
 */
typedef struct Q_FastGRNN_Scales {
  MYSCL input;
  MYSCL mean;
  MYSCL meanSub;
  MYSCL stdDev;
  MYSCL normFeaturesHDStdDev;
  MYSCL W;
  MYSCL normFeaturesMVW;
  MYSCL H1W;
  MYSCL H2W;
  MYSCL U;
  MYSCL hiddenStateMVU;
  MYSCL H1U;
  MYSCL H2U;
  MYSCL mV1AddMV2;
  MYSCL mV2AddMV1;
  MYSCL mV1AddMV2Out;
  MYSCL pC1AddBg;
  MYSCL Bg;
  MYSCL pC1AddBgOut;
  MYSCL sigmoidScaleIn;
  MYSCL sigmoidScaleOut;
  MYSCL pC1AddBh;
  MYSCL Bh;
  MYSCL pC1AddBhOut;
  MYSCL tanhScaleIn;
  MYSCL tanhScaleOut;
  MYSCL gateHDHiddenState;
  MYSCL hiddenStateHDGate;
  MYSCL qOneScale;
  MYSCL qOneSubGate;
  MYSCL qOneSubGateOut;
  MYSCL sigmoidZeta;
  MYSCL sigmoidZetaMulQOneSubGate;
  MYSCL sigmoidNu;
  MYSCL sigmoidNuAddQOneSubGate;
  MYSCL sigmoidNuAddQOneSubGateOut;
  MYSCL sigmoidNuAddQOneSubGateHDUpdate;
  MYSCL updateHDSigmoidNuAddQOneSubGate;
  MYSCL pC3AddPC1;
  MYSCL pC1AddPC3;
  MYSCL hiddenStateOut;
  MYINT div;
  MYINT add;
  MYINT sigmoidLimit;
  MYINT qOne;
} Q_FastGRNN_Scales;

/**
* @brief Buffers required for computation of FastGRNN
* @var   preComp1     pointer to buffer space, must be initalized to atleast hiddenDims size
* @var   preComp2     pointer to buffer space, must be initalized to atleast hiddenDims size
* @var   preComp3     pointer to buffer space, must be initalized to atleast hiddenDims size
* @var   normFeatures pointer to buffer space, must be initalized to atleast inputDims size
*/
typedef struct Q_FastGRNN_Buffers {
  MYINT* preComp1;
  MYINT* preComp2;
  MYINT* preComp3;
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
 * @param[in]       scales       pointer to model scales
 * @param[in]       backward     direction of the pass, 0 for forward, 1 for backward
 * @param[in]       normalize    apply mean-var normalization, 0 for no, 1 for yes
 * @return     The function returns <code>0</code> on success
 *             <code>ERR_PRECOMP_NOT_INIT</code> if preComp1 not allocated
 *             <code>ERR_PRECOMP_NOT_INIT</code> if preComp2 not allocated
 *             <code>ERR_PRECOMP_NOT_INIT</code> if preComp3 not allocated
 *             <code>ERR_NORMFEAT_NOT_INIT</code> if normFeatures not allocated
*/
int q_fastgrnn(MYINT* const hiddenState, MYITE hiddenDims,
               const MYINT* const input, MYITE inputDims, MYITE steps,
               const void* params, void* buffers, const void* scales,
               int backward, int normalize);

#endif
