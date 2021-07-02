// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __RNN_BRICKED_H__
#define __RNN_BRICKED_H__

/*  All the matrices are stored in the row major format
    
    NOTES for using the layers
->  Single-directional Computation
    While using the bricked fastgrnn layers, the user needs to adhered to the two following constraints
    1) in_time % hop = 0
    2) fwd_window % hop = 0 and bwd_window % hop = 0

    Violation of the above two constraints (1 & 2), will cause segmentation faults
    The layers first compute all the Wx steps and then compute Uh for all the windows parallelly
    Hence, the user needs to adhered to the constraints 1 & 2

->  Bi-directional Computation
    For bi-directional cases, there are 2 additionally constraints that would need to be followed
    A) sample_first_brick and sample_last_brick = 1
    B) An offset of rnn_hidden would need to be given to the output_signal pointer during the backward function call
        Each function will only process its given context(forward/backward). The other context will need to be called separately.
        E.g : 1st step -> forward(output, ..., input, ..., bi-direction=1, ...)
              2nd step -> backward(output + rnn_hidden, ..., input, ..., bi-direction=1, ...)

    The two extra constraints (A & B) are only for bi-directional cases and can be ignored if only forward (or only backward) is used
    Violating the conditions would cause index mis-matches or data corruption
    If the first (last) brick is not sampled, the first few (last few) time steps would be missing in the forward (backward) result 
    If the offset is not passed during the backward function call, the backward pass will overwrite the forward result (bi-directional case only)
*/

/**
 * @brief Model parameters for the 1D Convolution Layer
 * @var   W1                     pointer to first low-rank component of W. shape = [rank * in_dims]
 * @var   W2                     pointer to second low-rank component of W. shape = [rnn_hidden * rank]
 * @var   wRank                  rank of W matrix
 * @var   U1                     pointer to first low-rank component of U. shape = [rank * rnn_hidden]
 * @var   U2                     pointer to second low-rank component of U. shape = [rnn_hidden * rank]
 * @var   uRank                  rank of U matrix
 * @var   Bg                     pointer to bias for sigmoid
 * @var   Bh                     pointer to bias for tanh
 * @var   sigmoid_zeta           first weight parameter for update from input from next step
 * @var   sigmoid_nu             second weight parameter for update from input from next step
 * @var   block_size_w_to_lr     block/tile size for the cache. Used for tiled MatMul. For W1 * x
 * @var   block_size_w_from_lr   block/tile size for the cache. Used for tiled MatMul. For W2 * result(W1 * x) 
 * @var   block_size_u_to_lr     block/tile size for the cache. Used for tiled MatMul. For U1 * h
 * @var   block_size_u_from_lr   block/tile size for the cache. Used for tiled MatMul. For U2 * result(U1 * h)
 */
typedef struct BrickedFastGRNN_LR_Params {
  float* W1; 
  float* W2;
  unsigned wRank;
  float* U1;
  float* U2; 
  unsigned uRank;
  float* Bg; 
  float* Bh;
  float sigmoid_zeta;
  float sigmoid_nu;
  unsigned block_size_w_to_lr;
  unsigned block_size_w_from_lr;
  unsigned block_size_u_to_lr;
  unsigned block_size_u_from_lr;
} BrickedFastGRNN_LR_Params;

/** Forward Bricking and application of the forward RNN for an input signal
 * @param[out]       output_signal        pointer to output signal. size = out_time * rnn_hidden
 * @param[in]        rnn_hidden           output dimension for the current cell
 * @param[in]        input_signal         pointer to input signal. size = in_time * in_dims
 * @param[in]        in_time              number of input time steps.
 * @param[in]        in_dims              input dimensions
 * @param[in]        window               window length for each brick. For the final brick, the left over time steps are used(need not be window in length for the last brick) 
 * @param[in]        hop                  hop distance for between bricks
 * @param[in]        params               pointer to the parameters for the RNN
 * @param[in]        bi_direction         determine if the ouput if for a bi-directional RNN. 
 * @param[in]        sample_first_brick   determine if the 1st brick should also be sampled
 *                                        -> if = 0, only the last hidden state of each brick is sampled. out_time = (in_time-window)/hop + 1
 *                                        -> if = 1, for the 1st brick, we sample every hop index(similar to ::hop). For all the bricks(including the 1st) we sample the final hiddens state. out_time = in_time/hop + 1
 */
int forward_bricked_fastgrnn_lr(float* output_signal, unsigned rnn_hidden, 
  float* input_signal, unsigned in_time, unsigned in_dims, 
  unsigned window, unsigned hop, const void* params,
  unsigned bi_direction, unsigned sample_first_brick);

/** Backward Bricking and application of the backward RNN for an input signal
 * @param[out]       output_signal        pointer to output signal. size = out_time * rnn_hidden
 * @param[in]        rnn_hidden           output dimension for the current cell
 * @param[in]        input_signal         pointer to input signal. size = in_time * in_dims
 * @param[in]        in_time              number of input time steps.
 * @param[in]        in_dims              input dimensions
 * @param[in]        window               window length for each brick. For the final brick, the left over time steps are used(need not be window in length for the last brick)
 * @param[in]        hop                  hop distance for between bricks
 * @param[in]        params               pointer to the parameters for the RNN
 * @param[in]        bi_direction         determine if the ouput if for a bi-directional RNN. 
 * @param[in]        sample_last_brick    determine if the last brick should also be sampled
 *                                        -> if = 0, only the first(last in reverse) hidden state of each brick is sampled. out_time = (in_time-window)/hop + 1
 *                                        -> if = 1, for the last brick, we sample every hop index in reverse(similar to ::hop in reverse). For all the bricks(including the last) we sample the first hiddens state(last in reverse). out_time = in_time/hop + 1
 */
int backward_bricked_fastgrnn_lr(float* output_signal, unsigned rnn_hidden, 
  float* input_signal, unsigned in_time, unsigned in_dims, 
  unsigned window, unsigned hop, const void* params,
  unsigned bi_direction, unsigned sample_last_brick);

#endif
