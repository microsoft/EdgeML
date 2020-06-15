// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __QUANTIZED_RNNPOOL_H__
#define __QUANTIZED_RNNPOOL_H__

#include "quantized_utils.h"

typedef int (*rnn_t)(MYINT* const, MYITE, const MYINT* const, MYITE, MYITE, const void*, void*, int, int);

/**
 * @param[in]        patch          pointer to activation of patch (row, col, channel)
 * @param[in]        inputDims      dimemsion of each input pixel
 * @param[in]        patchDim       number of rows and columns in a square patch
 * @param[in]        stride         stride lenghth in the larger image to get to next row
 * @param[in]        rnn1           function pointer to RNN1
 * @param[in]        hiddenDims1    dimension of the hidden state of RNN1
 * @param[in]        rnn1_params    pointer to parameters of RNN1
 * @param[in]        rnn1_buffers   pointer to buffers needed for RNN1
 * @param[in]        rnn2           function pointer to RNN2
 * @param[in]        hiddenDims2    dimension of the hidden state of RNN2
 * @param[in]        rnn2_params    pointer to parameters of RNN2
 * @param[in]        rnn2_buffers   pointer to buffers needed for RNN2
 * @param[out]       output         pointer to output, initialized to size 4 * hiddenDims2
 * @param[in,out]    buffer         pointer to buffer, intialized to size hiddenDims1 * max{nrows, cols}
 */
int q_rnnpool_block(const MYINT* const patch, MYITE inputDims, MYITE patchDim,
					MYITE stride, rnn_t rnn1, MYITE hiddenDims1,
					const void* rnn1_params, void* rnn1_buffers, rnn_t rnn2,
					MYITE hiddenDims2, const void* rnn2_params,
					void* rnn2_buffers, MYINT* const output, MYINT* const buffer);

#endif