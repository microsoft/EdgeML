// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __QUANTIZED_MBCONV_H__
#define __QUANTIZED_MBCONV_H__

#include "quantized_utils.h"

/**
 * @brief Model parameters for Quantized MBConv Layer
 * Note: This implementation doesn't support dilations yet.
 * @param[in]        input          pointer to the input buffer
 * @param[in]        filter1        pointer to the first convolution filter buffer
 * @param[in]        BN1W           pointer to the buffer holding the multiplication factor of the first BatchNorm computation
 * @param[in]        BN1B           pointer to the buffer holding the additive factor of the first BatchNorm computation
 * @param[in]        filter2        pointer to the second convolution filter buffer
 * @param[in]        BN2W           pointer to the buffer holding the multiplication factor of the second BatchNorm computation
 * @param[in]        BN2B           pointer to the buffer holding the additive factor of the second BatchNorm computation
 * @param[in]        filter3        pointer to the third convolution filter buffer
 * @param[in]        BN3W           pointer to the buffer holding the multiplication factor of the third BatchNorm computation
 * @param[in]        BN3B           pointer to the buffer holding the additive factor of the third BatchNorm computation
 * @param[out]       output         pointer to the output buffer
 * @param[in]        convBuffer1    pointer to the buffer used for storing intermediate values for the first Convolution
 * @param[in]        convBuffer2    pointer to the buffer used for storing intermediate values for the second Convolution
 * @param[in]        treesumBuffer  pointer to the buffer used for storing intermediate values for TreeSum
 * @param[in]        N              number of batches passed to the layer
 * @param[in]        H              height of a single input tensor
 * @param[in]        W              width of a single input tensor
 * @param[in]        CIn            number of input channels in an input tensor
 * @param[in]        CTemp          number of channels in the intermediate convolution output
 * @param[in]        HF             height of a filter
 * @param[in]        WF             width of a filter
 * @param[in]        COut           number of channels in the final output
 * @param[in]        HOut           height of a single output tensor
 * @param[in]        WOut           width of a single output tensor
 * @param[in]        HPadU          pad to the top of the input tensor, along its height dimension
 * @param[in]        HPadD          pad to the bottom of the input tensor, along its height dimension
 * @param[in]        WPadL          pad to the left of the input tensor, along its width dimension
 * @param[in]        WPadR          pad to the right of the input tensor, along its width dimension
 * @param[in]        HStride        stride of the filter along the height dimension
 * @param[in]        WStride        stride of the filter along the height dimension
 * @param[in]        depth1         depth of the first TreeSum computation
 * @param[in]        depth2         depth of the second TreeSum computation
 * @param[in]        depth3         depth of the third TreeSum computation
 * @param[in]        limit1         maximum output value of the first relu_six computation
 * @param[in]        limit2         maximum output value of the first relu_six computation
 * @param[in]        shrU1          scale to divide the first TreeSum output by
 * @param[in]        shrB1          scale to divide the first BatchNorm addition factor by
 * @param[in]        shrX1          scale to divide the first Convolution output by
 * @param[in]        shrU2          scale to divide the second TreeSum output by
 * @param[in]        shrB2          scale to divide the second BatchNorm addition factor by
 * @param[in]        shrX2          scale to divide the second Convolution output by
 * @param[in]        shrU3          scale to divide the third TreeSum output by
 * @param[in]        shrB3          scale to divide the third BatchNorm addition factor by
 * @param[in]        shrW3          scale to divide the third Convolution output by
 * @param[in]        shlU1          scale to multiply with the first TreeSum output
 * @param[in]        shlB1          scale to multiply with the first BatchNorm addition factor
 * @param[in]        shlX1          scale to multiply with the first Convolution output
 * @param[in]        shlU2          scale to multiply with the second TreeSum output
 * @param[in]        shlB2          scale to multiply with the second BatchNorm addition factor
 * @param[in]        shlX2          scale to multiply with the second Convolution output
 * @param[in]        shlU3          scale to multiply with the third TreeSum output
 * @param[in]        shlB3          scale to multiply with the third BatchNorm addition factor
 * @param[in]        shlW3          scale to multiply with the third Convolution output
 * @return           none
 *
 * @brief The function computes the following three sub-parts:
 * 1) Convolution(input, filter1) -> Batch Normalization(BN1W, BN1B) -> ReLU(limit1) -> convBuffer1
 * 2) Depthwise Separable Convolution(convBuffer1, filter2) -> Batch Normalization(BN2W, BN2B) -> ReLU(limit2) -> convBuffer2
 * 3) Convolution(convBuffer2, filter3) -> Batch Normalization(BN3W, BN3B) -> output
 * Variables depth1, depth2 and depth3 are used along with treesumBuffer for accumulating the sums during convolutions.
 * Rest of the variables are used as indicated.
 *
 * @example          Please refer the file: c_reference/tests/mbconv/test_quantized_mbconv.c
 */
void q15_mbconv_block(const Q15_T* const input, const Q15_T* const filter1,
  const Q15_T* const BN1W, const Q15_T* const BN1B, const Q15_T* const filter2,
  const Q15_T* const BN2W, const Q15_T* const BN2B, const Q15_T* const filter3,
  const Q15_T* const BN3W, const Q15_T* const BN3B, Q15_T* const output,
  Q15_T* const convBuffer1, Q15_T* const convBuffer2, INTM_T* const treesumBuffer,
  ITER_T N, ITER_T H, ITER_T W, ITER_T CIn, ITER_T CTemp, ITER_T HF, ITER_T WF,
  ITER_T COut, ITER_T HOut, ITER_T WOut, S_ITER_T HPadU, S_ITER_T HPadD,
  S_ITER_T WPadL, S_ITER_T WPadR, ITER_T HStride, ITER_T WStride, SCALE_T depth1,
  SCALE_T depth2, SCALE_T depth3, INTM_T limit1, INTM_T limit2, L_SCALE_T shrU1,
  L_SCALE_T shrB1, L_SCALE_T shrX1, L_SCALE_T shrU2, L_SCALE_T shrB2,
  L_SCALE_T shrX2, L_SCALE_T shrU3, L_SCALE_T shrB3, L_SCALE_T shrW3,
  L_SCALE_T shlU1, L_SCALE_T shlB1, L_SCALE_T shlX1, L_SCALE_T shlU2,
  L_SCALE_T shlB2, L_SCALE_T shlX2, L_SCALE_T shlU3, L_SCALE_T shlB3,
  L_SCALE_T shlW3);

#endif
