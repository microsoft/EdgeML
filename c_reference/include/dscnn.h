// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __DSCNN_H__
#define __DSCNN_H__

// Function pointer for the Conv layer to be passed as a parameter. (conv1d or conv1d_lr only)
typedef int (*conv_layer)(float*, unsigned, unsigned, const float*, 
                          unsigned, unsigned, unsigned, unsigned, 
                          const void*, unsigned, unsigned);

/**
 * @brief Model definition for the 1D Convolution block applied before the RNN
 * @brief sub-layers : batchnorm1d -> conv1d_lr
 * @param[out]   output_signal       pointer to the final output signal, minimum size = out_time * in_channels. out_time has to be calculated based on the reduction from all the conv and pool layers
 * @param[in]    input_signal        pointer to the input signal. size = in_time * in_channels
 * @param[in]    cnn                 function pointer for the CNN layer. (any of the conv layers can be passed with appropriate params)
 * @param[in]    in_time             number of time steps in the input_signal
 * @param[in]    in_channels         number of input channels
 * @param[in]    mean                pointer to the mean for the batch normalization, size = in_channels. Pass NULL/0 for affine_config = 2
 * @param[in]    var                 pointer to the variance for the batch normalization, size = in_channels. Pass NULL/0 for affine_config = 2
 * @param[in]    affine_config       whether the affine operations are applied
 *                                   if affine_config = 0, then only mean and var are used
 *                                   if affine_config = 1, then mean, var, gamma and beta are used for the final computation.
 *                                   if affine_config = 2, then only the gamma and beta are used. gamma = original_gamma/sqrt(var), beta = original_beta - gamma * mean/sqrt(var)
 *                                   Note: Use affine_config = 2 for faster calculations. The new gamma and beta would need to be pre-computed, stored and passed
 * @param[in]    gamma               pointer to the scaling factors for the post-norm affine operation, size = in_channels. Pass NULL/0 for affine_config = 0
 * @param[in]    beta                pointer to the offsets for the post-norm affine operation, size = in_channels. Pass NULL/0 for affine_config = 0
 * @param[in]    in_place            in-place computation check for the batchnorm. Storage efficient
 * @param[in]    cnn_hidden          hidden state/out_channels dimensions for the low-rank CNN. The final channel size of this block
 * @param[in]    cnn_padding         padding for the low-rank CNN layer. Note: applied to both sides of the input 
 * @param[in]    cnn_kernel_size     kernel size of the low-rank CNN
 * @param[in]    cnn_params          weights, bias and other essential parameters for the low-rank CNN
 * @param[in]    cnn_stride          stride factor for the low-rank CNN
 * @param[in]    cnn_activation      an integer to choose the type of activation function.
 *                                   0: none
 *                                   1: sigmoid
 *                                   2: tanh
 *                                   3: relu
 */
int phon_pred_lr_cnn(float* output_signal, float* input_signal,
  conv_layer cnn, unsigned in_time, unsigned in_channels,
  const float* const mean, const float* const var,
  unsigned affine_config, const float* const gamma, const float* const beta, unsigned in_place,
  unsigned cnn_hidden, unsigned cnn_padding, unsigned cnn_kernel_size,
  const void* cnn_params, unsigned cnn_stride, unsigned cnn_activation);

/**
 * @brief Model definition for the 1D Convolution block applied after the RNN
 * @brief sub-layers : custom nonlinearity(semi_sigmoid_tanh) -> batchnorm1d -> conv1d_depth -> conv1d_lr -> avgpool1d
 * @param[out]   output_signal          pointer to the final output signal, minimum size = out_time * in_channels. out_time has to be calculated based on the reduction from all the conv and pool layers
 * @param[in]    input_signal           pointer to the input signal. size = in_time * in_channels
 * @param[in]    point_cnn              function pointer for the point-wise CNN. (any of the conv layers can be passed with appropriate params)
 * @param[in]    in_time                number of time steps in the input
 * @param[in]    in_channels            number of input channels
 * @param[in]    mean                   pointer to the mean for the batch normalization, size = in_channels. Pass NULL/0 for affine_config = 2
 * @param[in]    var                    pointer to the variance for the batch normalization, size = in_channels. Pass NULL/0 for affine_config = 2
 * @param[in]    affine_config          whether the affine operations are applied
 *                                      if affine_config = 0, then only mean and var are used
 *                                      if affine_config = 1, then mean, var, gamma and beta are used for the final computation.
 *                                      if affine_config = 2, then only the gamma and beta are used. gamma = original_gamma/sqrt(var), beta = original_beta - gamma * mean/sqrt(var)
 *                                      Note: Use affine_config = 2 for faster calculations. The new gamma and beta would need to be pre-computed, stored and passed
 * @param[in]    gamma                  pointer to the scaling factors for the post-norm affine operation, size = in_channels. Pass NULL/0 for affine_config = 0
 * @param[in]    beta                   pointer to the offsets for the post-norm affine operation, size = in_channels. Pass NULL/0 for affine_config = 0
 * @param[in]    in_place               in-place computation of the batchnorm. Storage efficient
 * @param[in]    depth_cnn_padding      padding for the depth CNN layer. Note: applied to both sides of the input to the depth CNN
 * @param[in]    depth_cnn_kernel_size  kernel size of the depth CNN
 * @param[in]    depth_cnn_params       weights, bias and other essential parameters used to describe the depth CNN
 * @param[in]    depth_cnn_stride       stride factor for the depth CNN
 * @param[in]    depth_cnn_activation   an integer to choose the type of activation function.
 *                                      0: none
 *                                      1: sigmoid
 *                                      2: tanh
 *                                      3: relu
 * @param[in]    point_cnn_hidden       hidden state/out_channels dimensions for the point CNN. The final channel size of this block
 * @param[in]    point_cnn_padding      padding for the point CNN layer. Note: applied to both sides of the input to the point CNN
 * @param[in]    point_cnn_kernel_size  kernel size of the point CNN
 * @param[in]    point_cnn_params       weights, bias and other essential parameters used to describe the point CNN
 * @param[in]    point_cnn_stride       stride factor for the point CNN
 * @param[in]    point_cnn_activation   an integer to choose the type of activation function.
 *                                      0: none
 *                                      1: sigmoid
 *                                      2: tanh
 *                                      3: relu
 * @param[in]    pool_padding           padding for the pool layer. Note: applied to both sides of the input to the pool 
 * @param[in]    pool_kernel_size       kernel size of the pool
 * @param[in]    pool_stride            stride factor for the pool
 * @param[in]    pool_activation        an integer to choose the type of activation function.
 *                                      0: none
 *                                      1: sigmoid
 *                                      2: tanh
 *                                      3: relu
 */
int phon_pred_depth_point_lr_cnn(float* output_signal, float* input_signal,
  conv_layer point_cnn, unsigned in_time, unsigned in_channels,
  const float* const mean, const float* const var,
  unsigned affine_config, const float* const gamma, const float* const beta, unsigned in_place,
  unsigned depth_cnn_padding, unsigned depth_cnn_kernel_size,
  const void* depth_cnn_params, unsigned depth_cnn_stride, unsigned depth_cnn_activation,
  unsigned point_cnn_hidden, unsigned point_cnn_padding, unsigned point_cnn_kernel_size,
  const void* point_cnn_params, unsigned point_cnn_stride, unsigned point_cnn_activation,
  unsigned pool_padding, unsigned pool_kernel_size, unsigned pool_stride, unsigned pool_activation);

#endif
